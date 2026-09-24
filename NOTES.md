# NOTES — environment, runbooks, operations

Practical material for running this project on the remote training box. Results and
methods live in `PROGRESS.md`; the milestone plan in `PLAN.md`.

## Environment

- conda env **`robodiff`**: torch **2.8.0+cu128** (sm_120), robosuite 1.2.0, robomimic
  0.2.0, mujoco_py 2.0.2.13, numcodecs 0.10.2, wandb 0.15.12.
  **Do not recreate the env from `conda_environment.yaml`** — its `pytorch=1.12.1` pin is
  stale and unusable on the RTX 5090s. The live env was upgraded in place.
- `sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`, or robosuite
  fails to import. Offscreen rendering works with osmesa, EGL, and the default backend.
- **Camera control** (no `CameraMover` in robosuite 1.2): `sim.model.cam_pos/cam_quat`
  plus `sim.forward()`. `cam_quat` is **wxyz**; `robosuite.utils.transform_utils` is
  **xyzw** — convert explicitly. `hard_reset=False` (set by the stock runner) makes camera
  moves persist across soft resets, so the harness re-applies the viewpoint at every reset
  from a base pose captured on first use — it never compounds.
- **The zarr cache (`<hdf5>.zarr.zip`) is keyed only by hdf5 path**, not by `shape_meta`.
  A single-view run caches one camera; a later two-camera run on the same path `KeyError`s
  on the wrist key. Delete the cache to switch.
- Dropping the wrist camera from `shape_meta` removes encoder FLOPs but **not** render
  cost: `camera_names` comes from the hdf5 `env_args`, so robosuite still renders both
  cameras every step.
- **`numcodecs 0.10.2` has no `jpeg2k`**, so anything opening the multi-view zarrs must
  import `diffusion_policy.dataset.multiview_image_dataset` first (`register_codecs()`).
- `wandb` is unattended-safe here via `~/.netrc` — proven with detached runs reaching
  `logging synced files` with no TTY.

## Runbook

All commands run from the repo root. **201 epochs, not 200**: checkpoint and rollout fire
on `epoch % 50 == 0` and there is no save at the end of training, so 201 makes epoch 200
fire (200 would stop at 150).

### Training

```bash
# M1 single-view baseline
python train.py --config-dir=. --config-name=train_diffusion_unet_image_workspace.yaml \
  task=<task>_image_abs_single training.seed=42 training.device=cuda:0 \
  training.num_epochs=201 dataloader.num_workers=8 checkpoint.topk.k=1 \
  logging.project=diffusion_policy_view \
  hydra.run.dir=data/outputs/run_<task>_abs_single_s42_200ep

# L1 view diversity (7-view pool, M1's architecture)
#   task=randview_image_abs_multiview  training.rollout_every=50  dataloader.num_workers=10

# M3 view-conditioned encoder (flags select the 2x2 cell; both false = m3off)
python train.py --config-name=train_diffusion_unet_image_workspace_m3 \
  task=m3_plucker_image_abs_multiview task.task_name=square \
  policy.obs_encoder.use_plucker=false policy.obs_encoder.use_eef_hist=false \
  training.seed=42 training.num_epochs=201 training.device=cuda:0 \
  dataloader.num_workers=14 val_dataloader.num_workers=2 checkpoint.topk.k=1 \
  logging.project=diffusion_policy_view \
  hydra.run.dir=data/outputs/run_square_m3off_s42_200ep
#   N=1 fidelity gate: task=m3_plucker_image_abs_n1
#   single-view cell:  task.dataset.view_count_range=[1,1]

# M4 aux heads (control: policy.aux_loss_weight=0.0)
python train.py --config-name=train_diffusion_unet_image_workspace_m4 \
  task=m4_aux_image_abs_multiview task.task_name=square policy.aux_loss_weight=1.0 \
  training.seed=42 training.num_epochs=201 training.device=cuda:0 \
  dataloader.num_workers=10 val_dataloader.num_workers=2 checkpoint.topk.k=1 \
  logging.project=diffusion_policy_view \
  hydra.run.dir=data/outputs/run_square_m4on_s42_200ep
```

Multi-seed on a multi-GPU box: `ray start --head --num-gpus=3` then
`ray_train_multirun.py --config-dir=. --config-name=<cfg> --seeds=42,43,44
--monitor_key=test/mean_score -- multi_run.run_dir='...' multi_run.wandb_name_base='...'`.

### View count (N per sample)

`view_count_range` is drawn in **one place**, `multiview_image_dataset.py` `_m3_slots`:
`k_active = np.random.randint(lo, hi + 1)` (uniform over the range, inclusive) then
`np.random.choice(view_pool, size=k_active, replace=False)` (uniform over *subsets*, and the
order within a draw is random too, which is what stops the encoder keying off slot identity).
Redrawn every `__getitem__`, so N and the subset move each epoch. Active slots are always a
prefix (`0..k_active-1`); the rest are zero with mask 0.

- **A guard that rejected `[1, 2]` was removed on 2026-09-24.** It raised for any
  non-degenerate range whose `hi` was below the slot count — so `[1, 2]` at K=7 failed while
  `[2, 2]` was allowed, though both strand the same slots. Redundant (`n_slots <= len(view_pool)`
  plus `hi <= n_slots` already give `hi <= len(view_pool)`, all `np.random.choice(..., replace=False)`
  needs) and inert for every committed cell (`[1, 7]` has `hi == n_slots`, `[1, 1]` has
  `lo == hi`). Pinned by `tests/test_view_conditioned_obs_encoder.py::test_view_count_range_is_honoured`,
  which cannot construct its first dataset against the old code. **Legal ranges now: any
  `1 <= lo <= hi <= n_slots`.**
- **The view draws are NOT the global numpy RNG leaking across workers — do not "fix" this.**
  An earlier note in this project claimed torch does not re-seed numpy per worker, so every
  worker replays one draw stream. That is **false for this torch version**. Torch seeds it
  explicitly, in `torch/utils/data/_utils/worker.py`:
  `np_seed = _generate_state(base_seed, worker_id); np.random.seed(np_seed)`, alongside
  `random.seed(seed)` / `torch.manual_seed(seed)`. `base_seed` is redrawn for every new
  iterator, and `persistent_workers: False` means a new iterator each epoch — so draw streams
  differ **across workers and across epochs**. Adding a `worker_init_fn`, a `generator=`, or
  `persistent_workers=True` would *change* the stream and break exact comparability with every
  committed cell, for no benefit.
- **Cost model: encoder cost tracks the MEAN active view count, not K.** Masked slots are
  skipped in the encode loop (`view_conditioned_obs_encoder.py`), so K only sets the loop
  length; parameters are K-independent. Measured ≈ **7.3 s/epoch per mean active view + ~14 s
  fixed** — anchors: `[1,1]` (mean 1.0) 21 s/epoch, `[1,7]` (mean 4.0) 43 s/epoch, `[1,2]`
  (mean 1.5) **24 s/epoch observed**. Use it as a launch gate: if a rung's epoch time matches
  a different range's prediction, the override did not take.
- **N is drawn per sample, so the eval N is a design decision, not a detail.** Every
  evaluation here is **N=1** (`eval_novel_view.py::_serve_m3` puts the single live camera in
  slot 0 and zeroes the rest). At N=1 the fusion softmax is over one unmasked key, so the
  learnable query has *no effect* — that path is a degenerate corner of the module. M3
  randomises N precisely so that corner stays in-distribution; a cell trained at `[2, 2]` or
  `[k, k]` forfeits that, and a floor from it cannot be read as a diversity statement.

### Evaluation

```bash
# novel-view sweep (M3-era flags; omit --m3-slots for M1/L1)
python eval_novel_view.py -c <run>/checkpoints/latest.ckpt -o data/eval_interp_square_m3off \
  -d cuda:0 --preset azimuth_interp --m3-slots 7 --eef-hist-steps 4 --n-envs 14 --n-test-vis 0
#   presets: azimuth_sweep5 / azimuth_interp / azimuth_sweep3 / elevation_az0
#   fast smoke: --preset azimuth_sweep3 --n-test 4 --n-test-vis 2 --n-envs 4

python summarize_novel_view.py <dir>/eval_log.json     # degradation table
python eval.py -c <run>/checkpoints/latest.ckpt -o data/eval_orig -d cuda:0   # sanity
```

`--m3-slots` is **7 for all M3 cells including `m3off`** — the assert compares against the
checkpoint's own `shape_meta`, and the cam/mask/history normalizer entries exist
regardless of the flags. `eval_log.json` is a flat dict keyed
`test/<viewpoint>/{mean_score,success_rate,sim_max_reward_<seed>}`, so any number can be
re-derived without re-running. `test_start_seed: 100000` in the env_runner config is what
makes the episodes **paired** across viewpoints.

### Relational probe, step 1a (`probe_relpose.py`)

Measures what geometry a **frozen** checkpoint's latents already encode -- no training:
absolute camera pose and camera-frame EE position from one view's `z_v`, relative pose
from a view **pair**, and how stable `z_g` is across view subsets of the same state. It
reads the checkpoint's own cfg, so it adapts to `use_plucker` / `use_eef_hist`.

```bash
# 0. step 1a is committed (87d1708). This checkout IS the box -- verified 2026-09-24: 2x
#    RTX 5090, the run_square_m3*/checkpoints and data/multiview/square_ph_ring13.zarr all
#    present locally -- so the runbook below runs in place. From another machine only:
#    git push <remote> main && ssh <box> 'cd <repo> && git pull'

# 0b. do the checkpoints still exist? disk has deleted weights here before (M1's went
#     on 2026-09-19), so never assume the ones a runbook names are on the box
ls -la data/outputs/run_square_m3*/checkpoints/ data/multiview/square_ph_ring13.zarr

# 1. CPU checks first -- catches convention errors before trusting any number
python tests/test_relpose_probe.py                  # seconds, no GPU, no data

# 2. smoke: does the wiring work with the real zarr + checkpoint? MUST pass --mlp-steps,
#    or the decisive column is skipped (see Traps -- the smoke passed while fit_mlp was
#    broken). At this n the rel_pose numbers are underdetermined and mean nothing.
python probe_relpose.py -c data/outputs/run_square_m3on_s42_200ep/checkpoints/latest.ckpt \
  -o /tmp/probe_smoke -d cuda:1 --n-samples 64 --stability-states 8 --mlp-steps 200

# 3. the pair that matters: rays on vs rays off (~50 s each). Runs 2026-09-24; results in
#    PROGRESS.md *Relational probe (step 1a)*.
for c in m3on m3off; do
python -u probe_relpose.py -c data/outputs/run_square_${c}_s42_200ep/checkpoints/latest.ckpt \
  -o data/probe_relpose_square_${c} -d cuda:1 --mlp-steps 2000
done
# m4on/m4off work too (same encoder, plus an inert aux head) -- optional third cell
# -d cuda:1 unless cuda:0 is free -- another user habitually holds ~13 GB there
```

Cost: ~2000 dataset reads + one encoder pass each (~30 s), ridge fits (a few seconds), the
MLP ~1-2 min. Feature matrices are ~200 MB at the defaults.

**Check before believing any of it**

- `shuffled_target` must be **worse** than `ridge`/`mlp`, or the probe is reading
  something other than geometry.
- `mean_predictor` is the collapse floor: a fit that does not beat it has found nothing.
- `stats.mean_active` should be ~4 of 7 slots (per-sample N∈[1,7]).

**How to read the four measurements**

- `abs_pose` -- for `use_plucker=True` this is near-trivial, because the ray map *is* an
  input. The informative column is `m3off`: pose decoding there came from the image.
- `cam_eef` -- the quantity M4's aux head read (6.4% of a mean-collapsing floor). This
  column doubles as a check that the probe works at all, so a null here means something is
  broken, not that the model is empty.
- `rel_pose` -- **a ridge null here is NOT evidence of absence.** Relative pose is
  bilinear in the two camera poses, so a *linear* probe cannot fit it even when the
  geometry is fully present (measured: 250.6 cm vs the mean predictor's 248.2 cm on
  synthetic data where it was exactly present). Read the `mlp` column and pass
  `--mlp-steps`.
- `latent_stats` -- `z_g_across_view_subsets` vs `z_v_across_views_within_draw` is the
  "`z_v` view-aware, `z_g` view-invariant" claim as two numbers.

Send back each run's `data/probe_relpose_*/probe_relpose.json`; the numbers go into
`PROGRESS.md` once read. Decision rule: if the `m3off` MLP column already recovers the
geometry, step 1b's head is a post-hoc fit and the *objective* -- not the head -- is what
has to change.

### Resume and long campaigns

Resume with the same run dir and `training.num_epochs=<epochs still wanted>`; the loop runs
`num_epochs` iterations from the restored counter (from epoch 150, `num_epochs=51` → 200),
and the cosine LR schedule is re-based to the new `num_epochs`. `TopKCheckpointManager`
state is in-memory only, so on resume its map starts empty and stale topk files on disk are
never evicted — delete them yourself.

- **Long jobs die with the Claude Code session.** Launch anything that must outlive it
  disowned: `setsid bash -c '<cmd> > log 2>&1' &`.
- **To re-plan a running batch without killing the job:** the driver is
  `bash -c 'for ...; do python train.py ...; done'`; `kill -TERM <bash pid>` stops the loop
  while the running `python` child survives and is reparented to init. Append progress
  markers to `data/m3_campaign.log` so a hand-over is auditable.
- `training.resume: True` **silently resumes** an existing run dir — set it deliberately.
- **Disabling rollouts takes two flags, not one.** The workspace tests
  `epoch % rollout_every == 0`, which is true at epoch 0, so a large `rollout_every` alone
  still fires the first rollout — and on the multi-view configs that dies with
  `KeyError: 'view_06_image'`. Set `training.rollout_every=1000000` **and**
  `checkpoint.topk.monitor_key=val_loss checkpoint.topk.mode=min`, the second because
  `TopKCheckpointManager` does an unguarded `data[monitor_key]` lookup.
- Four config defaults that cost time if the CLI overrides above are dropped:
  `checkpoint.topk.k` is **5** (up to ~23 GB/run), `dataloader.num_workers` is **4**
  (81 s/epoch vs 43 at 14), `logging.project` is `diffusion_policy_debug`, and
  `training.resume` is `True`.

### Data generation (M2)

```bash
python tests/test_multiview_dataset.py                      # CPU-only, no render
python generate_multiview_dataset.py \
  --dataset data/robomimic/datasets/square/ph/image_abs.hdf5 \
  --output data/multiview/square_ph_ring13.zarr --montage data/multiview/square_ring13.png
```

- Use `--workers 4`: `max_inflight = workers*5` and each in-flight future pins a whole
  `(T,13,84,84,3)` buffer, so the default 24 can reach ~8 GB of RAM.
- **The `--limit-demos 5` pilot (~5 min) is not optional.** The gates run only at the very
  end of a full run, so a broken render loop surfaces after hours; the pilot reproduces the
  full run's gate values exactly.
- **There is no resume**; `--overwrite` is the only recovery and it wipes the store. Wipe
  each output before starting, and abort the batch if gate 1 does not PASS (`grep -aq
  "gate 1.*PASS" data/gen_$task.log`).
- A killed run is **silent** — zarr returns fill-value 0 for unwritten chunks and the gates
  never read the zarr back. Check for all-zero *frames* (not zero pixels; square and can
  legitimately contain 3 and 1 pure-black pixels per 84,672).

## Machine and timing

2× RTX 5090 (32 GB each), 24 cores, 125 GB RAM, shared with other users — load spikes from
~5 to ~25 and gives 2× slowdowns. `n_envs: 28` is fine here (the old 6 GB-laptop warning
is moot).

| run | cost |
|---|---|
| M1 square / can / lift | ~22 / ~16 / ~7 s per epoch (440/335/127 batches @ bs 64) |
| M1 rollout | ~2–3 min (28 envs, 56 episodes) |
| M3 step (B=64) | **94 ms** (min 92, max 98), 6.9 GB peak GPU — ~1.9× M1 per step, since the shared resnet runs over N∈[1,7] views instead of 1 |
| M3 square | **43 s/epoch** at `num_workers=14` — **compute-bound**: 8 workers still gave 41 s under 2-GPU contention; **4 workers fell to 81 s** (decode-bound) |
| M3 N=1 gate / can | 23 / ~30 s per epoch |
| M3 rollout | ~6 min (n_envs=14, 56 episodes) |
| `azimuth_interp` sweep | ~40 min (11 viewpoints × 50 episodes); ~46 min with two sweeps sharing the box |
| `azimuth_sweep3` / `elevation_az0` | ~12 min |
| fixed-N=1 cell | 21 s/epoch |

- **≥8 dataloader workers**; ~2 concurrent runs is the practical ceiling on 24 cores. The
  limit is CPU, not GPU memory (7 GB of 32 GB used).
- **2 GPUs give ~1.7×, not 2×**, on a 3-run tail: independent processes run at full speed
  in parallel, but the last run has no partner.
- **Epoch cost is load-dependent — re-measure it every campaign.** The same M4 config ran
  at **373 ms/step** while another user held both GPUs and **84 ms/step** once they
  finished — a 4× swing that brackets M3's 94 ms.

## Disk

Chronic constraint: the root volume ran at 97–100% during M1 and was still ~98% (47.8 GB
free) at the start of the N>1 session. `/data` (15 TB) exists but is **owned by another
user and not writable**. Check `df -h` as step 0 of any campaign — the M4 run matrix
originally had no disk gate and that was the binding constraint all session.

- A checkpoint is **4.62 GB** (policy + EMA + Adam state); `topk.k=1` plus `latest.ckpt`
  ≈ **9.2 GB per run**. Nine M3 runs needed ~83 GB.
- **Size a run with `du`, never by counting checkpoint files.** `topk.k=1` writes a second
  file only when the best rollout is *not* the final epoch — `m3off`'s checkpoints dir is
  8.7 GiB (topk byte-identical to `latest.ckpt`, see below) while `m3plucker`'s is 4.4 GiB
  (no topk written). Estimating 3 × 4.4 GiB for a deletion that in fact released 17 GiB is
  exactly the error this note exists to prevent.
- **The top-k file is often byte-identical to `latest.ckpt`** — the workspace saves both
  back-to-back from the same in-memory state, so a topk named `epoch=0200-*` on a 201-epoch
  run is a duplicate. **Compare with `cmp`, never by size**: of 19 checkpoints, exactly one
  was byte-identical while six others were the same size and genuinely different models.
- **The M1 (`*_abs_single`) weights were deleted on 2026-09-19** to make room for M3. Their
  `logs.json.txt`, `media/` and `.hydra/` survive and every number derived from them is
  committed. Recovery is a 1–2.5 h retrain per task from the runbook above.
- **`run_square_{m3plucker,m3eef,m3fixedn1}_s42_200ep/checkpoints/` were deleted 2026-09-24**
  (17 GiB, 20 → 37 GB free) for the N-diversity ladder. Delete **only** `checkpoints/` — the
  git-tracked `logs.json.txt`, `media/` and `.hydra/` live beside it. All three cells' numbers
  and sweeps are committed and their questions closed: `m3plucker`/`m3eef` by 1a's finding
  that the conditioning is inert behaviourally, `m3fixedn1` by the resolved N>1 confound.
  **`m3off` is never a deletion candidate** — it is the ladder's anchor and the only cheap
  eval-path-drift check.
- **`data/robomimic_image.zip` (84.75 GB) does not exist on this box** — only the three
  `ph` tasks (`square`, `can`, `lift`) are available, with both `image.hdf5` and
  `image_abs.hdf5`. Anything sized against that archive needs re-checking.
- No cleanup was needed for M2: the full run consumed 4.5 GB and left 23 GB free.

## Traps

### A check that cannot fail proves nothing

Twice in this project a convention check was written that could not fail, and both times it
was caught only by writing a *mutation* test alongside it. The deleted M3 draft's Plücker
check compared a camera-frame ray direction against a world-frame expected direction
without applying the rotation `R`, so its wrong output (dots of −0.85, −0.18, −0.60 instead
of ≈1.0) said nothing about the code. A test at an **identity** camera is equally vacuous
(`R == Rᵀ == I`). `tests/test_view_conditioned_obs_encoder.py` therefore pins the
convention against an independent numpy projector at **non-identity** poses and adds five
mutation power checks; `tests/test_aux_action_heads.py` does the same for the rot6d
transform, including the executable assertion that the wrong 6×6 shortcut *passes* at
`R_c == I` and fails everywhere else.

### Silent degradation

The expensive failures in this project produced no error and no implausible number. Eight
were caught before M3 ran (a `keepdim=True` that yielded a `(3,3,1)` rotation matrix; a
`reshape` that folded a view dim into channels; a shadowed loop variable that broke the
row-major slot-to-history correspondence; two crop-offset desynchronisations between image
and ray map), and two more were fatal at epoch 0 and found only by *running*
(a normalizer key mismatch between the rollout runner, the eval harness and the policy —
`predict_action` normalizes every obs key with no fallback — and a struct-mode `DictConfig`
that raises on `del shape_meta['obs'][k]` only when the cfg comes from a checkpoint's dill
payload). The positive half: the M2 gate-1 flip check exists precisely because an
upside-down dataset looks plausible in a montage.

### Verify by running, not by importing

`preview_viewpoints.py`'s render path and both M3 runtime paths went from
"import-checked, never executed" to working only by execution; import checks and `__mro__`
assertions passed and proved nothing about the obs-key contract at runtime. **Budget for a
probe phase before any long run** — the M3 probe cost ~35 min and saved ~15 h.

**Step 1a is the fourth instance, and the sharpest.** `probe_relpose.py` shipped with a green
CPU suite that pins every geometric convention to ~1e-15 with mutation power — and the two
functions that decide the result had never run. The smoke found `report_target` crashing on
the translation-only `cam_eef` target (`tgt_R is None` fed to `rot6d_from_mat`), and the
first *full-scale* run found `fit_mlp` dying at `loss.backward()` — float64 targets against
a float32 `nn.Linear`, which type-promotes in the forward pass and only fails in the
backward. The MLP is the **only** column that can read `rel_pose` at all. Two lessons: a
convention test says nothing about the paths that touch the checkpoint, and **the smoke
passed while the decisive path was still broken**, because `--mlp-steps` defaults to 0 —
smoke the flags you intend to run with, not just the wiring. Corollary: at `--n-samples 64`
the `rel_pose` ridge scored 23× *worse* than the mean predictor purely from
underdetermination (566 train pairs against 1536 features); at full scale it beats it. A
small-`n` ridge null on a bilinear target reads as "no geometry" and means nothing.

### M2-specific

The generator's gates are computed only at the end of a run; `--limit-demos 5` is the
mitigation. The driver should wipe each output before starting and abort the batch on a
failed gate (see the runbook).

## Artifacts

| what | where | in git |
|---|---|---|
| novel-view sweeps + viewpoint videos | `data/eval_*/eval_log.json`, `media/<viewpoint>/*.mp4` | ✅ |
| training logs (`logs.json.txt`) | `data/outputs/run_*/` | ✅ for M1/L1/M3/M4 runs |
| aux-probe evidence | `data/probe_m4_square_logs.json.txt` | ✅ |
| relational-probe (step 1a) results | `data/probe_relpose_square_{m3on,m3off}/probe_relpose.json` | ✅ |
| N-diversity ladder sweeps | `data/eval_{interp,el}_square_m3v12/` (and `m3v13…` as rungs land) | ✅ |
| ladder failure-mode videos | `data/eval_smoke_square_m3v12/media/<viewpoint>/*.mp4` | ✅ |
| weights (4.6 GB each) | `data/outputs/run_*/checkpoints/latest.ckpt` | ❌ — rsync only |
| campaign logs (ordered, timestamped) | `data/m3_campaign.log`, `data/m4_campaign.log` | ❌ on the box |
| multi-view zarrs | `data/multiview/<task>_ph_ring13.zarr` (819k images, 4.5 GB) | ❌ n/a |
| robomimic PH datasets | `data/robomimic/datasets/<task>/ph/{image,image_abs}.hdf5` | ❌ n/a |

Move weights between machines with
`rsync -av data/outputs <user>@<other>:/path/to/diffusion_policy/data/outputs`.

Training curves and rollout videos are on wandb, project **`diffusion_policy_view`**
(account `zihan-wa23-tsinghua-university`); the M1 runs are square `runs/1h4oj5p6`, can
`runs/q95ylpsc`, lift `runs/poearmxq`.
