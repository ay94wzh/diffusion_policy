# NOTES — environment, runbooks, operations

Practical material for running this project. The module, its results and the open questions live
in `PROGRESS.md`; the plan in `PLAN.md`. **The runbooks below execute on `miroc-server`** (user
`zihan`, checkout `/home/zihan/repos/diffusion_policy`) — the box with the 2× RTX 5090s, the
multi-view zarrs, the hdf5s and every checkpoint. The **laptop** (hostname `ay`, user
`zihan-wang`, checkout `/home/zihan-wang/diffusion_policy`) is where code and documents are
written; no GPU, no `/data`. **Name machines, never "this box"**: two revisions of this file got
that wrong (the original "coding-and-documents machine" story, and the
2026-10-01 "this box is the training machine" correction, which was written in the training box's
checkout and reads false on the laptop); that is the same confusion under which a guard failure
skipped every evaluation of three finished runs for five days — see *A guard that discards its
own error* under Traps. The closed runbooks (the step-1a relational-probe runbook
and the never-launched proprioception-dropout runbook) are at git tag `docs-full-20260930`.

## Environment

- **The multi-view zarrs, the PH hdf5s and every checkpoint are on `miroc-server`**, and
  `PROGRESS.md` paths that name `data/outputs/run_*/checkpoints/latest.ckpt` or
  `data/multiview/*_ring13.zarr` resolve **there** (not in the laptop checkout). Since 2026-10-01
  the `data/outputs` weights are **symlinks** into `/data/zihan/ckpt_store/`, so reads through
  those paths work but `du` on `data/outputs` no longer reports them.
- **Disk is the binding constraint of any campaign on `miroc-server`.** Check `df -h` there as
  step 0 — but check the right volume. `/` (the repo) is chronically near-full and `/data` has
  terabytes, which is why new run dirs go to `/data/zihan/runs/run_<name>/` and only
  `logs.json.txt` is copied back. A checkpoint is 4.62 GB; `topk.k=1` plus `latest.ckpt` is
  ~9.2 GB per run.
- conda env **`robodiff`**: torch **2.8.0+cu128** (sm_120), robosuite 1.2.0, robomimic
  0.2.0, mujoco_py 2.0.2.13, numcodecs 0.10.2, wandb 0.15.12.
  **Do not recreate the env from `conda_environment.yaml`** — its `pytorch=1.12.1` pin is
  stale and unusable on the RTX 5090s. The live env was upgraded in place.
- **Never call a bare `python` in a script.** The shell's `python` is conda **base**
  (`/home/zihan/anaconda3/bin/python`) and has **no torch**; `robodiff` is not active by
  default. Pin the interpreter, as every driver script in `data/` does:
  `PY=/home/zihan/anaconda3/envs/robodiff/bin/python`. This cost 5 h of finished training
  once — see *A guard that discards its own error*, below.
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
- `wandb` is unattended-safe on `miroc-server` via `~/.netrc` — proven with detached runs.

## Runbook

All commands run from the repo root, on `miroc-server`. **201 epochs**.

### Training

```bash
# M1 single-view baseline
python train.py --config-dir=. --config-name=train_diffusion_unet_image_workspace.yaml \
  task=<task>_image_abs_single training.seed=42 training.device=cuda:0 \
  training.num_epochs=201 dataloader.num_workers=8 checkpoint.topk.k=1 \
  logging.project=diffusion_policy_view \
  hydra.run.dir=data/outputs/run_<task>_abs_single_s42_200ep
#   L1 view diversity: task=randview_image_abs_multiview, dataloader.num_workers=10

# M3 view-conditioned encoder (flags select the 2x2 cell; both false = m3off)
python train.py --config-name=train_diffusion_unet_image_workspace_m3 \
  task=m3_plucker_image_abs_multiview task.task_name=square \
  policy.obs_encoder.use_plucker=false policy.obs_encoder.use_eef_hist=false \
  training.seed=42 training.num_epochs=201 training.device=cuda:0 \
  dataloader.num_workers=14 val_dataloader.num_workers=2 checkpoint.topk.k=1 \
  logging.project=diffusion_policy_view \
  hydra.run.dir=data/outputs/run_square_m3off_s42_200ep
#   N=1 fidelity gate: task=m3_plucker_image_abs_n1
#   N-diversity ladder: the m3off line with ONE override --
#     task.dataset.view_count_range="[1,N]" -> hydra.run.dir=data/outputs/run_square_m3v1<N>_s42_200ep
#   Committed rungs (mean active N): [1,1] 1.0 m3fixedn1 | [1,2] 1.5 m3v12 | [1,3] 2.0 m3v13
#                                    [1,5] 3.0 m3v15    | [1,7] 4.0 m3off

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
`np.random.choice(view_pool, size=k_active, replace=False)`. Redrawn every `__getitem__`, so N and
the subset move each epoch. Active slots are always a prefix (`0..k_active-1`); the rest are zero
with mask 0. A guard that rejected `[1, 2]` was removed 2026-09-24 (redundant and internally
inconsistent; inert for every committed cell); **legal ranges now: any `1 <= lo <= hi <= n_slots`**.

- **The view draws are NOT the global numpy RNG leaking across workers — do not "fix" this.**
  Torch seeds numpy per worker explicitly (`torch/utils/data/_utils/worker.py`), `base_seed` is
  redrawn per iterator, and `persistent_workers: False` means a new iterator each epoch — so draw
  streams differ across workers and epochs. Adding a `worker_init_fn`, a `generator=`, or
  `persistent_workers=True` would *change* the stream and break comparability with every
  committed cell, for no benefit.
- **Cost model: encoder cost tracks the MEAN active view count, not K.** Masked slots are skipped
  in the encode loop, so K only sets the loop length; parameters are K-independent. Measured
  ≈ **7.3 s/epoch per mean active view + ~14 s fixed** — anchors: `[1,1]` 21 s/epoch, `[1,7]`
  43 s/epoch, `[1,2]` 24 s/epoch. Use it as a launch gate.
- **N is drawn per sample, so the eval N is a design decision, not a detail.** Every evaluation
  here is **N=1** (`eval_novel_view.py::_serve_m3` puts the single live camera in slot 0 and
  zeroes the rest). At N=1 the fusion softmax is over one unmasked key, so the learnable query has
  *no effect* — a degenerate corner of the module. Keeping N=1 in range is what makes each ladder
  rung differ from `m3off` on exactly one axis; a cell trained at `[2, 2]` or `[k, k]` forfeits
  that, and a floor from it cannot be read as a diversity statement. (M3's original rationale —
  keeping the N=1 corner in-distribution — is refuted by `[2,7]`, which infers at N=1 as well as
  `[1,7]` without ever training there.)

### Evaluation

```bash
# novel-view sweep (M3-era flags; omit --m3-slots for M1/L1)
python eval_novel_view.py -c <run>/checkpoints/latest.ckpt -o data/eval_interp_square_m3off \
  -d cuda:0 --preset azimuth_interp --m3-slots 7 --eef-hist-steps 4 --n-envs 14 --n-test-vis 0
#   presets: azimuth_sweep5 / azimuth_interp / azimuth_sweep3 / elevation_az0
#   ladder rungs sweep BOTH presets: -o data/eval_interp_square_<run> and -o data/eval_el_square_<run>
#   fast smoke: --preset azimuth_sweep3 --n-test 4 --n-test-vis 2 --n-envs 4

python summarize_novel_view.py <dir>/eval_log.json     # degradation table
python eval.py -c <run>/checkpoints/latest.ckpt -o data/eval_orig -d cuda:0   # sanity
```

`--m3-slots` must equal the checkpoint's own slot count — the assert compares against its
`shape_meta`: **7 for all M3 cells including `m3off`**, and `5` for a five-slot model.
`eval_log.json` is a flat dict keyed
`test/<viewpoint>/{mean_score,success_rate,sim_max_reward_<seed>}`, so any number can be
re-derived without re-running. `test_start_seed: 100000` is what makes the episodes **paired**
across viewpoints.

### Collapse screen (`screen_collapse.py`)

**Run this before believing any training run is healthy.** It detects the encoder outputting a
near-constant vector (the policy then acts open-loop and fails at the TRAINED pose), which is
`[1,2]`'s floor and what separates the tasks L1 solves from the tasks it destroys.

```bash
# the SAME --view-count-range for every cell being compared
python screen_collapse.py -c <run>/checkpoints/latest.ckpt -d cuda:0 \
  --view-count-range 7,7 -o data/screen_collapse/<cell>_latest.json
python screen_collapse.py -c <run>/checkpoints/latest.ckpt -d cuda:0 \
  --view-count-range 7,7 --random-init -o data/screen_collapse/<cell>_RANDOM_INIT.json
```

- **A matched `--view-count-range`, or the screen is not comparable across cells** (live-view
  count moves the fused spread). **The baseline is per-architecture and must be measured, never
  borrowed**: 1.27e-02 (lift) / 1.25e-02 (square) for L1's `MultiImageObsEncoder`, **5.1e-03**
  for M3's `ViewConditionedObsEncoder`.
- **Anchors** (matched `[7,7]`): `m3off` 1.72e-02, `m3v15` 2.23e-02, `m3v13` 1.36e-02 (all
  healthy), `m3v12` **1.83e-07** (collapsed), L1 lift 2.49e-02 (healthy) against L1 square
  1.49e-04 / 1.99e-04 and can 3.05e-05 (collapsed). Read the **ordering**: ~1e-02 healthy,
  ~1e-04 and below degenerate.
- **A screen is not a verdict on failure.** `m3v13` is at the floor with a *healthy* encoder.
  Screen for collapse, not for success.
- It is a screen (16 consecutive frames), **correlation, not causation**. Some cells cannot be
  matched (`m3n1gate` is K=1 with a singleton pool, so `[7,7]` is refused correctly) and their
  numbers should not be quoted as matched.

### Conditioning screen (`screen_conditioning.py`)

**The one that can see a policy failing while its representation is fine** — it holds the
diffusion sampling noise fixed and varies one input path at a time, so a change in the output is
attributable to that input. Reported: `std` across the observations, over the action chunk,
normalised by the chunk's mean absolute value.

```bash
python screen_conditioning.py -c <run>/checkpoints/latest.ckpt -d cuda:0 \
  -o data/screen_conditioning/<cell>_latest.json      # --n-obs 8 --seed 0 by default
```

- **Read the two paths separately, never their sum.** `global_cond` is
  `concat([z_global, low-dim])` and the low-dim keys are 9 dims of proprioception that vary
  across samples too. Measured together, the *collapsed* cell comes out the **most**
  observation-sensitive — the opposite of the truth.
- **`image_only` near 0.0000 means the image path is severed**; `m3v12`'s image arm is
  2.4–5.0e-05, ~256× below the others. **The proprio arm is the draw-independent one** and
  reproducible to ~0.3% (0.3470–0.3489 across all four rungs); the image arm is not — at matched
  inputs the cell rank order flips across seeds, which is why `image→action sensitivity` cannot
  be used as a scalar. The n=8 anchors are superseded as method (readings scale with `n`).
- Smoke ONE cell first (`-o /tmp/smoke.json`): the tool has no assertions, and `n_obs=1` would
  print 0 for both arms.

### Latent probe (`probe_relpose.py`) — cross-cell comparison

Measures what geometry a **frozen** checkpoint's latents already encode — no training, one
forward pass per sample, then a closed-form ridge and an MLP readout. **Use the schema-2 grid,
not a bare `--stability-ranges`, when comparing CHECKPOINTS.**

```bash
python -u probe_relpose.py -c <ckpt> -o data/probe_relpose_grid_square_<cell> -d cuda:0 \
  --view-count-range 1,7 --stability-ranges "1,2;1,4;1,7;7,7" \
  --stability-states 256 --stability-repeats 6 --n-samples 2000 --mlp-steps 2000 \
  --num-threads 4          # add --random-init-control on the anchor
```

**Reading order, and the trap.** (1) `zv_across_states` — **check this FIRST**, it is the
collapse detector; (2) `zv_pair_ratio` (fusion-free, measures the backbone alone); (3)
`zg_across_view_subsets` per range. A collapsed encoder yields a `zv_pair_ratio` that is a ratio
of two ~1e-5 numbers and can read as *extreme* view-invariance (`m3v12` read 1.277 against a
working 0.581 — a naive reading would call it a confirmation). Anchors on square: healthy
random-init **1.51**, `m3off` **0.58**; `m3v12` collapsed (`zv_across_states` 2.7e-05 vs
`m3off`'s 0.572).

**Gates, all of them, before any number is believed:** `stats.mean_active` tracks the EFFECTIVE
range (~4.0 under a `[1,7]` override; 1.5 means the override was silently ignored);
`zg_permutation_floor` ~1e-8; `zv_n_agnostic_max_abs_diff` < **1e-4** (it is 2e-05 — cuDNN picks
different kernels for batch-1 vs batch-7, so `== 0` is the wrong gate); and the
`draw_fingerprint` must MATCH between the cells being compared. `shuffled_target` must be worse
than `ridge`/`mlp`, and `mean_predictor` is the collapse floor. **A ridge null on `rel_pose` is
NOT evidence of absence** — relative pose is bilinear, so a linear probe cannot fit it even when
the geometry is present; **read the `mlp` column and pass `--mlp-steps`.** Cost: ~30 s encode +
ridge fits + 1–2 min MLP; feature matrices ~200 MB at the defaults.

**Single-view `z_g` decode** — is the FUSED latent decodable at the inference condition (N=1)?
4096 single-view draws, equal `n` for every cell. Read the rotation column only (translation
dims diverge in every cell, including the one that works). The readout is
`latent_grid["1,1"].zg_abs_pose` in the same JSON:

```bash
python -u probe_relpose.py -d cuda:0 --view-count-range 1,2 --stability-ranges "1,1" \
  --stability-states 512 --stability-repeats 8 --n-samples 200 --num-threads 4 \
  -c data/outputs/run_square_<cell>_s42_200ep/checkpoints/latest.ckpt \
  -o data/probe_zg_square_<cell>
```

Send back each run's `probe_relpose.json`; the numbers go into `PROGRESS.md` once read.

### Data generation (M2)

```bash
python tests/test_multiview_dataset.py                      # CPU-only, no render
python generate_multiview_dataset.py \
  --dataset data/robomimic/datasets/square/ph/image_abs.hdf5 \
  --output data/multiview/square_ph_ring13.zarr --montage data/multiview/square_ring13.png
```

- Use `--workers 4` (`max_inflight = workers*5`; the default 24 can reach ~8 GB of RAM), and
  **the `--limit-demos 5` pilot (~5 min) is not optional** — the gates run only at the very end,
  so a broken render loop otherwise surfaces after hours; the pilot reproduces the full run's
  gate values exactly.
- **There is no resume**; `--overwrite` wipes the store. Wipe each output before starting, and
  abort the batch if gate 1 does not PASS. A killed run is **silent** (zarr returns fill-value 0
  for unwritten chunks) — check for all-zero *frames*.
- **The generated zarrs are on `miroc-server`**; a runbook step that reads them must begin
  by checking they exist.

### Resume and long campaigns

Resume with the same run dir and `training.num_epochs=<epochs still wanted>` (from epoch 150,
`num_epochs=51` → 200); the cosine LR schedule is re-based. `TopKCheckpointManager` state is
in-memory only, so stale topk files are never evicted on resume — delete them yourself.

- **Long jobs die with the Claude Code session.** Launch anything that must outlive it disowned:
  `setsid bash -c '<cmd> > log 2>&1' &`.
- **To re-plan a running batch without killing the job:** `kill -TERM` the driver `bash` — the
  running `python` child survives and is reparented to init.
- `training.resume: True` **silently resumes** an existing run dir — set it deliberately.
- **Disabling rollouts takes two flags, not one**: `training.rollout_every=1000000` **and**
  `checkpoint.topk.monitor_key=val_loss checkpoint.topk.mode=min` — the workspace tests
  `epoch % rollout_every == 0` (true at epoch 0) and `TopKCheckpointManager` does an unguarded
  `data[monitor_key]` lookup.
- Four config defaults that cost time if the CLI overrides above are dropped:
  `checkpoint.topk.k` is **5** (up to ~23 GB/run), `dataloader.num_workers` is **4**
  (81 s/epoch vs 43 at 14), `logging.project` is `diffusion_policy_debug`, and
  `training.resume` is `True`.

## Machine and timing (`miroc-server`)

2× RTX 5090 (32 GB each), 24 cores, 125 GB RAM, shared with other users — load spikes from
~5 to ~25 and gives 2× slowdowns. `n_envs: 28` is fine.

| run | cost |
|---|---|
| M1 square / can / lift | ~22 / ~16 / ~7 s per epoch (440/335/127 batches @ bs 64) |
| M1 rollout | ~2–3 min (28 envs, 56 episodes) |
| M3 step (B=64) | **94 ms** (min 92, max 98), 6.9 GB peak GPU — ~1.9× M1 per step |
| M3 square | **43 s/epoch** at `num_workers=14` — **compute-bound**; 4 workers fell to 81 s (decode-bound) |
| M3 N=1 gate / can | 23 / ~30 s per epoch |
| M3 rollout | ~6 min (n_envs=14, 56 episodes) |
| `azimuth_interp` sweep | ~46 min (11 viewpoints × 50 episodes); ~52 min with two sweeps sharing the box |
| `azimuth_sweep3` / `elevation_az0` | ~12 min |
| fixed-N=1 cell | 21 s/epoch |

- **≥8 dataloader workers**; ~2 concurrent runs is the practical ceiling on 24 cores. The limit
  is CPU, not GPU memory. 2 GPUs give ~1.7×, not 2×.
- **Epoch cost is load-dependent — re-measure it every campaign.** The same M4 config ran at
  **373 ms/step** while another user held both GPUs and **84 ms/step** once they finished.

## Disk

Chronic constraint on **`miroc-server`**: check `df -h` as step 0 of any campaign.

- A checkpoint is **4.62 GB** (policy + EMA + Adam state); `topk.k=1` plus `latest.ckpt` ≈
  **9.2 GB per run**. Size a run with `du`, never by counting checkpoint files — `topk.k=1` writes
  a second file only when the best rollout is *not* the final epoch.
- **Before deleting anything, `cmp` it against its `latest.ckpt`** — the top-k file is sometimes
  byte-identical to it. **Compare with `cmp`, never by size**: of 19 checkpoints checked on
  2026-09-24, exactly one was byte-identical while six others were the same size and genuinely
  different models.
- **Delete only `checkpoints/`**, never the run dir: the git-tracked `logs.json.txt`, `media/`
  and `.hydra/` live beside it.
- **`m3off` is never a deletion candidate** — it is the ladder's anchor and the only cheap
  eval-path-drift check.
- **Do not delete a cell's weights until its LATENT has been probed, not merely its behaviour
  measured.** `m3fixedn1` was deleted the day its behavioural question closed, and within hours a
  mechanistic question arose for which it was the natural pole. Probing is free.
- **`data/robomimic_image.zip` (84.75 GB) does not exist on `miroc-server`** — only the
  three `ph` tasks are available, with both `image.hdf5` and `image_abs.hdf5`.

## Traps

### A guard that discards its own error

On 2026-09-26 `data/m5_campaign.sh` trained all three main-line runs to completion, then
declared every checkpoint corrupt and skipped every sweep and screen. Nothing was wrong with
the checkpoints: the guard called a bare `python` (**base, no torch** — above), and its error
went into `grep -q`, so the log recorded only `CKPT_LOAD_FAILED`. **Three 4.6 GB loads reported
inside one second is the tell** — a real load takes ~30 s, so a failure that fast is an import
error, never a truncated file. Two rules came out of it:

- **Print or log the error a guard is testing for.** A guard whose failure text is discarded
  cannot be told apart from the condition it is guarding against.
- **A guard must be able to fail, and be able to pass.** `out=$(python ... | tail -1)` reports
  `tail`'s status, which is always 0 — the same class of vacuous check as an identity-camera
  Plücker test (below). `data/m5_sweeps.sh` keeps the status un-piped for this reason.

The generalisable version: **a skipped stage must be loud.** The campaign's real damage was not
the bad guard, it was that "guard failed" and "work skipped" were one quiet line in a log nobody
re-read for five days.

### A check that cannot fail proves nothing

Twice in this project a convention check was written that could not fail, and both times it
was caught only by writing a *mutation* test alongside it — a Plücker check comparing a
camera-frame direction against a world-frame expected direction without applying the rotation
`R`, and a test at an **identity** camera (`R == Rᵀ == I`) is equally vacuous.
`tests/test_view_conditioned_obs_encoder.py` therefore pins the convention against an
independent numpy projector at **non-identity** poses with five mutation power checks, and
`tests/test_aux_action_heads.py` does the same for the rot6d transform.

### Silent degradation

The expensive failures in this project produced no error and no implausible number: a
`keepdim=True` that yielded a `(3,3,1)` rotation matrix; a `reshape` that folded a view dim into
channels; a shadowed loop variable; two crop-offset desynchronisations; a normalizer key
mismatch between the rollout runner, the eval harness and the policy (fatal at epoch 0, found
only by *running*); and a struct-mode `DictConfig` that raises only when the cfg comes from a
checkpoint's dill payload. An upside-down dataset looks plausible in a montage — which is why
M2's gate-1 flip check exists.

### Verify by running, not by importing

Import checks and `__mro__` assertions proved nothing about the obs-key contract at runtime —
the M3 render paths went from "import-checked" to working only by execution. **Budget for a
probe phase before any long run** (M3's probe cost ~35 min and saved ~15 h). The sharpest case:
`probe_relpose.py` shipped with a green CPU suite pinning every convention to ~1e-15, while the
two functions that decide the result had never run — the smoke passed while the decisive path
was still broken, because `--mlp-steps` defaults to 0. **Smoke the flags you intend to run
with, not just the wiring.**

### Hydra struct mode: new policy kwargs need a `+`

A key that is declared in **no** config cannot be set with a bare `key=value` — Hydra 1.2 sets
struct mode unconditionally on the config root and rejects the override at launch. Use
`+policy.<key>=...`. M4's `aux_loss_weight` works bare only because it lives in
`train_..._m4.yaml`. **Do not "fix" it by declaring the key in the shared m3 config**: that
config is used by plain-M3 runs whose `_target_` is `DiffusionUnetImagePolicy`, which would
swallow the unknown kwarg into `self.kwargs` and forward it to `scheduler.step` — a `TypeError`
at **rollout** time, hours in.

### Regenerating the multi-view zarr: the action convention is not checked

`generate_multiview_dataset.py` converts actions with
`_convert_actions(raw_actions, abs_action=True)`, which converts **only the rotation** to rot6d —
the position dims pass through untouched. So it cannot repair a wrong position convention, and
the generator's gates check **images** (gate 1) and the **projected gripper** (gate 2) only: **a
wrong-action zarr would pass every gate silently.** If the zarrs are ever regenerated, gate the
actions deliberately.

## Artifacts

| what | where | in git |
|---|---|---|
| novel-view sweeps | `data/eval_*/eval_log.json` (plus `media/*.mp4` only when `--n-test-vis > 0`) | ✅ |
| training logs (`logs.json.txt`) | `data/outputs/run_*/` | ✅ |
| collapse screens (trained + `_RANDOM_INIT`) | `data/screen_collapse/*.json` | ✅ |
| conditioning screens | `data/screen_conditioning/*.json` | ✅ |
| relational probe (step 1a + schema-2 grid + `z_g`) | `data/probe_relpose_*/`, `data/probe_zg_square_*/` | ✅ |
| aux-probe evidence | `data/probe_m4_square_logs.json.txt` | ✅ |
| weights (4.62 GB each) | `data/outputs/run_*/checkpoints/latest.ckpt` | ❌ — **on `miroc-server`**; rsync only |
| multi-view zarrs | `data/multiview/<task>_ph_ring13.zarr` | ❌ — **on `miroc-server`** |
| robomimic PH datasets | `data/robomimic/datasets/<task>/ph/{image,image_abs}.hdf5` | ❌ — **on `miroc-server`** |

Move weights between the laptop and `miroc-server` with
`rsync -av data/outputs <user>@<other>:/path/to/diffusion_policy/data/outputs`.

Training curves and rollout videos are on wandb, project **`diffusion_policy_view`**
(account `zihan-wa23-tsinghua-university`); the M1 runs are square `runs/1h4oj5p6`, can
`runs/q95ylpsc`, lift `runs/poearmxq`.
