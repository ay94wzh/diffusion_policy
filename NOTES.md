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
#   N-diversity ladder: the m3off line above with ONE override --
#     task.dataset.view_count_range="[1,N]" -> hydra.run.dir=data/outputs/run_square_m3v1<N>_s42_200ep
#   Committed rungs (mean active N): [1,1] 1.0 m3fixedn1 | [1,2] 1.5 m3v12 | [1,3] 2.0 m3v13
#                                    [1,5] 3.0 m3v15    | [1,7] 4.0 m3off

# Proprioception dropout -- the `[1,3]` balance test. No new config: a policy
#   _target_ override, and the dropout lives in compute_loss only (so rollouts,
#   eval, screen_conditioning.py and probe_relpose.py are untouched).
#
#   `+policy.proprio_dropout=0.5` -- the `+` is REQUIRED, and the runbook's original
#   bare `policy.proprio_dropout=0.5` (commit 61e82c4) is rejected at launch. Hydra 1.2
#   sets struct mode unconditionally on the config root
#   (hydra/_internal/config_loader_impl.py:256-259: "One must use + to add new fields
#   to them"), and the key is declared in NO config -- unlike M4's `aux_loss_weight`,
#   which works bare only because it lives in train_..._m4.yaml.
#   DO NOT "fix" it by declaring the key in the shared m3 config instead: that config
#   is used by plain-M3 runs whose _target_ is DiffusionUnetImagePolicy, which would
#   swallow the unknown kwarg into self.kwargs and forward it to scheduler.step --
#   a TypeError at ROLLOUT time, hours in (the same trap the policy docstring names).
#
#   Gate, pre-registered 2026-09-25 BEFORE the run: a NON-NULL lift at [1,3] is
#   consistent with the balance mechanism but does not establish it -- proprio dropout
#   is also a generic regulariser. The separating control is an arm the mechanism
#   predicts NO lift for: m3v12 ([1,2]), which fails by a severed image path (collapsed
#   encoder), so dropout should not rescue it (~1.35 h + a sweep). The mechanism claim
#   is not made until that control is run and does not lift. A NULL at [1,3] needs no
#   control and is decisive on its own.
#
#   No p=0 control run is needed either: at proprio_dropout=0.0 the class delegates to
#   super().compute_loss before drawing any RNG, so a p=0 rerun is bit-identical to
#   m3v13 -- m3v13 IS the p=0 arm. The p=0.5 arm is NOT RNG-locked to it (the mask draw
#   precedes the noise draw), so this is a run-to-run comparison at the usual n=1
#   resolution: a delta below ~0.1 is unmeasured.
python train.py --config-name=train_diffusion_unet_image_workspace_m3 \
  task=m3_plucker_image_abs_multiview task.task_name=square \
  task.dataset.view_count_range="[1,3]" +policy.proprio_dropout=0.5 \
  policy._target_=diffusion_policy.policy.diffusion_unet_image_policy_propdrop.DiffusionUnetImagePolicyPropDrop \
  policy.obs_encoder.use_plucker=false policy.obs_encoder.use_eef_hist=false \
  training.seed=42 training.num_epochs=201 training.device=cuda:1 \
  dataloader.num_workers=14 val_dataloader.num_workers=2 checkpoint.topk.k=1 \
  logging.project=diffusion_policy_view training.resume=false \
  hydra.run.dir=data/outputs/run_square_m3v13_pdrop_s42_200ep
#   launch gate: grep -q propdrop <log>  -- the policy prints an ACTIVE banner at
#   construction, so a silently-dropped override cannot be read as "dropout didn't help"
#   ~29 s/epoch at mean-N 2.0; ~43 s means view_count_range did not take
#   val_loss is a THIRD incomparable val_loss: self.training is always True in this
#   workspace (it evals the EMA copy, never putting self.model in eval mode), so
#   dropout is live during validation. Compare it only at matched epochs.
#   CPU test first: python tests/test_prop_dropout.py

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
#   ladder rungs sweep BOTH presets: -o data/eval_interp_square_<run> and -o data/eval_el_square_<run>
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

### Latent grid (schema 2) — cross-cell comparison

**Use this, not a bare `--stability-ranges`, when comparing CHECKPOINTS.** The old
`latent_stats` are not comparable across cells and the numbers below supersede them: the
subset sizes compared were governed by each cell's own `view_count_range`, which is not a
small effect — the *same* `m3off` encoder reads `z_g_across_view_subsets` = **0.335 at
`[1,2]` vs 0.202 at `[1,7]`**. The override plus the grid is what removes that.

```bash
python -u probe_relpose.py -c <ckpt> -o data/probe_relpose_grid_square_<cell> -d cuda:0 \
  --view-count-range 1,7 --stability-ranges "1,2;1,4;1,7;7,7" \
  --stability-states 256 --stability-repeats 6 --n-samples 2000 --mlp-steps 2000 \
  --num-threads 4          # add --random-init-control on the anchor
```

**Reading order, and the trap.** (1) `zv_pair_ratio` — fusion-free, so it measures the
backbone alone; (2) `zv_across_states` — **check this FIRST**, because it is the collapse
detector; (3) `zg_across_view_subsets` per range. A **collapsed** encoder yields near-constant
`z_v`, which makes `zv_pair_ratio` a ratio of two ~1e-5 numbers and can read as *extreme*
view-invariance — it produced exactly that on `m3v12` (1.277 vs a working 0.581, which a
naive reading would call a confirmation). Anchors on square: healthy random-init **1.51**,
`m3off` **0.58**, `m3v12` collapsed (`zv_across_states` 2.7e-05 vs `m3off`'s 0.572).
Gates, all of them, before any number is believed: `stats.mean_active` tracks the EFFECTIVE
range (~4.0 under a `[1,7]` override; 1.5 means the override was silently ignored);
`zg_permutation_floor` ~1e-8 (float noise — if it is not tiny, something non-deterministic is
live and nothing else is trustworthy); `zv_n_agnostic_max_abs_diff` < **1e-4** (it is 2e-05 —
cuDNN picks different kernels for batch-1 vs batch-7, so `== 0` is the wrong gate and would
fail a correct implementation); and the `draw_fingerprint` must MATCH between cells being
compared, which is what makes the pairing checkable rather than assumed.

**`[1,2]` and `[1,4]` grid entries carry `zv_pair_ratio: null` by construction** — a full
draw needs every slot, so only `[1,7]` and `[7,7]` can carry it. That is why the grid exists
rather than a single range.

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

**Single-view `z_g` decode** -- is the FUSED latent decodable at the inference condition
(N=1)? 4096 single-view draws (512 states x 8 repeats), **equal `n` for every cell** so the
comparison is valid even where regularisation biases the absolute values. Read the rotation
column only -- the translation dims diverge in every cell, including the one that works.

```bash
python -u probe_relpose.py -d cuda:0 --view-count-range 1,2 --stability-ranges "1,1" \
  --stability-states 512 --stability-repeats 8 --n-samples 200 --num-threads 4 \
  -c data/outputs/run_square_<cell>_s42_200ep/checkpoints/latest.ckpt \
  -o data/probe_zg_square_<cell>
```

The readout is `latent_grid["1,1"].zg_abs_pose` in the same `probe_relpose.json`.

### Collapse screen (`screen_collapse.py`)

**Run this before believing any training run is healthy. Always with `--random-init`, always
with a matched `--view-count-range`, and always with `-o`** (the flag is what writes the
artifact; without it the number exists only in your scrollback, which is how one of them got
misread as `0.0000`).

The failure it detects: the encoder outputs a near-constant vector, so the policy acts
open-loop and fails at the TRAINED pose rather than only off-axis. That is `[1,2]`'s floor, and
it is what separates the tasks L1 solves from the tasks it destroys (PROGRESS.md *Collapse is
a failure mode*).

```bash
# the SAME --view-count-range for every cell being compared
python screen_collapse.py -c <run>/checkpoints/latest.ckpt -d cuda:0 \
  --view-count-range 7,7 -o data/screen_collapse/<cell>_latest.json
python screen_collapse.py -c <run>/checkpoints/latest.ckpt -d cuda:0 \
  --view-count-range 7,7 --random-init -o data/screen_collapse/<cell>_RANDOM_INIT.json
```

- **A matched `--view-count-range`, or the screen is not comparable across cells.** By default
  each cell draws from its OWN training range, so a `[1,2]` cell sees 1–2 live views while a
  `[1,7]` cell sees up to 7 — and live-view count moves the fused spread. This is the *same*
  confound the probe's grid removes; it was reproduced here once already, and the tell was
  `[1,3]` reading **above** `m3off` (4.13e-02 vs 3.31e-02).
- **The baseline is per-architecture and must be measured, never borrowed**: 1.27e-02 (lift) /
  1.25e-02 (square) for L1's `MultiImageObsEncoder`, **5.1e-03** for M3's
  `ViewConditionedObsEncoder`. A threshold taken from one family is wrong for the other.
- **The screen cannot be matched for every cell.** `m3n1gate` is K=1 with a singleton
  `view_pool=[6]`, so `--view-count-range 7,7` is refused (correctly) and it can only be
  measured at its own `[1,1]`. A cell whose pool is smaller than the requested `hi` is not
  comparable, and the tool failing loudly is the intended behaviour.
- **Anchors** (relative spread, matched `[7,7]`): baselines 5.1e-03 / 1.3e-02; L1 lift (works)
  2.49e-02, L1 square s42/s43 (fails) 1.49e-04 / 1.99e-04, L1 can (fails) 3.05e-05; `m3off`
  (works) 1.72e-02, `m3v15` (works) 2.23e-02, `m3v13` (floor) 1.36e-02, `m3v12` (floor)
  **1.83e-07**. Read the **ordering**: ~1e-02 healthy, ~1e-04 and below degenerate.
- **A screen is not a verdict on failure.** `m3v13` is at the floor with a *healthy* encoder, so
  a healthy reading does not mean a cell will work. Screen for collapse, not for success.
- It is a screen (16 consecutive frames), not a measurement, and it is **correlation, not
  causation** — collapse may be a symptom of something deeper. Seed-robust on L1 square
  (s42 1.49e-04, s43 1.99e-04) and control-backed (random-init baselines match to 1.6% within
  the L1 family).
- Works on any image encoder, which is why it is separate from `probe_relpose.py` (that one
  needs M3's per-slot cam keys). Since 2026-09-25 it also handles `view_subset` datasets (L1's),
  which have no `view_count_range` at all.

### Conditioning screen (`screen_conditioning.py`)

**This is the one that can see a policy failing while its representation is fine** — the other
two tools measure whether information is *present*, and `m3v13` is the case where it is present
and the policy still fails.

```bash
python screen_conditioning.py -c <run>/checkpoints/latest.ckpt -d cuda:0 \
  -o data/screen_conditioning/<cell>_latest.json      # --n-obs 8 --seed 0 by default
```

It holds the diffusion sampling noise fixed (`torch.manual_seed(0)` before every
`predict_action`) and varies one input path at a time, so a change in the output is
attributable to that input rather than the sampler. Reported: `std` across the observations,
over the action chunk, normalised by the chunk's mean absolute value.

- **`image-only` near `0.0000` means the image path is severed** — the policy cannot see the
  scene at all, which is what a collapsed encoder must produce. `m3v12` reads **2.59e-05**
  against `m3off`'s 0.0068.
- **Read the two paths separately, never their sum.** `global_cond` is
  `concat([z_global, low-dim])` and the low-dim keys are 9 dims of proprioception that vary
  across samples too. Measured together, the *collapsed* cell comes out the **most**
  observation-sensitive (`m3v12` 0.0686) — its constant image riding along with normally
  varying proprioception — which is the opposite of the truth.
- Anchors: `m3off` 0.0068 / 0.0386 (image / proprio), `m3v13` 0.0067 / 0.0674, `m3v12`
  2.59e-05 / 0.0686.
- **Replicate before believing the balance reading.** Those anchors are `--n-obs 8 --seed 0`, and
  the `[1,3]`-vs-`m3off` proprio contrast is the *only* mechanism anyone has proposed for
  `[1,3]`'s floor — so it gets re-measured at scale before it is worth GPU-hours. Pre-registered
  rule: the hypothesis lives only if `m3v13`'s proprio/image ratio exceeds `m3off`'s by **≥1.5×
  across all three seeds**; at parity, the intervention below is off and the missing instrument
  is the only route left.

**Gate reading, formalised and pre-registered 2026-09-25 — committed BEFORE any n=64 screen
was run, so it cannot be renegotiated against the numbers.** With `R_cell(s) =
proprio_only / image_only` read from each run's JSON:

- **Pass** iff `min over s in {0,1,2} of R_m3v13(s) / R_m3off(s) >= 1.5`.
- **2-of-3 seeds above 1.5 is NOT support.** The rule says "across all three seeds"; a partial
  result is recorded as *gate failed, reading ambiguous* — neither pass nor a clean parity.
- **Clean parity** (every seed below 1.5, or ratios ≈ 1) is the branch that retires the
  intervention: the only proposed mechanism for `[1,3]`'s floor dies, and the missing
  instrument becomes the only route left.
- **Structural note, so the statistic is not misread as more than it is.** The two image arms
  are *equal by measurement* (0.0067 vs 0.0068 — no difference at all), so `R_m3v13/R_m3off`
  is carried almost entirely by the proprio arm: 0.0674/0.0386 = **1.75×** at the default.
  The gate therefore asks whether a 1.75× proprio gap survives replication, not whether a
  two-factor contrast does.
- `m3v12` rides along as a **control, not part of the rule**: its encoder is collapsed, so its
  image arm should stay ~2.6e-05 and its reading cannot support anything.

```bash
# -d cuda:1: GPU 0 holds ~13 GB from another user (NOTES.md *Machine and timing*).
# -o is NOT optional -- without it the number exists only in scrollback.
for s in 0 1 2; do for c in m3v13 m3off m3v12; do
python screen_conditioning.py -d cuda:1 --n-obs 64 --seed $s \
  -c data/outputs/run_square_${c}_s42_200ep/checkpoints/latest.ckpt \
  -o data/screen_conditioning/square_${c}_n64_s${s}.json
done; done
# reference points, also on disk: L1 lift (works, non-slot encoder) and L1 square (collapsed)
# smoke ONE cell (-o /tmp/smoke.json) first: the tool has no assertions, and n_obs=1 would
# print 0 for both arms -- the same misreading class as the :.4f bug below.
```

**Gate outcome, 2026-09-25 — FAILED, so the intervention is off and the propdrop run was NOT
launched.** Read from the nine JSONs, not the console:

| seed | `R_m3v13` | `R_m3off` | ratio | gate (≥1.5) |
|---|---|---|---|---|
| 0 | 38.53 | 22.59 | 1.705 | PASS |
| 1 | 40.05 | 12.60 | 3.179 | PASS |
| 2 | 16.20 | 28.56 | **0.567** | fail |

`min` over seeds = **0.567** → the pre-registered *gate failed, reading ambiguous* branch
(2-of-3 is not support). **Two defects in the instrument, both verified in code, are why it is
ambiguous rather than clean parity:**

1. **The observation draw was never seeded.** `screen_conditioning.py` seeds `torch` only
   (line 75); `multiview_image_dataset` draws each sample's view subset with `np.random`
   (lines 626/630). So every invocation draws a *different* view ensemble. `image_only` swings
   **2.47×** (`m3v13`) and **2.27×** (`m3off`) across the three seeds — the whole instability.
2. **The two cells were drawn at different view counts.** The dataset is instantiated from each
   checkpoint's own cfg, so `m3v13` drew `[1,3]` (mean 2.0) while `m3off` drew `[1,7]`
   (mean 4.0). Fewer live views → smaller ensemble spread → lower `image_only`, which is the
   direction of the entire seed-0 gap. **This is the same confound the project already
   corrected for `screen_collapse` on 2026-09-25.**

**The clean quantity is decisive, and it is what the write-up rests on.** `proprio_only` is
draw-independent — the proprio keys are the same 64 low-dim rows whichever views are live — and
it is stable to 0.2–0.4% within each cell across all three runs: `m3v13` 0.35003, `m3off`
0.34726, `m3v12` 0.34704. The proprio contrast is **1.008×**, so **the n=8 claim of a 1.75×
proprio gap does not reproduce and the balance hypothesis as stated is refuted** — on the one
arm the view draw cannot touch, independently of the gate's instability. This *confirms* the
suspicion already recorded above ("what is solid here is the severed path, not the balance
reading") rather than contradicting it. The control still works: `m3v12`'s image arm is
2.4–5.0e-05, ~256× below the others.

**Instrument fix, pre-registered 2026-09-25 before it was run.** `--view-count-range`
(mirroring `screen_collapse.py`'s, via `probe_relpose._apply_range`) plus `np.random.seed(seed)`
— **optional, default None = unchanged behaviour** so the flagless path stays as-is, and
`cell_view_count_range` / `effective_range` are recorded in the JSON the way
`screen_collapse.py` does, so the draw is auditable from the artifact.

- **Role of the matched reading, fixed now.** It is a pre-declared *confound control*, following
  the project's own precedent for this exact confound (the 2026-09-25 collapse-screen
  correction, which re-measured matched and let the matched table supersede the first). It
  **does not revisit the gate verdict** — that is closed: the intervention is off. Its purpose
  is forward-looking, to give the project a *use*-sensitive instrument that is not
  draw-confounded.
- **Prediction, registered before the run.** If the ladder's latent anti-correlation extends
  into behaviour-space, `image_only` at a matched `[7,7]` draw declines monotonically with
  mean-N: `m3off` (4.0) > `m3v15` (3.0) > `m3v13` (2.0) > `m3v12` (1.5, at ~1e-05 — severed).
  **The competing outcome is equally informative: `image_only` flat across cells**, meaning the
  ladder's behavioural differences are invisible to this instrument too. Both are recorded as
  results; neither is read as the other.
- **Consequence to record, not to hide:** the three committed `*_latest.json` came from the
  *unseeded* path, so they are **not reproducible** under the fixed code even at identical
  flags. Their qualitative finding (m3v12's severed path) stands; their method is superseded.

- **Print with `:.6g`, not `:.4f`.** The `:.4f` format turned 2.59e-05 into `0.0000`, which
  was then written up as "bit-identical"; the number is 256× below the other cells, not zero.

**Read the two arms at very different resolutions — this is the operational rule, and it cost
this session a gate.** After the fix the tool is **bit-reproducible** at a fixed
`(checkpoint, flags, seed)` (verified: `0.00752622` twice to 6 s.f.), so any remaining
difference is real. But the two arms have different real variance:

| arm | reproducibility | usable resolution |
|---|---|---|
| `proprio_only` | across-seed spread ≤ **0.4%**; draw-independent | resolves ~1% differences; **trust this one** |
| `image_only` | across-seed spread **1.9–2.5×**, *even with the draw matched* | **cannot resolve anything below ~2×**; only a severed path (~300× down) is readable |

- **Matching the draw does NOT reduce the image arm's spread** — measured, 1.92–2.50× matched
  against 2.27–2.47× unmatched, so the earlier explanation (unseeded `np.random` + per-cell
  ranges) was wrong even though both defects were real and the fix is still worth having.
- **The deeper problem: `image_only` is not a scalar.** At a fixed `[7,7]` range `k` is always
  7, so every cell sees the *same* view order for a given seed (verified directly) — the
  comparison is matched, differing only in the weights — and **the cell rank still flips**:
  the `m3v13`/`m3off` ratio runs 0.512 / 0.272 / 1.362 across seeds 0/1/2. Same inputs,
  opposite conclusion. So a cross-cell `image_only` comparison needs either a design whose
  verdict does not depend on the probe ensemble, or many seeds — **not** more `--n-obs`.
- Corollary for the gate that used it: the `R = proprio/image` ratio inherits the image arm's
  instability and cannot carry a 1.5× decision. The pre-registered n=64 gate failed
  (min ratio 0.567) for that reason, and the balance hypothesis was settled instead on
  `proprio_only`, which is the arm that is stable.

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
- **2026-09-25 session, in order — 15 GB → 18 GB free, having first paid 17.6 GB out.** Freed
  four topks whose cells' questions were closed (`m3n1gate`, `square_randview_s42`,
  `can_m3on`, `lift_randview`), each keeping its `latest.ckpt` so no cell lost probeability;
  `m3off` untouched as always. Then two new runs (`m3v27`, `abs_single_..._retrain200ep`) took
  17.4 GB. Then the **first byte-identical duplicate this project has found**: the M1 re-train's
  `epoch=0200-*` topk was byte-identical to its own `latest.ckpt` (the case predicted above — a
  topk named for the *final* epoch is saved back-to-back from the same in-memory state), which
  released 4.4 GB for nothing. Prior `cmp` sweeps across 25 files had found none, so the rule had
  released nothing until now.
- **Before deleting anything, `cmp` it against its `latest.ckpt`** — and note that the M1 case
  shows the payoff is real rather than theoretical. Deleting a *topk* never removes a cell's
  probeability as long as `latest.ckpt` stays, which is what makes these deletions safe under the
  probe-first rule.
- **Size a run with `du`, never by counting checkpoint files.** `topk.k=1` writes a second
  file only when the best rollout is *not* the final epoch, which is a per-run coin flip. As of
  2026-09-25 (end of session), **two-file (8.7 GiB)** dirs: `m3v13`, `m3v15`, `m3v27`, `m4on`,
  `m4off`, `can_randview`, `lift_n1gate`. **One-file (4.4 GiB)**: `m3off`, `m3on`, `m3v12`,
  `m3n1gate`, both `square_randview` seeds, `can_m3on`, `lift_randview`,
  `abs_single_retrain`. The *same* run dir can move between the two columns — `m3n1gate` went
  2 → 1 when its topk was freed, `m3v27` arrived at 2 — so never carry an old count forward.
  Estimating "3 × 4.4 GiB" for a deletion that in fact released 17 GiB is exactly the error this
  note exists to prevent.
- **The top-k file is often byte-identical to `latest.ckpt`** — the workspace saves both
  back-to-back from the same in-memory state, so a topk named `epoch=0200-*` on a 201-epoch
  run is a duplicate. **Compare with `cmp`, never by size**: of 19 checkpoints checked on
  2026-09-24, exactly one was byte-identical while six others were the same size and genuinely
  different models. **Re-checked 2026-09-25 across the then-current 25 files: none were
  byte-identical**, so this rule released nothing on that date — the 4 GiB reclaimed came from
  deleting an already-probed topk outright, not from a duplicate.
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
- **Rule, learned the hard way on 2026-09-25: do not delete a cell's weights until its LATENT
  has been probed, not merely its behaviour measured.** `m3fixedn1` was deleted the same day
  its behavioural question closed — and within hours a mechanistic question arose (is the
  floor an encoder collapse?) for which it was the natural second floor pole. It survives by
  luck: `m3v12` turned out to be a *better* pole, because it moves mean-N where `m3fixedn1`
  does not. A behavioural null does not close a cell; it only closes the behavioural question.
  Probing is free and needs only the checkpoint, so the insurance costs nothing but a rule.
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
| novel-view sweeps | `data/eval_*/eval_log.json` (plus `media/<viewpoint>/*.mp4` only when `--n-test-vis > 0`; the M3-era sweeps kept none) | ✅ |
| training logs (`logs.json.txt`) | `data/outputs/run_*/` | ✅ for M1/L1/M3/M4 runs |
| aux-probe evidence | `data/probe_m4_square_logs.json.txt` | ✅ |
| relational-probe (step 1a) results | `data/probe_relpose_square_{m3on,m3off}/probe_relpose.json` | ✅ |
| probe grid + fused-latent (`z_g`) runs | `data/probe_relpose_grid_square_*`, `data/probe_zg_square_*` | ✅ **since 2026-09-25 — before that, the `probe_zg_*` row was a false claim: none of the four dirs were tracked, and PROGRESS.md's *second failure mode* Stage 2 reads `latent_grid["1,1"].zg_abs_pose` straight out of them. The `.gitignore` whitelist covered `probe_relpose_*` only. Fixed, and all four are now committed.** |
| **collapse screens** | `data/screen_collapse/*.json` (trained + `_RANDOM_INIT` controls) | ✅ |
| **conditioning screens** | `data/screen_conditioning/*.json` | ✅ |
| N-diversity ladder sweeps | `data/eval_{interp,el}_square_{m3v12,m3v13,m3v15}/` | ✅ |
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
