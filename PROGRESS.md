# PROGRESS

Last updated: 2026-09-12. See `PROPOSAL.md` (research direction) and `PLAN.md`
(milestones M1–M5).

**M1 is COMPLETE.** Single-view DP baselines were trained for can, lift, and
square (robomimic PH, agentview only, 201 epochs each) on the 2× RTX 5090
machine, and the novel-view degradation curves were measured at azimuth
0°, ±15°, ±30°. The headline result: success collapses essentially to zero at
±15° on all three tasks.

**M2 (multi-view data): code written and CPU-verified; rendering not yet run.**
See §7 for what to run on the render machine and what must be on it.

---

## 1. Results — novel-view degradation curves

`success_rate` at each viewpoint (50 paired test episodes, harness
`azimuth_sweep5`; strict `EnvRobosuite.is_success()` metric):

| viewpoint | square | can | lift |
|---|---|---|---|
| az_0 | 0.82 | 0.98 | 0.76 |
| az_m15 | 0.02 | 0.08 | 0.08 |
| az_p15 | 0.00 | 0.00 | 0.08 |
| az_m30 | 0.00 | 0.00 | 0.00 |
| az_p30 | 0.00 | 0.00 | 0.00 |

`mean_score` (max-reward, the stock runner's metric):

| viewpoint | square | can | lift |
|---|---|---|---|
| az_0 | 0.84 | 0.98 | 1.00 |
| az_m15 | 0.02 | 0.08 | 0.46 |
| az_p15 | 0.00 | 0.00 | 0.50 |
| az_m30 | 0.00 | 0.00 | 0.00 |
| az_p30 | 0.00 | 0.00 | 0.22 |

**The baseline is extremely view-tied**: on all three tasks success drops to
≈0 at the smallest perturbation tested (±15°), even though at az_0 the
policies are at or near the published performance band (paper: Lift ≈ 1.00,
Square ≈ 0.9–1.0; ours: 0.98 / 0.82). This sharp collapse is the reference
curve M3–M5 must beat.

Notes on the numbers:
- **Framing was verified, not assumed.** The displaced views keep their
  texture (frame std 66–69 vs az_0's 65) and differ from az_0 by only
  ~10–11% mean pixel value on same-seed first frames — the scene is fully
  visible at ±15°/±30°, so the collapse is a property of the policy, not a
  camera-looking-somewhere-useless artifact (see PROGRESS §4's caveat).
- **Scores are noisy ±~0.05**: diffusion sampling noise is unseeded, so
  repeated rollouts of the *same* checkpoint vary (square epoch-150 scored
  0.86 once and 0.94 on re-eval; epoch 100/200 both 0.88). Treat single-digit
  differences between checkpoints as noise.
- `success_rate` (is_success) is stricter than `mean_score` (max reward) —
  e.g. lift at az_0: success 0.76 while mean_score 1.00, and at ±15° the
  policy still gets near the cube (mean 0.46–0.50) but rarely closes the grip.

Artifacts (all under `data/`, gitignored):
```
data/outputs/run_{square,can,lift}_abs_single_s42_200ep/checkpoints/latest.ckpt   (epoch-200 models)
data/outputs/.../logs.json.txt                                                   (train + rollout history)
data/eval_az5_{square,can,lift}_ep200/eval_log.json + media/<viewpoint>/*.mp4    (sweeps)
```
Summarize any sweep with `python summarize_novel_view.py <dir>/eval_log.json`.

### 1.1 Weights and wandb

- **Trained weights are NOT in git** (4.6 GB/checkpoint exceeds GitHub's
  100 MB file limit). They exist only on the machine that trained them, at
  `data/outputs/run_<task>_abs_single_s42_200ep/checkpoints/` (`latest.ckpt`
  = epoch-200 model per task). To move them to another machine:
  ```bash
  rsync -av data/outputs <user>@<other>:/path/to/diffusion_policy/data/outputs
  ```
  or re-train from the runbook in §3 (~1–2.5 h per task).
- **Training curves and rollout videos are on wandb**: project
  `diffusion_policy_view` (account zihan-wa23-tsinghua-university) — square
  `runs/1h4oj5p6`, can `runs/q95ylpsc`, lift `runs/poearmxq`.
- **Committed to git** (so a fresh clone has the numbers): the sweep
  `eval_log.json` files + all 66 viewpoint videos under
  `data/eval_az5_*_ep200/`, and the per-batch training logs at
  `data/outputs/run_*_abs_single_s42_200ep/logs.json.txt`.

## 2. Configs and scripts added for M1

| File | Purpose |
|---|---|
| `diffusion_policy/config/task/{square,lift,can}_image_abs_single.yaml` | single-view variants of the stock `_abs` configs (wrist camera dropped) |
| `eval_novel_view.py` | novel-view eval harness (from the M1 planning commit) |
| `summarize_novel_view.py` | aggregates `eval_log.json` files into the degradation table |

Note: the prepared `{square,lift}_image_single.yaml` (relative actions) are
unused — the baselines were trained with `_abs_single` variants, matching the
published robomimic setup (`image_abs.hdf5`, `abs_action: True`, 10-dim
actions). Verified single-view at runtime: the zarr conversion reads exactly
one camera's worth of images per task (e.g. square: 30154 = raw step count,
not 2×).

## 3. Runbook used on this machine (verified)

```bash
conda activate robodiff        # exists; torch 2.8.0+cu128 (sm_120), robosuite 1.2.0, robomimic 0.2.0
cd <repo root>

# train (201 epochs: checkpoint+rollout fire on epoch % 50 == 0, and there is
# NO save at the end of training — 201 makes epoch 200 fire; 200 would stop at 150)
python train.py --config-dir=. --config-name=train_diffusion_unet_image_workspace.yaml \
  task=<task>_image_abs_single training.seed=42 training.device=cuda:0 training.num_epochs=201 \
  dataloader.num_workers=8 checkpoint.topk.k=1 logging.project=diffusion_policy_view \
  hydra.run.dir=data/outputs/run_<task>_abs_single_s42_200ep

# novel-view sweep (az_0, ±15°, ±30°, paired seeds)
python eval_novel_view.py -c <run>/checkpoints/latest.ckpt -o data/eval_az5_<task>_ep200 \
  -d cuda:0 --preset azimuth_sweep5

# resume (e.g. after a kill): num_epochs = epochs still wanted; the loop runs
# that many epochs from the restored counter (checkpoint epoch + num_epochs − 1
# is the last epoch saved, iff num_epochs % 50 lands right — 51 from 150 → 200)
python train.py <same args, minus task switch> training.num_epochs=51 checkpoint.topk.k=1 \
  hydra.run.dir=<same run dir>
```

Timings on this box (idle; fp32, no AMP — the workspace has none):
- square ~22 s/epoch (440 batches @ bs 64), can ~16 s/epoch, lift ~7 s/epoch
- rollout ~2–3 min (28 envs, 56 episodes); epoch-0 rollout is the worst case
- full sweep ~18 min (5 viewpoints × 50 episodes)
- checkpoint = 4.62 GB (policy + EMA + Adam state); `topk.k=1` + latest ≈
  9.2 GB/run. Root disk is 97–100% full — check `df -h` before running more.
  `/data` (15 TB) exists but is owned by another user and not writable.

## 4. Verified environment facts (all checked by running)

- `robodiff` env: torch **2.8.0+cu128** (the `conda_environment.yaml` pin
  `pytorch=1.12.1` is stale and unusable on RTX 5090 — do NOT recreate the
  env from it), robosuite 1.2.0, robomimic 0.2.0, mujoco_py 2.0.2.13.
  Offscreen rendering works with osmesa, EGL, and the default backend.
- Camera control at the mujoco_py level (no `CameraMover` in robosuite 1.2):
  `sim.model.cam_pos/cam_quat` + `sim.forward()`. `cam_quat` is **wxyz**;
  robosuite utils are **xyzw** — the harness converts explicitly.
- `hard_reset=False` (set by the stock runner) persists camera moves across
  soft resets; the harness re-applies the viewpoint at every reset, computed
  from the base pose captured on first use (never compounds).
- `eval_novel_view.py`'s checkpoint-loading path **has been exercised against
  real checkpoints** (all three sweeps above) — it mirrors `eval.py` and
  works, including the `${eval:...}` resolver dependency on importing the
  workspace class.
- The zarr cache (`<hdf5>.zarr.zip`) is keyed **only by hdf5 path**, not by
  `shape_meta` — a single-view run caches one camera; a later 2-camera run on
  the same path would `KeyError` on the wrist key. Delete the cache to switch.
- Dropping the wrist camera from `shape_meta` removes encoder FLOPs but not
  render cost: `camera_names` comes from the hdf5 `env_args`, so robosuite
  still renders both cameras every step.

## 5. Machine / operational notes

- This is the remote training machine: 2× RTX 5090 32 GB, 24 cores, 125 GB
  RAM, multiple users (load can spike from ~5 to ~25; expect 2× slowdowns).
- `n_envs: 28` is fine here (PROGRESS §5's old 6 GB-laptop warning is moot).
- Long jobs die when the Claude Code session ends (background tasks are killed
  with the session). For anything that must outlive the session, launch
  disowned: `setsid bash -c '<cmd> > log 2>&1' &` — used successfully for the
  can/lift sweeps.
- The epoch loop does not skip completed epochs on resume: it runs
  `num_epochs` iterations from the restored counter. LR on resume is the
  cosine schedule re-based to the new `num_epochs` (clamped ≥ 0, harmless in
  practice).
- `TopKCheckpointManager` state is in-memory only: on resume its map starts
  empty, so existing topk files on disk are never evicted by the new run (and
  delete stale ones yourself if space is tight).

## 6. Not yet done / next

- **M2 rendering** — see §7. Code is ready; the render itself has not been run.
- Open design questions to settle before M3: fusion module choice, and
  whether aux heads condition on camera-frame action history, Plücker map, or
  both (PROPOSAL.md §7).

## 7. M2 — multi-view data (code ready, rendering pending)

Goal: for every demo timestep, N simultaneous renders of the same scene state
from N fixed camera poses, plus camera parameters, so M3 (Plücker conditioning),
M4 (camera-frame aux heads) and M5 (single-novel-view inference) have training
data. Decisions: all three tasks; azimuth ring every 15° to ±90° (13 views);
full data (every demo, every step).

### Files added (all new; no upstream package file modified)

| File | Purpose |
|---|---|
| `generate_multiview_dataset.py` | offline re-renderer → ReplayBuffer-compatible zarr |
| `diffusion_policy/dataset/multiview_image_dataset.py` | `MultiViewImageDataset` + camera-math helpers |
| `tests/test_multiview_dataset.py` | CPU-only checks on a synthetic zarr (no simulator) |
| `config/task/{square,can,lift}_image_abs_multiview.yaml` | 13-view task configs |

### Verified locally (no rendering)

- `python tests/test_multiview_dataset.py` → **ALL MULTIVIEW DATASET CHECKS PASSED**:
  camera math (quaternion convention, intrinsics from fovy, projection sign and
  depth), zarr schema round-trip through `ReplayBuffer`/`SequenceSampler`,
  `__getitem__` shapes/dtypes/range, per-view content mapping, normalizer rules
  (image [0,1]→[-1,1], lowdim, invertible 10-dim abs action), `camera_params`
  round-trip, validation split, `get_all_actions`.
- `generate_multiview_dataset.py` imports cleanly and its pure helpers
  (`view_key`, `make_compressor`, `resolve_render_size`, `_draw_cross`) pass
  checks; the three task configs compose and resolve under Hydra with 13 rgb
  views, the right zarr/hdf5 paths and 10-dim abs actions.
- The render loop itself (`reset_to` + camera move + `sim.render`) **cannot be
  exercised here** — no robosuite env data on the dev laptop, and rendering is
  deliberately reserved for the GPU box.

### What must be on the render machine

Nothing new to download — the generator reads the hdf5 M1 already trained on:

| Needed | Status |
|---|---|
| `data/robomimic/datasets/{square,can,lift}/ph/image_abs.hdf5` | already there (M1) |
| `robodiff` env (robosuite 1.2.0, robomimic 0.2.0, torch 2.8) | already there |
| this code (`git pull`) | — |
| **≥ 20 GB free disk** | needs cleanup; see the warning below |

The hdf5 supplies everything the renderer consumes: `data/demo_*/states`
(replayed with `reset_to`), `data/demo_*/actions` (copied, converted to 10-dim
abs), `data/demo_*/obs/robot0_eef_{pos,quat}` + `robot0_gripper_qpos` (copied),
and the `env_args` used to build the env. No calibration files, depth or meshes
are needed — intrinsics come from `sim.model.cam_fovy`, extrinsics from the sim.

⚠️ **Disk cleanup is your call (destructive):** `data/robomimic_image.zip` is
84.75 GB and holds only transport/tool_hang/all-MH splits, which square/can/lift
PH do not use. Confirm it is re-downloadable before removing it. Stale topk
checkpoints are ~4.6 GB each.

### Runbook

```bash
cd <repo root> && git pull
python tests/test_multiview_dataset.py          # CPU-only sanity, no render
df -h .                                         # need ~20 GB

# pilot: 5 demos of one task + montage, inspect before committing hours
python generate_multiview_dataset.py \
  --dataset data/robomimic/datasets/square/ph/image_abs.hdf5 \
  --output data/multiview/square_ph_ring13.zarr \
  --limit-demos 5 --montage /tmp/ring5.png

# full run, disowned so it survives the session ending
for t in square can lift; do
  setsid bash -c "python generate_multiview_dataset.py \
    --dataset data/robomimic/datasets/$t/ph/image_abs.hdf5 \
    --output data/multiview/${t}_ph_ring13.zarr \
    > data/gen_$t.log 2>&1" &
done
```

The script prints three gates at the end of every run:

1. **az_0 re-render vs the hdf5's stored agentview image** — `mean|diff|` should
   be ≲ 3/255. A mean of ~40+ means the `[::-1]` flip is missing or doubled
   (mujoco's readPixels is bottom-up; robomimic flips it once, and the hdf5
   stores the flipped form). This is the check that catches an upside-down
   dataset, which otherwise looks plausible in a montage.
2. **projected gripper site** — how many (view, step) pairs the derived
   intrinsics/extrinsics put inside the frame. This validates the camera-parameter
   chain M3's Plücker maps depend on.
3. **ring montage PNG** — eyeball that the scene is visible at every angle and
   decide whether ±75°/±90° are worth keeping.

Expected cost: ~400k images total (31k square / 23.5k can / 12k lift steps × 13
views); ~18 GB raw, ~6–9 GB at Jpeg2k(level=50); ≈4 h render on an idle machine,
~2× under load. The jpeg2k encode is threaded over `--workers`.

### Deliberately out of scope for M2

Elevation views, the wrist camera as an extra view, and a live multi-view
env_runner (the stock runner builds obs from `shape_meta`, so it cannot serve
`view_XX_image` keys; M3's encoder will accept a *subset* of view slots so the
existing single-camera novel-view harness can evaluate it). Plücker maps and
camera-frame actions are not stored — both are deterministic functions of the
stored camera params (+ base actions), so M3/M4 compute them at train time.

Sizing note for M3: 13 views ⇒ ~13× encoder FLOPs per sample; per-sample view
subsampling will likely be needed to keep epoch time near M1's.
