# PROGRESS — Milestone 1: single-view DP baseline + novel-view eval harness

Last updated: 2026-09-10. See `PROPOSAL.md` (research direction) and `PLAN.md`
(milestones M1–M5).

**Training has intentionally NOT been run.** All files are prepared and the
non-training parts are verified; the runbook in §3 is to be executed on a
remote machine with a larger GPU.

---

## 1. What was done

All changes are **new files**; no existing package file was modified (the fork
stays as close to upstream as possible).

| File | Status |
|---|---|
| `diffusion_policy/config/task/square_image_single.yaml` | new — single-view (agentview only, wrist dropped) |
| `diffusion_policy/config/task/lift_image_single.yaml` | new — same for Lift |
| `eval_novel_view.py` | new — standalone novel-view eval harness |
| `tests/test_novel_view_harness.py` | new — self-contained harness test (random policy, no checkpoint needed) |
| `PROPOSAL.md` | rewritten for clarity |
| `PLAN.md` | milestone plan (M1 baseline … M5 single-novel-view inference) |
| `CLAUDE.md` | commands section updated |

Everything else in §2 below is *verified*, not assumed.

## 2. Verified on this machine (no training)

**A. Camera control works (robosuite 1.2.0 + mujoco_py 2.1.2).**
- robosuite 1.2.0 has **no `CameraMover`** (added in 1.3). A camera is moved at
  the mujoco_py model level: `cid = sim.model.camera_name2id(name)`, set
  `sim.model.cam_pos[cid]` / `sim.model.cam_quat[cid]`, then `sim.forward()`.
  Offscreen rendering reads these model fields on every render call, so this
  takes effect immediately.
- **Quaternion conventions differ**: `sim.model.cam_quat` is **wxyz** (mujoco);
  `robosuite.utils.transform_utils` is **xyzw**. The harness converts
  explicitly. Getting this backwards silently renders a wrong view.
- Only **fixed** cameras can be perturbed. `agentview` is world-fixed.
  `robot0_eye_in_hand` is attached to the gripper — perturbing it is not
  meaningful and is not supported.
- The camera move **persists across soft resets** (`hard_reset=False` is
  already set by the stock runner) — the harness still re-applies the
  viewpoint at every reset, computed from the base pose captured on first use,
  so poses never compound.

**B. Renders are exactly reproducible (verified bit-exact).**
- Same state + same camera pose ⇒ identical image (`max|diff| = 0`, repeated
  renders).
- az_0 ⇒ az_30 ⇒ az_0 restores the base render **exactly** (`mean|diff| = 0`),
  so paired evaluation across viewpoints is sound.
- 30° azimuth moves the camera on a circle about world z: base `[0.5, 0, 1.35]`
  → `[0.433, 0.25, 1.35]`, and the render changes by ~7.5% mean pixel
  difference.
- Gotcha found while testing: an unseeded reset takes robosuite's *random*
  reset branch (different state, different image). The harness always seeds
  every rollout, so this cannot happen — but do not call `set_viewpoint` +
  `reset()` without a seed.

**C. Pose math unit-verified** against robosuite's real `transform_utils`:
rotation directions, look-at construction (camera −z toward the base look-at
point, +y near world up), quaternion composition order, wxyz↔xyzw round trip,
and the identity (az_0) fast path.

**D. Success metric is available.** `EnvRobosuite.is_success()` returns a dict
with a `"task"` key. The stock runner never reads it (it logs only
`max(reward)`); the harness's wrapper subclass exposes it so
`AsyncVectorEnv.call('is_success')` works and per-viewpoint `success_rate` is
logged.

**E. Harness test** — `tests/test_novel_view_harness.py`, run from the repo
root with the dataset present:
```
python tests/test_novel_view_harness.py
```
It checks (1) the camera move changes the render, (2) the identity viewpoint
reproduces the base render exactly, (3) a 3-viewpoint sweep with a random
policy produces per-viewpoint `mean_score` / `success_rate` / `success_<seed>`
keys and one video per viewpoint.
Status: **PASSES end-to-end on this machine** (verified with a random policy;
final line `ALL NOVEL-VIEW HARNESS CHECKS PASSED`, ~2 min including env
creation).

Note on the test's structure: checks 1–2 create a rendering env in-process,
and check 3 forks workers that each create their own rendering env. Creating a
GL context *before* forking deadlocks the workers (found the hard way), so the
render check runs as a `--render-check` subprocess. Keep that separation if
you extend the test.

Two real defects were found and fixed this way: the fork-after-GL deadlock, and
a `past_action` name collision (the config flag and the action array shared one
name, so `bool(array)` raised when `past_action=True`).

**F. Datasets present locally** (gitignored, must be re-downloaded remotely):
```
data/robomimic/datasets/square/ph/image.hdf5   (2.5 GB, 200 demos, states stored)
data/robomimic/datasets/lift/ph/image.hdf5     (799 MB, 200 demos, states stored)
```
Both verified readable with 200 demos and the expected obs keys
(`agentview_image`, `robot0_eye_in_hand_image`, `robot0_eef_pos/quat`,
`robot0_gripper_qpos`) and stored `states` (needed later for the M2 re-render
pipeline).

## 3. Runbook for the remote machine

### 3.1 Environment

```bash
# system deps (mujoco_py needs these on Linux)
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

mamba env create -f conda_environment.yaml    # env name: robodiff
conda activate robodiff
```

Two fixes were needed locally; apply them if you hit the same errors:

```bash
# 1) patchelf missing -> mujoco_py/cymj build fails on import
#    (if you cannot sudo, the pip package works:)
pip install patchelf

# 2) huggingface_hub too new for this repo's diffusers==0.11.1
#    symptom: ImportError: cannot import name 'cached_download' from 'huggingface_hub'
pip install "huggingface_hub==0.14.1"
```

Verify:

```bash
python -c "import robosuite, robomimic; print(robosuite.__version__)"   # expect 1.2.0
python -c "from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace; print('ok')"
```

### 3.2 Data

```bash
cd <repo root>
python <site-packages>/robomimic/scripts/download_datasets.py \
  --tasks square lift --dataset_types ph --hdf5_types image \
  --download_dir data/robomimic/datasets
```
(or direct: `http://downloads.cs.stanford.edu/downloads/rt_benchmark/{task}/ph/image.hdf5`)

### 3.3 Smoke test the harness (no training, no checkpoint)

```bash
python tests/test_novel_view_harness.py
```
Expect `ALL NOVEL-VIEW HARNESS CHECKS PASSED` (~2 min: it creates 3 robosuite
envs and runs 18 short rollouts with a random policy).

### 3.4 Train the single-view baseline

Wandb is used by the workspaces — either `wandb login` first, or add
`logging.mode=disabled` for a smoke run.

```bash
# Lift, smoke (1 seed)
python train.py --config-dir=. --config-name=train_diffusion_unet_image_workspace.yaml \
  task=lift_image_single training.seed=42 training.device=cuda:0 \
  logging.project=diffusion_policy_view \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_s42'

# Square, 3 seeds
python ray_train_multirun.py --config-dir=. --config-name=train_diffusion_unet_image_workspace.yaml \
  --seeds=42,43,44 --monitor_key=test/mean_score \
  task=square_image_single \
  multi_run.run_dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
  multi_run.wandb_name_base='square_image_single_dp_singleview'
```
Checkpoints land in `data/outputs/<ts>_<name>_<task>/train_<i>/checkpoints/`
(`latest.ckpt` and top-k by `test_mean_score`).

### 3.5 Evaluate at novel viewpoints

```bash
# original view (unmodified eval.py, sanity check)
python eval.py -c <run>/train_0/checkpoints/latest.ckpt -o data/eval_orig -d cuda:0

# novel views
python eval_novel_view.py -c <run>/train_0/checkpoints/latest.ckpt \
  -o data/eval_az5 -d cuda:0 --preset azimuth_sweep5

# quick smoke of the harness against a real checkpoint
python eval_novel_view.py -c <ckpt> -o data/eval_smoke -d cuda:0 \
  --preset azimuth_sweep3 --n-test 4 --n-test-vis 2 --n-envs 4
```

Outputs: `eval_log.json` with keys
`test/<viewpoint>/{mean_score, success_rate, success_<seed>, sim_max_reward_<seed>, sim_video_<seed>}`
and videos under `media/<viewpoint>/`.

Presets: `azimuth_sweep5` (0°, ±15°, ±30°) and `azimuth_sweep3` (0°, ±30°);
the same seeds are used at every viewpoint (paired evaluation).

## 4. What to look for

- At `az_0` the baseline should approach the Diffusion Policy paper's robomimic
  image numbers (Lift ≈ 100%, Square ≈ 90–95% PH) — treat as a sanity band, not
  a gate.
- The result of interest is the **degradation curve** across ±15°/±30°. That
  curve is the reference the M3–M5 method must beat.
- Eyeball videos at ±30° before concluding anything: large offsets can move
  the arm or object partially out of frame, and "the camera is now looking
  somewhere useless" is a different finding from "the policy cannot handle a
  shifted view".

## 5. Machine notes and caveats

- The development machine used here has a 6 GB laptop GPU and ~15 GB RAM.
  **The stock eval config `n_envs: 28` assumes a 16-core/64 GB instance**
  (~1 GB RAM per robosuite env). On a smaller machine reduce it, e.g.
  `--n-envs 4 --n-test 8 --n-test-vis 2`. `train.py` rollouts use the same
  `n_envs` from the task config — lower it there too if memory is tight.
- `eval.py` and `eval_novel_view.py` both refuse an existing output directory
  (they prompt to overwrite) — use fresh `-o` paths.
- `max_steps` comes from the checkpoint config (400 for PH, 500 for MH).
- The harness never touches `media/` creation logic in the stock runner; it
  creates `media/<viewpoint>/` itself with `parents=True`.

## 6. Not yet done / next

- Train and evaluate the Square/Lift single-view baselines (§3.4–3.5) and
  record the per-viewpoint table. **Training has never been run** (by request
  it is deferred to the remote machine) — the checkpoint-loading path of
  `eval_novel_view.py` mirrors `eval.py` but has not been exercised against a
  real checkpoint yet. §3.5's `--n-test 4 --n-envs 4` smoke run is the first
  thing to try after the first training completes.
- After the baseline is reviewed: M2 (multi-view re-render pipeline using the
  stored `states` in the hdf5), then M3–M5 in `PLAN.md`.
- Open design questions to settle before M3: the exact fusion module and
  whether the auxiliary heads condition on camera-frame action history, on the
  Plücker map, or both (see `PROPOSAL.md` §7).
