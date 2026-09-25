# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Fork of the Columbia/Stanford [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) codebase: visuomotor policies learned as conditional denoising diffusion models over action trajectories. Training is Hydra-config-driven; evaluation rolls the policy out in a gym environment.

## Commands

All commands run from the repo root. The package is **not pip-installed and needs no build step** — every script self-bootstraps `sys.path`, and tests `os.chdir` to the repo root.

```console
# Environment (conda env name: robodiff, python 3.9, torch 2.8.0+cu128 — the 1.12.1 pin in conda_environment.yaml is stale; the live env was upgraded for the RTX 5090s and must not be recreated from the yaml)
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
mamba env create -f conda_environment.yaml   # or: conda env create -f conda_environment.yaml

# Train, single seed (wandb.login required — workspaces call wandb.init)
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# Train, multiple seeds via Ray
export CUDA_VISIBLE_DEVICES=0,1,2
ray start --head --num-gpus=3
python ray_train_multirun.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml --seeds=42,43,44 --monitor_key=test/mean_score -- multi_run.run_dir='...' multi_run.wandb_name_base='...'

# Evaluate a checkpoint (writes eval_log.json + media/*.mp4)
python eval.py --checkpoint data/0550-test_mean_score=0.969.ckpt --output_dir data/pusht_eval_output --device cuda:0

# Novel-view eval (standalone harness, no package changes; per-viewpoint success_rate/videos)
python eval_novel_view.py -c data/outputs/<ts>_train_diffusion_unet_image_<task>/train_0/checkpoints/latest.ckpt \
  -o data/eval_az5 -d cuda:0 --preset azimuth_sweep5   # or azimuth_sweep3 + --n-test/--n-test-vis/--n-envs for smoke

# Multi-view data (M2). Needs a GPU box with robosuite; nothing renders on a laptop.
python tests/test_multiview_dataset.py          # CPU-only schema/sampling check, no simulator
python generate_multiview_dataset.py \
  --dataset data/robomimic/datasets/square/ph/image_abs.hdf5 \
  --output data/multiview/square_ph_ring13.zarr --montage /tmp/ring13.png
# the run ends with 3 gates (az_0 vs the hdf5's stored image, gripper projection, ring montage);
# run it disowned: setsid bash -c '... > data/gen_square.log 2>&1' &
```

**Tests**: no pytest — each `tests/test_*.py` is a standalone script with its own `test()` function. Run individually, e.g. `python tests/test_replay_buffer.py`. Some contain hardcoded personal paths and are best-effort dev checks, not CI.

**Data**: `data/` is gitignored except narrow tracked exceptions (see `.gitignore`): every sweep under `data/eval_*/` (`eval_log.json`, plus videos where kept), the per-run training logs `data/outputs/run_*/logs.json.txt`, and the probe/screen JSONs (`data/probe_*`, `data/screen_*`) are **committed**. **This checkout is the coding-and-documents machine.** The robomimic PH datasets, the multi-view zarrs and every checkpoint live on the **remote training machine**; a bare `ls data/outputs/*/checkpoints` here legitimately returns nothing, and that is not a loss to repair. Trained checkpoints (NOT in git; 4.62 GB each) live under `data/outputs/run_*/checkpoints/` *there*, as do the zarrs `data/multiview/<task>_ph_ring13.zarr`; hydra outputs otherwise go to `data/outputs/<date>/<time>_<name>_<task_name>/`. **Corrections (2026-09-26):** the paths this file used to call "present locally" are on the training machine; the 84.75 GB archive `data/robomimic_image.zip` does **not** exist on either box — only the three `ph` tasks are available (no transport/tool_hang/MH splits) — and the M1 `run_*_abs_single_*/checkpoints/` were deleted on 2026-09-19 to free disk (their numbers are committed). Check `df -h` on the **training machine** before a campaign: a checkpoint is 4.62 GB and `topk.k=1` roughly doubles a run's footprint.

## Architecture

Organized for O(N+M) cost to add N tasks and M methods: tasks and methods interact only through a narrow interface. Adding a task = copy `dataset/`, `env_runner/`, and `config/task/` files for an existing task; adding a method = copy a `policy/` + `workspace/` + workspace config.

**Task side** (observation/data source):
- `Dataset` (`diffusion_policy/dataset/`) adapts a zarr replay buffer or robomimic hdf5 to samples `{'obs': ..., 'action': (Ta, Da)}`. Must implement `get_normalizer()`, `get_validation_dataset()`, `get_all_actions()`. Zarr loading goes through `common/replay_buffer.py` (`ReplayBuffer.copy_from_path(zarr_path, keys=[...])`) with `common/sampler.py` (`SequenceSampler` with pad/repeat semantics at episode edges, `get_val_mask`/`downsample_mask`).
- `EnvRunner` (`diffusion_policy/env_runner/`) builds vectorized gym envs (`AsyncVectorEnv` over `MultiStepWrapper` → `VideoRecordingWrapper` → raw env), runs `policy.predict_action` in a rollout loop, returns metrics dict.
- `config/task/<task>.yaml` wires dataset + env_runner + `shape_meta` + horizon (`n_obs_steps` = To, `n_action_steps` = Ta, `horizon` = T).

**Method side** (learning):
- `Policy` (`diffusion_policy/policy/`) wraps a diffusion model + noise scheduler: `predict_action(obs_dict)` does iterative denoising (conditioning region overwritten each step), `compute_loss(batch)` adds noise and regresses the target over the masked region. Masks come from `model/diffusion/mask_generator.py`.
- `Workspace` (`diffusion_policy/workspace/`) owns the whole training loop: optimizer, EMA (rollouts use EMA weights), LR scheduler, validation loss, periodic rollouts, top-k checkpointing, wandb + `logs.json.txt`. `train_diffusion_unet_lowdim_workspace.py` is the canonical implementation; other workspaces are near-copies.
- `config/train_<...>_workspace.yaml` composes a task via Hydra `defaults: [task: <task>]` and references its fields as `${task.obs_dim}` etc.

**Policies come in two observation flavors** — dataset and policy must match:
- *Low-dim*: obs is a `(B, To, Do)` tensor; policy ctor takes explicit `obs_dim`/`action_dim`.
- *Image*: obs is a dict keyed per `shape_meta.obs` entries (`type: rgb` or `type: low_dim`); the policy takes `shape_meta` and builds/instantiates an obs encoder from it — `MultiImageObsEncoder` (torchvision ResNet / R3M, `model/vision/`) for `DiffusionUnetImagePolicy`, robomimic `bc_rnn` for `DiffusionUnetHybridImagePolicy`. Features are flattened over `To` into the UNet's `global_cond_dim`.

**Hydra conventions**:
- Every config node uses `_target_: fully.qualified.ClassName`, instantiated by `hydra.utils.instantiate`; class paths mirror file paths.
- `${eval: ...}` (resolver registered in `train.py`) allows Python arithmetic inside YAML for derived dims.
- Workspace files have `@hydra.main(config_name=<own filename>)` at the bottom, so they are also runnable standalone.
- Checkpoints are `dill`-pickled payloads (`BaseWorkspace.save_checkpoint`) holding state dicts plus `payload['cfg']`; `eval.py` rebuilds the workspace class from the saved config and restores state.

## Fork-Specific State

- **The active work is the M1–M5 research line** (view-generalizable visuomotor policies) — see `PROPOSAL.md` (direction), `PLAN.md` (the live plan, read it first), **`PROGRESS.md`** (the module and its measured behaviour: M1–M4 in full — single-view baseline, multi-view data, view-conditioned encoder, per-view aux heads — plus one conclusion block per follow-up experiment), **`PROGRESS_DETAIL.md`** (the parked detail: full protocol, the investigation log section by section, the code map, an index of every results table in **Appendix A**, five parked investigations in **Appendix B**, and the **record of corrections in Appendix C** — read C before quoting any number), and `NOTES.md` (runbooks, timings, traps; closed runbooks in its appendices N1/N2). One-paragraph summary as of 2026-09-26: **M1, L1, M2, M3, M4, the N>1 confound, the N-diversity ladder, `[2,7]` and the M1 re-screen are done, and M5's capability is measured rather than pending** — every M3 number is already a single-camera N=1 inference at a novel pose. The headline: **PROPOSAL §2.2's fusion works** (square/can 0.55/0.74 held-out) while §2.1's Plücker/EE-history conditioning and §2.4's aux heads are **behaviourally inert**; **N>1 sampling is the load-bearing ingredient**; diversity has a **knee** between mean-N 2.0 and 3.0, not a slope; `[2,7]` shows N=1 *inference* works without ever training at N=1; and M1's baseline was **not** collapsed, so view-tiedness and collapse are distinct failure modes. **Next**: three runs that fill recorded holes — M3-on-lift, a second seed on `[1,5]`, a ±60°-pool run — plus the M5 write-up. Everything lands as new files with the upstream package untouched except documented seams; every experiment is a Hydra CLI override on an existing config.
- **Out of scope (per `PLAN.md`) — the mujoco image dataset.** `diffusion_policy/dataset/mujoco_image_dataset.py` loads a zarr with keys `robot_0_camera_images`, `robot_0_tcp_xyz_wxyz`, `robot_0_gripper_width`, `action_0_tcp_xyz_wxyz`, `action_0_gripper_width`, producing obs `image (T,3,H,W)` in [0,1] + `agent_pos (T,8)` (tcp xyz+wxyz+gripper) and 8-dim actions. **Known likely bug** at `mujoco_image_dataset.py:64`: `get_normalizer()` fits `agent_pos` on `tcp_xyz_wxyz` concatenated with *itself* (14 dims) instead of with `gripper_width`, which will fail against the 8-dim `agent_pos` produced by `_sample_to_data`.
- `image_pusht_diffusion_policy_cnn.yaml` at the repo root is a fully-resolved snapshot config (no `defaults` block) used as the template for new runs (stock task configs point at `data/pusht/...`, which is not on this box).
- `common/replay_buffer.py::copy_from_path` was patched to skip non-array (zarr.Group) `meta` entries — required by the mujoco zarr layout.
- `conda_environment.yaml` has an uncommitted one-line change: `free-mujoco-py==2.1.6` → `mujoco==2.3.7`.
- Compared to newer upstream versions, this checkout has **no** `diffusion_policy_3d/`, no DINOv2 encoders, no `@register_environment`/`make_policy` helpers, no `_TASK_CONFIG_PATH`, and no `train_benchmark.py`. Available vision encoders: torchvision ResNet / R3M (`model/vision/model_getter.py`, `MultiImageObsEncoder`) and robomimic `bc_rnn`.
- `PROPOSAL.md` describes the research direction: a view-aware temporal encoder over multi-view images conditioned on camera Plücker maps + camera-frame action history, with per-view auxiliary action heads and a fused latent feeding a base-frame diffusion policy. `PLAN.md` turns it into milestones: M1 = single-view DP baseline + novel-view eval harness (robomimic Square/Lift PH); M2-M5 = multi-view re-render data → conditioned encoder + fusion → aux heads → single-novel-view inference (M1-M4, the N>1 confound, the ladder, `[2,7]` and the M1 re-screen are done; M5's **inference capability is measured**, its distillation stage is not coded).
- Milestone 1 additions (all new files, no package changes): `config/task/{square,lift,can}_image_abs_single.yaml` (agentview-only, wrist dropped; the earlier `{square,lift}_image_single.yaml` relative-action variants are unused), `eval_novel_view.py` — standalone harness that perturbs a fixed camera (`agentview`) at the mujoco_py model level (`sim.model.cam_pos/cam_quat` + `sim.forward()`) per viewpoint, sweeps azimuth presets with paired seeds, and logs per-viewpoint `success_rate` via `EnvRobosuite.is_success()` — and `summarize_novel_view.py` (aggregates `eval_log.json` files into a degradation table). M1 is complete: baselines trained 201 epochs on can/lift/square and swept at az 0°/±15°/±30°; results in `PROGRESS.md` (success collapses to ≈0 at ±15° on all three tasks).
- Milestone 2 (multi-view data) additions — all new files, no package changes: `generate_multiview_dataset.py` (offline re-renderer: replays each hdf5 `states` through `env.reset_to`, orbits the fixed camera to 13 azimuth poses computed with `eval_novel_view._compute_perturbed_pose`, writes a ReplayBuffer-compatible zarr — images `uint8 (T,H,W,C)` + `Jpeg2k(level=50)`, lowdim/actions float32, per-view camera params in `meta`. It applies **exactly one `[::-1]`**: mujoco's readPixels is bottom-up and robomimic's `EnvRobosuite.get_observation` flips it once, which is the orientation stored in the hdf5); `diffusion_policy/dataset/multiview_image_dataset.py` (`MultiViewImageDataset` reading the zarr lazily via `ReplayBuffer.create_from_path`, plus `camera_params` for M3/M4 and the `fovy_to_intrinsics` / `project_world_to_pixel` / `quat_wxyz_to_mat` helpers); `tests/test_multiview_dataset.py` (CPU-only, synthetic zarr, no simulator); and `config/task/{square,can,lift}_image_abs_multiview.yaml` (13 rgb view keys `view_00..view_12`, rollouts disabled — the stock runner cannot serve multi-view obs; M3's encoder will accept a subset of view slots). Code is CPU-verified and **generation ran** (2026-09-15, all three tasks, 819k images, 4.5 GB, 8h) — but **the generated zarrs live on the training machine, not in this checkout**; see `PROGRESS.md` *M2* for results, verification and the M3-relevant findings. If they are ever regenerated, note the action-convention trap in `NOTES.md` *Traps*.
