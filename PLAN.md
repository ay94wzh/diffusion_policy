# Plan: View-Aware Temporal Encoder (see PROPOSAL.md)

## Constraints

- This is a fork of the original diffusion_policy repo — **modify it as little as possible**. Each milestone lands as **new files**; existing files are touched only at documented seams and only when unavoidable.
- Do the baseline first, then add the model step by step. Models are planned here but coded only in their own milestone.

## Milestone 1: Single-View DP Baseline + Novel-View Eval Harness

**✅ DONE (2026-09-12).** Single-view DP baselines trained on can/lift/square
(robomimic PH, agentview only, 201 epochs, 1 seed each) and swept at az
0°/±15°/±30° with the novel-view harness. Headline: success collapses to ≈0
at ±15° on all three tasks (square 0.82→0.02, can 0.98→0.08, lift 0.76→0.08
at az_0→az_m15). Full tables, runbook, and artifacts in `PROGRESS.md`.

Platform: robomimic + robosuite. Tasks: Square primary + Lift smoke, PH split, 3 seeds. Single-view DP (agentview only, wrist dropped). No ACT. No 2-view contrast.

Deliverables:
1. `diffusion_policy/config/task/{square,lift}_image_single.yaml` — copies of stock task configs without `robot0_eye_in_hand_image` (new files).
2. `eval_novel_view.py` (repo root) — standalone novel-view eval harness. Reuses repo classes via import (`create_env`, `RobomimicImageWrapper`, `MultiStepWrapper`, `VideoRecordingWrapper`, `VideoRecorder`, `AsyncVectorEnv`); **no package files are modified**. Perturbs a fixed camera pose at the mujoco_py model level (`sim.model.cam_pos/cam_quat` + `sim.forward()`) per viewpoint; sweeps viewpoints × seeds with paired seeds; logs per-viewpoint `success_rate`/`mean_score`/videos to `eval_log.json`.
3. Result: per-viewpoint degradation curve (azimuth 0°, ±15°, ±30°) for the single-view baseline — the reference the proposed method must beat.

Verified environment facts (robosuite 1.2.0, robomimic 0.2.0, env `robodiff`):
- No `CameraMover` in robosuite 1.2.0 (added in 1.3). Offscreen rendering reads mjModel cam fields on every render → mutating them + `sim.forward()` moves the camera. `cam_quat` is **wxyz** (mujoco); `robosuite.utils.transform_utils` is **xyzw** — convert explicitly.
- `hard_reset = False` (already set by the stock runner) makes camera pose persist across soft resets; the harness re-applies the viewpoint at every reset (idempotent, computed from the base pose captured on first use).
- Fixed cameras only: `agentview` is world-fixed; `robot0_eye_in_hand` is body-attached (perturbation not meaningful there).
- `EnvRobosuite.get_observation()` re-renders on demand (`force_update=True`) — call after moving the camera to avoid a stale first frame.
- `EnvRobosuite.is_success()` → dict with `"task"` key (the stock runner never reads it; the harness's script-local wrapper subclass exposes it).
- Blocker: `patchelf` must be installed (`sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`) or robosuite import fails.

Runbook:

```bash
# train (lift smoke, 1 seed)
python train.py --config-dir=. --config-name=train_diffusion_unet_image_workspace.yaml \
  task=lift_image_single training.seed=42 training.device=cuda:0 logging.project=diffusion_policy_view \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_s42'

# train (square baseline, seeds 42/43/44 — one at a time on the 6GB GPU;
# on a multi-GPU machine use ray_train_multirun.py --seeds=42,43,44 instead)
for s in 42 43 44; do
python train.py --config-dir=. --config-name=train_diffusion_unet_image_workspace.yaml \
  task=square_image_single training.seed=$s training.device=cuda:0 logging.project=diffusion_policy_view \
  hydra.run.dir="data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_s$s"
done

# original-view sanity check via unmodified eval.py
python eval.py -c <ckpt> -o data/eval_orig -d cuda:0
# novel-view eval
python eval_novel_view.py -c <ckpt> -o data/eval_az5 -d cuda:0 --preset azimuth_sweep5
# fast smoke
python eval_novel_view.py -c <ckpt> -o data/eval_smoke -d cuda:0 --preset azimuth_sweep3 --n-test 4 --n-test-vis 2 --n-envs 4
```

Reference targets (DP paper, robomimic image-obs PH — verify, don't gate): Lift ≈ 100%, Square ≈ 90–95% at the original view. Degradation at ±15°/±30° is the expected result.

## Roadmap: adding the model step by step (planned, NOT coded yet)

Each step lands as new files, has an experiment gate before proceeding, and keeps `DiffusionUnetImagePolicy` untouched (the obs encoder is injected via `cfg.policy.obs_encoder`, so new encoders swap in through config).

- **M2 — Multi-view data with camera poses.** 🟡 **Code ready, rendering pending — see `PROGRESS.md` §7.** New files: `generate_multiview_dataset.py`, `diffusion_policy/dataset/multiview_image_dataset.py`, `tests/test_multiview_dataset.py` (CPU-only, passes locally), `config/task/{square,can,lift}_image_abs_multiview.yaml`. Original plan: re-renders each demo (subsampled steps) at N camera poses via stored mujoco `states` + `env.reset_to({'states': s})` + camera move + render; stores per-view images + per-view camera params (pose, fovy → intrinsics; extrinsics from pose) + per-view camera-frame actions (base-frame actions transformed by extrinsics) in a new zarr. New dataset class mirroring `RobomimicReplayImageDataset`. *Seam:* new shape_meta obs types (e.g. `camera`) need small branches in dataset + encoder key-classification loops (currently `rgb`/`low_dim` only). *Gate:* rendered views look right; N=1 training reproduces the single-view baseline. Verified locally (no rendering): dataset schema/sampling/normalizers/camera-parameter round-trip and config composition. To verify on the render box, each generation run prints gate 1 (az_0 re-render vs the hdf5's stored image — catches a wrong `[::-1]` flip), gate 2 (gripper projection through the derived intrinsics/extrinsics) and gate 3 (ring montage).
- **M3 — View-conditioned encoder + fusion.** New `ViewConditionedObsEncoder` in `model/vision/` (same dict-in → 1-D-out contract as `MultiImageObsEncoder`): shared backbone; per-view conditioning = Plücker map (channel-concat or feature embedding) + camera-frame action-history embedding; fusion = small transformer/cross-attention over per-view features that supports N=1. *Gate:* multi-view training ≈ stock 2-view DP at original views.
- **M4 — Per-view aux heads.** New policy subclass (e.g. `DiffusionUnetImagePolicyAux`): encoder exposes per-view latents + fused latent (small interface extension, e.g. `forward_full`); per-view MLP head predicts the camera-frame action chunk; aux loss added in a `compute_loss` override. Workspace untouched (single optimizer covers all params). *Gate:* aux loss improves novel-view generalization; ablations: conditioning on/off, aux on/off.
- **M5 — Single-novel-view inference (+ optional distillation).** Fusion already supports N=1; at eval the novel camera pose is known from the sim, so Plücker maps + action-history transforms are computed for the novel view. Optional teacher→student distillation (single-view encoder regresses the multi-view fused latent). *Gate:* final sweep tables vs the Milestone 1 baseline curves.

## Out of scope for now

ACT, mujoco pipeline (incl. `mujoco_image_dataset.py` normalizer bug), and any M2–M5 coding until the baseline is done and reviewed.
