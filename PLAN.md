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

- **M2 — Multi-view data with camera poses.** ✅ **DONE (2026-09-15) — all three tasks rendered and verified; see `PROGRESS.md` §7** (results, verification, and two findings that change M3's design). New files: `generate_multiview_dataset.py`, `diffusion_policy/dataset/multiview_image_dataset.py`, `tests/test_multiview_dataset.py` (CPU-only, passes locally), `config/task/{square,can,lift}_image_abs_multiview.yaml`. Original plan: re-renders each demo (subsampled steps) at N camera poses via stored mujoco `states` + `env.reset_to({'states': s})` + camera move + render; stores per-view images + per-view camera params (pose, fovy → intrinsics; extrinsics from pose) + per-view camera-frame actions (base-frame actions transformed by extrinsics) in a new zarr. New dataset class mirroring `RobomimicReplayImageDataset`. *Seam:* new shape_meta obs types (e.g. `camera`) need small branches in dataset + encoder key-classification loops (currently `rgb`/`low_dim` only). *Gate:* rendered views look right; N=1 training reproduces the single-view baseline. Verified locally (no rendering): dataset schema/sampling/normalizers/camera-parameter round-trip and config composition. To verify on the render box, each generation run prints gate 1 (az_0 re-render vs the hdf5's stored image — catches a wrong `[::-1]` flip), gate 2 (gripper projection through the derived intrinsics/extrinsics) and gate 3 (ring montage).
- **L1 — view-randomized single-slot baseline ("view diversity only").** ✅ **DONE, all three tasks (2026-09-17).** M1's *exact* architecture with its one camera slot filled per sample from a randomly drawn training view (7 of the 13 ring poses; the odd azimuths ±15/±45/±75 are never trained on). **The effect is task-dependent in both directions:** on **lift** it *solves* the problem — 0.76–0.96 success at every viewpoint out to ±75° where M1 collapses to 0.08/0.00 — while on **square and can** it is catastrophic (≤0.08 everywhere, including poses it trained on, where M1 scores 0.82/0.98). What separates lift from square/can is **not identified**; the leading hypothesis (lift needs no goal-directed placement, so less precise visual localization) has one task per side, and a scene-complexity confound can't be ruled out. Full tables and stated limits in `PROGRESS.md` §8. **Consequence for M3: it must win on square/can, and explain why lift didn't need it.**
- **M3 — View-conditioned encoder + fusion.** 🔵 **IN PROGRESS (2026-09-17).** New `ViewConditionedObsEncoder` in `model/vision/` (same dict-in → 1-D-out contract as `MultiImageObsEncoder`): shared backbone; per-view conditioning = Plücker map (channel-concat or feature embedding) + camera-frame action-history embedding; fusion = small transformer/cross-attention over per-view features that supports N=1. *Gate:* multi-view training ≈ stock 2-view DP at original views. **Required ablation: conditioning-off** (the only remaining control that holds capacity constant).
- **M4 — Per-view aux heads.** New policy subclass (e.g. `DiffusionUnetImagePolicyAux`): encoder exposes per-view latents + fused latent (small interface extension, e.g. `forward_full`); per-view MLP head predicts the camera-frame action chunk; aux loss added in a `compute_loss` override. Workspace untouched (single optimizer covers all params). *Gate:* aux loss improves novel-view generalization; ablations: conditioning on/off, aux on/off.
- **M5 — Single-novel-view inference (+ optional distillation).** Fusion already supports N=1; at eval the novel camera pose is known from the sim, so Plücker maps + action-history transforms are computed for the novel view. Optional teacher→student distillation (single-view encoder regresses the multi-view fused latent). *Gate:* final sweep tables vs the Milestone 1 baseline curves.

## M3 design spec — ViewConditionedObsEncoder

Implements PROPOSAL.md §2 steps 1–2. **Status: specified, not yet working** (see
`PROGRESS.md` §9 for the honest state of the draft code).

### Interface contract (why this is a small change)

`ViewConditionedObsEncoder` must be a **drop-in for `MultiImageObsEncoder`**:
`__init__(shape_meta, ...)`, `forward(obs_dict) -> (B·To, D)`, `output_shape()`.
`DiffusionUnetImagePolicy` reads `output_shape()[0]` and sets
`global_cond_dim = D * n_obs_steps`, so the encoder swaps in **through config
only** — no policy change. This is the same trick M3's predecessor encoders use.

### Architecture

```
view_0: image ─┐
        plucker ├─► [ shared resnet18, conv1 widened to 9ch ] ─► z_0 ─┐
view_1: image ─┤                                                     │
        plucker ├─► [ same shared backbone ]              ─► z_1 ─► [ fusion: MHA,
        ...     ─┘                                                     learnable query ] ─► z_g ─► + lowdim ─► out
```

- **Per view**: image (3ch) concatenated with its **Plücker ray map** (6ch) →
  shared resnet18 whose `conv1` is widened from 3 to 9 input channels (the
  pretrained RGB filters are copied into the first 3). Output `z_v` (512-d).
- **Fusion**: `nn.MultiheadAttention` over the N view tokens with a single
  **learnable query** → `z_g` (512-d). Degenerates correctly at N = 1, which is
  what makes M5's single-novel-view inference work.
- **Output**: `[z_g | lowdim keys]`, matching the stock layout.

### The matched-capacity property (important)

With `fused_dim = 512`, M3's `output_shape()` is **521 = 512 + 9**, i.e. exactly
M1/L1's. So M3 and M1 differ in *training distribution and conditioning*, **not in
downstream capacity** — the UNet's `cond_dim` is identical. Without this, any M3
gain would be confounded with simply feeding the UNet a wider conditioning
vector.

### Camera-parameter plumbing

Poses arrive as ordinary obs keys, one per view: image key `view_07_image` is
paired with `view_07_cam` of shape `(10,)` =
`[pos(3), quat_wxyz(4), fovy(1), h(1), w(1)]`. The encoder consumes these and
**excludes them from the low-dim block**.

Passing them per-sample (rather than as a fixed per-slot buffer) is deliberate:
at test time the camera sits at a *novel* pose, so the pose cannot be baked in at
construction. Required changes:

- **Dataset** (`multiview_image_dataset.py`): optionally emit `view_XX_cam` keys
  from the existing `camera_params` property (already validated by M2's gate 2).
- **Eval** (`eval_novel_view.py`): emit the *perturbed* pose as the cam key. The
  harness already computes it in `ViewpointImageWrapper._apply_viewpoint`, so it
  knows the answer; it just has to publish it into the obs dict.

### Required ablation

`use_plucker=False` drops the ray-map channels, keeping everything else — same
N views, same `fused_dim`, same capacity. **This is the only remaining control
for attributing an M3 gain to conditioning rather than to having more slots**
(the K-slot unconditioned variant was dropped). It is a constructor flag, so the
ablation costs one config, not one code path.

### Gates

1. **N=1 sanity**: one view + its pose should roughly reproduce the M1/L1 single
   view result. A large gap means the Plücker path is corrupting the image path.
2. **The headline**: 7 views on **square** (where view randomization alone fails,
   ≤0.08). Conditioning-on must beat conditioning-off there.
3. **Lift control**: lift is already solved by the conditioning-free baseline
   (0.76–0.96 across ±75°), so M3 should *not* be headlined on lift.

### Open risks

- **Crop alignment.** The stock encoder random-crops 76×76 from 84×84. The
  Plücker map must be cropped with the *identical* window or the conditioning
  desynchronises from the pixels. The draft does this itself for that reason
  (it cannot use `CropRandomizer`, which returns no offsets).
- **Novel-pose eval is the real integration risk**, not the encoder: the harness
  must publish the perturbed pose, and `robomimic`'s obs-modality mapping has
  already bitten this project twice (see `PROGRESS.md` §7.4).
- **Action-history conditioning** (PROPOSAL §2.1, second condition) is *not* in
  this spec. Staging it after the Plücker path works keeps the first experiment
  interpretable; it is M4-adjacent and shapes the per-view interface.

## Out of scope for now

ACT, mujoco pipeline (incl. `mujoco_image_dataset.py` normalizer bug), and any M2–M5 coding until the baseline is done and reviewed.
