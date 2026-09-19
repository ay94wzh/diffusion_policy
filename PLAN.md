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
- **M3 — View-conditioned encoder + fusion.** ✅ **DONE (2026-09-19).** Trained and swept on square and can. **Two findings, pointing opposite ways:**
  - ✅ **It solves square and can** — the two tasks L1 destroyed. At azimuths **excluded from the training pool**, M3 scores 0.50–0.57 (square) and 0.74–0.77 (can) where M1 is 0.00–0.02 and L1 is at the noise floor. The load-bearing half is that **L1 also fails at the poses it trained on** (~0.02 at ±30°, which it saw constantly), so L1's failure was never a failure to generalise — it never solved the task. Per-slot fusion turns view diversity from harmful into sufficient.
  - ❌ **The conditioning contributes nothing.** `m3off` (`use_plucker=False`, `use_eef_hist=False`) matches or beats `m3on` on both tasks at every viewpoint, inside the ±0.05 noise. **PROPOSAL §2/§7's central claim is not supported** — neither signal moved the number. What moved it was 7 slots, per-sample N∈[1,7], and attention fusion, which arrived as plumbing *for* the conditioning rather than as the hypothesis.
  - Elevation is a partial extrapolation win: 0.18–0.34 at +15° elevation where M1 is 0.00–0.04, but square collapses at −15° (0.00–0.04). Unexplained asymmetry.
  - The 2×2's `m3off` cell carried the attribution exactly as §10.6 predicted — M3-on vs L1 alone would have produced the false claim "Plücker conditioning fixes square".
  - Full tables, the two fatal bugs found before any long run, and the stated limits (n=1/±0.05 noise; four confounded architectural changes): `PROGRESS.md` §11. **Next question is architectural, not conditioning — see `PROGRESS.md` §6.**
- **M4 — Per-view aux heads.** New policy subclass (e.g. `DiffusionUnetImagePolicyAux`): encoder exposes per-view latents + fused latent (small interface extension, e.g. `forward_full`); per-view MLP head predicts the camera-frame action chunk; aux loss added in a `compute_loss` override. Workspace untouched (single optimizer covers all params). *Gate:* aux loss improves novel-view generalization; ablations: conditioning on/off, aux on/off. *(Its hooks exist: the encoder already keeps per-view latents separate before fusion, and the camera-frame action transform M4 needs is `eef_hist_to_cam` in `multiview_image_dataset.py`.)*
- **M5 — Single-novel-view inference (+ optional distillation).** Fusion already supports N=1 **and M3 is trained with randomized N precisely so this works**; at eval the novel camera pose is known from the sim, and `eval_novel_view.py --m3-slots K` already publishes the perturbed pose into the slot's cam key. Optional teacher→student distillation (single-view encoder regresses the multi-view fused latent). *Gate:* final sweep tables vs the Milestone 1 baseline curves.

## M3 design spec — ViewConditionedObsEncoder

Implements PROPOSAL.md §2 steps 1–2. **Status: IMPLEMENTED, CPU-verified, and
TRAINED — see `PROGRESS.md` §11 for the results and §10 for the implementation
record (parameter counts, the eight bugs caught before the run, the CPU test).**
Two deviations from the text below were decided after it was written and are
marked inline: the EE history is conditioned by **AdaGN**, and **N is randomized
per sample**.

> **Outcome note (2026-09-19):** the ablation below was designed to answer "does
> *either* conditioning signal suffice, or are both needed". The measured answer
> is **neither**: all four cells are indistinguishable (`PROGRESS.md` §11.3). The
> spec's other predictions held — matched capacity, the exact zero-gradient
> ablation, and the N=1 path. The architectural changes that came in as plumbing
> (7 slots, per-sample N, MHA fusion) are what produced the view-robustness gain,
> which the spec did not anticipate and did not test for. Anyone extending this
> should treat "which architectural ingredient" as the open question, not
> "which conditioning signal".

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
  shared resnet18 whose `conv1` is widened from 3 to 9 input channels. Output
  `z_v` (512-d). *(Correction: the config uses `weights: null`, so the resnet is
  randomly initialised — there are no pretrained RGB filters to copy, and the
  copy step is vestigial. It is kept so an `IMAGENET1K_V1` config would work
  unchanged, but it buys nothing today.)*
- **Modulation (added after this spec)**: `z_v`'s computation is additionally
  **modulated** by the camera-frame EE-pose history via AdaGN/FiLM at each
  residual stage — PROPOSAL §2.1's second condition, read literally. The FiLM
  heads are zero-init so training starts unconditioned.
- **Fusion**: `nn.MultiheadAttention` over the N view tokens with a single
  **learnable query** → `z_g` (512-d). Degenerates correctly at N = 1, which is
  what makes M5's single-novel-view inference work. **N is randomized per sample
  over `[1, K]`** (added after this spec) so that the N=1 case is in
  distribution rather than a shift — the slots stay fixed in number and a
  `view_mask` marks the live ones, so batches remain rectangular.
- **Output**: `[z_g | lowdim keys]`, matching the stock layout.

### The matched-capacity property (important)

With `fused_dim = 512`, M3's `output_shape()` is **521 = 512 + 9**, i.e. exactly
M1/L1's. So M3 and M1 differ in *training distribution and conditioning*, **not in
downstream capacity** — the UNet's `cond_dim` is identical. Without this, any M3
gain would be confounded with simply feeding the UNet a wider conditioning
vector.

**Measured, not asserted** (`PROGRESS.md` §10.2): `output_shape() == (521,)` for
all four flag combinations, `global_cond_dim == 1042`, and building both UNets
shows **all 148 parameter tensors shape-identical** — exactly the twelve
`cond_encoder.1.weight` matrices depend on this width. Scope limit worth stating
plainly: this is *downstream* capacity. The encoder itself grows 11,176,512 →
12,532,928 params (+12.14%), which is 0.47% of the total model.

### Camera-parameter plumbing

Poses arrive as ordinary obs keys, one per view: image key `view_07_image` is
paired with `view_07_cam` of shape `(10,)` =
`[pos(3), quat_wxyz(4), fovy(1), h(1), w(1)]`. The encoder consumes these and
**excludes them from the low-dim block**.

Passing them per-sample (rather than as a fixed per-slot buffer) is deliberate:
at test time the camera sits at a *novel* pose, so the pose cannot be baked in at
construction. Required changes:

- **Dataset** (`multiview_image_dataset.py`): ✅ emits the cam keys from
  `camera_params` (already validated by M2's gate 2), plus `view_mask` and the
  camera-frame `view_eef_hist`.
- **Eval** (`eval_novel_view.py`): ✅ `--m3-slots K` publishes the perturbed pose
  as the cam key. It is read **back from the sim** (`sim.model.cam_pos/cam_quat/
  cam_fovy`) rather than recomputed from the viewpoint spec, so the published
  pose is guaranteed to be the one that rendered the frame.

  One addition this spec did not anticipate: the env-side `shape_meta` must be
  **reduced** to the single rgb key robosuite actually renders, because
  `create_env` builds robomimic's obs-modality mapping from it and
  `EnvRobosuite.get_observation` would otherwise try to emit 7 cameras. The
  wrapper supplies the other slots itself.

### Required ablation

Two flags, so the control is a **2×2** rather than a single off-switch:
`use_plucker` (the ray-map channels) and `use_eef_hist` (the AdaGN modulation).
All four cells have **identical parameter counts** (12,532,928 — verified), so
each cell differs from another only in the conditioning signal it receives.
`use_plucker=False` keeps the widened `conv1` and feeds zeros, which drives the
gradient into those 6 channels to *exactly* 0.0.

The two single-signal cells are what answer PROPOSAL §7's own question ("do the
per-view auxiliary heads need the camera-frame action history as *conditioning*
as well as the Plücker map, or is one of the two sufficient?"). **The
`M3-off` cell carries the attribution** — M3-on vs L1 conflates "variable
multi-view" with "conditioning".

### Gates

1. **N=1 sanity**: one view + its pose should roughly reproduce the M1/L1 single
   view result. A large gap means the Plücker or FiLM path is corrupting the
   image path. Run **before** the bulk — it is the cheapest way to avoid
   spending ~15 h on a broken encoder. → ✅ **PASSED** (az_0 0.94 vs M1's 0.82,
   collapse to 0.00 at ±30 — M1's whole curve, §11.1).
2. **The headline**: 7 views on **square** (where view randomization alone fails,
   ≤0.08). Conditioning-on must beat conditioning-off there, and the square 2×2
   is read *before* committing to the can and lift runs. → ⚠️ **The prediction
   was wrong.** M3-on reaches 0.50–0.57 at held-out views, so the gate's first
   half passes; but conditioning-on does **not** beat conditioning-off — the two
   are indistinguishable, and `m3off` is nominally ahead (§11.3). The gate read
   *before* committing to can is what made the remaining cells cheap to skip.
3. **Lift control**: lift is already solved by the conditioning-free baseline
   (0.76–0.96 across ±75°), so M3 should *not* be headlined on lift — one M3-on
   run there is a regression check, not a result. → ⏸ **NOT RUN**; still open
   (`PROGRESS.md` §6).

Full ordering, the preflight, and the elevation sweep: `PROGRESS.md` §10.6.

### Open risks

- ~~**Crop alignment.**~~ **Resolved and verified.** The encoder samples the crop
  offsets once and applies them to both the image and its Plücker map (the stock
  `CropRandomizer` cannot be used — its `forward_in` discards the offsets,
  `crop_randomizer.py:88`). Checked exactly by spying on the offsets, and the
  eval path's `(4,4)` offset reproduces `torchvision.center_crop`.
- **Novel-pose eval is the real integration risk**, not the encoder — and it
  remains unproven. The harness now publishes the perturbed pose *and* must
  reduce the env-side `shape_meta` to one rgb key; `robomimic`'s obs-modality
  mapping has already bitten this project twice (§7.4). Implemented, never
  executed (`PROGRESS.md` §10.5).
- ~~**Action-history conditioning** is *not* in this spec.~~ **Now in.** It is
  implemented as **AdaGN** modulation of the backbone rather than as extra input
  channels: the history is a global vector, so modulation is the path that
  matches its shape, and a channel broadcast would be spatially constant. This
  does mean the "first experiment interpretable" staging is gone — mitigated by
  the 2×2, which isolates each signal's contribution.
- **New risk, introduced by randomization:** because N varies per sample, the
  dataset must read all K slot arrays, so the IO saving that a fixed single-slot
  config would get is not available. Bounded and measured in §7.5 terms (≈13 ms
  cold per sample; the store fits in page cache).

## Out of scope for now

ACT, and the mujoco pipeline (incl. the `mujoco_image_dataset.py` normalizer bug
noted in `CLAUDE.md`).

*(The line that used to sit here — "any M2–M5 coding until the baseline is done
and reviewed" — is discharged: M1/L1/M2 are done and M3 is done and trained.
M4 and M5 remain uncoded. **M3's results change what M4 and M5 are for, so both
entries above should be re-read with `PROGRESS.md` §11 and §6 in hand** — the
conditioning M4 would add aux losses to is measurably inert, and M5's
distillation premise assumes a multi-view latent carries something a single view
cannot, which M3's N=1 result has not yet demonstrated.)*
