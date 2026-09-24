# PROGRESS

Training a diffusion policy that keeps working from camera viewpoints it was never
trained on, and at inference time from a **single camera placed at a novel pose**.
`PROPOSAL.md` holds the direction and the original predictions; `PLAN.md` what remains;
`NOTES.md` the operational detail (runbooks, timings, disk, recurring traps).

Last updated 2026-09-24.

## The ladder, and what each rung found

| rung | what changed | verdict |
|---|---|---|
| **M1** | single-view DP baseline (agentview at az_0), three tasks | collapses to ≈0 at ±15° azimuth on all three |
| **L1** | view diversity only: M1's exact model, its single camera slot filled per sample from a randomly drawn training view | task-split **in both directions** — solves lift out to ±75°, destroys square/can |
| **M2** | multi-view data: a 13-pose azimuth ring re-rendered from the demos' stored simulator states | validated by the N=1 gate; ±75°/±90° views are low value |
| **M3** | view-conditioned encoder: per-view latents, Plücker + camera-frame-history conditioning, K=7 slots, per-sample N∈[1,7], MHA fusion | **solves square and can at held-out views**; its conditioning contributes nothing |
| **M4** | per-view camera-frame auxiliary action heads | mechanism is real (`aux_loss` falls 52×), behaviour is null (+0.02/+0.04) |
| **N>1** | M3's model with one CLI line changed so every sample sees a single view | **the load-bearing ingredient** — reproduces L1, not M3 |

Internal labels, used in the code and configs: **L1** names this fork's second rung
(M1's architecture, randomized view) — L0 is M1 itself, and L2–L4 are M3, M4 and M5.

Every number in this document is **n=1 model, 50 paired episodes**, and the effective
noise on such a sweep is larger than the ±0.05 the plan assumed — two sweeps of the same
checkpoint on the *same* camera pose differ by 0.14 (see *Noise and resolution*). Nulls
here mean "indistinguishable at this resolution", not "proven identical".

The three conclusions, in the order the project reached them:

1. **View diversity alone is neither sufficient nor harmless.** On lift it is
   sufficient — a conditioning-free baseline reaches 0.76–0.96 at every viewpoint out
   to ±75°, where the single-view baseline is at 0.08 and 0.00. On square and can the
   same intervention is catastrophic: ≤0.08 everywhere, *including the poses it trained
   on*.
2. **M3's architecture fixes square and can; its conditioning does not.** The two tasks
   L1 destroyed reach 0.55 / 0.74 success at azimuths excluded from the training pool.
   Turning off the Plücker map and the camera-frame history changes nothing measurable,
   and neither do the auxiliary heads M4 added.
3. **Multi-view sampling is the ingredient.** Of the four things that changed together
   between L1 and M3, the one that matters is N>1: the same encoder forced to one view
   per sample scores 0.04 / 0.05, indistinguishable from L1.

Honest summary against the proposal: **§2.2's fusion works; §2.1's geometric
conditioning and §2.4's auxiliary heads are both inert.**

## Setup and protocol

**Platform.** robomimic PH demonstrations for **square** (primary), **can**, and
**lift**, in the `_abs` variant (absolute 10-dim actions). Observations are 84×84 RGB
from the **agentview** camera only — the wrist camera is dropped, matching the
single-view inference setting. Diffusion Policy: conditional UNet over action chunks,
horizon 16, `n_obs_steps` 2, `n_action_steps` 8, batch 64, AdamW + cosine LR + EMA,
201 epochs, seed 42, fp32 (the workspace has no AMP). Rollouts use the EMA weights.

**Novel-view evaluation** (`eval_novel_view.py`). A fixed camera is moved at the
mujoco_py level (`sim.model.cam_pos/cam_quat` + `sim.forward()`), and the viewpoint is
re-applied at every reset from a base pose captured on first use, so repeated resets
never compound. Policy rollouts at each viewpoint use **paired episodes** — the same
episode seeds across viewpoints — so a difference is attributable to the camera rather
than to the episode draws. 50 episodes per viewpoint. Two metrics: strict
`EnvRobosuite.is_success()` (`success_rate`) and max-reward (`mean_score`).
`success_rate` is the stricter of the two (M1 lift at az_0: 0.76 success vs 1.00 mean).

**Viewpoints.** Azimuth presets `azimuth_sweep5` (0, ±15°, ±30°) and `azimuth_interp`
(0, ±15°, ±30°, ±45°, ±60°, ±75°). Elevation is `elevation_az0` — an **orbit** about the
look-at point at 0/±15°, which is the off-manifold test. Azimuth past ±90° was rejected
for extrapolation: those views are dominated by the table edge. `el_0` is the same
camera pose as `az_0`, which makes it the internal consistency check.

**Multi-view data (M2).** A 13-pose azimuth ring every 15° out to ±90°. The **even
indices** (az −90/−60/−30/0/+30/+60/+90, every 30°) form the *training pool*; the **odd
indices** (±15/±45/±75) are **never sampled during training**, so every "held out" column
in this document is held out by construction. The single rgb key keeps the live camera's
name, `agentview_image` — it is a slot label, not a zarr array name — which is what lets
training rollouts and the stock evaluation path run unchanged.

### Noise and resolution

These three measurement properties govern how every table below should be read.

- **±0.05 was the assumed rollout noise; it is optimistic.** Two 50-episode sweeps of the
  same checkpoint through two presets that place the camera at the *same* pose differ by
  **0.14** (`m4on`: `el_0` 0.70 vs `az_0` 0.84). Unseeded diffusion sampling adds
  ~0.08 spread on repeat evals of one checkpoint (square epoch-150 scored 0.86 once and
  0.94 on re-eval). Treat differences below ~0.1 as unmeasured.
- **`mean_score` hides effects that `success_rate` shows.** In-training rollouts log
  `mean_score` only; on lift it saturates at 1.000 for a policy that succeeds 0.76 of the
  time. All four M3 square cells sit at 0.76–0.94 in `mean_score` while spanning 0.12 in
  the strict sweep.
- **`val_loss` does not track rollout behaviour.** All four square M3 cells end at
  0.0568–0.0602 — essentially L1's 0.060 — while rolling out like M1 rather than L1. It
  rises ~3× after epoch ~50 for every model (overfitting on the 2% val split): `m3off`
  0.0190 → 0.0602, the N=1 gate 0.0189 → 0.0499, L1 0.0198 → 0.0599. Compare it only at
  matched epochs. M4's `val_loss` additionally contains the aux term (0.0186 diff + 0.017
  aux = 0.037, observed 0.0372) and is not comparable to M3's at all.

## M1 — single-view baseline: the reference curve

**Method.** Diffusion Policy on one camera at the training pose (az_0), 201 epochs, seed
42, one run per task.

**Results** — `success_rate`, 50 paired episodes (`azimuth_sweep5`):

| viewpoint | square | can | lift |
|---|---|---|---|
| az_0 | 0.82 | 0.98 | 0.76 |
| az_m15 | 0.02 | 0.08 | 0.08 |
| az_p15 | 0.00 | 0.00 | 0.08 |
| az_m30 | 0.00 | 0.00 | 0.00 |
| az_p30 | 0.00 | 0.00 | 0.00 |

`mean_score`, same sweeps:

| viewpoint | square | can | lift |
|---|---|---|---|
| az_0 | 0.84 | 0.98 | 1.00 |
| az_m15 | 0.02 | 0.08 | 0.46 |
| az_p15 | 0.00 | 0.00 | 0.50 |
| az_m30 | 0.00 | 0.00 | 0.00 |
| az_p30 | 0.00 | 0.00 | 0.22 |

**Conclusion.** The baseline is extremely view-tied: at az_0 it sits at or near the
published band (paper: Lift ≈ 1.00, Square ≈ 0.9–1.0; here 0.98 and 0.82), and success
collapses to ≈0 at the smallest perturbation tested. This is the reference curve the
later milestones must beat. The collapse is a property of the policy, not of the camera
looking somewhere useless: the displaced views keep their texture (frame std 66–69 vs
az_0's 65) and differ from az_0 by only ~10–11% mean pixel value on same-seed first
frames, so the scene is fully visible at ±15°/±30°.

## M2 — multi-view data (rendered 2026-09-15)

**Method.** For every demo timestep: replay the stored mujoco `states` through
`env.reset_to`, orbit the fixed camera to each of 13 azimuth poses, render, and write a
ReplayBuffer-compatible zarr — per-view 84×84 images (`uint8`, `Jpeg2k(50)`), low-dim
observations, absolute actions, and per-view camera parameters in `meta`. Camera-frame
actions and Plücker maps are **not** stored: both are deterministic functions of the
stored camera parameters (plus the base actions), so M3/M4 compute them at train time.

**Results.** `--workers 4`, one task at a time, sequential; **8h00m total** at ~2.2
steps/s.

| task | steps | images (×13) | gate 1 mean\|diff\| | gate 2 in-frame | time | on disk |
|---|---|---|---|---|---|---|
| square | 30154 | 392k | 1.454/255 PASS | 33/39 | 3h50m | 1.6 GB |
| can | 23207 | 302k | 2.801/255 PASS | 39/39 | 2h59m | 2.4 GB |
| lift | 9666 | 126k | 1.502/255 PASS | 39/39 | 1h11m | 496 MB |

**819k images, 4.5 GB on disk.** Three gates run at the end of every generation:

1. **az_0 re-render vs the hdf5's stored image** — per-255 mean |diff| should be ≲ 3, and
   ~40 is the signature of a missing or doubled `[::-1]` flip (mujoco's `readPixels` is
   bottom-up and robomimic flips it once). This is what catches an upside-down dataset,
   which otherwise looks plausible in a montage. Verified at demos 0/100/199: square
   1.28–1.30, can 2.78–2.80, lift 1.24–1.35 — so the generated az_0 view is provably the
   original dataset camera. The per-task spread is scene-dependent re-render fidelity,
   not compression.
2. **Projected gripper site** — how many (view, step) pairs the derived intrinsics and
   extrinsics put inside the frame. This validates the camera-parameter chain the Plücker
   maps depend on.
3. **Ring montage** — the scene is visible at every angle.

**Findings that changed M3's design.**

- **Per-sample view subsampling saves encoder FLOPs but not IO.** `SequenceSampler` reads
  *every* zarr key regardless of `shape_meta`, so a one-view config still reads all 13 —
  at full 26-frame length, making it ~6× *slower* per sample than the 13-view config
  (156 ms vs 24 ms, reproducible and order-independent). M3 must restrict the sampler's
  keys, not just `shape_meta`. (13 views costs ~24 ms/sample cold; the whole square store
  is 1.6 GB, so warm training IO is far cheaper than that.)
- **±75°/±90° are low value** — dominated by the table edge, with the object small or
  partly out of frame. They were kept (2/13 of the bytes) because the encoder accepts a
  subset of view slots, so no re-render was needed.

**The N=1 gate: why everything later is measured on the generated zarr.** A run trained
on the generated data using **only** its az_0 view — the same single-camera-pose setup as
M1, from a different data source. It is a *fidelity control* on M2, not a new method:

| viewpoint | M1 success | N=1 success | M1 mean | N=1 mean |
|---|---|---|---|---|
| az_0 | 0.760 | **0.840** | 1.000 | 1.000 |
| az_m15 | 0.080 | 0.040 | 0.460 | 0.280 |
| az_p15 | 0.080 | 0.160 | 0.500 | 0.620 |
| az_m30 | 0.000 | 0.000 | 0.000 | 0.000 |
| az_p30 | 0.000 | 0.000 | 0.220 | 0.240 |

Trained on the generated zarr, the model reproduces M1's **whole degradation curve** —
high at az_0, collapse at ±15°, zero at ±30° — not merely its az_0 score. Residual
differences sit inside the rollout noise.

## L1 — view diversity only: does showing many views buy invariance?

**Method.** M1's *exact* architecture — one rgb key, `MultiImageObsEncoder`, resnet18 —
with its single camera slot filled per sample from a randomly drawn **training** view (the
ring's seven even indices, every 30°). Both `n_obs_steps` frames come from that same
view, so no viewpoint changes within a sample. There is **no pose information anywhere**:
the model is never told which view it is looking from. The rgb key is named
`agentview_image`, the live camera's name, which keeps the rollout environment and the
evaluation harness stock.

**Results** — `success_rate`, `azimuth_interp`, 50 paired episodes:

| viewpoint | M1 lift | **L1 lift** | M1 can | L1 can | M1 square | L1 square (s42 / s43) |
|---|---|---|---|---|---|---|
| az_0 | 0.760 | **0.960** | 0.980 | 0.020 | 0.820 | 0.020 / 0.040 |
| az_p15 | 0.080 | **0.920** | 0.000 | 0.020 | 0.000 | 0.020 / 0.020 |
| az_m15 | 0.080 | **0.900** | 0.080 | 0.020 | 0.020 | 0.000 / 0.020 |
| az_p30 | 0.000 | **0.940** | 0.000 | 0.020 | 0.000 | 0.000 / 0.080 |
| az_m30 | 0.000 | **0.880** | 0.000 | 0.020 | 0.000 | 0.000 / 0.040 |
| az_p45 | — | 0.960 | — | 0.020 | — | 0.020 / 0.040 |
| az_m45 | — | 0.860 | — | 0.060 | — | 0.000 / 0.040 |
| az_p60 | — | 0.900 | — | 0.040 | — | 0.060 / 0.040 |
| az_m60 | — | 0.840 | — | 0.000 | — | 0.020 / 0.000 |
| az_p75 | — | 0.840 | — | 0.000 | — | 0.040 / 0.000 |
| az_m75 | — | 0.760 | — | 0.020 | — | 0.020 / 0.020 |

In-training rollouts at az_0 (`mean_score`, for the record — note it saturates on lift):
lift M1 1.000 / L1 1.000; square M1 0.880 / L1 0.020–0.080; can M1 0.980 / L1 0.020.

**Conclusion.** The effect is task-dependent **in opposite directions**.

- **On lift, 7-pose randomization solves the problem outright.** L1 holds 0.76–0.96 at
  every viewpoint out to ±75°, where M1 collapses to 0.08 (±15°) and 0.00 (±30°). A
  conditioning-free baseline already achieves the project's stated goal on this task.
- **On square and can, the same intervention is catastrophic:** ≤0.08 *everywhere*,
  including the poses it trained on, where M1 scores 0.82 and 0.98.

Read L1's low numbers as a **noise floor, not as weak success**: 0.00–0.08 over 50
episodes is 0–4 episodes, and that is the floor at az_0 — the one pose it definitely
trained on. The can column confirms the square pattern exactly: uniform ~0 across all 11
viewpoints, trained poses included. There is no third behaviour; the tasks split 2–1.

**What separates lift from square/can is not identified.** The cleanest structural
difference is that lift requires no goal-directed placement — grasp-and-raise, where the
target is wherever the object already is, versus nut-onto-peg and can-into-bin. That
would make the axis "how much precise spatial localization from the image the task
needs". It is a **hypothesis with one task per side, not a finding**, and a
scene-complexity confound cannot be ruled out (lift's plain table and cube versus can's
cluttered shelf). A supporting limit: L1's `val_loss` on square is ~2× M1's (0.060 vs
0.029), so it does fit worse — but a 2× loss gap does not explain a 44× rollout gap, and
M3 below shows `val_loss` does not track rollout behaviour at all.

## M3 — view-conditioned encoder

### Method

One shared resnet18 is applied to each of **K=7 slots**. `conv1` is widened from 3 to 9
input channels; the 6 extra channels carry the view's **Plücker ray map** (per-pixel ray
origin and direction, encoding intrinsics and extrinsics). The per-view latent `z_v` is
additionally **modulated** by the camera-frame end-effector history through **AdaGN /
FiLM** at each residual stage; the FiLM heads are zero-init, so training starts
unconditioned. Fusion is `nn.MultiheadAttention` over the N view tokens with a single
**learnable query** → `z_g` (512-d), which degenerates correctly at N=1. **N is
randomized per sample** over [1, K] while the slot count stays fixed at K with a
`view_mask` marking the live ones, so batches stay rectangular and the N=1 setting used
by rollouts, evaluation, and M5 is *in distribution* rather than a shift.

**Matched capacity.** With `fused_dim = 512`, `output_shape()` is **521 = 512 + 9**,
identical to M1's and L1's — so M3 differs from the baselines in *training distribution
and conditioning*, not in downstream capacity. Verified, not asserted: all **148**
`ConditionalUnet1D` parameter tensors are shape-identical to a stock-encoder policy's,
and `global_cond_dim` is 1042 (2 × 521) in both.
Scope limit: this is downstream capacity; the encoder itself grows 11,176,512 →
**12,532,928** parameters (+12.14%, of which the fusion MHA is 1,050,624 and the widened
`conv1`'s six extra channels 18,816), which is 0.47% of the total model (UNet
277,632,138), so the M3 encoder is 0.49% of the UNet.

**The ablation.** Two constructor flags, so the control is a 2×2 rather than a single
off-switch: `use_plucker` (the ray-map channels) and `use_eef_hist` (the AdaGN
modulation). All four cells instantiate **12,532,928** parameters — each differs from
another only in the conditioning signal it receives. `use_plucker=False` keeps the
widened `conv1` and feeds zeros, which drives the gradient into those 6 channels to
*exactly* 0.0; `use_eef_hist=False` provably ignores the history tensor. The EMA weights
confirm the ablation applied: the six ray channels are non-zero only in the two
`use_plucker=True` cells, and all four checkpoints' `conv1` and `fusion_query` hashes are
distinct.

**Plumbing decisions that matter for reading the results.**

- Poses arrive as ordinary obs keys, one per view: the image key `view_07_image` is paired
  with `view_07_cam` of shape `(10,)` = `[pos(3), quat_wxyz(4), fovy(1), h(1), w(1)]`.
  Passing the pose **per sample** rather than as a fixed per-slot buffer is deliberate —
  at test time the camera sits at a *novel* pose, so it cannot be baked in at construction.
- The cam keys, the mask, and the EE history live **outside `shape_meta`**, and
  `shape_meta` keeps exactly **one** rgb key. This is load-bearing: `create_env` builds
  robomimic's obs-modality mapping from `shape_meta`, so seven rgb keys would make the
  rollout environment demand seven cameras — which is why the 13-view M2 configs disable
  rollouts. Keeping one key is what lets M3 keep training rollouts at all.
- Non-image keys get **identity normalizers, registered explicitly but never fitted**.
  Mandatory (the policy normalizes every key present), but they must not be fitted: under
  `mode='limits'` a constant dim maps to 0, which would silently destroy a novel camera
  pose at evaluation.
- Evaluation uses `eval_novel_view.py --m3-slots 7`, which reduces the env-side
  `shape_meta` to the single renderable key and serves the perturbed pose into slot 0 —
  an **N=1 inference**, M5's setting. The pose is read back **from the simulator**
  (`sim.model.cam_pos/cam_quat/cam_fovy`) rather than recomputed from the viewpoint spec,
  so the published pose is guaranteed to be the one that rendered the frame.
- Correctness of the conventions is pinned by `tests/test_view_conditioned_obs_encoder.py`
  (Plücker convention to 6.66e-16 against the numpy projector M2's gate 2 validated,
  with five mutation power checks; crop alignment; matched capacity; exact ablations) and
  `tests/test_multiview_dataset.py` (camera math, zarr schema, normalizers).

**The N=1 fidelity gate** (single slot, `view_pool=[6]` = az_0) — a *fidelity control*,
not a result:

| viewpoint | N=1 gate | M1 (hdf5, 1 view) | L1 (7 views) |
|---|---|---|---|
| az_0 | **0.94** | 0.82 | 0.02 |
| az_p30 | 0.00 | 0.00 | 0.00 |
| az_m30 | 0.00 | 0.00 | 0.00 |

It reproduces M1's whole curve. 0.94 is *above* M1's 0.82, which is not a leak
(`view_pool=[6]` with `view_count_range=[1,1]` admits no other view) — the same
+0.08–0.12 pattern appeared in the lift gate above, so the generated zarr behaves as a
slightly cleaner source than the hdf5 pipeline.

### Results

`*` marks a pose that **is** in the training pool (the ring's even indices). Unmarked
columns are never trained on and are the actual test. Strict `success_rate`, 50 paired
episodes, `azimuth_interp` (11 viewpoints) + `elevation_az0` (3).

**Square**:

| cell | m75 | m60* | m45 | m30* | m15 | 0* | p15 | p30* | p45 | p60* | p75 | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m3off | 0.40 | 0.78* | 0.60 | 0.78* | 0.68 | 0.78* | 0.42 | 0.70* | 0.58 | 0.78* | 0.62 | 0.88 | 0.30 | 0.04 |
| m3plucker | 0.34 | 0.78* | 0.50 | 0.66* | 0.56 | 0.80* | 0.56 | 0.78* | 0.50 | 0.78* | 0.54 | 0.76 | 0.24 | 0.02 |
| m3eef | 0.48 | 0.84* | 0.66 | 0.66* | 0.68 | 0.80* | 0.48 | 0.74* | 0.50 | 0.80* | 0.64 | 0.82 | 0.18 | 0.02 |
| m3on | 0.50 | 0.72* | 0.56 | 0.72* | 0.66 | 0.76* | 0.52 | 0.74* | 0.58 | 0.74* | 0.46 | 0.78 | 0.18 | 0.00 |
| *L1 s42* | 0.02 | 0.02* | 0.00 | 0.00* | 0.00 | 0.02* | 0.02 | 0.00* | 0.02 | 0.06* | 0.04 | 0.02 | 0.02 | 0.00 |
| *M1* | — | — | — | 0.00 | 0.02 | 0.82 | 0.00 | 0.00 | — | — | — | 0.88 | 0.00 | 0.00 |

**Can**:

| cell | m75 | m60* | m45 | m30* | m15 | 0* | p15 | p30* | p45 | p60* | p75 | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m3off | 0.70 | 0.88* | 0.76 | 0.86* | 0.74 | 0.92* | 0.80 | 0.92* | 0.80 | 0.90* | 0.64 | 0.90 | 0.34 | 0.22 |
| m3on | 0.84 | 0.86* | 0.74 | 0.86* | 0.74 | 0.86* | 0.82 | 0.92* | 0.82 | 0.84* | 0.64 | 0.88 | 0.28 | 0.22 |
| *L1* | 0.02 | 0.00* | 0.06 | 0.02* | 0.02 | 0.02* | 0.02 | 0.02* | 0.02 | 0.04* | 0.00 | 0.02 | 0.00 | 0.06 |
| *M1* | — | — | — | 0.00 | 0.08 | 0.98 | 0.00 | 0.00 | — | — | — | 0.98 | 0.04 | 0.00 |

Means, split by whether the pose was trained:

| model | square trained | square **held out** | can trained | can **held out** |
|---|---|---|---|---|
| m3off | 0.764 | **0.550** | 0.896 | **0.740** |
| m3plucker | 0.760 | **0.500** | — | — |
| m3eef | 0.768 | **0.573** | — | — |
| m3on | 0.736 | **0.547** | 0.868 | **0.767** |
| *L1* | *0.02* | *0.00–0.06* | *0.02* | *0.00–0.06* |

There *is* a real trained/held-out gap (square 0.76 → 0.55, can 0.90 → 0.74). But the
held-out **floor is 0.40–0.84, not 0.00**, where M1 is 0.00–0.02 off-axis and L1 is at
the noise floor everywhere.

### Conclusion

**It solves square and can — the two tasks L1 destroyed.** At azimuths excluded from the
training pool, M3 scores 0.50–0.57 on square and 0.74–0.77 on can, where M1 is 0.00–0.02
and L1 is at the noise floor.

The **load-bearing comparison is L1 at trained poses**: L1 trained on the same pool and
still scores ~0.02 at ±30°, poses it saw constantly. So L1's failure was never a failure
to *generalise* — it failed to solve the task at all. "M3 generalises to novel views" is
therefore the weaker of the two claims on offer; the stronger one is that **per-slot
fusion turns view diversity from harmful into sufficient**.

**The conditioning contributes nothing.** `m3off` (`use_plucker=False`,
`use_eef_hist=False`) matches or beats `m3on` on both tasks at every viewpoint, and
`m3on` is never the best square cell; can's held-out means run the *wrong* way for the
hypothesis (0.740 vs 0.767). Square held-out means span 0.500–0.573 across all four
cells and trained means 0.736–0.768 — all inside the noise. So PROPOSAL §2.1's central
claim is **not supported**: neither signal moved the number. What moved it was 7 slots,
per-sample N∈[1,7], and attention fusion — three changes that arrived as plumbing *for*
the conditioning rather than as the hypothesis.

**Elevation is a partial extrapolation result.** M3 holds 0.18–0.34 at `el_p15` on both
tasks, where M1 is 0.00–0.04, and 0.22 at `el_m15` on can, where M1 is 0.00 and L1 is
0.06. But on square, `el_m15` collapses to 0.00–0.04 for every M3 cell. Extrapolation
holds going up and not down, on one of two tasks; the asymmetry is unexplained.

**Costs and limits.** M3 costs a little in-distribution: az_0 0.76–0.80 vs M1's 0.82.
Every cell is n=1 at the noise resolution above. The four architectural changes are
confounded (resolved later, in favour of N>1). Only square and can were swept — **lift
was deliberately not run**, and it is the one task where L1 already wins, so it remains
the place M3 could regress. The four single-signal cells reduce to two on can: the square
2×2 established the conditioning null across the full 2×2, so can's `m3plucker` and
`m3eef` were dropped as redundant.

**M1 and L1 on the same elevation axis** (captured before the M1 weights were deleted).
`el_0` reproduces each model's committed az_0 `success_rate`, which validates the orbit's
zero point:

| task | model | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|
| square | M1 | 0.88 | 0.00 | 0.00 |
| square | L1 | 0.02 | 0.02 | 0.00 |
| can | M1 | 0.98 | 0.04 | 0.00 |
| can | L1 | 0.02 | 0.00 | 0.06 |
| lift | M1 | 0.74 | 0.00 | 0.02 |
| lift | L1 | **0.96** | **0.36** | **0.26** |

The lift row extends L1's task-split advantage onto a second axis: L1's lift lead
survives into elevation, degraded from its azimuth performance but far above M1
everywhere. One more task-side datapoint for the open "what separates lift" question.

## M4 — per-view auxiliary action heads

### Method

A shared per-view head (`PerViewAuxActionHead`, **151,888** parameters, 0.052% of the
policy) predicts the next `n_action_steps` (8) actions **in that camera's frame** — the
base-frame actions transformed by that view's extrinsics — as mean-squared error, summed
with the diffusion loss at weight **1.0**. The encoder's forward path is unchanged; this
is one thing added to a fixed M3 model.

Four design points decide whether the test is meaningful:

- **The chunk must be `n_action_steps` (8), not `horizon` (16).** Obs step `to` reads
  `action[to : to+C]`, so at `To=2` a full-horizon target reaches index 16, which does not
  exist — it is `pad_after` edge-repeat, i.e. fake supervision the head would happily fit.
  The dataset constructor rejects `C + n_obs_steps − 1 > horizon`.
- **The rot6d rotation round-trips through the full 3×3 matrix.** The 6d encoding keeps
  only rows 0 and 1 of `R`, but `rows01(R_cᵀ R_b)` depends on all three rows, so rotating
  the 6d vector with a 6×6 linear map is *not* the transform — and **is exactly right when
  `R_c == I`**, which is why a test at the base camera would prove nothing.
- **The target is a top-level sample key, never an obs key.** It derives from
  demonstrated future actions, which do not exist at rollout, and the policy normalizes
  every obs key.
- **Zero-init output layer**, so an aux-on run starts as M3; the head's own weights get
  gradient immediately, so the trunk sees the aux gradient from step 1.

**The pair is RNG-locked**, which is stronger than the parameter-identical ablations
above: the aux head is built unconditionally and consumes no RNG, and the encoder runs
exactly once per step in both arms, so every crop, noise draw and timestep is identical
for the whole run and the two arms differ by exactly one scalar term and its gradient.
Verified exactly, not statistically: `m4on`'s batch-0 `train_loss − aux_loss` =
**1.081642270** equals `m4off`'s `train_loss` to a difference of `0.000e+00`, and the
weight-0 arm is asserted bit-identical to the parent policy's loss and **all 179**
gradients.

The weight is a **guess** (1.0, PROPOSAL's literal "summed"), with no sweep. It was
checked before committing hours, and the config's own prediction of `aux/diff ≈ 0.1–0.3`
at init was wrong: the measured batch-0 ratio is **0.316** (diff 1.0816, aux 0.3415), and
the CPU suite independently says 0.301 on synthetic data. The reason is that
the absolute-action normalizer range-normalizes only the 3 position dims and leaves the
rot6d and gripper dims at scale 1. A rule was fixed in advance (proceed unless the ratio
exceeded 1.0), so 1.0 stayed — no post-hoc tuning.

### Results

Strict `success_rate`, 50 paired episodes, `azimuth_interp` + `elevation_az0`:

| model | trained\* | **held-out** | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|---|
| m3off | 0.76 | **0.55** | 0.88 | 0.30 | 0.04 |
| m3plucker | 0.76 | 0.50 | 0.76 | 0.24 | 0.02 |
| m3eef | 0.77 | 0.57 | 0.82 | 0.18 | 0.02 |
| m3on | 0.74 | 0.55 | 0.78 | 0.18 | 0.00 |
| **m4on** | **0.76** | **0.56** | 0.70 | 0.00 | 0.00 |
| **m4off** | **0.74** | **0.52** | 0.80 | 0.14 | 0.00 |
| *delta* | *+0.02* | *+0.04* | *−0.10* | *−0.14* | *0.00* |

Per-viewpoint deltas run −0.12 to +0.18 with the sign flipping freely (at ±15°, ±45° and
±75° `m4on` is behind; at ±45° well ahead). The means (+0.02, +0.04) are the noise floor.
**The aux term buys nothing measurable**, and M4 does not regress either — it sits inside
M3's 0.50–0.57 held-out band.

**The mechanism, by contrast, is alive.** `aux_loss` collapses 52× from its batch-0 value:

| epoch | 0 (mean) | 5 | 15 | 25 | 50 | 200 |
|---|---|---|---|---|---|---|
| `aux_loss` | 0.2389 | 0.0304 | 0.0206 | 0.0173 | 0.0147 | **0.0066** |

Flatness at the init value would have meant "unlearnable from one view" — M3's failure
mode, caught within 50 steps. And it is **not a collapse to the mean**: the loss is
mean-over-active-views then mean-over-frames MSE in normalized camera-frame action space,
so a *constant* head scores the pooled target variance, measured on 1024 real samples as
**0.2692**. The epoch-25 loss (0.0173) is **6.4%** of that floor — the head is 15.6× below
what a mean-collapsing predictor can reach, which means it reads view-relative geometry
out of `z_v`. Pipeline check: the same script's zero-predictor estimate (0.3469) matches
the actual batch-0 value (0.3415) to 1.6%, comparing 1024 samples against 64.

**The trunk was reshaped about as much as a re-seed.** Relative L2 on the EMA weights
(what rollouts use):

| module | params | aux effect (on vs off @200) | **RNG-only reference** |
|---|---|---|---|
| `obs_encoder` | 12,532,928 | 0.596 | **0.471** |
| `model` (UNet) | 277,632,138 | 0.856 | **0.824** |
| `aux_head` | 151,888 | 1.167 | — |
| `normalizer` | 2,526 | **0.00000** | 0.00000 |

The reference column is `m4off` versus the committed `m3on`: same seed, same shapes,
different RNG stream only — which is what makes it a clean *scale*. The aux term moves
the weights about as much as re-seeding does, so the null is **not** explained by "the
gradient was too weak to move anything"; that hypothesis is dead. Two controls confirm
the comparison: `m4off`'s aux head drifts 1e-5 over 50 epochs (provably never touched),
and `normalizer` differs by exactly zero. Caveat: weight-space distance is crude — SGD
accumulates drift along directions that need not matter functionally, so the defensible
claim is "a perturbation of the same order as run-to-run variation, producing no
behavioural change".

**Elevation cuts both ways, and the cut is measured.** At face value M4 is *worse* than
M3 at elevation (`m4on` 0.00 vs `m3on` 0.18 at `el_p15`). That reading does not survive
its own internal check: `el_0` is the *same camera pose* as `az_0`, and for `m4on` they
read 0.70 and 0.84 — a 0.14 spread on an identical pose from two separate sweeps. So the
elevation dip is not established, and the null above is correspondingly less precise than
"+0.04 versus a ±0.05 noise floor" makes it sound.

### Conclusion

**The mechanism is real and the behavioural effect is null.** The surviving explanation
is the one the design did not consider: **the information the aux head needs is already in
`z_v`.** M3 reaches 0.55 held-out with no aux supervision at all, and a 151,888-parameter
head (0.052% of the policy) can fit camera-frame targets post hoc from a latent that
already encodes the scene. The aux pressure then adds nothing the diffusion objective had
not already produced — which is why it can reshape the weights substantially while
changing no behaviour. The unswept weight remains a limit, but it is no longer the
leading explanation.

Only square was run for M4: the gate said can only after the square pair was read, and the
readout made the *mechanism* question the live one.

## The architectural confound, resolved: N>1 is the load-bearing ingredient

**Method.** M3-off's exact model, with **one CLI line changed** —
`task.dataset.view_count_range=[1,1]` — so every sample has exactly one active slot,
drawn from the same 7-pose pool. It is L1's training distribution routed through the M3
encoder (7 slots, 1 active, widened `conv1` fed zeros, MHA fusion degenerating on a single
token). Cost: 21 s/epoch, 201 epochs.

**The prediction was registered before the run**: *this cell fails like L1, therefore N>1
fusion is the load-bearing ingredient.* The competing outcome — that it still solves
square — would have meant the gain lives in the encoder path itself. Stating this first is
what keeps the result from being read post hoc.

**Results.** In-training rollouts (az_0 `mean_score`) against the two committed references:

| epoch | m3off (N∈[1,7]) | L1 (1 view) | **fixed-N=1** |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 50 | 0.38 | 0.02 | 0.06 |
| 100 | 0.56 | 0.04 | 0.06 |
| 150 | 0.74 | 0.00 | 0.04 |
| 200 | 0.80 | 0.02 | 0.00 |

Five of five checkpoints track L1's floor, not `m3off`. The strict sweep agrees —
per-viewpoint, `m3fixedn1` never exceeds 0.06 at any of the eleven azimuths:

| model | trained\* | **held-out** | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|---|
| m3off | 0.76 | **0.55** | 0.88 | 0.30 | 0.04 |
| L1 s42 | 0.02 | 0.02 | 0.02 | 0.02 | 0.00 |
| L1 s43 | 0.04 | 0.02 | — | — | — |
| **m3fixedn1** | **0.04** | **0.05** | 0.02 | 0.02 | 0.04 |

Gap to `m3off`: **0.72 trained / 0.50 held-out** — an order of magnitude beyond noise. Gap
to L1: **+0.02 / +0.03** — zero. The `el_0` = 0.02 is the sharpest single number: on the
*trained* camera pose this model cannot do the task at all.

**Conclusion.** Multi-view training samples are necessary. Every other candidate for the
L1 → M3 gain is now closed:

| candidate | verdict | where |
|---|---|---|
| Plücker rays + camera-frame history conditioning | inert | M3 |
| per-view auxiliary heads | learnable, behaviourally inert | M4 |
| encoder architecture at N=1 (widened conv1, fusion query, per-view path) | reproduces nothing | this section |
| **multi-view sampling (N>1)** | **load-bearing** | this section |

This is the project's first *positive* identification, and it reframes the story: the
proposal's §2.2 fusion is what works, while §2.1's geometric conditioning and §2.4's aux
heads are both inert. The gain came from a plumbing change — variable-N fusion — not from
the mechanism the proposal proposed.

**What it does not establish.** It does not say N=1 *inference* fails: M3 trains with
N∈[1,7] and infers at N=1 well (0.55 held-out), so the capability lives in the multi-view
training signal and the inference path rides on it. It does not separate "N>1 needed" from
"*variable* N needed", since the cell never sees N>1 at all. And it does not say how much
view diversity is enough.

## Relational probe (step 1a) — the geometry is already in `z_v`

**Why.** `PLAN.md` opened relational supervision on `z_v` to *make geometry necessary*
rather than better-injected, on the argument that a static scene leaks pose through the
image, so a richer Plücker injection carries nothing the image lacks. Step 1b's
relative-pose head rests on the unmeasured assumption that `z_v` does **not** already carry
that geometry — which is also M4's surviving explanation. This probe measures it on frozen
checkpoints: no training, no rollouts, one forward pass per sample, then a closed-form ridge
readout and a 2000-step MLP readout against two baselines that make the numbers mean
something.

**Results.** Square, 2000 dataset indices (4000 frames, mean **3.97** of 7 slots active —
the N∈[1,7] draw is what it should be), 80/20 split, `n_train` 12,715 (`abs_pose`,
`cam_eef`) / 18,076 (`rel_pose`). Rotation in degrees, translation in cm, each against the
target's own median scale. **Bold is the MLP column**, which is the one to read — relative
pose is bilinear in the two camera poses, so a linear probe underfits it even when the
geometry is fully present (the CPU test measures exactly that, 250.6 cm vs the mean
predictor's 248.2 cm on synthetic data where it was present by construction).

| target (median scale) | model | **mlp** | ridge | mean predictor | shuffled |
|---|---|---|---|---|---|
| `abs_pose` (144.0 cm) | `m3on` | **3.12°** / 5.59 cm | 2.65° / 4.47 cm | 37.94° / 42.19 cm | 50.16° / 58.68 cm |
| | `m3off` | **4.13°** / 18.11 cm | 12.53° / 24.45 cm | 37.94° / 42.19 cm | 49.62° / 61.39 cm |
| `cam_eef` (67.6 cm) | `m3on` | **3.66 cm** | 3.84 cm | 22.85 cm | 32.43 cm |
| | `m3off` | **10.05 cm** | 19.19 cm | 22.85 cm | 34.59 cm |
| `rel_pose` (70.7 cm) | `m3on` | **2.02°** / 3.14 cm | 21.42° / 30.41 cm | 65.35° / 55.75 cm | 75.26° / 75.64 cm |
| | `m3off` | **9.34°** / 15.36 cm | 31.07° / 51.23 cm | 65.35° / 55.75 cm | 74.18° / 83.40 cm |

`latent_stats` — the proposal's core claim as two numbers, "`z_v` view-aware, `z_g`
view-invariant":

| | `m3on` | `m3off` |
|---|---|---|
| `z_g` across view subsets of one state | 0.171 | 0.162 |
| `z_v` across views within one draw | **0.466** | **0.224** |

**All four gates pass**, so the probe is measuring what it claims to. `shuffled_target` is
worse than the fit in all six cells (the probe is not reading something other than
geometry); `cam_eef` — the positive control, the quantity M4's aux head read — beats its
mean predictor by 6.2× / 2.3×, so the wiring is real and a null there would have meant
"broken", not "empty".

### Conclusion

**1. The geometry is already there without any pose conditioning.** `m3off` has *no*
Plücker map and *no* EE history — a pure image encoder — and a small readout recovers its
own camera's absolute pose to **4.13°**, the camera-frame EE position to **10.05 cm**, and
the relative pose between two views to **9.34° / 15.36 cm**, against mean-predictor floors
of 37.94°, 22.85 cm and 65.35°. These are not floor-level numbers. The static-scene leak
that `PLAN.md` reasoned about is now **measured rather than assumed**.

**2. Plücker is not inert in the latent — it was inert in the behaviour.** `m3on` is 4.6×
better at relative-pose rotation (2.02° vs 9.34°), 3.0× better at absolute-pose rotation,
and its `z_v` is **2.1× more view-discriminative** (0.466 vs 0.224) while its `z_g` stays
about as view-invariant (0.171 vs 0.162). That is the conditioning doing exactly what
PROPOSAL §2.1 designed it to do, and it is the **first live mechanistic signal the
conditioning has shown** — `PLAN.md` asked for this test precisely because the rollout
numbers could not see it. It does not contradict the M3/M4 behavioural nulls; it sharpens
them: the conditioning measurably reorganizes the representation along a direction the
policy did not need.

**3. The pre-registered decision rule fires — with a caveat.** `NOTES.md` said: if
`m3off`'s MLP column already recovers the geometry, then 1b's head is a post-hoc fit and
the *objective*, not the head, is what has to change. `m3off` recovers it well above any
floor, so the head alone is not enough. But the `m3on`/`m3off` gap is real and large
(4.6×), which is the part the rule did not anticipate: there is headroom the ray map
already partly occupies, so a relational *objective* has somewhere to act — it just cannot
claim to be supplying information the encoder did not have.

**Fidelity control.** The probe's own paths were executed for the first time here, and the
smoke found two blocking bugs (both recorded in `NOTES.md`'s traps): `report_target` crashed
for the translation-only `cam_eef` target, and `fit_mlp` died at `loss.backward()` on
float64 targets against a float32 `nn.Linear` — the latter only at `--mlp-steps 2000`, i.e.
in the column that decides this section. Both now have CPU regression tests with mutation
power demonstrated. One smoke artifact worth recording because it nearly misled: at
`--n-samples 64` the `rel_pose` ridge had 566 training pairs against 1536 features,
**severely underdetermined**, and scored 23× *worse* than the mean predictor; at full scale
it beats it. Ridge on `rel_pose` needs ≳10× more rows than features to be readable at all.

**Limits.** One task (square), one seed, one checkpoint pair — n=1 at this project's usual
resolution. Decodability by a 2000-step MLP over 18k pairs is an **upper bound** on "the
information is present"; it is not a claim about how the policy routes it. And `z_v` is an
encoding of a *specific* scene, so the leak is about pose recovery in a static scene, which
is the regime the whole argument was made in.

## Open questions

- **How much diversity is enough?** `view_count_range=[2,2]` answers PROPOSAL §7's "2
  demo views, or many poses?" directly — one config line, ~2 h. `[2,7]` separates "N>1
  needed" from "variable N needed".
- **Lift is untested for M3.** It is the one task where L1 already wins (0.76–0.96), so an
  M3-on run there is a genuine "did we break it" question rather than a result. ~1 h
  (lift is 127 batches/epoch).
- **Second seed.** Every number in this document is n=1 at a resolution worse than ±0.05.
  A second seed on one M3 or M4 cell would materially strengthen the nulls, which are
  "indistinguishable at this resolution", not "proven identical".
- **The elevation asymmetry.** Square collapses at `el_m15` for every M3 cell while can
  holds 0.22. Unexplained.
- **What separates lift from square/can** — the question L1 opened and nothing since has
  closed. One task per side of the hypothesis, with a scene-complexity confound.
- **M5's premise.** M3 already does N=1 novel-view inference well, so distillation is only
  worth it if the fused latent demonstrably carries something the single-view path cannot,
  which no result so far shows.

## Code map

Every milestone lands as new files; these are them.

| milestone | files added |
|---|---|
| M1 | `eval_novel_view.py`, `summarize_novel_view.py`, `config/task/{square,lift,can}_image_abs_single.yaml` |
| M2 | `generate_multiview_dataset.py`, `dataset/multiview_image_dataset.py`, `tests/test_multiview_dataset.py`, `config/task/{square,can,lift}_image_abs_multiview.yaml`, `config/task/single_view_image_abs_multiview.yaml` |
| L1 | `config/task/randview_image_abs_multiview.yaml` |
| M3 | `model/vision/plucker.py`, `model/vision/view_conditioned_obs_encoder.py`, `env_runner/cam_key_image_runner.py`, `config/task/m3_plucker_image_abs_{multiview,n1}.yaml`, `config/train_diffusion_unet_image_workspace_m3.yaml`, `tests/test_view_conditioned_obs_encoder.py`, `preview_viewpoints.py` |
| M4 | `policy/diffusion_unet_image_policy_aux.py`, `model/vision/per_view_aux_head.py`, `config/task/m4_aux_image_abs_multiview.yaml`, `config/train_diffusion_unet_image_workspace_m4.yaml`, `tests/test_aux_action_heads.py` |
| step 1a | `probe_relpose.py`, `tests/test_relpose_probe.py` (a measurement, not a method — no training) |

**Edited seams** — the only changes to files this fork did not itself add:
`multiview_image_dataset.py` (cam table, per-sample view draw, mask, camera-frame EE
history, `action_to_cam`), `eval_novel_view.py` (`--m3-slots`, `--eef-hist-steps`, the
elevation orbit), `summarize_novel_view.py` (sort key for `el_*` names), and one
`getattr`-guarded `step_log['aux_loss']` line in the upstream
`train_diffusion_unet_image_workspace.py`. No other upstream package file is modified.
Note that `{square,lift}_image_single.yaml` (relative-action single-view variants) exist
but are unused — the baselines use the `_abs_single` variants.

## Artifacts

Sweeps, training logs, videos, and weights are listed in `NOTES.md` (which also holds the
runbooks, timings, and the operational traps). Committed to git: every sweep's
`eval_log.json` with its viewpoint videos, and the per-batch training logs. Weights
(4.6 GB per checkpoint) are **not** in git — rsync only.
