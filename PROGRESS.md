# PROGRESS

Training a diffusion policy that keeps working from camera viewpoints it was never
trained on, and at inference time from a **single camera placed at a novel pose**.
`PROPOSAL.md` holds the direction and the original predictions; `PLAN.md` what remains;
`NOTES.md` the operational detail (runbooks, timings, disk, recurring traps).

Last updated 2026-09-25.

## Current state

| rung | what changed | verdict |
|---|---|---|
| **M1** | single-view DP baseline (agentview at az_0), three tasks | collapses to ≈0 at ±15° azimuth on all three |
| **L1** | view diversity only: M1's exact model, its single camera slot filled per sample from a randomly drawn training view | task-split **in both directions** — solves lift out to ±75°, destroys square/can |
| **M2** | multi-view data: a 13-pose azimuth ring re-rendered from the demos' stored simulator states | validated by the N=1 gate; ±75°/±90° views are low value |
| **M3** | view-conditioned encoder: per-view latents, Plücker + camera-frame-history conditioning, K=7 slots, per-sample N∈[1,7], MHA fusion | **solves square and can at held-out views**; its conditioning contributes nothing |
| **M4** | per-view camera-frame auxiliary action heads | mechanism is real (`aux_loss` falls 52×), behaviour is null (+0.02/+0.04) |
| **N>1** | M3's model with one CLI line changed so every sample sees a single view | **the load-bearing ingredient** — reproduces L1, not M3 |
| **1a** | relational probe on frozen checkpoints: what geometry does `z_v` already carry? | geometry is there *without* conditioning; **Plücker is live in the latent, inert in the behaviour** |
| **`[1,2]`** | N-diversity ladder, first rung | **floor** — 0.028/0.043, indistinguishable from `[1,1]`; the N>1 gain is not reachable at max-N = 2 |
| **ladder** | the N-diversity ladder closed: mean-N 1.0 / 1.5 / 2.0 / 3.0 / 4.0 | **knee between 2.0 and 3.0** (0.080 → 0.456), then graded to 0.764; and `[1,5]` is *more* view-general than `[1,7]` |
| **collapse** | measured the encoder's output spread across every surviving checkpoint, matched-draw, with random-init controls | **a failure mode, not *the* failure mode** — explains `[1,2]`'s floor and answers L1's task split, but `[1,3]` fails while healthy, so a second failure mode exists |

Internal labels, used in the code and configs: **L1** names this fork's second rung
(M1's architecture, randomized view) — L0 is M1 itself, and L2–L4 are M3, M4 and M5.

**The five conclusions, in the order the project reached them.**

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
4. **"How much" has a knee, not a slope.** Mean-N 1.0 / 1.5 / 2.0 are all at the floor
   (0.036 / 0.028 / 0.080); the knee is between 2.0 and **3.0** (0.456), and it rises to 0.764
   at 4.0. Unexpectedly, the 3.0 cell is the *more* view-general of the two working ones
   (82% of trained retained against 72%).
5. **The floor is not one thing, and the encoder is not always the problem.** `[1,2]`'s floor
   *is* an encoder collapse — five orders of magnitude below its neighbours, and its image path
   is behaviourally severed. `[1,3]`'s floor is **not**: its representation beats the working
   cell on variance, `z_v` decodability, `z_g` decodability *and* image→action sensitivity, and
   it still scores 0.080. Collapse is a real failure mode and it answers L1's task split, but a
   healthy screen does not mean a working policy.

Honest summary against the proposal: **§2.2's fusion works; §2.1's geometric
conditioning and §2.4's auxiliary heads are both inert.**

**The open question** is conclusion 5's second half: the difference between `[1,3]` and the
working cells is evidently not whether information is *present*, and every instrument in this
repository asks exactly that. See *The second failure mode — `[1,3]`*.

Every behavioural number below is **one model, 50 paired episodes**, at the resolution of
*Noise and resolution* — treat differences below ~0.1 as unmeasured. The probe and screen
sections are not episode-based at all and each states its own noise floor.

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

These rules govern how every table below should be read.

- **±0.05 was the assumed rollout noise; it is optimistic.** Two 50-episode sweeps of the
  same checkpoint through two presets that place the camera at the *same* pose differ by
  **0.14** (`m4on`: `el_0` 0.70 vs `az_0` 0.84). Unseeded diffusion sampling adds
  ~0.08 spread on repeat evals of one checkpoint (square epoch-150 scored 0.86 once and
  0.94 on re-eval). Treat differences below ~0.1 as unmeasured.
- **On a *mean* over viewpoints the working threshold is 0.15 — a stated convention, not a
  derivation.** The 0.14 above is the spread on a *single* viewpoint. A mean over the 5 trained
  or 6 held-out azimuths cancels the episode-draw and sampling components but not the
  systematic ones, so it is tighter than 0.14 without being as tight as 0.14/√5 would suggest.
  The N-diversity ladder adopts **|Δ| < 0.15 unmeasured, 0.15–0.30 weak, > 0.30 real** so its
  branch decisions are fixed in advance rather than chosen after the fact. **No ladder decision
  depends on the exact value**: `[1,1]`/`[1,2]`/`[1,3]` sit 0.07–0.12 below it and `[1,5]`
  0.31 above it, so the floor branch and the working branch are both robust to any plausible
  revision.

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

**What "n=1" covers, and what it does not.** The *behavioural* tables — M1 through the
N-diversity ladder — are one model, 50 paired episodes, at the noise above. The **probe and
screen sections are not episode-based at all** — they report readouts over 2000 samples,
4096 draws, or 16 frames — and each states its own noise floor (a half-split gap, or a matched
random-init baseline). Everywhere, a null means "indistinguishable at this resolution", not
"proven identical".

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
Runbook: `NOTES.md`.

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
   maps depend on. 33/39 on square, 39/39 on can and lift.
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
differences sit inside the rollout noise, and the same +0.08–0.12 pattern recurs in M3's
gate below, so the generated zarr behaves as a slightly cleaner source than the hdf5
pipeline.

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
| az_m45 | — | 0.860 | — | 0.020 | — | 0.000 / 0.040 |
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

**What separates lift from square/can is not identified here.** The cleanest structural
difference is that lift requires no goal-directed placement — grasp-and-raise, where the
target is wherever the object already is, versus nut-onto-peg and can-into-bin. That
would make the axis "how much precise spatial localization from the image the task
needs". It is a **hypothesis with one task per side, not a finding**, and a
scene-complexity confound cannot be ruled out (lift's plain table and cube versus can's
cluttered shelf). A supporting limit: L1's `val_loss` on square is ~2× M1's (0.060 vs
0.029), so it does fit worse — but a 2× loss gap does not explain a 44× rollout gap, and
M3 shows `val_loss` does not track rollout behaviour at all. *Answered later, for L1*:
its encoder **collapses** on the two tasks it destroys — see *Collapse is a failure mode*.

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
  (Plücker convention to 6.66e-16 against the numpy projector M2's gate 2 validated, with
  five mutation power checks; crop alignment; matched capacity; exact ablations) and
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
+0.08–0.12 pattern appeared in the lift gate above.

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
read 0.70 and 0.84 — the 0.14 spread of *Noise and resolution*, on an identical pose from
two separate sweeps. So the elevation dip is not established, and the null above is
correspondingly less precise than "+0.04 versus a ±0.05 noise floor" makes it sound.

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

## N>1 — the architectural confound, resolved

**Method.** M3-off's exact model, with **one CLI line changed** —
`task.dataset.view_count_range=[1,1]` — so every sample has exactly one active slot,
drawn from the same 7-pose pool. It is L1's training distribution routed through the M3
encoder (7 slots, 1 active, widened `conv1` fed zeros, MHA fusion degenerating on a single
token).

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
something. Runbook: `NOTES.md`.

**Setting.** One run per cell, square, seed 42: `m3on` (`use_plucker=True`,
`use_eef_hist=True`) and `m3off` (both false). Artifacts:
`data/probe_relpose_square_{m3on,m3off}/probe_relpose.json`.

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

> **Superseded — do not quote, do not compare to any later number.** These two rows came
> from the pre-schema-2 code path, in which the compared subset sizes were governed by
> *each checkpoint's own* draw range — the incomparability the grid was built to remove.
> `m3off`'s `z_g_across_view_subsets` is **0.162** here, **0.202** at its own range in the
> grid run, and **0.252** at `[1,7]` in the same run: three numbers for one quantity, the
> differences being the draw and the code path, not the model. **The comparison the
> conclusion rests on survives**, because both cells above were measured identically: `z_v`
> is 2.1× more view-discriminative under Plücker (0.466 vs 0.224) and `z_g` is similar in
> both (0.171 vs 0.162). See *Record of corrections*.

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
smoke found two blocking bugs (both recorded in `NOTES.md`'s traps): `report_target`
crashed for the translation-only `cam_eef` target, and `fit_mlp` died at `loss.backward()`
on float64 targets against a float32 `nn.Linear` — the latter only at `--mlp-steps 2000`,
i.e. in the column that decides this section. Both now have CPU regression tests with
mutation power demonstrated. One smoke artifact worth recording because it nearly misled:
at `--n-samples 64` the `rel_pose` ridge had 566 training pairs against 1536 features,
**severely underdetermined**, and scored 23× *worse* than the mean predictor; at full
scale it beats it. Ridge on `rel_pose` needs ≳10× more rows than features to be readable
at all.

**Limits.** One task (square), one seed, one checkpoint pair — n=1 at this project's usual
resolution. Decodability by a 2000-step MLP over 18k pairs is an **upper bound** on "the
information is present"; it is not a claim about how the policy routes it. And `z_v` is an
encoding of a *specific* scene, so the leak is about pose recovery in a static scene, which
is the regime the whole argument was made in.

## N-diversity ladder — how much multi-view signal is needed?

**Why every rung keeps N=1 in the range.** Every published number here is an **N=1
inference**: `eval_novel_view.py` serves the one live camera into slot 0 (`mask[0] = 1.0`,
the rest zero and masked out), and at N=1 the fusion softmax is over a *single* unmasked key
— the learnable query has no effect at all, so that path is a **degenerate corner** of the
module rather than a scaled-down version of it. M3's stated rationale was that randomising N
makes that corner in-distribution. A `[2,2]` cell never trains there, so its failure would be
confounded three ways (≤2 views insufficient / N=1 inference out-of-distribution / no N=1
samples at all). **So each rung varies only the upper end of the range**, and differs from
`m3off` on exactly one axis. That answers PROPOSAL §7's "2 demo views, or many poses?" as
well as the N>1 section's "N>1 vs *variable* N".

**Setting.** Five runs on **`m3_plucker_image_abs_multiview`** with `use_plucker=false
use_eef_hist=false` — `m3off`'s exact configuration — varying **one integer**,
`task.dataset.view_count_range`. Seed 42, 201 epochs, K=7 slots throughout. Rungs `[1,1]`
and `[1,7]` are the pre-existing `m3fixedn1` and `m3off`; `[1,2]`, `[1,3]`, `[1,5]` are
`m3v12`, `m3v13`, `m3v15`. **No control run exists** — the ladder has no RNG-locked null.
Each rung is swept with both presets, 50 paired episodes. Launch and sweep commands:
`NOTES.md`. Artifacts: `data/eval_{interp,el}_square_{m3v12,m3v13,m3v15}/eval_log.json`,
committed with each rung's `logs.json.txt`.

**Pre-registered 2026-09-24, before Stage 1 was launched.** Read on the **trained mean**
`success_rate` (mean over `az_m60,m30,0,p30,p60`), held-out mean as a supporting read. Noise
floor 0.15 on a mean. References: `m3off` `[1,7]` = 0.764 / 0.550; `m3fixedn1` `[1,1]` = 0.036
/ 0.047.

**Prediction: `[1,2]` lands partial — trained mean 0.15–0.40**, above the floor band and well
short of 0.764. Registered as such, with the competing outcomes named in advance:

| Stage 1 trained mean | pre-registered reading | Stage 2 |
|---|---|---|
| **≥ 0.40** | saturating — max-2 plus an N=1 component captures most of the gain | `[2,2]`: the confound probe at its cheapest point |
| **≤ 0.15**, no trained viewpoint > 0.25 | floor — max-2 insufficient *even with N=1 in-distribution*, which also kills the out-of-distribution explanation for any future `[2,2]`-style floor | `[1,3]`, then `[1,5]` to bisect |
| **0.15 < mean < 0.40** | graded dose-response | `[1,4]` to bracket the knee |

**Stop rule:** two consecutive rungs within 0.15 → the ladder has saturated; spend what
remains on a **second seed** on the most interesting rung, since every null in this document
is n=1 and this ladder has no RNG-locked null.

**Resolution of a conflict between that stop rule and the follow-up rule — recorded
2026-09-25 *before* `[1,3]`'s number was read.** If `[1,3]` lands at the floor, both rules
fire and they disagree: the stop rule says two consecutive within-noise rungs mean
saturation, while the follow-up says bisect with `[1,5]`. **The stop rule governs the
*working* regime, not the floor regime.** Its purpose is to stop chasing noise once rungs stop
differing *at a level worth having*; here every rung sits at the floor, the effect is known to
exist above mean N = 2.0, and the open question is *where* it turns on rather than *how much*
it is worth. Bisecting with a large jump serves the stop rule's intent — do not spend
GPU-hours on a crawl — rather than violating it. The second seed is therefore spent on
whichever rung first *works*, where a null would actually mean something. Written down and
committed ahead of the number so it cannot be read as a post-hoc rescue.

**The one code change it needed.** `MultiViewImageDataset.__init__` rejected any
non-degenerate range whose `hi` was below the slot count, so `[1, 2]` at K=7 raised. That
guard was redundant (`n_slots <= len(view_pool)` plus `hi <= n_slots` already give
`hi <= len(view_pool)`, all `np.random.choice(..., replace=False)` needs) and internally
inconsistent (it allowed `[2,2]`, which strands the same five slots). It evaluated **False**
for both committed cells, so removing it is inert for every number published above; pinned by
a new regression test that cannot even construct its first dataset against the old code.

### Stage 1 — `[1,2]` lands at the floor, indistinguishable from `[1,1]`

**The prediction above was falsified, in the direction that carries information.** `[1,2]`
was pre-registered as *partial* (trained mean 0.15–0.40), on the reasoning that mean active
N = 1.5 would capture part of the dose-response. It captured none of it and fell on the
**floor branch**.

In-training rollouts (az_0, `mean_score`, the N>1 section's protocol):

| epoch | `m3off` `[1,7]` | `m3fixedn1` `[1,1]` | **`m3v12` `[1,2]`** |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 50 | 0.38 | 0.06 | **0.00** |
| 100 | 0.56 | 0.06 | **0.04** |
| 150 | 0.74 | 0.04 | **0.02** |
| 200 | 0.80 | 0.00 | **0.04** |
| **mean** | **0.50** | **0.03** | **0.02** |

Strict `success_rate`, 50 paired episodes, `azimuth_interp`:

| viewpoint | `m3off` `[1,7]` | `m3fixedn1` `[1,1]` | **`m3v12` `[1,2]`** | |
|---|---|---|---|---|
| az_m75 | 0.40 | 0.02 | 0.00 | held out |
| az_m60 | 0.78 | 0.02 | 0.06 | trained |
| az_m45 | 0.60 | 0.06 | 0.04 | held out |
| az_m30 | 0.78 | 0.00 | 0.02 | trained |
| az_m15 | 0.68 | 0.02 | 0.04 | held out |
| az_0 | 0.78 | 0.04 | 0.02 | trained |
| az_p15 | 0.42 | 0.06 | 0.04 | held out |
| az_p30 | 0.70 | 0.06 | 0.00 | trained |
| az_p45 | 0.58 | 0.06 | 0.06 | held out |
| az_p60 | 0.78 | 0.06 | 0.04 | trained |
| az_p75 | 0.62 | 0.06 | 0.08 | held out |
| **trained mean** | **0.764** | 0.036 | **0.028** | |
| **held-out mean** | **0.550** | 0.047 | **0.043** | |

| model | trained | held-out | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|---|
| `m3off` `[1,7]` | 0.764 | 0.550 |
| `m3fixedn1` `[1,1]` | 0.036 | 0.047 |
| **`m3v12` `[1,2]`** | **0.028** | **0.043** |

Elevation is at the floor for both floor cells too (`m3v12` 0.02 / 0.04 / 0.06 at
`el_0`/`el_p15`/`el_m15`; `m3fixedn1` 0.02 / 0.02 / 0.04), against `m3off`'s 0.88 / 0.30 /
0.04.

**The two nulls are exact, not approximate.** `[1,2]` against `[1,1]`: **Δ trained 0.008,
Δ held-out 0.004** — an order of magnitude inside the 0.15 noise floor, and below the ±0.05
that the original plan *assumed* was noise. Against `m3off`: **0.736 trained / 0.507
held-out**, five to ten times the floor. The rung is not a weakened version of the effect;
it is the same floor as a single view, on both axes.

**Gates.** `|el_0 − az_0| = 0.00` on the identical camera pose (the internal check passes
exactly); the 50 episode-seed keys are **set-identical to `m3off`'s**, the mechanical proof
the episodes are paired across cells; the sampling gate read **mean 1.510 / max 2** on 100
real draws, where a silently-ignored override would read ~4.0; epoch-0 `val_loss` **0.0844**
sits between `m3fixedn1`'s 0.0822 and `m3off`'s 0.0862, in the same order as their mean
active view counts, confirming identical initialisation.

**What it establishes.** The `[1,1] → [1,7]` gain is **not reachable by allowing a second
view**. Combined with the N>1 section, the ingredient is therefore neither "any N>1" nor a
smooth dose in the low-N regime. Two limits, stated because they bound the claim: mean active
N moved only 1.0 → 1.5, so this says nothing about where between 1.5 and 4.0 the effect turns
on — which is what Stage 2 is for; and the smoke shows the failure is *not* view confusion —
at the **trained** pose the model scores 0.02, so it never learned the task, L1's failure mode
rather than "generalises badly".

**Secondary result, and it is free.** This rung **trains at N=1** and still fails, so the
out-of-distribution explanation for a `[2,2]`-style floor is dead: a floor no longer needs
"N=1 inference was never trained on" to be explained.

**Protocol note.** `m3v12`'s checkpoints dir held two *different* models — `latest.ckpt`
(epoch 200, swept here and the runbook default) and an `epoch=0100` topk, which `cmp` shows
is not byte-identical. The epoch-100 topk scored 0.04 on the same in-training protocol, so
the choice cannot move the conclusion, but note that the reference sweeps record no
provenance at all (`data/eval_interp_square_*/` holds only `eval_log.json`), so which
checkpoint produced a committed number is recoverable only from the runbook convention.

### Stage 2 — `[1,3]` (mean N 2.0) is also at the floor

Per the pre-registered floor branch, the next rung was `[1,3]`. Strict `success_rate`, 50
paired episodes:

| viewpoint | `[1,1]` | `[1,2]` | **`[1,3]`** | `[1,7]` |
|---|---|---|---|---|
| az_m75 | 0.02 | 0.00 | 0.04 | 0.40 |
| az_m60 | 0.02 | 0.06 | 0.10 | 0.78 |
| az_m45 | 0.06 | 0.04 | 0.08 | 0.60 |
| az_m30 | 0.00 | 0.02 | 0.06 | 0.78 |
| az_m15 | 0.02 | 0.04 | 0.08 | 0.68 |
| az_0 | 0.04 | 0.02 | 0.06 | 0.78 |
| az_p15 | 0.06 | 0.04 | 0.10 | 0.42 |
| az_p30 | 0.06 | 0.00 | 0.06 | 0.70 |
| az_p45 | 0.06 | 0.06 | 0.06 | 0.58 |
| az_p60 | 0.06 | 0.04 | 0.12 | 0.78 |
| az_p75 | 0.06 | 0.08 | 0.08 | 0.62 |
| **trained mean** | 0.036 | 0.028 | **0.080** | **0.764** |
| **held-out mean** | 0.047 | 0.043 | **0.073** | **0.550** |
| max trained viewpoint | 0.06 | 0.06 | 0.12 | 0.78 |

`[1,3]` is nominally the strongest floor cell (Δ ≈ 0.05 over the other two) but that is
inside the 0.15 noise floor and no trained viewpoint exceeds 0.25, so the pre-registered
reading is **floor**, not partial. Gates: `|el_0 − az_0| = 0.04` (0.02 vs 0.06), and the 50
episode seed keys are set-identical to `m3off`'s. Elevation is at the floor on all three
poses (0.02/0.08/0.02).

**The ladder now reads: mean-N 1.0, 1.5 and 2.0 all at the floor; 4.0 works.** The transition
lies somewhere between 2.0 and 4.0. Stage 3 is `[1,5]` (mean 3.0), launched 2026-09-25 — a
large jump, per the stop-rule resolution, rather than a crawl.

**And the transition is not a collapse boundary.** `[1,3]` fails with a *healthy* encoder
(shown in *Collapse is a failure mode*), so locating where behaviour switches on will not
simultaneously locate where collapse ends. The two phenomena have come apart, which is what
makes the second failure mode an open question rather than a restatement of the first.

### Stage 3 — `[1,5]` (mean N 3.0) breaks the floor, and the ladder is graded after all

Per the stop-rule resolution, a large jump rather than a crawl. In-training rollouts (az_0,
`mean_score`, the same protocol throughout):

| epoch | `[1,1]` | `[1,2]` | `[1,3]` | **`[1,5]`** | `[1,7]` |
|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 0.06 | 0.00 | 0.06 | 0.08 | 0.38 |
| 100 | 0.06 | 0.04 | 0.08 | **0.30** | 0.56 |
| 150 | 0.04 | 0.02 | 0.08 | **0.44** | 0.74 |
| 200 | 0.00 | 0.04 | 0.08 | **0.40** | 0.80 |
| **mean** | 0.03 | 0.02 | 0.06 | **0.24** | **0.50** |

**`[1,5]` leaves the floor and plateaus at roughly half of `[1,7]`.** So the ladder is *both*
things at once, which is a correction to how the previous stage read it:

- a **knee** between mean-N 2.0 and 3.0, where three rungs pinned at 0.04–0.08 break open, and
- a **graded rise** above it — 0.24 at mean-N 3.0 against 0.50 at 4.0, on both the mean and
  the plateau value.

"How much diversity is enough" therefore has a two-part answer: below mean-N 3 it is not
enough at all, and above it, more still buys more.

**Confirmed by the strict 50-episode paired sweep**, which is what the pre-registration reads:

| cell | mean N | trained | held-out | trained→held-out drop |
|---|---|---|---|---|
| `m3fixedn1` `[1,1]` | 1.0 | 0.036 | 0.047 | — |
| `m3v12` `[1,2]` | 1.5 | 0.028 | 0.043 | — |
| `m3v13` `[1,3]` | 2.0 | 0.080 | 0.073 | — |
| **`m3v15` `[1,5]`** | **3.0** | **0.456** | **0.373** | **0.083** |
| `m3off` `[1,7]` | 4.0 | 0.764 | 0.550 | **0.214** |

Gates for `[1,5]`: `|el_0 − az_0| = 0.08` (0.38 vs 0.46), and the 50 episode seed keys are
set-identical to `m3off`'s. Elevation `0.38 / 0.26 / 0.00` — it holds *up* and not down, the
same asymmetry M3 showed.

**A second result in the last column, which the ladder was not designed to find.** `[1,5]`
retains **82%** of its trained performance at held-out views; `[1,7]` retains **72%**. So
`[1,5]` is the *more view-general* model even though its absolute held-out score is lower
(0.373 against 0.550). More diversity bought absolute performance **and** cost generalization
— the opposite of a monotone story, and a caution against reading the ladder as "more views
is better" beyond the point where it stops being.

**How this section was read, in sequence, and each reading's error.** `[1,2]`'s null was
first read as "a second view buys nothing"; then the shape was called a threshold; then a
threshold-plus-dose. The five-rung curve is the last of those, and each earlier reading was
overconfident about a curve drawn from two points. The pre-registered Stage-1 prediction
(0.15–0.40 for `[1,2]`) was falsified outright. That is the record; the ladder closed at five
rungs and no further rung is planned.

## Collapse is a failure mode

**Why this was measured at all.** Every evaluation here is N=1 inference, and at N=1 the
fusion softmax is over a *single* unmasked key, so the learnable query is inert. Large-N
training therefore cannot be teaching the fusion anything, which pointed at the **shared
backbone**: the hypothesis was that seven views of one scene press the encoder toward a
canonical, scene-level representation that two views do not. It predicted that a fusion-free
view-invariance statistic would separate the working cell from the floor cell.

### `[1,2]`'s encoder has collapsed

**Two confounds had to be fixed before the comparison meant anything.** `probe_relpose.py`'s
`measure_stability` instantiated the dataset from the checkpoint's own config, so the compared
subset sizes were governed by that cell's `view_count_range` — and that is not a small effect:
the *same* `m3off` encoder reports `z_g_across_view_subsets` = **0.335 at `[1,2]` vs 0.202 at
`[1,7]`**. And at N=1 the fusion reduces to `z_g = A z_v + b` with `A` trained per cell, so a
raw `z_g` comparison conflates the backbone with that cell's value-projection gain. Both were
removed: a `--view-count-range` override applied to a *grid* of ranges (`1,2;1,4;1,7;7,7`), and
a fusion-free primary statistic — pairwise `z_v` distances over the view pool at a full draw
(`zv_pair_ratio`), against the `z_v` distance *between states at a fixed view* as the
denominator. `z_v` is N-agnostic by construction (GroupNorm, eval-mode centre crop, no
dropout), which is what licenses it; measured on the box at **2e-05**, i.e. cuDNN
kernel-selection noise four orders of magnitude below the quantities being read.

**The result inverts the hypothesis.** `m3v12`'s encoder has **collapsed**:

| | `m3off` `[1,7]` works | `m3v12` `[1,2]` floor |
|---|---|---|
| `z_v` spread across states | 7.6e-02 | **2.7e-06** |
| `z_v` norm (mean) | 0.476 | **0.104** |
| `zv_pair_ratio` @ `7,7` | 0.581 | 1.277 |
| `zv_across_states` @ `7,7` | 0.572 | **2.7e-05** |
| permutation floor | 4.2e-08 | 0 |

Its `z_v` is a near-constant function of its input — a factor of ~**23,000** less
state-to-state variation than the working cell — while the images feeding it differ by 0.93 in
max pixel value. It is not an EMA artefact: the raw (non-EMA) model shows the same (3.2e-06).
So `[1,2]` does not generalise badly to novel views; it barely encodes the scene at all, and a
constant `z_g` means the policy acts **open-loop** — which is what a 0.028 success rate, at
the *trained* pose, looks like.
*(The `z_v` spread here and the screen's "relative spread" below are different statistics on
different draws — read each against its own definition.)*

**This is also a false confirmation that the design caught.** Under a naive reading,
`m3v12`'s `zv_pair_ratio` of 1.277 against `m3off`'s 0.581 is *above* the working cell's —
which the pre-registration said would **confirm** the canonicity hypothesis. But both the
numerator and the denominator of that ratio are ~1e-5, because a constant representation has
neither view-distance nor state-distance to measure; the ratio is meaningless. What caught it
is exactly the guard the design added for this trap: the explicit denominator
(`zv_across_states`, 2.7e-05) plus the random-init control, which shows a *healthy* encoder
sits at 1.51 and a working trained one at 0.58 — so 1.277 is not "even more canonical", it is
degenerate. The pre-registered **unreadable** branch is the correct reading.

**What it means.** The invariance-pressure hypothesis is neither confirmed nor refuted — it
is **unreadable on this cell**. What replaces it is simpler and better supported: the ladder's
floor at low mean-N is a **representation collapse**, so what large-N sampling supplies is not
canonicity pressure but *prevention of collapse*. That reframes the N>1 headline rather than
overturning it, and it is testable: if `[1,3]`'s encoder also collapses while `[1,7]`'s does
not, the collapse boundary coincides with the behavioural boundary. **It does not** — which is
what makes a second failure mode necessary.

**A practical consequence worth pinning.** `z_v` variance is a cheap, direct collapse
detector, and it can be read at epoch ~25 instead of after 200 epochs plus a 52-minute sweep.
It also may not be specific to the ladder: L1 — the cell that destroyed square and can —
failed at the *trained* pose too, which is the same signature, and its checkpoints are still
on disk for a direct read.

**Caveats.** One task (square), one seed. The collapse is measured on the encoder's output,
not traced to a layer or a cause; "collapse" here is a description of the representation, not
a mechanism.

### It answers L1's task split — and it is *a* failure mode, not *the* one

Since L1 this document had said, and repeated through M3, M4 and the N>1 section, that *"what
separates lift from square/can is not identified"*. L1's three runs are the controlled
comparison the question always needed: **identical architecture, identical procedure,
identical hyperparameters — only the task differs.** The matched screen below shows **within
L1's three runs, every failing cell is a collapsed encoder and the working cell is not**
(lift 2.49e-02 against square 1.49e-04 / 1.99e-04 and can 3.05e-05, with a random-init
baseline of 1.3e-02). Training **collapsed** square and can while **enriching** lift. This
comparison is confound-free by construction: all three runs draw views identically
(`view_subset`), so they are matched to each other.

**What it unifies.** L1's task split and the N-ladder's floor are the *same* failure mode, and
it explains the detail that never fit anywhere else: why these cells fail at the **trained**
pose. A constant `z_g` means the policy acts **open-loop** — it is not generalising badly, it
has nothing to condition on. That single fact accounts for M1's collapse at ±15°, L1's uniform
~0.02 on square/can, and `[1,2]`'s 0.02 at az_0.

**But collapse is not necessary for failure.** `[1,3]` is at the floor with a healthy
encoder — 2.7× *above* its own baseline, within ~1.3× of the working cell — so collapse is a
*sufficient-looking* explanation where it appears, not a necessary one for failure. Read every
claim in this section as "collapse explains these cells", never as "collapse explains
failure". A healthy screen means a cell will not fail *this way*, not that it will work.

**Caveats, and they matter.** The metric is crude: 16 consecutive frames, `std` across states
normalised by mean norm, so read the **ordering** as the finding and the absolute values as
indicative. This is **correlation, not causation** — collapse may be a symptom of something
deeper rather than the proximate cause, and no layer or cause is identified here. It is one
metric on one view of the data. What makes it worth acting on is that it is **seed-robust**
(s42 and s43 agree to 26%), **control-backed** (random-init baselines match to 1.6%), and it
separates works from fails across **two independently trained families** — L1's three tasks
and the ladder's two poles.

**Consequence for the ladder.** The pre-registered reading of the invariance probe is moot
for the floor cell: there is no representation to ask about invariance of. The open question
is not canonicity but **what makes training collapse**, and probing `[1,3]` cost ~11 min —
free.

### The whole table, screened — and `m3n1gate` sharpens the mechanism

**Setting.** `screen_collapse.py`, one run per checkpoint, ~2 min each, no training
(commands: `NOTES.md`). It reports the encoder's **relative output spread**: `std` across 16 consecutive dataset
states, max over dims, divided by the mean feature norm — dividing by the norm is what makes
the number comparable between encoders of different magnitude. `--random-init` discards the
weights and re-measures, which is the control that distinguishes "training collapsed this"
from "this architecture always looks like that". Artifacts: `data/screen_collapse/*.json`.
**Always with `--random-init`, always with a matched `--view-count-range`, and always with
`-o`** (`NOTES.md`).

**Anchors.** The random-init baseline is **per-architecture** and must be measured, never
borrowed: **1.27e-02** (lift) and **1.25e-02** (square) for L1's `MultiImageObsEncoder`,
**5.1e-03** for M3's `ViewConditionedObsEncoder`.

**Correction (2026-09-25).** The first version of this table measured each cell drawing from
**its own** `view_count_range` — a confound, since `[1,2]` sees 1–2 active views while `[1,7]`
sees up to 7 and the live-view count moves the fused output's spread. The tell was `[1,3]`
appearing *above* `m3off` (4.13e-02 vs 3.31e-02). Everything below is re-measured with a
**matched draw**. See *Record of corrections*.

Results, every surviving checkpoint, matched at `[7,7]`:

| cell | behaviour | relative spread | vs its random-init | verdict |
|---|---|---|---|---|
| random init, `MultiImageObsEncoder` | — | 1.27e-02 / 1.25e-02 | — | baseline |
| random init, `ViewConditionedObsEncoder` | — | 5.1e-03 | — | baseline |
| L1 **lift** | works | 2.49e-02 | 2.0× above | healthy |
| L1 **square** s42 / s43 | fails | 1.49e-04 / 1.99e-04 | 65–84× below | **collapsed** |
| L1 **can** | fails | 3.05e-05 | 408× below | **collapsed** |
| `m3off` `[1,7]` | works | 1.72e-02 | 3.4× above | healthy |
| `m3v15` `[1,5]` | **works** | 2.23e-02 | 4.4× above | healthy |
| `m3v13` `[1,3]` | **floor** | 1.36e-02 | 2.7× above | healthy |
| `m3v12` `[1,2]` | floor | **1.83e-07** | **28,000× below** | **collapsed** |

**Conclusion.** `[1,2]`'s floor *is* collapse — five orders of magnitude below every other
cell. And the **L1 task-split correlation stands**: square and can collapsed, lift did not.

**The band, and the one cell that sits oddly in it.** Healthy cells span 1.36e-02–2.49e-02
and collapsed ones 1.8e-07–3.1e-05, with nothing between — a gap of two to three orders of
magnitude. `m3n1gate` (5.93e-02) is measured at its **own** range because it *cannot* be
matched: it is a **K=1** model (`n_slots=1`) with a **singleton** `view_pool=[6]`, so a
`[7,7]` override is structurally impossible — `_apply_range` correctly refuses — and its
number therefore carries both confounds. It should not be quoted as a matched comparison.

**`m3n1gate` is the cell that sharpens the mechanism: it trains at N=1 and is perfectly
healthy (5.93e-02).** So collapse is *not* caused by "too few views" — suggestive, with the
caveat just stated. The contrast it *does* support is with `m3v12` — both have N ≤ 2 samples,
but `m3n1gate` always sees the **same** view while `m3v12` sees a **random** one from a 7-view
pool, and L1 (random single view, collapsed on square/can) fits the same pattern. The
destabiliser is therefore **view variation the encoder cannot yet reconcile**: with a fixed
view the image→action mapping is learnable and the encoder stays healthy; with a randomly
varying view at low N, the same scene arrives from different angles carrying the same action
label, and if the encoder cannot build a view-invariant representation from too few views,
ignoring the input entirely is the loss-minimising degenerate solution. Many views supply
enough signal to build the invariant instead. That reframes the N>1 headline once more: the
ingredient is not "more than one view" but **enough views to make the varying-view objective
solvable**. It also gives a reason lift resists — it is the task whose actions depend least on
precise spatial localisation, so conflicting view information costs it least. Both statements
are hypotheses on this evidence, not measurements.

**When the collapse happens, as far as existing runs can say.** `m3v12` was screened at both
of its checkpoints: already fully collapsed at **epoch 100** (6.3e-07) and unchanged at epoch
200 (5.4e-07). Collapse is therefore not a late-training artefact — it is established by a
third of the way through and stable after. **That bound is no longer reproducible**: the
epoch-100 checkpoint was **deleted on 2026-09-25** to free disk (4 GiB, reclaimed when the
volume hit 100% during the `[1,5]` run), so
`data/outputs/run_square_m3v12_s42_200ep/checkpoints/` holds `latest.ckpt` only while `m3v13`
and `m3v15` each still hold their topk. The numbers were real when measured; the measurement
now stands on an artifact that is gone. See *Record of corrections*.

**What cannot be established from any existing run is whether collapse *precedes* the
behavioural failure**, because the workspace saves only `topk` and `latest` per run — there is
no per-epoch series to order the two against. Answering that needs a run configured to
checkpoint periodically, and recording it as a limitation here is the honest alternative to
inferring an ordering the data does not contain.

**Limits.** `random init` is per-architecture and must never be borrowed across families
(5.1e-03 vs 1.3e-02 here). `m3fixedn1` — the cell that would have tested "N=1 varying view"
directly — cannot be screened: its weights were deleted before this question existed. M1's
baselines cannot either, for the same reason; re-training one would establish whether *the
original single-view baseline was itself collapsed*, which would make this one story from M1
onward rather than two.

## The second failure mode — `[1,3]` fails with a healthy representation

**The hypothesis this tests.** A healthy *spread* is not information — an encoder can vary
richly along directions carrying nothing about the scene. So the natural account of `[1,3]`
was "it varies, but uselessly". Three measurements test it at three stages, and it is
refuted at every one.

**Stage 1 — the encoder (`z_v`).** `probe_relpose.py` under its schema-2 grid (see
*N-diversity ladder* for the rewrite), one run per cell on frozen checkpoints,
`--view-count-range 1,7` so the geometry target is comparable across cells. Ridge readouts
(closed-form, so they cannot diverge — see the anomaly below), same grid and seed for every
cell:

| target | `m3off` `[1,7]` WORKS | `m3v12` `[1,2]` collapsed | **`m3v13` `[1,3]` floor** | mean-predictor floor |
|---|---|---|---|---|
| `abs_pose` | 24.45 cm / 12.53° | 41.45 cm / 38.49° | **23.98 cm / 13.44°** | 42.19 cm / 37.94° |
| `cam_eef` | 19.19 cm | 22.78 cm | **16.62 cm** | 22.85 cm |
| `rel_pose` | 51.23 cm / 31.07° | 55.13 cm / 64.77° | **44.16 cm / 29.71°** | 55.75 cm / 65.35° |

`m3v12` sits at the floor on every column, which is what a collapsed encoder must look like —
a useful confirmation that the probe tracks the collapse it was built alongside. But
**`m3v13` beats the working cell on every single column** while scoring 0.080. So `[1,3]`'s
representation is healthy *and* more linearly decodable than the cell that works, and it still
fails. **Therefore the failure is entirely downstream of the encoder** — in the fusion, or in
whether the policy uses its conditioning at all. This is the first time in this document that
a failure has been localised *past* the encoder.

**Consistency check.** The grid run reproduces the earlier step-1a probe on `m3off` exactly
(`abs_pose` ridge 24.45 cm / 12.53° in both), so the `--view-count-range` override and the
schema-2 changes did not perturb what is collected — the numbers in this table are the same
quantity as the ones in the step-1a section.

**Anomaly, flagged rather than explained.** The MLP readout *diverged* on `m3v13`'s
translation dims — 727 cm against a 42 cm floor, i.e. **worse than predicting the mean** —
while its rotation dims were fine (4.04°). A fit landing worse than the mean predictor
indicates divergence, not absent information, so it is an artefact; but it is unexplained, and
the MLP column should not be quoted for this cell until it is. The ridge columns are
authoritative here precisely because a closed-form fit cannot diverge.

**Stage 2 — the fusion (`z_g`).** The policy never sees `z_v`; `forward` hands the UNet `z_g`
alone, so a decodable `z_v` does not imply a decodable `z_g`, and the fusion was the obvious
remaining suspect. A schema-2 extension decodes each frame's own camera pose from `z_g` at
single-view draws — the inference condition, and the case where `z_g` is an unambiguous
function of one view's `z_v`. 4096 single-view draws, `n_train` 3276 / `n_test` 820, **equal
`n` for every cell** so the comparison is valid even where regularisation biases the absolute
values. Artifacts: `data/probe_zg_square_{m3off,m3v12,m3v13}/probe_relpose.json`, field
`latent_grid["1,1"].zg_abs_pose`. The `z_v` column is from the readable `[1,7]` grid run
(`n_train` ≈ 12,700).

| cell | `z_g` rot err | gain over floor | `z_v` rot err (readable run) |
|---|---|---|---|
| `m3off` `[1,7]` works | 19.58° | 1.98× | 12.53° |
| **`m3v13` `[1,3]` floor** | **18.68°** | **2.08×** | 13.44° |
| `m3v12` `[1,2]` collapsed | 33.49° | 1.16× | 38.49° |

`m3v13` beats the working cell at **both** stages and still scores 0.080. So the refutation
chain is now: not the encoder's variance, not its decodability, and not the fusion.

**Read the rotation column only — the translation column is an unresolved artefact in *every*
cell, including the one that works.** Ridge `t_rmse_cm` for `z_g` is 190.95 for `m3off`
against a 42.60 floor (4.5× *worse* than predicting the mean), 30.53 for `m3v13` against
42.60, and 38.48 for `m3v12`. A fit landing worse than the mean predictor indicates
**divergence**, not absent information — the same signature already flagged above for
`m3v13`'s MLP column, and it is present for `m3off` too, which is what rules out reading it as
a property of the failing cells. At 6.4 rows per feature this decode is under the ≳10:1 the
ridge needs, so the translation dims — the hardest to fit — overfit first. Rotation is the
column that recovers signal above the floor, and it is the column the conclusion rests on.

**Stage 3 — the action head, which is not ignoring the image either.** `screen_conditioning.py`
holds the diffusion sampling noise fixed (`torch.manual_seed(0)` before every `predict_action`)
and varies one input path at a time, so a change in the output is attributable to that input
rather than the sampler. Reported: `std` across the 8 observations, over the action chunk,
normalised by the chunk's mean absolute value. Flags at defaults (`--n-obs 8`, `--seed 0`);
artifacts `data/screen_conditioning/square_<cell>_latest.json`.

| cell | image-only | proprio-only | artifact |
|---|---|---|---|
| `m3off` `[1,7]` works | 0.0068 | 0.0386 | `square_m3off_latest.json` |
| `m3v13` `[1,3]` floor | 0.0067 | **0.0674** | `square_m3v13_latest.json` |
| `m3v12` `[1,2]` collapsed | **2.59e-05** | 0.0686 | `square_m3v12_latest.json` |

**`[1,2]`'s image path is severed.** Changing its *entire* image moves the action by 2.6e-05
— about **256×** below `m3v13`'s 0.0067 — which is what an encoder collapsed to a constant
must produce. The collapse finding, previously inferred from the representation, is now
confirmed **end-to-end and behaviourally**.

**Read the two paths separately, never their sum.** `global_cond` is `concat([z_global,
low-dim])` and the low-dim keys are 9 dims of proprioception that vary across samples too.
Measured together, the *collapsed* cell comes out the **most** observation-sensitive
(`m3v12` 0.0686) — its constant image riding along with normally varying proprioception —
which is the opposite of the truth. See *Record of corrections*.

**So the two floor cells fail for different, now-measured reasons.** `[1,2]`: the image path is
severed. `[1,3]`: the image path is intact and its sensitivity is *identical* to the working
cell's (0.0067 vs 0.0068) — the only difference is balance, leaning ~1.75× harder on
proprioception (0.0674 vs 0.0386), and proprioception cannot see where the nut is.

**Caveats.** The image-sensitivity equality is 0.0067 vs 0.0068, which is no difference at
all — so what is solid here is the **severed** path (2.6e-05 against 0.0067), not the balance
reading. n=8 observations, one seed, one crude ratio, `mean` over the action chunk; the
`[1,2]` value is three orders of magnitude below the others, so its exact figure is not the
point and its order of magnitude is. The balance reading is a hypothesis with a plausible
mechanism, not a measurement.

**The question this leaves.** `[1,3]` sits at the floor (0.080) with a representation that
beats the working cell at four separately-measured stages: encoder variance, `z_v`
decodability, `z_g` decodability, and image→action sensitivity. Every instrument built here
asks *whether information is present*; the difference between those two cells is evidently not
presence. So the question is what the policy *learned to do* with correct information — and no
probe in this repository can currently see it.

## Record of corrections and superseded numbers

Dated, newest first. These are kept because a document that quietly repairs its own headline
is worth less than one that shows the repair — read them before quoting any number they
touch.

- **2026-09-25 — `screen_conditioning`'s image-only value was misread as `0.0000`.** The
  artifact says **2.5911e-05** for `m3v12`; the console showed `0.0000` because the script
  prints with `:.4f`, and the write-up said "bit-identical" from that console line rather than
  from the JSON the run had just written. The conclusion is unchanged — 2.6e-05 is ~256× below
  the other cells and ~2600× below their proprioception — but it rests on that margin, not on
  identity. **"Bit-identical" was never measured.**
- **2026-09-25 — the first collapse screen measured each cell at its own `view_count_range`.**
  A confound: live-view count moves the fused output's spread. Re-measured with a matched
  `[7,7]` draw; the tell was `[1,3]` reading *above* `m3off` (4.13e-02 vs 3.31e-02). Superseded
  with the table: the own-range ratios 1.9× (lift) / 2.5× (`m3off`) / 24,000× (`m3v12`), which
  the matched measurement replaces with 2.0× / 3.4× / 28,000×.
- **2026-09-25 — "collapse is *the* unifying failure mode" → "*a* failure mode."** Corrected
  once `[1,3]` screened healthy *at the floor*: `m3v13` sits 2.7× above its own random-init
  baseline while scoring 0.080, so collapse is not necessary for failure.
- **2026-09-25 — the `[1,2]` epoch-100 collapse bound is no longer reproducible.** That
  checkpoint was deleted for disk (4 GiB, volume at 100%). The numbers were real when measured
  (6.3e-07 at epoch 100, 5.4e-07 at epoch 200); the artifact is gone, so the bound now stands
  on a measurement that cannot be repeated without re-training.
- **2026-09-25 — Stage 1's pre-registered prediction was falsified.** `[1,2]` was registered as
  *partial* (trained mean 0.15–0.40, on mean active N = 1.5 capturing part of the
  dose-response) and landed on the floor branch at 0.028. Related, and the reason this
  document now states the ladder's shape once: the curve was read three times — "a second view
  buys nothing", then "a threshold", then "a threshold plus a dose" — and each earlier reading
  was overconfident about a curve drawn from two points.
- **2026-09-24 — the pre-schema-2 `latent_stats` are superseded.** `z_g_across_view_subsets`
  0.171 (`m3on`) / 0.162 (`m3off`) and `z_v_across_views_within_draw` 0.466 / 0.224 came from a
  code path in which the compared subset sizes were governed by each checkpoint's own draw
  range. Do not quote them or compare them to any later number: the *same* `m3off` encoder
  reads 0.162 there, 0.202 at its own range in the grid run, and 0.252 at `[1,7]` — three
  numbers for one quantity. The within-run contrast (2.1× more view-discriminative `z_v` under
  Plücker) is unaffected, because both cells were measured identically.
- **2026-09-24 — step 1a's pre-registered decision rule fired, with a caveat.** The rule said
  1b's head would be a post-hoc fit if `m3off` already recovered the geometry, and it does —
  but the rule did not anticipate the `m3on`/`m3off` gap (4.6×), which is the load-bearing
  half: there is headroom a relational *objective* has somewhere to act on, it just cannot
  claim to supply information the encoder did not have.
- **`m3n1gate`'s contrast is not matched, and was first quoted as though it were.** "Collapse
  is not caused by too few views" rests on a K=1 model with a singleton `view_pool=[6]`, which
  `_apply_range` correctly refuses to override; its number (5.93e-02) carries both the slot
  count and the draw as confounds. The supported contrast is the one with `m3v12` — same view
  every time versus a random one — and it is suggestive, not matched.

## Open questions

- **The second failure mode — the most open question here.** `[1,3]` sits at the floor (0.080)
  with a representation that beats the working cell at four separately-measured stages: encoder
  variance, `z_v` decodability, `z_g` decodability, and image→action sensitivity. Every
  instrument built here asks *whether information is present*; the difference between those two
  cells is evidently not presence. So the question is what the policy *learned to do* with
  correct information — and no probe in this repository can currently see it. See *The second
  failure mode — `[1,3]`*.
- **What makes training collapse?** Known: not "too few views" (`m3n1gate` trains at N=1 and is
  healthy), established by epoch ~100, and task-dependent (L1 collapses on square and can, not
  lift). Unknown: the mechanism, the layer, and whether collapse is a cause or a symptom. Also
  unknown and worth stating because it rules out the cheapest design — **whether collapse
  precedes the behavioural failure** — since no existing run has a per-epoch checkpoint series.
- **M1's baselines were never screened**, their weights having been deleted before the question
  existed. Re-training one would establish whether *the original single-view baseline was itself
  collapsed*, which would make this one story from M1 onward rather than two.
- ~~**How much diversity is enough?**~~ — **answered**: a knee between mean-N 2.0 (0.080) and
  3.0 (**0.456**), then graded to 4.0 (0.764). See *N-diversity ladder*. The related `[2,7]`
  cell ("N>1 needed" vs "*variable* N needed") remains untested and is now *more* interesting
  than it was: min-N 2 with mean-N 4.5, so if it works while `[1,2]` and `[1,3]` collapse, the
  ingredient is mean view count rather than the presence of N=1.
- **Lift is untested for M3.** It is the one task where L1 already wins (0.76–0.96), so an
  M3-on run there is a genuine "did we break it" question rather than a result. ~1 h
  (lift is 127 batches/epoch).
- **Second seed.** Every *behavioural* number in this document is n=1 at a resolution worse
  than ±0.05. A second seed on one M3 or M4 cell would materially strengthen the nulls, which
  are "indistinguishable at this resolution", not "proven identical".
- **The elevation asymmetry.** Square collapses at `el_m15` for every M3 cell while can
  holds 0.22; `[1,5]` holds `el_p15` (0.26) and not `el_m15` (0.00). Unexplained.
- ~~**What separates lift from square/can**~~ — **answered for L1**: its encoder **collapses**
  on the two tasks it destroys and not on the one it solves (lift 2.49e-02 vs square 1.49e-04
  and can 3.05e-05, against an identical random-init baseline of 1.3e-02; seed-robust; and
  matched by construction, since all three runs draw views the same way). See *Collapse is a
  failure mode*. Two better questions it opens: **what makes training collapse**, and — since
  `[1,3]` fails with a healthy encoder — **what is the second failure mode**?
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
| N-diversity ladder | **no new files**: five runs varying one CLI integer (`task.dataset.view_count_range`), plus the `view_count_range` guard relaxation in `dataset/multiview_image_dataset.py` (a fork-added file) and its regression test in `tests/test_view_conditioned_obs_encoder.py` |
| collapse + screens | `screen_collapse.py`, `screen_conditioning.py` (measurement tools, no training; `screen_collapse.py` works on any image encoder, including L1's non-slot one). `probe_relpose.py` gained the schema-2 grid, the fusion-free `zv_pair_ratio`, the null-not-NaN rule, the draw fingerprint, the random-init control, and the `zg_abs_pose` block |

**Edited seams** — the only changes to files this fork did not itself add:
`multiview_image_dataset.py` (cam table, per-sample view draw, mask, camera-frame EE
history, `action_to_cam`; and the `view_count_range` guard removed 2026-09-24 so the
N-diversity ladder can run `[1, 2]` — inert for both committed cells, see *N-diversity
ladder*), `eval_novel_view.py` (`--m3-slots`, `--eef-hist-steps`, the
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
