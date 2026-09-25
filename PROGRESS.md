# PROGRESS

Training a diffusion policy that keeps working from camera viewpoints it was never
trained on, and at inference time from a **single camera placed at a novel pose**.
`PROPOSAL.md` holds the direction and the original predictions; `PLAN.md` what remains;
`NOTES.md` the operational detail (runbooks, timings, disk, recurring traps).

**This file is the module and its measured behaviour.** The four milestones that build the
module — M1 single-view baseline, M2 multi-view data, M3 view-conditioned encoder, M4
per-view aux heads — are here in full, followed by one conclusion block for each of the
follow-up experiments. `PROGRESS_DETAIL.md` holds the parked detail: the full protocol, the
investigation log verbatim, the code map, and three appendices — including its
**Appendix C, the record of corrections and superseded numbers, which should be read before
quoting any number**.

Last updated 2026-09-26.

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
| **`m3v15` probe** | filled the one rung missing from every latent table (fingerprint gate passed) | **the `[1,3]` anomaly is a monotone ladder-wide trend** — as mean-N falls, `z_g` gets *more* view-invariant and `z_v` *less* view-aware while behaviour gets worse |
| **propdrop gate** | the balance hypothesis ("`[1,3]` leans 1.75× harder on proprioception") re-measured at n=64 × 3 seeds, pre-registered | **refuted; intervention retired unlaunched** — the draw-independent proprio arm is 1.008× and flat across all four rungs (0.5% spread), and the gate itself failed (min ratio 0.567) |
| **screen instrument** | fixed the unseeded draw + per-cell draw ranges, then ran the matched control the fix was pre-registered with | **the tool is now bit-reproducible, but the diagnosis was wrong** — matching the draw did not reduce the 2× spread, and at matched inputs the cell rank order *flips* across seeds, so `image_only` is not a scalar |
| **`[2,7]`** | the ladder's one remaining cell: min-N 2, mean-N 4.5, so it never trains at N=1 | **indistinguishable from `[1,7]`** (Δ trained +0.104 inside the 0.15 band, Δ held-out +0.003) — so the *availability* of N=1 samples is not the ingredient, and N=1 **inference** works without ever training at N=1 |
| **M1 re-train + screen** | re-trained the deleted single-view baseline (fidelity-verified against its committed curve), then collapse-screened it | **not collapsed** — 17.5× *above* its own random-init baseline, so M1's view-tiedness is a *different* failure mode from collapse; the story is not one story from M1 onward |

Internal labels, used in the code and configs: **L1** names this fork's second rung
(M1's architecture, randomized view) — L0 is M1 itself, and L2–L4 are M3, M4 and M5.

## The five conclusions

In the order the project reached them.

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

## Setup and protocol

**Platform.** robomimic PH demonstrations for **square** (primary), **can**, and **lift**, in
the `_abs` variant (absolute 10-dim actions). Observations are 84×84 RGB from the
**agentview** camera only — the wrist camera is dropped, matching the single-view inference
setting. Diffusion Policy: conditional UNet over action chunks, horizon 16, `n_obs_steps` 2,
`n_action_steps` 8, batch 64, AdamW + cosine LR + EMA, 201 epochs, seed 42, fp32 (the
workspace has no AMP). Rollouts use the EMA weights.

**Novel-view evaluation** (`eval_novel_view.py`). A fixed camera is moved at the mujoco_py
level (`sim.model.cam_pos/cam_quat` + `sim.forward()`), and the viewpoint is re-applied at
every reset from a base pose captured on first use. Policy rollouts at each viewpoint use
**paired episodes** — the same episode seeds across viewpoints — so a difference is
attributable to the camera rather than to the episode draws. 50 episodes per viewpoint. Two
metrics: strict `EnvRobosuite.is_success()` (`success_rate`) and max-reward (`mean_score`).

**Viewpoints.** Azimuth presets `azimuth_sweep5` (0, ±15°, ±30°) and `azimuth_interp`
(0, ±15°, ±30°, ±45°, ±60°, ±75°). Elevation is `elevation_az0` — an **orbit** about the
look-at point at 0/±15°, which is the off-manifold test. `el_0` is the same camera pose as
`az_0`, which makes it the internal consistency check.

**Multi-view data (M2).** A 13-pose azimuth ring every 15° out to ±90°. The **even indices**
(az −90/−60/−30/0/+30/+60/+90, every 30°) form the *training pool*; the **odd indices**
(±15/±45/±75) are **never sampled during training**, so every "held out" column in this
document is held out by construction. The single rgb key keeps the live camera's name,
`agentview_image` — it is a slot label, not a zarr array name — which is what lets training
rollouts and the stock evaluation path run unchanged.

The full protocol, including how the noise floor was derived: `PROGRESS_DETAIL.md` Part 1.

### Noise and resolution

- **Treat differences below ~0.1 on a single viewpoint as unmeasured.** Two 50-episode sweeps
  of one checkpoint through two presets that place the camera at the *same* pose differ by
  **0.14** (`m4on`: `el_0` 0.70 vs `az_0` 0.84), and unseeded diffusion sampling adds ~0.08
  spread on repeat evals of one checkpoint.
- **On a mean over viewpoints the working threshold is 0.15 — a stated convention, not a
  derivation.** The N-diversity ladder adopts **|Δ| < 0.15 unmeasured, 0.15–0.30 weak, > 0.30
  real** so its branch decisions were fixed in advance rather than chosen after the fact. No
  ladder decision depends on the exact value.
- **`mean_score` hides effects that `success_rate` shows.** On lift it saturates at 1.000 for
  a policy that succeeds 0.76 of the time; all four M3 square cells sit at 0.76–0.94 in
  `mean_score` while spanning 0.12 in the strict sweep.
- **`val_loss` does not track rollout behaviour.** All four square M3 cells end at
  0.0568–0.0602 — essentially L1's 0.060 — while rolling out like M1 rather than L1. Compare
  it only at matched epochs, and M4's is not comparable to M3's at all (it contains the aux
  term).

Every behavioural number below is **one model, 50 paired episodes**, at the resolution of
*Noise and resolution* — treat differences below ~0.1 as unmeasured. The probe and screen
sections are not episode-based at all and each states its own noise floor.

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

## Other experiments — key conclusions

The module's own numbers are above. What follows is what each follow-up experiment concluded;
each block is the result and the reading, not the reasoning. Full sections, with every
per-viewpoint table and every pre-registration: `PROGRESS_DETAIL.md` Part 2.

### L1 — view diversity alone

**Method.** M1's *exact* architecture, its single camera slot filled per sample from a randomly
drawn **training** view (the ring's seven even indices). No pose information anywhere — the
model is never told which view it is looking from.

**Result.** A task split, in opposite directions. On **lift** it holds 0.76–0.96 at every
viewpoint out to ±75°, where M1 is 0.08 at ±15° and 0.00 at ±30°. On **square and can** the
same intervention is ≤0.08 *everywhere*, including the poses it trained on, where M1 scores
0.82 and 0.98.

**Conclusion.** A conditioning-free baseline already achieves the project's stated goal on
lift, and destroys the other two tasks. Read L1's low numbers as a **noise floor, not as weak
success**: 0.00–0.08 over 50 episodes is 0–4 episodes. *Why* the tasks split was open until
the collapse screen answered it (*Collapse*, below).

### N>1 — the architectural confound

**Method.** `m3off`'s exact model with **one CLI line changed** (`view_count_range=[1,1]`), so
every sample has one active slot drawn from the same 7-pose pool. The prediction — *this cell
fails like L1, therefore N>1 fusion is the load-bearing ingredient* — was registered before the
run.

**Result.** Five of five in-training checkpoints track L1's floor, not `m3off`'s. Strict sweep:
trained **0.04** / held-out **0.05**, against `m3off`'s 0.764/0.550 (gap 0.72/0.50 — an order of
magnitude beyond noise) and L1's 0.02/0.02 (gap +0.02/+0.03 — zero). At `el_0`, the *trained*
camera pose, it scores 0.02: it cannot do the task at all.

**Conclusion.** Multi-view sampling is the ingredient, and every other candidate for the
L1 → M3 gain is closed: the Plücker/history conditioning is inert, the aux heads are inert, and
the encoder architecture at N=1 reproduces nothing. PROPOSAL §2.2's fusion works; §2.1's
geometric conditioning and §2.4's aux heads do not.

### N-diversity ladder — how much multi-view signal is enough

**Method.** Five runs on `m3off`'s exact configuration, varying **one integer**
(`task.dataset.view_count_range`), K=7 slots throughout; each rung swept with 50 paired
episodes. Read on the trained mean, pre-registered 2026-09-24.

| rung | mean-N | trained | held-out | retention |
|---|---|---|---|---|
| `m3fixedn1` `[1,1]` | 1.0 | 0.036 | 0.047 | — |
| `m3v12` `[1,2]` | 1.5 | 0.028 | 0.043 | — |
| `m3v13` `[1,3]` | 2.0 | 0.080 | 0.073 | — |
| `m3v15` `[1,5]` | 3.0 | **0.456** | **0.373** | **82%** |
| `m3off` `[1,7]` | 4.0 | **0.764** | **0.550** | 72% |

**Conclusion.** **A knee between mean-N 2.0 and 3.0** (0.080 → 0.456), then a graded rise to
4.0: below mean-N 3 view diversity is not enough at all, and above it more still buys more. Two
surprises: `[1,5]` is the *more view-general* of the two working cells (82% of trained retained
against 72%) — more diversity buys absolute performance **and** costs generalization — and
Stage 1's pre-registered prediction (`[1,2]` partial, trained mean 0.15–0.40) was **falsified**;
it landed on the floor. The ladder closed at five rungs and no further rung is planned.

### `[2,7]` — the availability of N=1 is not the ingredient

**Method.** min-N 2, mean-N 4.5, so it **never trains at N=1**; evaluated at N=1 inference like
every other number here. The only cell that separates "N>1 needed" from "*variable* N needed".

**Result.** Indistinguishable from `[1,7]` on both axes: Δ trained **+0.104** (inside the
pre-registered 0.15 band) and Δ held-out **+0.003**, while clearing the floor cells by an order
of magnitude. Retention 63%.

**Conclusion.** The ingredient is having *enough views on average* — neither the presence of
N>1 nor the availability of N=1. This also corrects M3's stated rationale: the N=1 **inference**
path works without ever training at N=1, so randomising N was not necessary for the capability
it was invoked to protect. Confound it cannot remove: mean-N 4.5 against 4.0 is ~12% more
encoder compute per sample.

### Collapse is a failure mode

**Method.** `screen_collapse.py` — the encoder's relative output spread (`std` across 16
consecutive dataset states over the mean feature norm), matched `[7,7]` draw, always against a
**per-architecture random-init baseline measured on the cell itself**.

**Result.** `[1,2]`'s encoder has **collapsed**: 1.83e-07, **28,000× below** its own random-init
baseline and five orders of magnitude below every other cell — and its image path is severed
behaviourally (moving the *entire* image moves the action by 2.6e-05, ~256× below the other
cells). The same screen explains **L1's task split**: square 1.49e-04 / 1.99e-04 and can
3.05e-05 collapsed, lift 2.49e-02 healthy, on three runs that are matched by construction.

**Conclusion.** Collapse is **a** failure mode, not **the** failure mode. It explains `[1,2]`
and L1's task split, and it explains the detail that never fit anywhere else — why those cells
fail at the **trained** pose: a constant `z_g` means the policy acts open-loop. But `[1,3]`
is at the floor with a *healthy* encoder (2.7× above its own baseline), so a healthy screen
does not mean a working policy. `m3n1gate`, which trains at N=1 but always on the *same* view,
is healthy too — pointing at view variation the encoder cannot yet reconcile rather than at
view count (*suggestive, not matched*).

### The second failure mode — `[1,3]` fails with a healthy representation

**Method.** Three instruments on frozen checkpoints, all at matched draws: `z_v` decodability
(ridge), `z_g` decodability at single-view draws, and `image→action sensitivity`.

**Result.** `[1,3]` **beats the working cell at every one of them** — `z_v` rotation 13.44° vs
12.53°, `z_g` rotation 18.68° vs 19.58°, image→action 0.0067 vs 0.0068 — and still scores
0.080. So the failure is downstream of the encoder, and it is not about whether the information
is *present*.

**Conclusion.** This is the project's open question: what the policy *learned to do* with
correct information. Every instrument built here asks whether information is present, and the
difference between those two cells is evidently not presence.

### Proprioception dropout — the balance test, retired at its own gate

**Method.** An n=8 reading said `[1,3]` leaned ~1.75× harder on proprioception than the working
cell, and proprioception cannot see where the nut is. Re-measured at n=64 × 3 seeds behind a
**pre-registered gate** (the hypothesis lives only if the ratio exceeds 1.5× across all three
seeds); the falsifying intervention was coded and stood ready.

**Result.** The gate **failed** (min ratio 0.567; 2-of-3 seeds above the bar), so by
pre-registration the run was **never launched** — decided before its justification was
examined. The clean quantity refutes the hypothesis outright: the draw-independent
`proprio_only` arm is **1.008×** between `m3v13` and `m3off`, and **flat across all four
rungs** (0.5% spread).

**Conclusion.** The balance hypothesis is dead ladder-wide, not just for `[1,3]`. Its
by-product is methodological and it sharpens the open question: `image→action sensitivity` is
**not a scalar** — at matched inputs the cell rank order *flips* across probe ensembles
(0.512 / 0.272 / 1.362 across seeds) even though the tool is now bit-reproducible. The missing
instrument cannot be built by tightening that statistic.

### The `m3v15` latent gap — the latent numbers point the wrong way

**Method.** Filled the only rung missing from every latent table (`[1,5]`), with the draw
fingerprint byte-identical to the three existing grid cells.

**Result.** `[1,3]`'s anomaly stops being an anomaly and becomes the middle of a monotone trend.
As mean-N falls 4.0 → 3.0 → 2.0, `z_g` gets *more* view-invariant (0.1469 → 0.1131 → 0.0636)
and `z_v` *less* view-aware (0.581 → 0.491 → 0.382) — toward what the proposal calls the
goal — while behaviour falls 0.764 → 0.456 → 0.080.

**Conclusion.** Across four independently trained rungs the representation looks better and the
policy works worse. The project's central latent claim **anti-correlates with success** over the
entire working range.

### Relational probe (step 1a) — the geometry is already in `z_v`

**Method.** Frozen checkpoints, no training: a closed-form ridge and a 2000-step MLP readout of
geometry from `z_v`, against mean-predictor and shuffled-target controls, for `m3on` (Plücker +
EE history) and `m3off` (neither).

**Result.** `m3off` — a pure image encoder — recovers its own camera's absolute pose to
**4.13°**, the camera-frame EE position to **10.05 cm**, and the relative pose between two views
to **9.34° / 15.36 cm**, against mean-predictor floors of 37.94°, 22.85 cm and 65.35°. `m3on` is
**4.6×** better at relative-pose rotation (2.02°) and its `z_v` is **2.1× more
view-discriminative**.

**Conclusion.** The static-scene pose leak `PLAN.md` reasoned about is now **measured rather
than assumed**, which is why a head alone would be a post-hoc fit — the pre-registered 1b
decision rule fired. But Plücker is **live in the latent, inert in the behaviour**: it
measurably reorganizes the representation along a direction the policy did not need.

### M1's baseline was not collapsed

**Method.** M1's weights were deleted in a disk reclaim, so its square baseline was re-trained —
fidelity-gated first against its committed rollout curve (max |Δ| 0.06, inside the ~0.14 noise)
— and then collapse-screened.

**Result.** **0.1221**: **17.5× above** its own random-init baseline, and higher in absolute
terms than even the working L1 lift.

**Conclusion.** M1's failure at ±15° is **view-tiedness with a healthy, richly-varying
encoder**, which is a *different* failure mode from collapse. The collapse story is not one
story from M1 onward.

## Open questions

- **The second failure mode — the most open question here.** `[1,3]` sits at the floor (0.080)
  with a representation that beats the working cell at four separately-measured stages: encoder
  variance, `z_v` decodability, `z_g` decodability, and image→action sensitivity. Every
  instrument built here asks *whether information is present*; the difference between those two
  cells is evidently not presence. So the question is what the policy *learned to do* with
  correct information — and no probe in this repository can currently see it. See *The second
  failure mode* (`PROGRESS_DETAIL.md`). **Updated 2026-09-25:** the one proposed mechanism
  (balance) is now refuted and the `m3v15` probe shows the latent/behaviour mismatch is
  monotone across all four rungs, so this is a property of the ladder rather than of `[1,3]`;
  and the natural candidate statistic, `image→action sensitivity`, is **not a scalar** — at
  matched inputs its cell ranking flips across probe ensembles, which is a specification for
  the missing instrument rather than a reason to iterate on this one.
- **What makes training collapse?** Known: not "too few views" (`m3n1gate` trains at N=1 and is
  healthy), established by epoch ~100, and task-dependent (L1 collapses on square and can, not
  lift). Unknown: the mechanism, the layer, and whether collapse is a cause or a symptom. Also
  unknown and worth stating because it rules out the cheapest design — **whether collapse
  precedes the behavioural failure** — since no existing run has a per-epoch checkpoint series.
- **Lift is untested for M3.** It is the one task where L1 already wins (0.76–0.96), so an
  M3-on run there is a genuine "did we break it" question rather than a result. ~1 h
  (lift is 127 batches/epoch).
- **Second seed.** Every *behavioural* number in this document is n=1 at a resolution worse
  than ±0.05. A second seed on one M3 or M4 cell would materially strengthen the nulls, which
  are "indistinguishable at this resolution", not "proven identical".
- **The elevation asymmetry.** Square collapses at `el_m15` for every M3 cell while can
  holds 0.22; `[1,5]` holds `el_p15` (0.26) and not `el_m15` (0.00). Unexplained.
- **M5's premise.** M3 already does N=1 novel-view inference well, so distillation is only
  worth it if the fused latent demonstrably carries something the single-view path cannot,
  which no result so far shows.

## Where the rest is

- **`PROGRESS_DETAIL.md`** — Part 1 the full protocol and the noise derivation; Part 2 the
  investigation log verbatim, section by section (L1, N>1, step 1a, the ladder with its
  pre-registration and stop rule, the collapse screen, the second failure mode, propdrop, the
  `m3v15` probe, `[2,7]`, M1's re-screen) plus the code map; **Appendix A** an index of every
  results table with the artifact that holds it; **Appendix B** five parked investigations;
  **Appendix C** the record of corrections and superseded numbers — **read C before quoting
  any number**.
- **`NOTES.md`** — runbooks (launch and sweep commands), timings, disk, and the recurring traps.
- **`PLAN.md`** — the milestone plan and what remains: ranked next steps, and the deliberately
  deferred runs.
- **`PROPOSAL.md`** — the direction and the original predictions.

Sweeps, training logs, videos, and weights are inventoried in `NOTES.md`. Committed to git:
every sweep's `eval_log.json` with its viewpoint videos, and the per-batch training logs.
Weights (4.6 GB per checkpoint) are **not** in git — rsync only.
