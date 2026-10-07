# PROGRESS

Training a diffusion policy that keeps working from camera viewpoints it was never
trained on, and at inference time from a **single camera placed at a novel pose**.
`PROPOSAL.md` holds the direction and the original predictions; `PLAN.md` what remains;
`NOTES.md` the operational detail (runbooks, timings, disk, recurring traps).

**This file is the module and its measured behaviour**: the milestones, their headline results,
the findings that shaped the story, and the open questions. The long-form record this file was
distilled from — the full protocol, the verbatim investigation log, every per-viewpoint table
and the corrections register — is at git tag **`docs-full-20260930`** (see *Where the rest is*).

Last updated 2026-10-07.

## Summary

**What works.** M3's per-slot fusion reaches **0.55 (square) / 0.74 (can)** at held-out
viewpoints, where the single-view baseline M1 scores 0.00–0.02 and the conditioning-free L1
baseline sits at the floor. **What is inert.**Plücker + camera-frame-history conditioning and auxiliary heads change nothing measurable (M3, M4). **What is load-bearing.** Multi-view *sampling*: the same encoder forced to one view per sample scores 0.04/0.05, and the ladder shows the ingredient is *enough views on average* — a knee between
mean-N 2.0 and 3.0, then a graded rise. **What is also load-bearing, and new (2026-10-01).** View
*quality*: `m3pm60` drops the two ±90° views from the pool at an unchanged mean-N of 3.0 and lifts
square's trained mean 0.456 → **0.728** and held-out 0.373 → **0.533** — reaching `[1,7]`'s
performance at two thirds of its mean-N. **What the healthy series adds (2026-10-06).** The clear
run's raw latents: the geometry (up to rotation) is decided by ~10–25 epochs, the scales by
~50–75, the view-vs-state balance not before **126–149** — and at inference the N=1 fused latent
is within ~0.5% (direction) of the 11-view one. **What the clear run's held-out sweep adds
(2026-10-07).** Its azimuth generalisation is **interpolation, not view-tiedness**: 7.5° off the
nearest trained pose it reads **0.716** against 0.747 in-distribution, while past the ring
(±82.5/±90) it falls to 0.310 and on the elevation orbit to **0 of 50 both ways** — so elevation is
not a view-count problem, and at K=11/mean-N 6.0 it is *worse* there than the K=7 cells.
**What is not explained.** Two distinct failure modes
sit at the floor — an encoder *collapse* (`[1,2]`, L1 on square/can) and a second one with no
candidate mechanism left: `[1,3]` fails while its representation beats the working cell on every
instrument built so far. Honest summary against the proposal: **§2.2's fusion works; §2.1's
geometric conditioning and §2.4's aux heads are both inert.**

## Status

| rung | what changed | verdict |
|---|---|---|
| **M1** | single-view DP baseline (agentview at az_0), three tasks | collapses to ≈0 at ±15° azimuth on all three |
| **M2** | multi-view data: a 13-pose azimuth ring re-rendered from the demos' stored simulator states | validated by an N=1 gate that reproduces M1's whole curve; ±75°/±90° views are low value |
| **L1** | view diversity only: M1's exact model, its single camera slot filled per sample from a randomly drawn training view | task-split **in both directions** — solves lift out to ±75°, destroys square/can |
| **M3** | view-conditioned encoder: per-view latents, Plücker + camera-frame-history conditioning, K=7 slots, per-sample N∈[1,7], MHA fusion | **solves square and can at held-out views**; its conditioning contributes nothing |
| **M4** | per-view camera-frame auxiliary action heads | mechanism is real (`aux_loss` falls 52×), behaviour is null (+0.02/+0.04) |
| **M5** | single-camera N=1 inference at a novel pose | **capability measured** — every M3 number is already this; the optional distillation stage is uncoded |
| **N>1** | M3's model with one CLI line changed so every sample sees a single view | **the load-bearing ingredient** — reproduces L1, not M3 |
| **ladder** | the N-diversity ladder closed: mean-N 1.0 / 1.5 / 2.0 / 3.0 / 4.0 | **knee between 2.0 and 3.0** (0.080 → 0.456), then graded to 0.764; `[1,5]` is *more* view-general than `[1,7]` |
| **`[2,7]`** | min-N 2, mean-N 4.5 — never trains at N=1 | **indistinguishable from `[1,7]`** — enough views *on average* is the ingredient |
| **collapse** | encoder output spread across every surviving checkpoint, matched draw, random-init controls | **a failure mode, not *the* one** — explains `[1,2]` and L1's task split; `[1,3]` fails while healthy |
| **M1 re-screen** | re-trained the deleted baseline, fidelity-gated, then collapse-screened | **not collapsed** — view-tiedness and collapse are distinct failure modes |
| **balance test** | `[1,3]`'s proprioception contrast re-measured at n=64 × 3 seeds, pre-registered | **refuted; intervention retired unlaunched** |
| **(a) lift + `m3on`** | recovered 2026-10-01 — trained 09-26, then skipped by a guard bug | **does not break lift** (held-out 0.870 vs L1 0.873); elevation is where it gives ground |
| **(b) `[1,5]` seed 43** | recovered 2026-10-01; first reproduction of any ladder rung | **Δ 0.052 / 0.083, inside the band** — the knee stands at two seeds |
| **(c) ±60° pool** | recovered 2026-10-01; pool `[2,4,6,8,10]` at matched mean-N 3.0 | **pool curation is worth +0.272 trained / +0.160 held-out; reaches `[1,7]` at 2/3 its mean-N** (size/content confounded) |

Internal labels, used in the code and configs: **L1** names this fork's second rung
(M1's architecture, randomized view) — L0 is M1 itself, and L2–L4 are M3, M4 and M5.

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
look-at point at 0/±15°, the off-manifold test. `el_0` is the same camera pose as `az_0`,
which makes it the internal consistency check.

**Multi-view data (M2).** A 13-pose azimuth ring every 15° out to ±90°. The **even indices**
(az −90/−60/−30/0/+30/+60/+90, every 30°) form the *training pool*; the **odd indices**
(±15/±45/±75) are **never sampled during training**, so every "held out" column below is held
out by construction.

**Noise and resolution.** Treat differences below **~0.1 on a single viewpoint as unmeasured**:
two 50-episode sweeps of one checkpoint through two presets that place the camera at the *same*
pose differ by **0.14** (`m4on`: `el_0` 0.70 vs `az_0` 0.84), and unseeded diffusion sampling
adds ~0.08 spread. On a **mean over viewpoints** the working threshold is **0.15** — a stated
convention, adopted by the ladder so its branch decisions were fixed in advance rather than
chosen after the fact. **Revised upward 2026-10-01** by the recovered runs, which measured both
axes worse than stated: the same-pose spread reaches **0.18** (`m3v15` s43, `az_0` 0.480 vs
`el_0` 0.300) and the across-seed spread of ONE config at ONE viewpoint reaches **0.200**
(`az_p30`, `[1,5]` s42 vs s43). Single-viewpoint cells are therefore noisier than this document
long assumed on both axes, and are never read individually. `mean_score` hides effects `success_rate` shows (on lift it saturates at
1.000 for a policy that succeeds 0.76), and `val_loss` does not track rollout behaviour (all
four square M3 cells end at 0.0568–0.0602, essentially L1's 0.060, while rolling out like M1).
Every behavioural number below is **one model, 50 paired episodes**; the probe and screen
sections are not episode-based and each states its own noise floor.

## M1 — single-view baseline: the reference curve

**Method.** Diffusion Policy on one camera at the training pose (az_0), 201 epochs, seed 42,
one run per task.

**Results** — `success_rate`, 50 paired episodes (`azimuth_sweep5`):

| viewpoint | square | can | lift |
|---|---|---|---|
| az_0 | 0.82 | 0.98 | 0.76 |
| az_m15 | 0.02 | 0.08 | 0.08 |
| az_p15 | 0.00 | 0.00 | 0.08 |
| az_m30 | 0.00 | 0.00 | 0.00 |
| az_p30 | 0.00 | 0.00 | 0.00 |

**Conclusion.** The baseline is extremely view-tied: at az_0 it sits at or near the published
band (paper: Lift ≈ 1.00, Square ≈ 0.9–1.0; here 0.98 and 0.82), and success collapses to ≈0 at
the smallest perturbation tested. This is the reference curve the later milestones must beat.
The collapse is a property of the policy, not of the camera looking somewhere useless: the
displaced views keep their texture (frame std 66–69 vs az_0's 65) and differ from az_0 by only
~10–11% mean pixel value on same-seed first frames, so the scene is fully visible at ±15°/±30°.

## M2 — multi-view data (rendered 2026-09-15)

**Method.** For every demo timestep: replay the stored mujoco `states` through
`env.reset_to`, orbit the fixed camera to each of 13 azimuth poses, render, and write a
ReplayBuffer-compatible zarr — per-view 84×84 images (`uint8`, `Jpeg2k(50)`), low-dim
observations, absolute actions, and per-view camera parameters in `meta`. Camera-frame actions
and Plücker maps are not stored: both are deterministic functions of the stored camera
parameters (plus the base actions), so M3/M4 compute them at train time. Runbook: `NOTES.md`.

**Results.** `--workers 4`, one task at a time, sequential — **8h00m total** at ~2.2 steps/s:

| task | steps | images (×13) | gate 1 mean\|diff\| | gate 2 in-frame | time | on disk |
|---|---|---|---|---|---|---|
| square | 30154 | 392k | 1.454/255 PASS | 33/39 | 3h50m | 1.6 GB |
| can | 23207 | 302k | 2.801/255 PASS | 39/39 | 2h59m | 2.4 GB |
| lift | 9666 | 126k | 1.502/255 PASS | 39/39 | 1h11m | 496 MB |

**819k images, 4.5 GB.** Three gates run at the end of every generation: az_0 re-render vs the
hdf5's stored image (per-255 mean |diff| ≲ 3, where ~40 is the signature of a missing or
doubled `[::-1]` flip — this is what catches an upside-down dataset); the projected gripper
site inside frame (validates the camera-parameter chain the Plücker maps depend on); the ring
montage. The **generated zarrs live on `miroc-server`**, not in git.

**The N=1 gate: why everything later is measured on the generated zarr.** A run trained on the
generated data using **only** its az_0 view — the same single-camera-pose setup as M1, from a
different data source (a *fidelity control*, not a new method):

| viewpoint | M1 success | N=1 success | M1 mean | N=1 mean |
|---|---|---|---|---|
| az_0 | 0.760 | **0.840** | 1.000 | 1.000 |
| az_m15 | 0.080 | 0.040 | 0.460 | 0.280 |
| az_p15 | 0.080 | 0.160 | 0.500 | 0.620 |
| az_m30 | 0.000 | 0.000 | 0.000 | 0.000 |
| az_p30 | 0.000 | 0.000 | 0.220 | 0.240 |

It reproduces M1's **whole degradation curve** — high at az_0, collapse at ±15°, zero at ±30° —
not merely its az_0 score.

**Two findings that changed M3's design.** Per-sample view subsampling saves encoder FLOPs but
not IO — `SequenceSampler` reads *every* zarr key regardless of `shape_meta`, so a one-view
config still reads all 13 (156 ms vs 24 ms per sample), and M3 must restrict the sampler's keys,
not just `shape_meta`. And **±75°/±90° are low value** (table edge, object small or out of
frame); they were kept (2/13 of the bytes) because the encoder accepts a subset of view slots.

## M3 — view-conditioned encoder + fusion

**Method.** One shared resnet18 is applied to each of **K=7 slots**. `conv1` is widened from 3
to 9 input channels; the 6 extra channels carry the view's **Plücker ray map**. The per-view
latent `z_v` is additionally **modulated** by the camera-frame end-effector history through
**AdaGN / FiLM** at each residual stage (heads zero-init). Fusion is `nn.MultiheadAttention` over
the N view tokens with a single **learnable query** → `z_g` (512-d), which degenerates correctly
at N=1. **N is randomized per sample** over [1, K] with the slot count fixed at K and a
`view_mask` marking the live ones. **Matched capacity:** `output_shape()` is **521 = 512 + 9**,
identical to M1's and L1's — all **148** `ConditionalUnet1D` parameter tensors shape-identical
(the encoder itself grows 11,176,512 → 12,532,928, +12.14%). **The ablation** is two constructor
flags — `use_plucker` and `use_eef_hist` — so the control is a 2×2; all four cells instantiate
12,532,928 parameters, and `use_plucker=False` drives the six ray channels' gradient to *exactly*
0.0. The pose arrives as an ordinary per-sample obs key, and `shape_meta` keeps exactly **one**
rgb key — which is what lets rollouts run at all.

**Results** — `success_rate`, 50 paired episodes, means split by whether the pose was trained
(`*` = in the training pool):

| model | square trained | square **held out** | can trained | can **held out** |
|---|---|---|---|---|
| m3off | 0.764 | **0.550** | 0.896 | **0.740** |
| m3plucker | 0.760 | **0.500** | — | — |
| m3eef | 0.768 | **0.573** | — | — |
| m3on | 0.736 | **0.547** | 0.868 | **0.767** |
| *L1* | *0.02* | *0.00–0.06* | *0.02* | *0.00–0.06* |

The N=1 fidelity gate (single slot, `view_pool=[6]` = az_0) reads **0.94** at az_0 against M1's
0.82, and 0.00 at ±30° — it reproduces M1's whole curve. The 0.94 is not a leak
(`view_count_range=[1,1]` admits no other view); the same +0.08–0.12 pattern appeared in the
lift gate above.

**Conclusion.** **It solves square and can — the two tasks L1 destroyed**: 0.50–0.57 on square
and 0.74–0.77 on can at azimuths excluded from the training pool, where M1 is 0.00–0.02 and L1
is at the noise floor. The load-bearing comparison is L1 **at trained poses**: L1 trained on the
same pool and still scores ~0.02 at ±30° — its failure was never a failure to *generalise*, it
failed to solve the task at all; so the stronger claim is that **per-slot fusion turns view
diversity from harmful into sufficient**. **The conditioning contributes nothing:** `m3off`
matches or beats `m3on` on both tasks at every viewpoint, and can's held-out means run the
*wrong* way (0.740 vs 0.767). **Elevation is a partial extrapolation result:** 0.18–0.34 at
`el_p15` where M1 is 0.00–0.04, and 0.22 at `el_m15` on can — but square collapses to 0.00–0.04
at `el_m15` for every cell, an unexplained asymmetry. **Costs and limits:** a little
in-distribution (az_0 0.76–0.80 vs M1's 0.82); every cell is n=1; the four architectural changes
are confounded (resolved later, in favour of N>1); and **lift was deliberately not run** — it is
the one task L1 already wins, so it is the place M3 could regress.

## M4 — per-view auxiliary action heads

**Method.** A shared per-view head (`PerViewAuxActionHead`, **151,888** parameters, 0.052% of
the policy) predicts the next `n_action_steps` (8) actions **in that camera's frame** as
mean-squared error, summed with the diffusion loss at weight **1.0** — a guess, no sweep: the
config's predicted init ratio was wrong (measured batch-0 `aux/diff` = **0.316**), and a
pre-fixed rule "proceed unless the ratio exceeds 1.0" kept it. The target is a top-level sample
key (never an obs key) and the output layer is zero-init. **The pair is RNG-locked:** `m4on`'s
batch-0 `train_loss − aux_loss` = **1.081642270** equals `m4off`'s `train_loss` to a difference
of `0.000e+00`, and the weight-0 arm is asserted bit-identical to the parent policy's loss and
**all 179** gradients.

**Results** — strict `success_rate`, 50 paired episodes:

| model | trained\* | **held-out** | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|---|
| m3off | 0.76 | **0.55** | 0.88 | 0.30 | 0.04 |
| m3plucker | 0.76 | 0.50 | 0.76 | 0.24 | 0.02 |
| m3eef | 0.77 | 0.57 | 0.82 | 0.18 | 0.02 |
| m3on | 0.74 | 0.55 | 0.78 | 0.18 | 0.00 |
| **m4on** | **0.76** | **0.56** | 0.70 | 0.00 | 0.00 |
| **m4off** | **0.74** | **0.52** | 0.80 | 0.14 | 0.00 |
| *delta* | *+0.02* | *+0.04* | *−0.10* | *−0.14* | *0.00* |

Per-viewpoint deltas run −0.12 to +0.18 with the sign flipping freely. **The mechanism, by
contrast, is alive:** `aux_loss` collapses **52×** from its batch-0 value (0.2389 → **0.0066**
at epoch 200), and it is not a collapse to the mean — a constant head scores 0.2692 on real
samples, so the epoch-25 loss is 6.4% of that floor. **The trunk was reshaped about as much as a
re-seed:** EMA relative-L2 on `obs_encoder` is 0.596 under aux versus 0.471 for the RNG-only
reference (`m4off` vs committed `m3on`), so the null is not "the gradient was too weak".

**Conclusion.** **The mechanism is real and the behavioural effect is null.** The surviving
explanation is the one the design did not consider: **the information the aux head needs is
already in `z_v`** — a 151,888-parameter head can fit camera-frame targets post hoc from a
latent that already encodes the scene. **Elevation cuts both ways and the cut is measured:**
`m4on` 0.00 vs `m3on` 0.18 at `el_p15` does not survive its own check — `el_0` is the same
camera pose as `az_0` and reads 0.70 vs 0.84, the 0.14 spread of *Noise and resolution*. Only
square was run for M4.

## M5 — single-novel-view inference

**Capability measured, not pending.** Every M3 number is already an N=1 inference at a novel
pose: `eval_novel_view.py --m3-slots K` reduces the env-side `shape_meta` to the single
renderable key and publishes the perturbed camera pose (read back **from the simulator**) into
a single slot; square infers at N=1 with 0.55 success on held-out azimuths. What remains is the
**optional distillation stage** — a single-view student encoder regressing the frozen teacher's
multi-view fused latent `z_g`. Its premise is unestablished: the capability lives in the
*training signal* (N>1 sampling), `[2,7]` shows N=1 inference works without ever training at
N=1, and no result shows a student would gain anything. The deciding measurement before coding
it: whether `z_g` at N=1 predicts behaviour better than the N=1 encoder's own output. Gate if
built: final sweep tables against M1's degradation curves. The clear-run analysis (2026-10-06)
gives the first latent-space evidence on the premise: at inference, dropping 10 of the 11 views
changes `z_g` by ~0.5% in direction (`cka` 0.998) — the fusion is nearly N-invariant, so a student
would be reproducing something the single view already determines at test time.

## The three recovered runs (2026-10-01)

### (a) lift + `m3on` — a regression test, and M3 passes it

Regression Test Passed: M3 matches L1's overall performance on the lift task, successfully dominating across all three tasks (square, can, and lift).

Elevation Asymmetry Confirmed: M3 exhibits performance drops at specific elevations (el_m15/el_p15), mirroring the asymmetry previously observed on square and can tasks.

Healthy Encoder: The collapse screen score (10.5$\times$ random initialization) confirms a robust, highly active encoder free of representation collapse.

### (b) second seed on `[1,5]` — the ladder survives it

A second random seed replication on the `[1,5]` cell successfully validated the experiment. Because the performance differences between the two seeds stayed within the pre-registered 0.15 threshold, the original ladder rung and its key performance metrics (retention, elevation) are confirmed to be robust and stable rather than mere seed artifacts.

### (c) the ±60° pool — view quality, and it is worth a lot

Same range `[1,5]`, same mean-N 3.0, same encoder, same seed as `m3v15` s42; the **only** change is
the pool — `[2,4,6,8,10]` = az (−60,−30,0,+30,+60), dropping the two ±90° views M2 had already
measured as low value (table edge, object small or out of frame).

| viewpoint | az_0 | ±15 | ±30 | ±45 | ±60 | ±75 |
|---|---|---|---|---|---|---|
| **m3pm60** | 0.680 | 0.520 / 0.480 | 0.740 / 0.720 | 0.600 / 0.620 | 0.700 / 0.800 | 0.540 / 0.440 |
| `m3v15` s42 | 0.460 | 0.420 / 0.380 | 0.360 / 0.540 | 0.340 / 0.340 | 0.440 / 0.480 | 0.340 / 0.420 |
| Δ | +0.220 | +0.100 / +0.100 | +0.380 / +0.180 | +0.260 / +0.280 | +0.260 / +0.320 | +0.200 / +0.020 |

**All eleven viewpoints improve.** Trained mean **0.728 vs 0.456 (Δ +0.272)** — far outside the
band and ~5× the measured seed spread. Held-out mean **0.533 vs 0.373 (Δ +0.160)** — just past the
band and ~2× the seed spread, so read it as suggestive rather than decisive. Retention 73% vs 82%.

**The result: at mean-N 3.0, `m3pm60` reaches what `[1,7]` needs mean-N 4.0 for** — trained
0.728 vs `m3off`'s 0.764, held-out 0.533 vs 0.550. Elevation agrees (0.680 / 0.020 / 0.220 vs
0.380 / 0.000 / 0.260), and its `el_0` equals its own `az_0` to the third decimal — the cleanest
same-pose consistency check any cell here has produced. Collapse screen: **4.70e-02, 7.8×** its
own baseline, the richest encoder in the project.

## The per-epoch latent series (the "clear" run, 2026-10-01)

**What it is.** The first run in this project with a per-epoch latent series: 11-view all-on
training (`m4_aux_image_abs_multiview_az75`, K=11 over ring 1..11, N ∈ [1,11], mean-N **6.0**),
whose workspace encodes a **fixed 128-state probe set every epoch** with the EMA weights, at a
full 11-view draw and at the N=1 `az_0` inference condition. Completed clean: 201 epochs in
3 h 32 m, `TRAIN_EXIT=0`, **201 snapshots** (1.71 MB each, 328 MB total, all finite), scalars in
both `logs.json.txt` and `latent_snapshots/snapshots.jsonl`, raw `z_v`/`z_g` in
`latent_snapshots/epoch_XXXX.npz`. **No held-out split** — all 11 ring views train — so it is an
*instrument*, not a ladder rung, and its behaviour is not comparable to the committed cells.

**The trajectory.**

| epoch | `zv_norm_mean` | `zv_rel_spread` | `zg_rel_spread` | `zv_pair_ratio` | `zg_pr` | rollout |
|---|---|---|---|---|---|---|
| 0 | 31.92 | 0.0403 | 0.0154 | 0.857 | 1.93 | 0.00 |
| 25 | 16.08 | 0.1501 | 0.1196 | 0.494 | 5.97 | — |
| 50 | 10.16 | 0.1958 | 0.1453 | 0.508 | 5.66 | 0.44 |
| 100 | 7.29 | 0.2258 | 0.1412 | 0.509 | 5.84 | 0.70 |
| 150 | 6.85 | 0.2280 | 0.1385 | 0.495 | 5.87 | 0.80 |
| 200 | 6.91 | 0.2240 | 0.1372 | 0.485 | 5.87 | 0.76 |

The encoder's geometry settles early (~10–25 epochs) while behaviour keeps improving until ~150

An azimuth_interp scan was performed along the ring during the final epoch.

**Pre-registered before the raw-tensor analysis (2026-10-06).** The scalar-only dry run
(`analyze_latent_series.py --scalars-only` on the committed log) fixes the settle convention at
**tol 5% of each curve's total excursion** and sharpens "saturates by epoch ~50–75": under that one
stated convention the scalars settle between **epoch 48** (`zg_n1_norm_mean`; `zg_norm_mean` 49)
and **epoch 149** (`zv_pr`), with the view-structure statistic `zv_pair_ratio` not until **126**.
The raw-tensor run (`data/clear_analysis.sh`, on `miroc-server`) is read against **measured** chance
levels and floors — the plain linear CKA's permutation floor is ≈ d/(n+d), i.e. ≈0.27 for `zv_flat`
(1408×512) and ≈0.80 for `zg` (128×512) at this run's shapes, and the fp16 storage floor is 4.9e-4
— never against zero. Expectations, registered before the run:

1. the recheck gate passes: npz-derived scalars within 5e-3 of the fp32 logged ones (expected ~1e-3);
2. the tensor drift settles somewhere in **50–150**; the view-structure scalars move until
   ~126–149, so a tensor freezing before epoch 50 would be the surprise — early settling is *not*
   assumed, late settling is *not* a falsification;
3. consecutive-epoch CKA stays ≥0.99, well above its floor;
4. per-view spreads settle within ~15 epochs of each other.

**Falsifier on record:** if any drift measure is still above the fp16 floor at epoch 150, then
"the encoder is decided early" is too strong a reading of this run and will be re-worded.

**The results (2026-10-06).** `data/clear_analysis.sh` ran the analyzer on `miroc-server` over
the run's 328 MB of `latent_snapshots/`: all **201 epochs** analyzed, every gate passed (contiguous
epochs, `state_idx` identical throughout), and the npz-derived scalars agree with the fp32 ones the
hook logged to **8.05e-05** worst-case over 2010 epoch-scalar pairs — fp16 storage costs the
*aggregates* almost nothing. Chance levels are **measured, not modelled**: the linear CKA's
permutation floor (max over 8 permutations) is **0.009** for `zv_flat`, **0.048** for `zg`,
**0.054** for `zg_n1` — 15–30× *below* the isotropic `d/(n+d)` estimate registered in advance
(0.27 / 0.80), because the latents are effectively ~6-dimensional (participation ratio 5.82 /
5.87; `PR/(n+PR)` fits the permutation *means* to within ~1.2×, so read it as an effective-rank
effect at the factor-2 level, not a law). The split-half gap on the final pair is ~1e-08 — the
reading there is saturated at ≈1.

*When does each thing settle?* (tol 5% of the curve's total excursion, so "settled at 53" means
the transient is over to within ~2%/epoch):

| measure | zv_flat | zg | zg_n1 | what it says |
|---|---|---|---|---|
| consecutive drift (`rel_prev`) | 53 | 65 | 70 | end of the 5%-convention transient |
| consecutive CKA | 3 | 3 | 4 | geometry up to rotation, settled almost immediately |
| consecutive Procrustes | 12 | 8 | 21 | no restructuring beyond a rigid rotation after ~20 |
| drift to the final epoch | 154 | 115 | 125 | distance to the final latent |
| scalars | `zv_norm_mean` 75, spreads 72/87, `zv_pair_ratio` **126**, `zv_pr` **149** | `zg_norm_mean` 49, rel_spread 75, `zg_pr` 55 | `zg_n1_norm_mean` 48, rel_spread 96 | the view-structure statistics are the slowest |
| per-view drift | 44 (`az_m75`) … 58 (`az_p15`/`az_p30`) | | | interior views keep moving longest |

*What keeps moving, and what it is.* The strict row-aligned drift decays smoothly but never
freezes: at epoch 150 it is 4.6e-03 — 12–23× the ~2–4e-04 statistic-level storage floor — and
7.3e-04 by epoch 199 (`frozen`, i.e. every later value below the floor, is never met for
`zv_flat`/`zg_n1`; the one for `zg` is the final single epoch — a razor edge, **not** a freeze).
Decomposed on `zv_flat`, it is a per-row **direction** change that decelerates, with magnitudes
nearly constant — θ = √(2·mean cosine distance):

| epoch | `rel_prev` | per-row rotation θ | `cka_prev` | norm change/epoch |
|---|---|---|---|---|
| 25 | 3.7e-02 | 1.59° | 0.999829 | −2.5% |
| 50 | 2.2e-02 | 1.05° | 0.999936 | −1.3% |
| 100 | 9.4e-03 | 0.53° | 0.999977 | −0.33% |
| 150 | 4.6e-03 | 0.28° | 0.999992 | −0.03% |
| 200 | 6.8e-04 | 0.04° | 1.000000 | +0.01% |

*Per-view, final epoch:* the view-vs-state ratio is U-shaped over the ring — most view-consistent
at `az_0` (0.388) and least at the edges (`az_m75` 0.707) — and norms are U-shaped too:

| ring view | `az_m75` | `az_m60` | `az_m45` | `az_m30` | `az_m15` | `az_0` | `az_p15` | `az_p30` | `az_p45` | `az_p60` | `az_p75` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| norm | 8.09 | 7.35 | 6.87 | 6.59 | 6.49 | 6.43 | 6.48 | 6.58 | 6.71 | 6.98 | 7.45 |
| rel_spread | 0.220 | 0.214 | 0.223 | 0.219 | 0.209 | 0.205 | 0.211 | 0.225 | 0.241 | 0.250 | 0.246 |
| pair/cross | 0.707 | 0.565 | 0.475 | 0.423 | 0.396 | 0.388 | 0.391 | 0.412 | 0.452 | 0.527 | 0.645 |
| drift settle | 44 | 47 | 52 | 54 | 56 | 57 | 58 | 58 | 55 | 51 | 48 |

Every per-view `rel_spread` (0.205–0.250) is four orders of magnitude above the collapse
screen's degenerate band — this run's encoder is richly varying in every view.

*The fused latent (M5's premise), at epochs 0 / 100 / 200:*

| relation | e0 | e100 | e200 |
|---|---|---|---|
| `rel(zg_n1, zg_full)` | 0.064 | 0.121 | 0.103 |
| mean per-row cosine distance | 0.0017 | 0.0073 | 0.0053 |
| `cka(zg_n1, zg_full)` | 0.993 | 0.997 | 0.998 |
| `cka(zg_full, zv_az0)` | 0.967 | 0.724 | 0.689 |
| `cka(zg_full, mean_pool)` | 0.966 | 0.707 | 0.677 |

Dropping 10 of the 11 views changes the fused latent by **~0.5% in direction** (cosine distance
0.0053; CKA 0.998; norm ratio 1.006): at inference the fusion output barely depends on how many
views it sees. Over training the full-draw fused latent nonetheless moves *away* from both the raw
`az_0` view latent (0.967 → 0.689) and the uniform mean-pool (0.966 → 0.677) — training changed
what the fusion computes, just not its dependence on N. Reading: **the multi-view signal changes
the encoder, not the inference-time fusion** — with the caveats that this is latent proximity, not
the deciding behavioural measurement, and that it was already true at epoch 0.

*The sweep* (epoch-200 `latest.ckpt`, 50 paired episodes per viewpoint; the checkpoint linkage
rests on the driver — `eval_log.json` itself carries no metadata):

| viewpoint | `az_0` | ±15 | ±30 | ±45 | ±60 | ±75 |
|---|---|---|---|---|---|---|
| `success_rate` | 0.76 | 0.84 / 0.82 | 0.76 / 0.84 | 0.70 / 0.78 | 0.72 / 0.76 | **0.54** / 0.70 |

* **Core Takeaway:** An encoder's coarse geometry settles early (~10–25 epochs), but a stable geometry doesn't mean a stable balance. Statistical evolution and behavioral improvement continue hand-in-hand to the end.
* **Key Findings:**
* **Viewpoints:** 11-view average success is 0.747, with latent statistics and behavior aligning on marginal views (e.g., `az_m75`).
* **Timeline:** Geometry sets by ~10–25 epochs $\rightarrow$ scales by ~50–75 $\rightarrow$ view-state balance matches behavior at ~126–149 epochs. Slow per-row rotation persists through the end.

### The held-out sweep (2026-10-07): the one measurement the clear run lacked

**Method.** `data/clear_heldout_eval.sh` swept the epoch-200 `latest.ckpt` through the
`azimuth_offgrid` preset — the **7.5°-offset ring**, azimuth ±7.5 … ±82.5 and ±90, *none* of which
the run ever trained on (it trained on every 15° pose from −75 to +75) — at 50 paired episodes per
viewpoint, `--m3-slots 11`, plus `elevation_az0` for cross-cell comparability. The smoke gate
(4 episodes × 14 viewpoints) passed first; the sweep ran 15:53–17:10 and elevation to 17:28
(`SWEEP_EXIT=0`, `ELEV_EXIT=0`).

**Registered before the run** (in the driver): the midpoints should land near the mean of their
neighbouring trained poses — an off-grid mean of roughly **0.6–0.8** over ±7.5…±67.5 means the
encoder *interpolates*; ±82.5/±90 may degrade; below ~0.4 overall would be real view-tiedness at
unvisited poses.

| | mean | n |
|---|---|---|
| in-distribution ring (trained poses, committed) | **0.747** | 11 |
| off-grid ±7.5…±67.5 (interpolation) | **0.716** | 10 |
| off-grid ±82.5/±90 (beyond the trained ±75) | **0.310** | 4 |
| off-grid, all 14 | 0.600 | 14 |

**Result — it interpolates.** Against the average of the two *neighbouring trained* poses per
viewpoint, 8 of 10 deltas land within ±0.07 (`az_m22.5` −0.12 and `az_p7.5` −0.21 are the
exceptions); the observed mean is **0.716 against a neighbour-average prediction of 0.760**, and
**0.031 below the in-distribution ring** (0.747). Both comparisons sit far inside the 0.15
mean-over-viewpoints threshold — and single cells are never read individually here
(*Noise and resolution*), so the registered quantity is the mean.

**Extrapolation degrades, and the cause is confounded.** ±82.5 reads 0.40/0.56 and ±90 reads
0.10/0.18 — a monotone falloff in |azimuth| past the trained ±75 (0.54/0.70). But M2 measured those
same poses as **low-value views** (table edge, object small or out of frame), so at ±82.5/±90 the
policy's extrapolation and the scene's visibility are confounded and this run cannot separate them.

**Elevation fails outright, and more views do not fix it.** `el_0` **0.700** — the same camera pose
as `az_0`, which reads 0.760 in-distribution, a 0.06 gap inside the documented 0.14–0.20 same-pose
spread, so the harness is consistent — then **0 of 50 at `el_m15` and 0 of 50 at `el_p15`**. That is
the pessimistic end of the M3 family (`m3off` 0.30/0.04, `m3pm60` 0.02/0.22, `m4on` 0.00/0.00), and
the clear run is the cell with the **most** views (mean-N 6.0, K=11). Reading: **elevation is not a
view-count problem** — the training distribution is an azimuth ring with no elevation variation
anywhere in it, and azimuth diversity does not teach a direction it never sees. The asymmetry
already on record (square at the floor at `el_m15` for every cell) is joined here by `el_p15` at the
floor, so for this cell elevation is not "partial extrapolation" in *either* direction.

### The latent-distribution figures (2026-10-07)

`visualize_latent_distribution.py compute` reads six of the run's `epoch_XXXX.npz` and fits **one**
2-D PCA basis on epoch 200's `z_v` ((S·K, D) = (1408, 512)), projecting every epoch's `z_v`, `z_g`
and `z_g_n1` into that frame (`evr` **0.353 / 0.152**); `plot` renders four figures into
`data/analysis_clear/figures/`. Compute is deterministic (fixed SVD sign convention) and takes ~1 s;
the committed coordinates are 219 KB, so any later latent can be projected into the same frame.

* **fig1 — `z_v` PCA small multiples, epoch 0/25/50/100/150/200.** At epoch 200 the 11 per-view
  centroids form a tight chain ordered by azimuth, and its spread is **~9% of the state cloud's**
  (both in the 2-D panel, ratio 0.118 at epoch 25 → 0.093 at 200) — the encoder keeps view identity
  as a minor, geometrically consistent axis rather than a dominant one. *Caveat, carried on the
  figure itself:* the basis is fit at epoch 200 and earlier epochs are projected into it (epoch 0's
  latents are 4.6× larger in norm), so **magnitudes are comparable only within a panel** — a
  cross-panel reading of the epoch-0 panel is a projection artifact, and the projection-free
  statistic for that question is fig4's.
* **fig4 — pairwise `z_v` distance across the ring.** The projection-free statistic: correlating the
  digest's own pairwise distances against |Δazimuth| gives **corr ≥ 0.986 at every epoch**,
  *including epoch 0* (0.9965). **The ring's ordering is not something training creates** — it is
  inherited — while training changes the *scale* of view separation relative to state
  (`zv_pair_ratio` 0.857 → 0.485) and the norms (31.9 → 6.9). At epoch 200 the distance grows
  near-linearly with separation (0.232 at 15°, 0.811 at 90°, 1.110 at 150°); the mean over all 55
  pairs rises 0.269 → 0.551 by epoch 50 and is flat to ±3% from epoch 100 on.
* **fig2 — `z_g` (11 views) vs `z_g_n1` (inference N=1, `az_0`).** The visual of the table above, and
  it sharpens one thing: the N-invariance is **not monotone**. True 512-D: cos-dist 0.17% (e0) →
  **1.57% peak (e22)** → 0.53% (e200); `rel` 6.4% → 17.7% peak (e20) → 10.3%. The fusion is *least*
  N-invariant at ~epoch 20, and the final state is slightly **less** N-invariant than
  initialization — consistent with "training changed what the fusion computes, just not its
  dependence on N", and worth knowing before anyone reads a single-epoch N=1 comparison off this
  instrument.
* **fig3 — the per-view series** (‖`z_v`‖, relative spread, pair/cross ratio over all 201 epochs),
  the visual of the tables above; no new number.

**Instrument defect found on the first render (fixed the same day).** fig2's on-panel numbers were
being computed from the *projected* points, whose origin inflates the denominator: it printed
`rel |dz|/|z|` 4.6% → 2.1% where the true 512-D series is 6.4% → 10.3% — understating by ~5× **and
flipping the trend's sign**. `relation_annotation()` now takes the digest's `fused/n1_vs_full`
values whenever `-d` is given, labels them `512-D`, falls back to the 2-D numbers only with an
explicit `2-D panel` label, and stops loudly on a digest whose fused series misses the epoch or
disagrees in length with `series.epochs`. No published number ever came from the broken path — the
tool was committed and first run the same day.

## Findings

What each follow-up experiment concluded — the result and the reading.

### L1 — view diversity alone

**Method.** M1's *exact* architecture, its single camera slot filled per sample from a randomly
drawn **training** view. No pose information anywhere — the model is never told which view it is
looking from.

**Result.** A task split, in opposite directions. On **lift** it holds 0.76–0.96 at every
viewpoint out to ±75°, where M1 is 0.08 at ±15° and 0.00 at ±30°. On **square and can** the same
intervention is ≤0.08 *everywhere*, including the poses it trained on, where M1 scores 0.82 and
0.98.

**Conclusion.** A conditioning-free baseline already achieves the project's stated goal on lift,
and destroys the other two tasks. Read L1's low numbers as a **noise floor, not as weak
success**: 0.00–0.08 over 50 episodes is 0–4 episodes. *Why* the tasks split was answered by the
collapse screen, below.

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
(`task.dataset.view_count_range`), K=7 slots throughout; each rung swept with 50 paired episodes.
Read on the trained mean, pre-registered 2026-09-24. Every published number here is an **N=1
inference** — `eval_novel_view.py` serves the one live camera into slot 0, and at N=1 the fusion
softmax is over a *single* unmasked key, so the learnable query is inert. So each rung varies
only the upper end of the range and differs from `m3off` on exactly one axis.

| rung | mean-N | trained | held-out | retention |
|---|---|---|---|---|
| `m3fixedn1` `[1,1]` | 1.0 | 0.036 | 0.047 | — |
| `m3v12` `[1,2]` | 1.5 | 0.028 | 0.043 | — |
| `m3v13` `[1,3]` | 2.0 | 0.080 | 0.073 | — |
| `m3v15` `[1,5]` | 3.0 | **0.456** | **0.373** | **82%** |
| `m3off` `[1,7]` | 4.0 | **0.764** | **0.550** | 72% |
| **`m3v27` `[2,7]`** | **4.5** | **0.868** | **0.547** | **63%** |

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
of magnitude.

**Conclusion.** The ingredient is having *enough views on average* — neither the presence of N>1
nor the availability of N=1. This also corrects M3's stated rationale: the N=1 **inference**
path works without ever training at N=1, so randomising N was not necessary for the capability
it was invoked to protect. Confound it cannot remove: mean-N 4.5 against 4.0 is ~12% more
encoder compute per sample.

### Collapse is a failure mode

`screen_collapse.py` compares various architectures against a random-initialization baseline based on relative dispersion: `[1,2]` collapses (performing 28,000× worse than the baseline) with the image pathway severed, and the L1 "square/can" configuration similarly collapses—explaining L1's task specialization, as a constant `z_g` renders the policy open-loop. However, collapse is merely **one** failure mode: `[1,3]` maintains a healthy encoder yet remains at the performance floor, demonstrating that a healthy screen does not guarantee a usable policy; meanwhile, `m3n1gate` suggests the issue may stem from view variation rather than the number of views.

## Open questions

- **What makes training collapse?** Known: not "too few views" (`m3n1gate` trains at N=1 and is
  healthy), established by epoch ~100, and task-dependent (L1 collapses on square and can, not
  lift). Unknown: the mechanism, the layer, and whether collapse is a cause or a symptom. Also
  unknown and worth stating because it rules out the cheapest design — **whether collapse
  precedes the behavioural failure** — since no existing run has a per-epoch checkpoint series.
  **Collected 2026-10-01, re-worded 2026-10-06:** the *clear* run produced the first per-epoch
  latent series in the project (see *The per-epoch latent series*, below). Its result, from the
  raw tensors: in a **healthy** run the geometry is decided by **~10–25** epochs, the scales by
  ~50–75, and the view-structure statistics not before **126–149**, while behaviour keeps
  improving to ~150 — so the ordering question is testable and the healthy reference is layered,
  not flat. **And the instrument has a floor:** a *deep* collapse is below fp16 resolution — the
  `[1,2]` cell read 1.8e-7 relative spread, four orders of magnitude under the 4.9e-4 storage
  floor — so a collapsing run's npz can only **bound** how early its collapse started; the fp32
  scalars in its `logs.json.txt` remain the deep instrument. It contains no collapse itself.
- **M5's premise.** M3 already does N=1 novel-view inference well, so distillation is only
  worth it if the fused latent demonstrably carries something the single-view path cannot,
  which no result so far shows. The clear-run analysis is the first evidence *against* the
  premise rather than merely its absence: at inference the fused latent at N=1 is within ~0.5%
  (direction) of the 11-view one, so whatever the extra views contribute is not visible in `z_g`
  at test time.

## Where the rest is

- **The long-form record** — full protocol, the investigation log section by section, the
  per-viewpoint tables, the pre-registration texts, the tables index, the parked investigations
  and the corrections register — lived in `PROGRESS_DETAIL.md`; it was folded into this file and
  removed on 2026-09-30 to keep the storyline clean. The complete text is at git tag
  **`docs-full-20260930`** (and in the commit history before it) — which is also where the
  provenance and the supersessions of any number above are recorded.
- **Do not quote two classes of superseded numbers.** (1) The pre-schema-2 `latent_stats` —
  `z_g_across_view_subsets` 0.171 (`m3on`) / 0.162 (`m3off`) and
  `z_v_across_views_within_draw` 0.466 / 0.224 — came from a code path whose compared subset
  sizes were governed by each checkpoint's own draw range; the *same* `m3off` encoder reads
  0.162, 0.202 and 0.252 in three different settings. The within-run contrast (2.1× more
  view-discriminative `z_v` under Plücker) is unaffected. (2) The n=8 conditioning anchors
  (`image_only` / `proprio_only`) are superseded *as method* — the readings scale with `n`
  (proprio moved 5.2× from n=8 to n=64) — though the severed-path finding survives at n=64.
- **`NOTES.md`** — runbooks (launch and sweep commands), timings, disk, and the recurring traps.
- **`PLAN.md`** — the milestone plan and what remains: the three runs, and the deferred work.
- **`PROPOSAL.md`** — the direction and the original predictions.

**Code map** — every milestone lands as new files; these are them.

| milestone | files added |
|---|---|
| M1 | `eval_novel_view.py`, `summarize_novel_view.py`, `config/task/{square,lift,can}_image_abs_single.yaml` |
| M2 | `generate_multiview_dataset.py`, `dataset/multiview_image_dataset.py`, `tests/test_multiview_dataset.py`, `config/task/{square,can,lift}_image_abs_multiview.yaml`, `config/task/single_view_image_abs_multiview.yaml` |
| L1 | `config/task/randview_image_abs_multiview.yaml` |
| M3 | `model/vision/plucker.py`, `model/vision/view_conditioned_obs_encoder.py`, `env_runner/cam_key_image_runner.py`, `config/task/m3_plucker_image_abs_{multiview,n1}.yaml`, `config/train_diffusion_unet_image_workspace_m3.yaml`, `tests/test_view_conditioned_obs_encoder.py`, `preview_viewpoints.py` |
| M4 | `policy/diffusion_unet_image_policy_aux.py`, `model/vision/per_view_aux_head.py`, `config/task/m4_aux_image_abs_multiview.yaml`, `config/train_diffusion_unet_image_workspace_m4.yaml`, `tests/test_aux_action_heads.py` |
| screens / probes | `screen_collapse.py`, `screen_conditioning.py`, `probe_relpose.py`, `tests/test_relpose_probe.py` (measurement tools, no training) |
| clear-run analysis | `analyze_latent_series.py`, `tests/test_analyze_latent_series.py`, `data/clear_analysis.sh` (the per-epoch latent-series digest; npz stay on `miroc-server`) |
| latent-distribution figures | `visualize_latent_distribution.py` (compute is numpy-only and runs where the npz are; plot is matplotlib), `tests/test_visualize_latent_distribution.py`, `data/clear_latent_viz.sh`, committed `data/analysis_clear/latent_viz_coords.json` + `data/analysis_clear/figures/` |

**Edited seams** — the only changes to files this fork did not itself add:
`multiview_image_dataset.py` (cam table, per-sample view draw, mask, camera-frame EE history;
and the `view_count_range` guard removed 2026-09-24 so the ladder can run `[1, 2]`),
`eval_novel_view.py` (`--m3-slots`, `--eef-hist-steps`, the elevation orbit),
`summarize_novel_view.py` (sort key for `el_*` names), and one `getattr`-guarded
`step_log['aux_loss']` line in the upstream
`train_diffusion_unet_image_workspace.py`. No other upstream package file is modified.
Note that `{square,lift}_image_single.yaml` (relative-action single-view variants) exist but are
unused — the baselines use the `_abs_single` variants.

Committed to git: every sweep's `eval_log.json`, the per-batch training logs, and the
probe/screen JSONs. Weights (4.6 GB per checkpoint) are **not** in git — rsync only, and they
live on `miroc-server`.
