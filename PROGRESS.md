# PROGRESS

Training a diffusion policy that keeps working from camera viewpoints it was never
trained on, and at inference time from a **single camera placed at a novel pose**.
`PROPOSAL.md` holds the direction and the original predictions; `PLAN.md` what remains;
`NOTES.md` the operational detail (runbooks, timings, disk, recurring traps).

**This file is the module and its measured behaviour**: the milestones, their headline results,
the findings that shaped the story, and the open questions. The long-form record this file was
distilled from — the full protocol, the verbatim investigation log, every per-viewpoint table
and the corrections register — is at git tag **`docs-full-20260930`** (see *Where the rest is*).

Last updated 2026-09-30.

## Summary

**What works.** M3's per-slot fusion reaches **0.55 (square) / 0.74 (can)** at held-out
viewpoints, where the single-view baseline M1 scores 0.00–0.02 and the conditioning-free L1
baseline sits at the floor. **What is inert.** PROPOSAL §2.1's Plücker + camera-frame-history
conditioning and §2.4's auxiliary heads change nothing measurable (M3, M4). **What is
load-bearing.** Multi-view *sampling*: the same encoder forced to one view per sample scores
0.04/0.05, and the ladder shows the ingredient is *enough views on average* — a knee between
mean-N 2.0 and 3.0, then a graded rise. **What is not explained.** Two distinct failure modes
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
chosen after the fact. `mean_score` hides effects `success_rate` shows (on lift it saturates at
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
montage. The **generated zarrs live on the training machine**, not in this checkout.

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
the one task L1 already wins, so it is the place M3 could regress (main-line run (a)).

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
built: final sweep tables against M1's degradation curves.

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

**Method.** `screen_collapse.py` — the encoder's relative output spread (`std` across 16
consecutive dataset states over the mean feature norm), matched `[7,7]` draw, always against a
**per-architecture random-init baseline measured on the cell itself**.

| cell | behaviour | relative spread | vs its random-init |
|---|---|---|---|
| random init, `MultiImageObsEncoder` | — | 1.27e-02 / 1.25e-02 | — |
| random init, `ViewConditionedObsEncoder` | — | 5.1e-03 | — |
| L1 **lift** | works | 2.49e-02 | 2.0× above |
| L1 **square** s42 / s43 | fails | 1.49e-04 / 1.99e-04 | 65–84× below |
| L1 **can** | fails | 3.05e-05 | 408× below |
| `m3off` `[1,7]` | works | 1.72e-02 | 3.4× above |
| `m3v15` `[1,5]` | **works** | 2.23e-02 | 4.4× above |
| `m3v13` `[1,3]` | **floor** | 1.36e-02 | 2.7× above |
| `m3v12` `[1,2]` | floor | **1.83e-07** | **28,000× below** |

**Result.** `[1,2]`'s encoder has **collapsed** — 28,000× below its own random-init baseline and
five orders of magnitude below every other cell — and its image path is severed behaviourally
(moving the *entire* image moves the action by 2.6e-05, ~256× below the other cells). The same
screen explains **L1's task split**: square and can collapsed, lift healthy, on three runs that
are matched by construction.

**Conclusion.** Collapse is **a** failure mode, not **the** failure mode. It explains `[1,2]`
and L1's task split, and the detail that never fit anywhere else — why those cells fail at the
**trained** pose: a constant `z_g` means the policy acts open-loop. But `[1,3]` is at the floor
with a *healthy* encoder (2.7× above its own baseline), so a healthy screen does not mean a
working policy. `m3n1gate` (trains at N=1, always on the *same* view, healthy at 5.93e-02)
points at view variation the encoder cannot reconcile rather than at view count — *suggestive,
not matched*.

### The second failure mode — `[1,3]` fails with a healthy representation

**Method.** Three instruments on frozen checkpoints, all at matched draws: `z_v` decodability
(ridge), `z_g` decodability at single-view draws, and `image→action sensitivity`.

**Result.** `[1,3]` **beats the working cell at every one of them** — `z_v` rotation 13.44° vs
12.53°, `z_g` rotation 18.68° vs 19.58°, image→action 0.0067 vs 0.0068 — and still scores
0.080. And with `m3v15` filled in, the anomaly becomes a **monotone ladder-wide trend**: as
mean-N falls 4.0 → 3.0 → 2.0, `z_g` gets *more* view-invariant (0.1469 → 0.1131 → 0.0636) and
`z_v` *less* view-aware (0.581 → 0.491 → 0.382) — toward what the proposal calls the goal —
while behaviour falls 0.764 → 0.456 → 0.080.

**Conclusion.** This is the project's open question: what the policy *learned to do* with
correct information. Every instrument built here asks whether information is *present*, and the
difference between those two cells is evidently not presence. The one proposed mechanism
(balance) is refuted: re-measured at n=64 × 3 seeds behind a **pre-registered gate** (which
failed, min ratio 0.567, so the falsifying intervention was **never launched**), the
draw-independent `proprio_only` arm is **1.008×** between `m3v13` and `m3off` and **flat across
all four rungs** (0.5% spread). Its by-product is methodological and sharpens the question:
`image→action sensitivity` is **not a scalar** — at matched inputs the cell rank order *flips*
across probe ensembles (0.512 / 0.272 / 1.362 across seeds) even though the tool is now
bit-reproducible.

### M1's baseline was not collapsed

**Method.** M1's weights were deleted in a disk reclaim, so its square baseline was re-trained —
fidelity-gated first against its committed rollout curve (max |Δ| 0.06, inside the ~0.14 noise)
— and then collapse-screened.

**Result.** **0.1221**: **17.5× above** its own random-init baseline (6.97e-03), and higher in
absolute terms than even the working L1 lift.

**Conclusion.** M1's failure at ±15° is **view-tiedness with a healthy, richly-varying encoder**,
which is a *different* failure mode from collapse. The collapse story is not one story from M1
onward.

## Open questions

- **The second failure mode — the most open question here.** `[1,3]` sits at the floor (0.080)
  with a representation that beats the working cell at four separately-measured stages: encoder
  variance, `z_v` decodability, `z_g` decodability, and image→action sensitivity. The balance
  mechanism is refuted and the latent/behaviour mismatch is monotone across the whole ladder, so
  this is a property of the ladder rather than of `[1,3]`; and the natural candidate statistic,
  `image→action sensitivity`, is not a scalar. The missing instrument needs a design whose
  verdict does not depend on which scenes are probed.
- **What makes training collapse?** Known: not "too few views" (`m3n1gate` trains at N=1 and is
  healthy), established by epoch ~100, and task-dependent (L1 collapses on square and can, not
  lift). Unknown: the mechanism, the layer, and whether collapse is a cause or a symptom. Also
  unknown and worth stating because it rules out the cheapest design — **whether collapse
  precedes the behavioural failure** — since no existing run has a per-epoch checkpoint series.
- **Lift is untested for M3.** It is the one task where L1 already wins (0.76–0.96), so an
  M3-on run there is a genuine "did we break it" question rather than a result (main-line run
  (a)).
- **Second seed.** Every *behavioural* number in this document is n=1 at a resolution worse
  than ±0.05. A second seed on one M3 or M4 cell would materially strengthen the nulls, which
  are "indistinguishable at this resolution", not "proven identical".
- **The elevation asymmetry.** Square collapses at `el_m15` for every M3 cell while can
  holds 0.22; `[1,5]` holds `el_p15` (0.26) and not `el_m15` (0.00). Unexplained.
- **M5's premise.** M3 already does N=1 novel-view inference well, so distillation is only
  worth it if the fused latent demonstrably carries something the single-view path cannot,
  which no result so far shows.

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
live on the training machine.
