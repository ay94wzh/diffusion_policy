# PROGRESS

Training a diffusion policy that keeps working from camera viewpoints it was never
trained on, and at inference time from a **single camera placed at a novel pose**.
`PROPOSAL.md` holds the direction and the original predictions; `PLAN.md` what remains;
`NOTES.md` the operational detail (runbooks, timings, disk, recurring traps).

**This file is the module and its measured behaviour**: the milestones, their headline results,
the findings that shaped the story, and the open questions. The long-form record this file was
distilled from — the full protocol, the verbatim investigation log, every per-viewpoint table
and the corrections register — is at git tag **`docs-full-20260930`** (see *Where the rest is*).

Last updated 2026-10-29.

## Summary

**What works.** M3's per-slot fusion reaches **0.55 (square) / 0.74 (can)** at held-out
viewpoints, where the single-view baseline M1 scores 0.00–0.02 and the conditioning-free L1
baseline sits at the floor. **What is inert.** Plücker + camera-frame-history conditioning and auxiliary heads change nothing measurable (M3, M4). **What is load-bearing.** Multi-view *sampling*: the same encoder forced to one view per sample scores 0.04/0.05, and the ladder shows the ingredient is *enough views on average* — a knee between
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
**What the base-frame arm adds (2026-10-08).**
The clear run's recipe re-run with the per-view aux target in the **base** frame (view-invariant)
instead of each slot's camera frame: the aux-supervision idea's second behavioural null,
pre-registered — all four registered means inside the 0.15 band — while the latent series
re-organizes as designed (`zv_pair_ratio` 0.485 → **0.239**, settling e57 vs e126, and the
latents become *less* pose-decodable), so re-shaping `z_v`'s view/state balance is not what
bounds behaviour. **What the `z_g`-supervision arm adds (2026-10-10).** The first supervision to
reshape the *fused* latent: aiming it at the live views' pooled camera-pose set (all rows scored,
dead ones against zero) makes `z_g` genuinely view-set-aware — digest N-dependence 0.199 vs the
parents' 0.089/0.103, probe `zg_across_view_subsets` 0.1434 vs 0.0663/0.0780, N=1 camera-pose
decodability **1.2° / 1.9 cm** vs 10.8–17.8° / 16.6–23.2 cm (outcome (i) of its pre-registration)
— and the behaviour on the registered means is a **third null** (0.735 in-dist / 0.714 off-grid,
inside the 0.15 band), with its one beyond-band move a *cost*: 0.140 vs 0.310 for both parents
past the ±75° ring. **What the joint arm adds (2026-10-14).** Both supervision lines at once
— the base-frame action head on `z_v` *and* the pose-set head on `z_g` — push the
representation further than either alone: `z_g`'s N-dependence **2.2–2.3×** `zgpose`'s, the
N=1 fused latent decodes the camera pose at 2.2° / 3.9 cm, while the base-frame term keeps the
encoding healthy (`zv_across_states` 0.916 ≈ the parents', collapse screen 12.3× its random
init). Behaviour is a **fourth in-distribution null** (0.751) and the **weakest held-out cell
of the four**: off-grid ±7.5…±67.5 **0.606** — 0.168 below `m4base`, the first *banded*
registered mean outside the 0.15 band — and past the ring **0.100**, no recovery of `zgpose`'s
cost. **What the probabilistic-fusion round adds (2026-10-16).** `PLAN_G.md` stage 1 — per-view
Gaussians (`mu_v`, `logvar_v`) fused by PoE, against a deterministic MHA-over-means control and
a plain-mean arm — is **behaviourally flat**: in-distribution all three cells sit inside the band
(0.793–0.829), the held-out contrast C-vs-B is **−0.074, inside the band and unresolved**, PoE is
the weakest of the three off-grid, and the best held-out cell is the control that never uses
uncertainty; the learned variances are real (view spread 0.447, clamp never binds) and buy
nothing. The completed battery (2026-10-16) reads all four registered representation readouts
**in-envelope** — by the round's own convention, no mechanism claim is available — and the
variance as **real spread that ranks no errors** (`rho(lvg, state residual)` −0.242): `PLAN_G.md`
stage 1 closes the probability axis with a null. **What the z-supervision round adds (2026-10-28).**
`PLAN_Z.md`'s two supervisions, one at a time and together, against `m4base`. **`z_v`
shared-semantics alignment is inert and degenerate**: cell `a` reproduces the baseline to three
decimals (in-dist 0.7964 vs 0.796, off-grid 0.764 vs 0.774) because the projection head collapses
to a constant — `cos_across` **1.0000** against a random head's 0.48, `L_align` 3.5e-09 — so the
loss is satisfied trivially and the encoder is never pressured (screen 12.2×, all four readouts
in-range). **`z_g` cross-view consistency destroys the encoder**: at the calibrated
`W_CONS = 190.46` (63× the driver's own `W > 3` warning) cells `b` and `c` collapse to **1.6× /
1.7×** their random-init baselines — below any arm in the project — `z_g` loses view-set
dependence (`zg_across_view_subsets` 0.0027 / 0.0054 against A's 0.0663) and camera-pose
decodability (47.6° / 27.5° against an envelope top of 17.8°), and behaviour falls to
**0.155 / 0.129** in-distribution and **0.150 / 0.150** off-grid. **`c` = `b` exactly** (off-grid
0.1500 both; `a` contributes nothing to either arm), so PLAN_Z §3's `C = A + B` decomposition
comes out clean with `A ≈ 0`. The round's new instrument — the semantic readout of the trained
head — is what turns `a`'s null from "alignment doesn't help" into "**this loss cannot help**".
The pre-registered R2 instrument (`diff_loss` rising above the family band) read **clear** at e200
for all three cells while the encoder and behaviour were collapsing, so that instrument does not
detect this failure mode; the collapse screen and the readouts do. **What is not explained.** Two distinct failure modes
sit at the floor — an encoder *collapse* (`[1,2]`, L1 on square/can) and a second one with no
candidate mechanism left: `[1,3]` fails while its representation beats the working cell on every
instrument built so far. Honest summary against the proposal: **§2.2's fusion works; §2.1's
geometric conditioning and §2.4's aux heads are both inert.** **What the encoder ladder adds
(2026-10-29).** `ENCODER_EXPERIMENT_PLAN.md`'s first two stages, B and S, were trained and
swept. **B**, the corrected EE-history protocol, is a clean one-variable pair — B0 `sample` vs
B1 `episode` — and is a **null on every registered mean**: in-distribution **0.751 vs 0.813**
(−0.062), interpolation 0.712 vs 0.784
(−0.072), past-ring 0.320 vs 0.365 (−0.045), elevation 0.367 vs 0.347 (+0.020, descriptive).
The history branch is behaviourally inert with the clean pair the M3/M4 null lacked. **S**, the
pure-RGB spatial-readout replacement, is **negative on the trained and interpolation rings**:
in-distribution **0.653 vs 0.813** (−0.160) and interpolation **0.618 vs 0.784** (−0.166), both
just outside the 0.15 band; past-ring is a null (0.355 vs 0.365, both at the ~0.36 floor), and
elevation is lower (0.293 vs 0.347, descriptive). B1 matches the committed `m4base` reference
(0.813/0.784 vs 0.796/0.774), so it is a valid internal baseline. S is one seed and its margins
are 0.010–0.016 over the threshold; **R is implemented and CPU-gated (2026-10-29) but untrained;
RH/RHG are not implemented**.

## Status

| rung | what changed | verdict |
|---|---|---|
| **M1** | single-view DP baseline (agentview at az_0), three tasks | collapses to ≈0 at ±15° azimuth on all three |
| **M2** | multi-view data: a 13-pose azimuth ring re-rendered from the demos' stored simulator states | validated by an N=1 gate that reproduces M1's whole curve; ±75°/±90° views are low value |
| **L1** | view diversity only: M1's exact model, its single camera slot filled per sample from a randomly drawn training view | task-split **in both directions** — solves lift out to ±75°, destroys square/can |
| **M3** | view-conditioned encoder: per-view latents, Plücker + camera-frame-history conditioning, K=7 slots, per-sample N∈[1,7], MHA fusion | **solves square and can at held-out views**; its conditioning contributes nothing |
| **M4** | per-view camera-frame auxiliary action heads | mechanism is real (`aux_loss` falls 52×), behaviour is null (+0.02/+0.04) |
| **M5** | single-camera N=1 inference at a novel pose | **capability measured** — every M3 number is already this; the optional distillation stage is uncoded, its premise **re-opened** 2026-10-10 by `zgpose` |
| **N>1** | M3's model with one CLI line changed so every sample sees a single view | **the load-bearing ingredient** — reproduces L1, not M3 |
| **ladder** | the N-diversity ladder closed: mean-N 1.0 / 1.5 / 2.0 / 3.0 / 4.0 | **knee between 2.0 and 3.0** (0.080 → 0.456), then graded to 0.764; `[1,5]` is *more* view-general than `[1,7]` |
| **`[2,7]`** | min-N 2, mean-N 4.5 — never trains at N=1 | **indistinguishable from `[1,7]`** — enough views *on average* is the ingredient |
| **collapse** | encoder output spread across every surviving checkpoint, matched draw, random-init controls | **a failure mode, not *the* one** — explains `[1,2]` and L1's task split; `[1,3]` fails while healthy |
| **M1 re-screen** | re-trained the deleted baseline, fidelity-gated, then collapse-screened | **not collapsed** — view-tiedness and collapse are distinct failure modes |
| **balance test** | `[1,3]`'s proprioception contrast re-measured at n=64 × 3 seeds, pre-registered | **refuted; intervention retired unlaunched** |
| **(a) lift + `m3on`** | recovered 2026-10-01 — trained 09-26, then skipped by a guard bug | **does not break lift** (held-out 0.870 vs L1 0.873); elevation is where it gives ground |
| **(b) `[1,5]` seed 43** | recovered 2026-10-01; first reproduction of any ladder rung | **Δ 0.052 / 0.083, inside the band** — the knee stands at two seeds |
| **(c) ±60° pool** | recovered 2026-10-01; pool `[2,4,6,8,10]` at matched mean-N 3.0 | **pool curation is worth +0.272 trained / +0.160 held-out; reaches `[1,7]` at 2/3 its mean-N** (size/content confounded) |
| **base-frame supervision** (`m4base`) | `m4base`, 2026-10-08 — the clear run's recipe with the per-view aux target's frame swapped camera→base (view-invariant) | **behavioural null (pre-registered), latent re-organized** — `zv_pair_ratio` 0.485→0.239, settles e57 vs e126; §2.4's frame axis closed |
| **`z_g` supervision** (`zgpose`) | 2026-10-10 — the fused latent supervised on the live views' pooled camera-pose set (all rows scored; dead zero-filled). Rev 1 killed pre-training (see the section) | **outcome (i) at the representation, behaviour inside band** — `z_g` became view-set-aware (both instruments ~2× the parents; pose decodable at 1.2°); registered means a third null, past the ring **−0.170**; M5's premise re-opened |
| **joint supervision** (`zvzg`) | 2026-10-14 — both aux lines at once: `m4base`'s base-frame action head on `z_v` (w 1.0) + `zgpose`'s pose-set on `z_g` (w 1.33) | **the strongest view-aware `z_g` of the four cells, with the encoder healthy** (N-dependence 2.2–2.3× `zgpose`; screen 12.3× its random init) — in-dist a fourth null (0.751); **off-grid 0.606: −0.168 vs `m4base`, the first banded mean outside its band** (−0.108 vs `zgpose`, inside); past-ring 0.100, no recovery of `zgpose`'s cost |
| **probabilistic fusion** (`prob_b/c/d`) | 2026-10-16 — `PLAN_G.md` stage 1: per-view Gaussians, three fusions — MHA-over-means (B, the deterministic control), PoE (C), plain mean (D); cell A = the committed `m4base`, not retrained | **behaviourally flat** — in-dist all inside band (0.829 / 0.804 / 0.793 vs A's 0.796); off-grid C−B **−0.074** and D−C **+0.060**, both unresolved, B the best held-out (0.738); past the ring **B 0.370** (above A) with C/D 0.210 / 0.195 (partial, no third floor); the variance learns (C: view spread 0.447, clamp 0.011/0.071) and buys nothing; **battery completed 2026-10-16**: all four registered representation readouts **in-envelope** for B/C/D, and the variance **spread but no signal** (R2 weak; R3/R4 null-to-against; double-count 2.23) |
| **z supervision** (`zalign`/`zcons`/`zboth`) | 2026-10-28 — `PLAN_Z.md`: L2-normalised `z_v` shared-semantics alignment (a), cross-view ε-consistency on `z_g` at W=190.46 (b), both (c); same two weights in all three so `c` decomposes as `a + b`; baseline = the committed `m4base`, not retrained | **`a` null, `b` and `c` collapse** — `a` reproduces A to three decimals (0.7964/0.764) because its head collapses to a constant (`cos_across` 1.0000, `L_align` 3.5e-09) and the encoder is untouched (screen 12.2×, all four readouts in-range); `b` 0.155/0.150 and `c` 0.129/0.150 with encoders at **1.6×/1.7×** random init and `z_g` uninformative (3 of 4 readouts MOVED); **`c` = `b` to four decimals off-grid**; the registered `diff_loss` R2 instrument **did not fire** |
| **encoder B** (`enc_b0`/`enc_b1`) | 2026-10-29 — corrected `eef_hist_window=episode` (B1) against the legacy `sample` (B0), one variable; K=11 pool [1..11], N ∈ [1,11], 201 epochs, seed 42 | **history null** — every registered mean inside the 0.15 band: in-dist −0.062, interp −0.072, past −0.045, elev +0.020 (descriptive); B1 is the encoder ladder's internal baseline |
| **encoder S** (`enc_s`) | 2026-10-29 — pure-RGB layer3 spatial tokens + one shared 512-d query + 8-head intra-image attention, B's masked cross-view fusion; structural, not RNG-locked to B | **negative on the ring** — in-dist 0.653 vs B1 0.813 (−0.160), interp 0.618 vs 0.784 (−0.166), both just outside the 0.15 band; past-ring null (0.355 vs 0.365); elevation 0.293 vs 0.347; seed 42 only |
| **encoder R** (`enc_r`) | 2026-10-29 — S + feature-level ray encoding: Plücker `[d, m/L]` at each layer3 cell's receptive-field centre (same crop window), MLP → 512-d added to the attention KEY; 5,284,928 encoder params | **implemented / CPU-gated / untrained** — code, configs and run/eval scripts landed; the CPU suite pins the ray convention, crop alignment, masking and the activation order |

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

**Viewpoints.** Azimuth presets `azimuth_sweep5` (0, ±15°, ±30°), `azimuth_interp`
(0, ±15°, ±30°, ±45°, ±60°, ±75°), and `azimuth_offgrid` (7.5° offsets — ±7.5° … ±67.5° —
plus ±82.5°/±90°). Elevation is `elevation_az0` — an **orbit** about the
look-at point at 0/±15°, the off-manifold test. `el_0` is the same camera pose as `az_0`,
which makes it the internal consistency check.

**Multi-view data (M2).** A 13-pose azimuth ring every 15° out to ±90°. The **even indices**
(az −90/−60/−30/0/+30/+60/+90, every 30°) form the *training pool*; the **odd indices**
(±15/±45/±75) are **never sampled during training**, so every "held out" column below is held
out by construction. **The encoder ladder (2026-10-29) uses a different pool** — indices 1..11
(az −75…+75 every 15°), all trained — so its in-distribution cell is the trained ring and its
`azimuth_offgrid` grid is the held-out azimuth set; see *The encoder ladder* below.

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
square was run for M4. **The frame axis is closed** (2026-10-08): a base-frame variant of the
same heads (`m4base`, *The base-frame arm*, below) is also behaviourally null while the latent
re-organizes — so the null is not about which frame the head predicts in.

## M5 — single-novel-view inference

**Capability measured, not pending.** Every M3 number is already an N=1 inference at a novel
pose: `eval_novel_view.py --m3-slots K` reduces the env-side `shape_meta` to the single
renderable key and publishes the perturbed camera pose (read back **from the simulator**) into
a single slot; square infers at N=1 with 0.55 success on held-out azimuths. What remains is the
**optional distillation stage** — a single-view student encoder regressing the frozen teacher's
multi-view fused latent `z_g`. Its premise is **re-opened, not established** (the `zgpose` and
`zvzg` arms; see *Open questions*): the capability lives in the
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

### The base-frame arm (`m4base`, 2026-10-08)

**What it is.** The clear run's recipe with ONE variable changed: each slot's `z_v` is supervised
by the raw stored absolute action chunk — the **base frame**, identical for every live slot, the
explicit form of the cross-view consistency pressure multi-view sampling supplies implicitly —
instead of the clear run's camera-frame chunk (view-specific, different per slot). The two task
configs differ by exactly `aux_action_frame: base`; data, slots, pool ([1..11]), N-range [1,11],
seed, aux weight (1.0) and the per-epoch latent probe are identical. **Pre-registered** in the
driver header before launch: the M4-informed expectation was *within-band* on the registered
means — another null, direction if any **not** predicted; three latent-series questions (settle
epochs of the view-structure statistics, the final view-vs-state balance, `z_g` N-invariance);
and the mechanism-live gate (aux loss falls below its batch-0 value, ratio recorded). The run
completed clean at 201 epochs.

**Result — the behavioural null held.** `success_rate`, 50 paired episodes, means vs `m4clear`:

| registered mean | m4base | m4clear | Δ |
|---|---|---|---|
| in-dist ring (trained, 11 vp) | 0.796 | 0.747 | +0.049 |
| off-grid ±7.5…±67.5 (10 vp) | 0.774 | 0.716 | +0.058 |
| past the ring, ±82.5/±90 (4 vp) | 0.310 | 0.310 | 0.000 |
| elevation el_0 / el_m15 / el_p15 | 0.840 / 0 / 0 | 0.700 / 0 / 0 | +0.140 / 0 / 0 |

All four registered quantities sit inside the band (the elevation entry is three single cells,
never read individually; el_0's +0.140 is inside the documented single-viewpoint spread,
*Noise and resolution*). The point estimates lean slightly positive everywhere and are identical
at the floor; the in-training rollout (N=1 at az_0, `test/mean_score`) reads 0.900 vs 0.760 at
e200, descriptively in the same direction.

**The instrument saw exactly what the design intended.** Against the clear run's digest (same
probe, same 128 states, same draws):

| final-epoch statistic | m4base | m4clear |
|---|---|---|
| `zv_pair_ratio` | **0.239** | 0.485 |
| … its settle epoch | **57** | 126 |
| `zv_norm_mean` | 8.39 | 6.91 |
| `zv_rel_spread_mean` (settle 148 vs 72) | 0.235 | 0.224 |
| `z_g` `rel(n1, full)` | 0.089 | 0.103 |
| `cka(n1, full)` | 0.9985 | 0.9976 |

The base-frame target **halved the view-specific component of `z_v`**, settling the view-vs-state
balance more than twice as fast (`zv_pair_ratio` e57 vs e126; `zv_pr` unchanged, 148 vs 149; the
within-view spread is the one statistic that stops *later*, 148 vs 72), while the fused latent
became slightly *more* N-invariant. The paired probe grid (range `[1,11]`; draw fingerprints
identical across the two cells, `mean_active` 6.046 both) adds the direction the digest cannot:
the representation became **less pose-decodable** — `z_v` readouts (MLP) absolute pose
5.25 cm / 2.28° vs 3.72 cm / 1.61°, relative pose 5.46 cm / 3.26° vs 3.35 cm / 1.38°, and the N=1
`z_g` camera-pose readout (mean rotation 17.8° vs 10.8° ridge, 16.9° vs 13.6° MLP, n=110/28) —
i.e. the view-invariant target removed the view-specific information it was designed to remove.
Mechanism live: batch-0 `aux/diff` **0.333** (m4clear 0.315); `aux_loss` decays **74×**
(0.361 → 0.0049). Both cells' collapse screens are healthy against their own random-init baselines
(m4base 2.53e-02 vs 4.41e-03; m4clear 3.46e-02 vs 2.84e-03). The per-view statistics rest on
`z_v` being N-agnostic; that check reads 2.0e-4 / 2.7e-4 here — negligible against the pair
distances (~0.3–0.6), and its absolute scale is cell-dependent: NOTES *Latent probe* carries the
corrected reading.

**The reading.** The supervision did its designed job at the representation level and the
behaviour did not move: per-view aux supervision is now behaviourally null in **both** frames
while measurably reshaping `z_v` in one — §2.4's frame axis is closed, and the behavioural
bottleneck is *not* `z_v`'s view/state balance, a quantity the instrument can now move at will.
Caveats: single seed, 50 episodes, so the ≈+0.05 lean is not distinguishable from run noise at
this budget; nothing is held out in this recipe (all 11 ring views train), so it is an
instrument, not a ladder cell. **One reading trap this run exposed:** the per-epoch probe fires
*after* the epoch's training (`train_diffusion_unet_image_workspace.py:225-231`), so
`epoch_0000.npz` is one *trained* epoch, not initialization — the cross-cell e0 contrast
(clear 0.857 vs m4base 0.484) is an early effect of the intervention, and e0 rows are not a
shared baseline (NOTES *Per-epoch latent snapshots*).

### The `z_g` pose-set arm (`zgpose`, 2026-10-10)

**What it is.** `PLAN.md` candidate 2 in its *geometry-prediction* form: the clear run's recipe
(K=11 over ring 1..11, N∈[1,11], seed 42, per-epoch latent probe, W = 0.33·diff0/aux0 calibrated
pre-launch to **1.33**) with ONE design-level change — the per-view aux head is retired and the
**fused latent `z_g`** predicts the live views' **pooled camera-pose set**: a fixed (V=11, 9)
target, row i == pool position i, `[pos(3) | rot6d(6)]`, **dead rows zero-filled and ALL rows
scored**. The driver measures the constant-predictor floor on 64 calibration samples
(**0.108586**) and refuses to launch unless some pool row is partially live over them — the gate
that makes a static target impossible to mistake for a test.

**Revision 2, and why it exists.** Attempt 1 scored the **live rows only**. The pose table is
static (the cameras never move), so the only sample-dependent content of the target is *which
views are live* — and a masked mean excludes exactly that: an input-independent head emitting the
table on every row matched every scored row exactly. The calibration's floor measured
**0.000000** (`ratio_aux0_floor` 4.5e11) and the registered tree degenerated — the "loss below
the floor" outcome was unreachable while the falsifier was vacuous. Attempt 1 was killed 2.5
minutes in, before any completed rollout; its `aux_loss` had already collapsed **0.4525 → 2.1e-06
within epoch 0's 439 steps** — the cheat, visible in the log. Revision 2 scores all V rows
against the zero-filled target, so the best input-independent predictor is the per-row constant
and beating it *requires* reading liveness out of `z_g`. Commits: `7da7ba5` (rev 1), `7b97ee5`
(rev 2, with `data/zgpose_run.sh`'s header carrying the revised pre-registration).

**Results — behavioural.** `success_rate`, 50 paired episodes, registered means vs the two
parents:

| registered mean | **zgpose** | m4base | m4clear |
|---|---|---|---|
| in-dist ring (11 vp, trained) | **0.735** | 0.796 | 0.747 |
| off-grid ±7.5…±67.5 (10 vp, held out) | **0.714** | 0.774 | 0.716 |
| past the ring ±82.5/±90 (4 vp) | **0.140** | 0.310 | 0.310 |
| elevation el_0 / el_m15 / el_p15 | 0.820 / 0 / 0 | 0.840 / 0 / 0 | 0.700 / 0 / 0 |

**Behaviour is inside the 0.15 band on both registered means** — the supervision line's third
behavioural null — **except past the ring**, where it is **0.170 below both parents** (az_m82.5
0.180 vs 0.440/0.400; az_p82.5 0.300 vs 0.660/0.560): the first registered behavioural number
this project has moved beyond band, and a *cost*. `el_0` (same pose as `az_0`) reads 0.820, the
harness's usual consistency.

**Results — the instruments (this run's distinguishing diagnostic), all at e200:**

| readout | **zgpose** | m4base | m4clear |
|---|---|---|---|
| digest `rel(zg_n1, zg_full)` — tail(last 25) | **0.199** (peak 0.914 @e7) | 0.0889 | 0.1031 |
| digest `cka(n1, full)` | 0.9975 | 0.9985 | 0.9976 |
| probe `zg_across_view_subsets(1,11)` | **0.1434** | 0.0663 | 0.0780 |
| probe `zv_pair_ratio` (1,11) | **2.255** | 0.267 | 0.544 |
| probe `zv_across_states` (1,11) | 0.412 | 1.115 | 1.086 |
| probe N=1 `zg` pose readout (ridge, 1,11) | **1.20° / 1.90 cm** | 17.8° / 23.2 cm | 10.8° / 16.6 cm |
| digest `cka(zg_full, zv_az0)` | **0.981** | 0.678 | 0.690 |
| digest `zv_pair_ratio`, final (settle) | **1.308** (91) | 0.239 (57) | 0.485 (126) |
| digest `zv_rel_spread_mean` / `zg_norm_mean` final | 0.093 / **65.8** | 0.235 / 5.4 | 0.224 / 3.5 |
| collapse screen, `[11,11]` (× its random-init) | 3.30e-03 (2.03×) | 2.53e-02 (5.7×) | 3.46e-02 (12.2×) |

The aux term is beaten by four orders of magnitude (`aux_e200` 8e-06 = 7e-05 of the floor).
**Both instruments moved, ~2× beyond the parents** (digest tail 0.199 > 0.13; subsets 0.1434 >
0.09) → **outcome (i), the "supervision bites the fusion" branch.** The probe's draw fingerprints
match the committed m4base probes in all four ranges (`mean_active` 6.046), so the cross-cell
probe comparison is on identical draws; the digest ran `--strict` with every gate passing
(scalar recheck 4.1e-04; no holes) and no warnings. The in-training e200 rollout (N=1 at az_0)
reads 0.820 (m4base 0.900, m4clear 0.760) and the e200 diffusion term 0.00423 sits at the
parents' level (0.00588 / 0.00692), so the policy is intact.

**The reading.** Everything the arm was designed to change, it changed: `z_g` is decodably
view-set-dependent; the N=1 fused latent carries the camera pose almost exactly (1.2° / 1.9 cm —
~15× the parents' decodability); the fusion output moved *towards* the per-view latents
(`cka(zg_full, zv_az0)` 0.98 vs 0.69 — the fusion now largely preserves view-token information);
and the pressure reached back into `z_v` (pair ratio 1.31–2.26 vs 0.24–0.54, within-view spread
halved). **And neither registered behavioural mean moved.** The project's central null therefore
extends to the fused latent itself: making `z_g` genuinely view-aware is *not* the behavioural
bottleneck — the third supervision target (per-view actions, base-frame actions, fused-latent
geometry) reshaped without buying success. **M5's distillation premise is re-opened** — the
pre-registration says to state this explicitly when (i) lands: a student regressing `z_g` would
now have something to learn that the N=1 path does not already determine, unlike the parents
(N-invariant to 0.5% / 0.998 CKA). The cost sits at extrapolation: past the ring, a third below
both parents.

**Caveats.** One seed, 50 episodes per viewpoint; an *instrument* cell (all 11 ring views train,
nothing held out), so its numbers are not ladder-comparable. The collapse screen reads 3.30e-03,
below the ~1e-02 healthy anchor though 2.0× its own random-init baseline and five orders above
the degenerate band; the digest explains the mechanism — `z_g`'s norm grew 4–19× (65.8) while its
*relative* state spread halved, i.e. the fused latent became view-dominated, and the screen's
absolute anchors were calibrated on state-dominated cells. This is a shift in the view/state
balance, not a collapse (the probe's `zv_across_states` 0.412 vs `m3v12`'s 2.7e-05). The
batch-0 `aux/diff` realized **0.20** rather than the nominal 0.33 — batch 0 drew a liveness set
2.4σ below the population mean (per-item sd 0.127 over 256 sampled items), a draw, not a bug.

### The joint arm (`zvzg`, 2026-10-14)

**What it is.** Both executed supervision arms on at once, on the clear run's recipe:
`m4base`'s per-view **base-frame** action head on `z_v` at its registered `w_v` = 1.0, **plus**
`zgpose`'s pooled camera-pose set on the fused `z_g` at the same pre-launch calibration
(`w_g` = 0.33·`diff0`/`aux0_pose`, asserting the committed `aux0_pose` 0.268672 / floor
0.108586 → **1.33**; the log confirms the realized weighting — batch-0 `aux_loss` 0.647792 =
0.372525 + 1.3300 × 0.206967). Data, K=11 over pool [1..11], N ∈ [1,11], seed 42, 201 epochs,
the per-epoch latent probe, all other knobs identical to both parents; the two targets come
from the ONE view draw. The question it targets: **do the two supervisions compose, or does one
pressure erase the other?** **Pre-registered in the driver header before launch**, against the
committed parents: another behavioural null if the supervisions are additive (in-dist and
off-grid means within 0.15 of BOTH `m4base` 0.796/0.774 and `zgpose` 0.735/0.714); past the
ring ±82.5/±90 **registered separately, no band** — "does the action term recover it toward
0.310, or does the pose term's cost persist?"; elevation descriptive only; which pressure wins
at the representation — the **tension readout** (probe `zv_pair_ratio`(1,11): < 0.5 base-frame
pressure wins / 0.5–1.5 intermediate / > 1.5 pose pressure wins, no direction predicted) plus
the two moved thresholds (digest tail > 0.13, subsets > 0.09); both aux terms' mechanism gates;
and the documented-ambiguity clause — if behaviour moves beyond band while *both* instruments
sit at the parents' values, report the movement and **do not attribute a mechanism**. Not
claimed (the header): draw-level RNG lockstep with `m4base` — constructing the pose head
consumes RNG, so both one-variable comparisons are design-level.

**Results — behavioural.** `success_rate`, 50 paired episodes, the registered means:

| registered mean | **zvzg** | m4base | m4clear | zgpose |
|---|---|---|---|---|
| in-dist ring (11 vp, trained) | **0.751** | 0.796 | 0.747 | 0.735 |
| off-grid ±7.5…±67.5 (10 vp, held out) | **0.606** | 0.774 | 0.716 | 0.714 |
| past the ring ±82.5/±90 (4 vp) | **0.100** | 0.310 | 0.310 | 0.140 |
| elevation el_0 / el_m15 / el_p15 | 0.820 / 0 / 0 | 0.840 / 0 / 0 | 0.700 / 0 / 0 | 0.820 / 0 / 0 |

**The in-distribution mean is inside both bands** (Δ −0.045 vs `m4base`, +0.016 vs `zgpose`),
and `el_0` equals its own `az_0` to the third decimal (0.820) — the harness's usual same-pose
consistency. **The off-grid mean is outside `m4base`'s band — 0.168 below, 0.018 past the 0.15
threshold** — while inside `zgpose`'s band (−0.108); it is not one cell: zvzg sits below both parents
at 7 of the 10 off-grid viewpoints. **Past the ring the cost persists rather than recovering:**
0.100, below `zgpose`'s 0.140 and far below the `m4base`/`m4clear` 0.310 — `az_m82.5` 0.260 vs
0.440/0.400/0.180, `az_p82.5` 0.140 vs 0.660/0.560/0.300, both ±90 at 0.000 — so the registered
question is answered on the pose term's side: the action term did not buy back the extrapolation
the pose supervision cost.

**Results — the instruments.** All at e200; the probe's draw fingerprints are identical across
the four cells (`mean_active` 6.046) and the digest ran `--strict` (201 epochs, no holes, scalar
recheck 1.16e-04, no warnings). This cell's `aux_loss` log line is the **weighted total** — the
per-term `aux_action_loss`/`aux_pose_loss` keys are the comparable ones:

| readout | **zvzg** | m4base | m4clear | zgpose |
|---|---|---|---|---|
| digest `rel(zg_n1, zg_full)` (tail, last 25) | **0.433** | 0.089 | 0.103 | 0.199 |
| digest `cka(n1, full)` | 0.9984 | 0.9985 | 0.9976 | 0.9975 |
| probe `zg_across_view_subsets`(1,11) | **0.3268** | 0.0663 | 0.0780 | 0.1434 |
| probe `zv_pair_ratio`(1,11) — the tension readout | **0.857** | 0.267 | 0.544 | 2.255 |
| probe `zv_across_states`(1,11) | 0.916 | 1.115 | 1.086 | 0.412 |
| probe N=1 `zg` pose readout (ridge) | 2.17° / 3.93 cm | 17.8° / 23.2 cm | 10.8° / 16.6 cm | 1.20° / 1.90 cm |
| digest `zv_pair_ratio`, final (settle) | 0.751 (107) | 0.239 (57) | 0.485 (126) | 1.308 (91) |
| digest `zv_norm_mean` / `zg_norm_mean` | 17.1 / 37.9 | 8.39 / 5.41 | 6.91 / 3.47 | 12.6 / 65.7 |
| collapse screen `[11,11]` (× its own random init) | 2.28e-02 (**12.3×**) | 2.53e-02 (5.7×) | 3.46e-02 (12.2×) | 3.30e-03 (2.03×) |

**Both instruments moved beyond `zgpose`'s** (digest tail 0.433 > the registered 0.13; subsets
0.327 > 0.09): `z_g` is now the project's most view-set-dependent fused latent — 2.2–2.3× the
pose-only parent — and the N=1 fused latent decodes the camera pose at 2.17° / 3.93 cm (the
pre-supervision parents read 10.8–17.8° / 16.6–23.2 cm; `zgpose` alone 1.20° / 1.90 cm). **The
tension readout landed in its registered intermediate bin** (0.857 ∈ [0.5, 1.5]): neither
pressure dominates — the base-frame term pulled `zgpose`'s 2.255 down, the pose term pulled
`m4base`'s 0.267 up, and the digest's independent statistic agrees (0.751, between 0.239 and
1.308). Unlike `zgpose`, the tug did not cost state discrimination: `zv_across_states` 0.916 ≈
the parents' (not `zgpose`'s 0.412) and the collapse screen reads 12.3× its own random init —
the healthy anchor — so this is the strongest view-awareness the project has produced **without**
a view-dominated encoder. Mechanism gates: both terms live — pose e200 2.45e-05 = 2.3e-04 of the
0.108586 floor (`zgpose` 7e-05); action 0.373 → 0.00721 epoch mean, 52× below its batch-0 and
37× below the 0.2692 constant floor. The batch-0 pose value 0.207 sits low in the first-16-step
spread (mean 0.250, −7% of the 0.269 CALIB — inside the calibration's own ~8% noise, and a
different draw: the header's non-claim). The in-training rollout (N=1 at az_0) reads 0.00 → 0.68
→ 0.82 → 0.80 → 0.82 at e0/50/100/150/200 (no collapse), and the e200 diffusion term — the
parents' last-step convention `train_loss − weighted aux` — reads 0.00287 (epoch mean 0.0042)
against the family's 0.00423–0.00692: the policy is intact.

**The reading.** The combination is not the clean null the registration anticipated: it is a
*stronger* `zgpose`. Every
representation-level effect the pose term produced is amplified (N-dependence 2.2×, subsets
2.3×, `z_v` pose-decodability kept far above the parents) **while the encoding stays healthy**,
behaviour at trained poses is unchanged (0.751, inside both bands), and **every held-out mean is
the lowest of the four cells** — off-grid 0.606 (the first *banded* registered mean this project
has moved outside its band; `zgpose`'s beyond-band move was the separately-registered, band-less
past-ring cell) and past-ring 0.100, no recovery. The documented-ambiguity clause did not
trigger (the instruments moved, they did not stand still), so the movement is reportable; what
it says is that the "more view-aware `z_g`" axis has now been pushed to its strongest setting
with the encoder kept healthy, and held-out success improved **nowhere** — it thinned. The
project's central null extends: **`z_g`'s view-set awareness is not the behavioural bottleneck**,
and the base-frame term's counterweight did not remove the pose supervision's cost pattern.

**Caveats.** One seed, 50 episodes per viewpoint; an *instrument* cell (all 11 ring views train,
nothing held out), so its numbers are not ladder-comparable. The off-grid move is 0.018 past the
band — outside by the stated convention, not by a large margin — and it is one mean over ten
cells that individually carry 0.14–0.20 spread; treat "weaker held-out generalization than both
parents" as the claim, not any single viewpoint. The past-ring mean mixes four floor-level
cells. Comparisons are design-level (the header's explicit non-claim of RNG lockstep with the
parents). "Intermediate" on the tension readout is the registered bin, not a predicted
direction.

## The probabilistic-fusion round (PLAN_G, 2026-10-16)

**What it is.** `PLAN_G.md` stage 1 asks whether a *probabilistic* view representation beats a
deterministic one. A small shared head now turns each view's latent `z_v` into a Gaussian — a
mean `mu_v` (that view's estimate of the 512-dim control state) and a log-variance `logvar_v`
(how unsure the view is). The mean branch is residual and zero-initialised
(`mu_v = z_v + f_mu(z_v)`), so at init it *is* the old latent, bit for bit; the variance branch
is zero-initialised and clamped to [−6, 2], so at init every view claims the same uncertainty.
Four cells, one variable apart:

| cell | fusion | what it isolates |
|---|---|---|
| **A** | MHA over `z_v` | the committed `m4base` run, **not retrained** |
| **B** | MHA over the means `mu_v` | the projection + the supervision move, with no probability in the model |
| **C** | **PoE** — multiply the live views' Gaussians | uncertainty-weighted fusion |
| **D** | the plain average of the means | whether PoE beats simple averaging |

The recipe is `m4base`'s: square, K=11 slots over pool [1..11], N ∈ [1,11], seed 42, 201 epochs,
base-frame aux at w = 1.0, the per-epoch latent probe. Two things change — the aux action head
reads `mu_v` instead of `z_v`, and the diffusion head consumes the fused **mean** `mu_g` — so
this round tests probabilistic state estimation and fusion, not stochastic action generation.
The head adds ~0.53 M parameters (0.18% of the policy) and draws from a private RNG stream, so no
other draw moves; the logs confirm it: B's batch-0 `train_loss` is bit-identical to `m4base`'s
(1.4437798), all three cells' batch-0 aux loss equals `m4base`'s `aux_loss` (0.360947847366333),
and C and D share a batch-0 loss (1.4688555) — at init C and D are the same model.

**Results — behaviour.** `success_rate`, 50 paired episodes, the registered means (band 0.15):

| registered mean | **B** | **C** | **D** | A = m4base |
|---|---|---|---|---|
| in-dist ring (11 vp, trained) | **0.829** | **0.804** | **0.793** | 0.796 |
| off-grid ±7.5…±67.5 (10 vp, held out) | **0.738** | **0.664** | **0.724** | 0.774 |
| past the ring ±82.5/±90 (4 vp) | **0.370** | **0.210** | **0.195** | 0.310 |
| elevation el_0 / el_m15 / el_p15 | 0.92 / 0 / 0 | 0.88 / 0 / 0 | 0.74 / 0 / 0 | 0.84 / 0 / 0 |

**In-distribution, all three cells are inside A's band** (Δ +0.033 / +0.008 / −0.003), so the
projection and the supervision move did no harm; all three also read at or above the other
committed arms (m4clear 0.747, zgpose 0.735, zvzg 0.751). **Off-grid — the round's headline
question — both registered contrasts land inside the band and are therefore unresolved**: C − B
= **−0.074** and D − C = **+0.060**. Both point away from the hypothesis: PoE is the weakest of
the three (−0.110 against A), C sits below B at 9 of the 12 shared held-out viewpoints, and D —
whose variance branch never trains — beats C at 8 of 12. **Past the ring the registered question
was "recovery, or a third floor?"; the answer is recovery for B (0.370, above A) and partial
recovery for C and D** (0.210 / 0.195 — above the pose arms' 0.140 / 0.100, below A). So there
is no third floor, but the arm that recovers is the one that never uses uncertainty. Elevation
is descriptive, as ever: 0 of 50 at both ±15° for every cell, and each cell's `el_0` differs
from its own `az_0` by 0.10 / 0.10 / 0.04 — inside the documented same-pose spread.

**Results — the instruments (the battery ran 2026-10-16).** The four *registered* readouts were
computed by `data/prob_analysis.sh` (digests, probes, screens and the variance read under
`data/analysis_prob_{b,c,d}/`, `data/probe_relpose_grid_square_prob_*/`,
`data/analysis_prob_variance/`, `data/screen_collapse/square_prob_*`), read against the four-cell
envelopes registered in `data/prob_run.sh:81-93` — a cell is "moved" iff its value is outside the
envelope by more than 10% of the envelope's own width:

| registered readout | envelope (A … zvzg) | **B** | **C** | **D** | read |
|---|---|---|---|---|---|
| probe `zv_pair_ratio`(1,11) | [0.267, 2.255] | 0.278 | 0.302 | 0.274 | all in-range, at the state-dominated end |
| probe `zg_across_view_subsets`(1,11) | [0.066, 0.327] | 0.0691 | 0.1270 | 0.1051 | all in-range |
| digest `rel(zg_n1, zg_full)` tail(25) | [0.089, 0.433] | 0.092 | 0.180 | 0.159 | all in-range |
| probe N=1 `zg` pose readout (ridge) | [1.20°, 17.8°] / [1.90, 23.2] cm | 12.1° / 21.7 cm | 11.1° / 20.7 cm | 13.6° / 19.3 cm | all in-range |

**No registered readout moved for any cell** — by the round's own convention every value sits
inside the committed four-cell envelope, so no mechanism claim is available from this battery (a
moved reading would have been "a lead to follow up, not a verdict"; an in-range one is "a bound,
not a demonstrated invariance"). Descriptively, within the envelope: B reproduces A on every
axis; C and D sit between A and `zgpose` on the two view-set-dependence readouts — C at ~80% of
the A→`zgpose` span (subsets 0.127 vs `zgpose`'s 0.1434; tail 0.180 vs 0.199), D at ~50–65%,
matching C > D > B — while C's and D's N=1 pose readouts stay at the A/`m4clear` level
(11–14° / 19–22 cm against `zgpose`'s 1.2° / 1.9 cm): **the liveness sensitivity the PoE fusion
adds is not camera-pose information**. All three cells also keep the encoder state-dominated
(`zv_pair_ratio` 0.27–0.30 ≈ A's 0.267): the projection alone does not push the representation
the way the supervision arms did. Collapse screens (matched range [11,11], against each cell's
own random init) read B 3.28e-02 (9.7×), C **9.11e-02 (28.6×)**, D 5.46e-02 (17.3×) — healthy,
and C's and D's are the **highest relative spreads any [11,11]-range screen has read**
(m4clear 3.46e-02, m4base 2.53e-02; m3pm60's 4.70e-02 was read at a [5,5] draw and is not
comparable). All three digests ran `--strict` clean (201 epochs, no holes, scalar recheck
≤1.6e-04), and the probe draw fingerprints are byte-identical across all six committed grids.
What the run logs already carry:

*Mechanism gates, all passed.* `mu_resid_rel` — how far the mean has moved off `z_v` — reads
0.015 / 0.015 / 0.014 after epoch 0 and rises to **0.321 / 0.279 / 0.293** by e200: the residual
head works in all three cells, so none is a re-seeded `m4base` (the exact 0.0 at init is pinned
by the CPU test, since the probe fires *after* epoch 0). All 201 epoch-end rows are finite and
complete in every cell. The in-training rollout (N=1 at az_0) climbs 0.00 → 0.58 / 0.62 / 0.66 →
0.80 / 0.84 / 0.82 — no collapse. **D's `lv_std` is 0.0 exactly**, as registered — and **B's is
too**: B's fusion never reads `logvar`, so its variance branch is dead by construction. So the
registration's "do not read D's uncertainty numbers as uncertainty" caveat extends to B as well —
neither cell's variance numbers mean anything.

*Variance quality — cell C only, the one cell where the variance is trained.* `lv_view_spread`
**0.447** > 0: the views genuinely disagree about their uncertainty, so PoE's weights are not
uniform (C is not a re-parameterized D). The clamp fractions are 0.011 / 0.071, far under the 0.9
line — the bound never binds. The fused variance is far tighter than any single view (`lvg_mean`
−4.20 vs `lv_mean` −1.32), and the N=1 degeneracy holds on the real model (`lvg_n1_vs_lv`
2.4e-07: one expert's product is that expert).

*The stage-2 variance read — is the uncertainty meaningful?* `analyze_variance_quality.py`, whose
thresholds were fixed in its header **before** any number from it existed, one forward pass per
state over a full 11-view draw at the frozen EMA checkpoint, run on cell C only (B's and D's
`logvar` branches are asserted **exactly zero** at this checkpoint — `max_abs_logvar_live` 0.0,
no fused logvar — so their uncertainty is unreadable by construction, the round's premise now
verified at the artifact rather than argued from the logs). Instrument gates: the per-pair aux
error re-derived by the script matches the policy's own loss to 2.2e-07; the full-draw override
took (`mean_active` == K); `state_idx` fixed throughout.

| read | what it measures | value (cell C) | verdict |
|---|---|---|---|
| **R1** view spread | per-state cross-view std of logvar | **0.4593** (log cross-check 0.4465) | **informative** |
| **R2** view quality | per-ring-view mean logvar vs mean error (11 poses) | ρ +0.209 / +0.164 (p 0.27 / 0.32) | **weak** (supported needs ≥0.5) |
| **R3** pair-level | logvar vs error over 5,632 live pairs, two error readings | +0.023 (at/below shuffle95 0.027) / **−0.166** | **null / against** |
| **R4** fused | `lvg` vs state-decode residual of `mu_g`, and vs diffusion loss | **−0.242** / −0.031 | **against / against** |
| **R5** N-sensitivity | `lvg` and fusion drift as N 1→11; leave-one-out over 11 views | both monotone as registered; az_0 drop 0.020403 vs +75° **0.020709** | **one formal violation** |
| **R6** overconfidence | implied σ_g vs the views' own disagreement | frac 0.338; median ratio 1.484; **double-count 2.23** | reported (1 = one expert dominates, √11 ≈ 3.32 = all independent) |

**The variance is spread, not signal.** The views genuinely disagree (R1), but that disagreement
ranks nothing: no ring view is systematically more uncertain than another (R2's ring means span
−1.30…−1.21 and the contrast runs the wrong way — az_0, one of the lowest-error views, carries
the *highest* mean logvar), pair-level uncertainty is indistinguishable from its shuffle control
and *anti*-correlated with the never-trained state residual, and the fused variance is
anti-correlated with error outright (R4 −0.242). R5 holds exactly where a product of experts must
hold it (more views → tighter, more stable fusion) and hands back one formal violation that is
really a flatness: dropping *any* view moves the fused mean by ~0.0204–0.0207, so "az_0 matters
most" is false because nothing dominates. R6 confirms `PLAN_G.md` §8 risk 2 in direction: with
correlated experts the product behaves as if ~2.2 experts were counted, halfway to treating all
11 as independent evidence.

*The digest scalars the earlier arms' tables carry* (same tool, same convention; `m4base`'s
reproduces its committed settle of 57, so the comparison is like for like):

| scalar, final (settle) | B | C | D | m4base |
|---|---|---|---|---|
| `zv_pair_ratio` | 0.244 (65) | 0.256 (**11**) | 0.235 (**32**) | 0.239 (57) |
| `zv_norm_mean` | 9.78 | 14.03 | 12.18 | 8.39 |
| `zg_norm_mean` | 5.59 | **13.03** | 11.15 | 5.41 |
| `zg_rel_spread` | 0.139 | 0.144 | 0.172 | 0.154 |

All three cells keep `z_v` in the baseline's state-dominated band (`zv_pair_ratio` 0.23–0.26
against `m4base`'s 0.239) — the projection alone does not push the representation the way the
supervision arms did (zgpose 1.308, zvzg 0.751) — while C and D's fused latent carries ~2–2.4×
`m4base`'s norm. The registration fixes mechanism claims to its four readouts only, so these are
context.

**The reading.** Round 1 does not show that probabilistic fusion helps. In `PLAN_G.md`'s terms:
**H2** (better out-of-distribution generalization) is not supported — the registered contrast is
unresolved and leans the wrong way (−0.074); **H1** (better fusion than the original MHA) is not
shown either — C reads 0.110 below A off-grid (inside band) and the three cells are
indistinguishable from one another; **H3** is not tested by this round, since all three cells
keep the base-frame supervision. What the round does add: the uncertainty machinery demonstrably
learns something — the views disagree, the clamp never binds, the product fuses and degenerates
exactly as designed — and buys nothing behaviourally. That is the same pattern the supervision
arms showed at the representation level, now on a different axis. Its one behavioural move is
outside the headline: past the ring all three cells leave the pose arms' floor, and the best of
them is the arm that never uses uncertainty. **The completed battery sharpens both halves.**
At the representation, the round's own registered axes find no cell outside the committed
four-cell envelope: PoE's fusion does react to which views are live (C sits ~80% of the way from
A to `zgpose` on both view-set readouts, descriptively), but that reactivity is not camera-pose
information (N=1 readout 11.1° / 20.7 cm against `zgpose`'s 1.2° / 1.9 cm) and does not tip the
encoder out of the state-dominated band any arm trained this way has lived in. At the variance,
PLAN_G §9's stage 6 ("judge whether the probability estimates are sensible") now has its answer:
the uncertainty is real spread that ranks no errors — R2 weak, R3 and R4 null-to-against, R6's
double-count 2.23. So the null is not a broken mechanism (every gate passes), not the projection
(B ≈ A on all four axes), and not a hidden movement in the latent (in-envelope); at this budget
and this seed, uncertainty-weighted fusion simply did not help. The stage-2 options PLAN_G §3.1
and §8 reserved — KL or calibration objectives on the variance — would be spending on the axis
this round has just measured as uninformative, which is a plan decision, not a result; the
evidence assembled for it is the six readings above.

**Caveats.** One seed, 50 episodes per viewpoint; the off-grid contrasts' registered read is
"unresolved inside 0.15", and the measured mean-scale spread is 0.05–0.08, so this round cannot
resolve differences below that — "PoE ≈ MHA-over-means" is a bound, not a demonstrated equality.
The cells are instrument-comparable to `m4base` (every ring view trains), not
ladder-comparable. C and D's batch-0 diffusion term is cell-specific by construction (they start
at the masked mean, not MHA), so their early loss curves are not read against A's. The
variance-quality thresholds for the stage-2 uncertainty read (`corr(logvar, error)` and
calibration) were fixed **before** those numbers existed, in `analyze_variance_quality.py`'s
header (written 2026-10-16); the script runs on cell C only, with a dead-logvar assertion in
place of a read for B and D — which passed **exactly** (`max_abs_logvar_live` 0.0). Its numbers
— and the four readouts above — landed 2026-10-16 from `data/prob_analysis.sh`. **Read the
"in-envelope" verdicts as the round registered them:** the moved-lines are the four-cell
envelope plus a 10%-of-width margin, a stated convention over unmeasured noise, so in-range is a
bound on how far the prob cells sit from the committed lineage, not a demonstrated invariance —
and the convention is deliberately conservative. Under the *earlier*, parent-anchored thresholds
the `zgpose` round used (probe subsets > 0.09, digest tail > 0.13), C would have read as moved on
both; its descriptive position inside the envelope (≈80% of the way from A to `zgpose`) is
recorded above for exactly that reason, and it is where any follow-up would start.

## The z-supervision round (PLAN_Z, 2026-10-28)

**What it is.** `PLAN_Z.md` proposes two supervisions on the multi-view encoder, each tested alone
and then together against the committed `m4base` baseline (not retrained):

| cell | supervision | mechanism |
|---|---|---|
| **a** `zalign` | `L_align` on `z_v` | a shared projection head `P_s` maps each live view's `z_v` into a "shared semantics" space; the loss is the **L2 between L2-normalised projections** of two views of the same state |
| **b** `zcons` | `L_cons` on `z_g` | two independent view draws give `z_g^(i)` and `z_g^(j)`; the **same** noisy action, timestep and noise pass through the UNet twice and the two ε-predictions are pulled together |
| **c** `zboth` | both | the same two weights as a and b, so `c` decomposes as `a + b` |

The recipe is `m4base`'s throughout: square, K=11 slots over pool [1..11], N ∈ [1,11], seed 42, 201
epochs, base-frame aux at w = 1.0, the per-epoch latent probe. The two weights are calibrated
pre-launch by the family convention `W = 0.33·diff0/loss0` from one train-mode forward at init
(`data/z_run.sh` stage 1): `diff0 = 1.078701`, `align0 = 0.041532`, `cons0 = 0.001869` →
**W_ALIGN = 8.571**, **W_CONS = 190.4601**. Both exceed the driver's `W > 3` warning and were
**registered, not adjusted**, per the pre-registration; the same pair went into all three cells so
that `c` is exactly `a + b`.

**Results — behaviour.** `success_rate`, 50 paired episodes, the registered means (band 0.15
against A):

| registered mean | **a** | **b** | **c** | A = m4base |
|---|---|---|---|---|
| in-dist ring (11 vp, trained) | **0.796** | **0.155** | **0.129** | 0.796 |
| off-grid ±7.5…±67.5 (10 vp, held out) | **0.764** | **0.150** | **0.150** | 0.774 |
| past the ring ±82.5/±90 (4 vp) | **0.375** | 0.135 | 0.125 | 0.310 |
| elevation el_0 / el_m15 / el_p15 | **0.840 / 0 / 0** | 0.080 / 0.04 / 0.04 | 0.020 / 0.04 / 0.08 | 0.840 / 0 / 0 |

**`a` is a null so exact that it is a replica of the baseline.** In-distribution 0.7964 against
0.796 (Δ +0.0004); off-grid 0.764 against 0.774 (Δ −0.010, far inside the band); `el_0` identical
at 0.840; the past-ring 0.375 is nominally the highest of any arm but that cell is registered
separately, band-less, and mixes two ±82.5° cells with two ±90° cells pinned at the floor. **`b`
is a collapse, not a null**: −0.641 in-distribution and −0.624 off-grid, impaired at every one of
the 11 *trained* viewpoints (0.06–0.26) and — uniquely in the lineage — at the untrained elevation
cells too (0.080 / 0.04 / 0.04, where every other arm reads 0 / 0). **`c` = `b`**: off-grid
matching to four decimals (0.1500 both), in-distribution 0.129 against 0.155, elevation the worst
of the three. PLAN_Z §3's `C = A + B` decomposition therefore comes out exact, with `A ≈ 0`.

**Results — the registered readouts.** `data/z_analysis.sh` (digests, probes, screens, plus the
round's new semantic head read), read against envelopes computed from the four committed grids
(`m4base`/`m4clear`/`zgpose`/`zvzg`); "moved" means outside the envelope by more than 10% of its
own width:

| registered readout | envelope (A … zvzg) | **a** | **b** | **c** |
|---|---|---|---|---|
| probe `zv_pair_ratio`(1,11) | [0.267, 2.255] | 0.2634 | 0.1472 | 0.1624 |
| probe `zg_across_view_subsets`(1,11) | [0.066, 0.327] | 0.0485 | **0.0027 MOVED** | **0.0054 MOVED** |
| digest `rel(zg_n1, zg_full)` tail(25) | [0.090, 0.433] | 0.0649 | **0.0045 MOVED** | **0.0085 MOVED** |
| probe N=1 `zg` pose readout (ridge) | [1.20°, 17.8°] / [1.9, 23.2] cm | 14.1° / 22.3 cm | **47.6° / 61.8 cm MOVED** | **27.5° / 39.6 cm MOVED** |

All four of `a`'s readouts are in-range — by the round's own convention that is a *bound*, and the
bound says the encoder did not move. `b` and `c` move the same three, in the same direction:
`z_g` loses view-set dependence (subsets 0.0027 / 0.0054 against A's 0.0663), the N=1 latent
becomes indistinguishable from the full-draw one (tail 0.0045 / 0.0085 against A's 0.0895), and
camera-pose decodability degrades to 47.6° / 27.5° from an envelope top of 17.8°. That is not
"more view-invariant"; it is `z_g` carrying less information about anything. The probe
comparability gate passes — draw fingerprints are byte-identical across all seven committed grids.

**Collapse screens** (matched [11,11] draw, each cell against its own random-init baseline) are
what separate the two arms:

| cell | trained spread | × own random-init | `mean_norm` |
|---|---|---|---|
| **a** | 4.02e-02 | **12.2×** | 5.15 |
| **b** | 4.41e-03 | **1.6×** | 44.05 |
| **c** | 7.31e-03 | **1.7×** | 5.66 |
| m4base (ref) | 2.53e-02 | 5.7× | 5.25 |
| zgpose (ref, previous worst) | 3.30e-03 | 2.0× | 63.3 |

`a`'s encoder is healthy and indistinguishable from m4clear's 12.2×. `b` and `c` sit at
**1.6× / 1.7×** — below every arm in the project, zgpose's 2.0× included. The digest agrees on
`b`: `zg_norm_mean` **44.8** (m4base 5.41) with `zg_rel_spread` **0.0298** (m4base 0.1045) — a huge
norm and almost no spread, the signature of features converging onto one common vector.

**The semantic readout** — `analyze_semantic_head.py`, the round's new instrument and risk R1's
check — asks whether the *encoder* or the *head* absorbed the alignment pressure:

| cell | trained `L_align` | trained `cos_within` | trained `cos_across` | re-init `cos_across` | `collapse_signature` |
|---|---|---|---|---|---|
| **a** | 3.5e-09 | 1.0000 | **1.0000** | 0.4838 | **True** |
| **b** | 0.0237 | 0.9881 | 0.5007 | 0.5007 (identical) | False |
| **c** | −0.0 | 1.0000 | **1.0000** | 0.5208 | **True** |

`a`'s head maps views of *different physical states* to the same direction — `cos_across` 1.0000
against a random head's 0.4838 — so `L_align → 0` is satisfied trivially and no gradient carrying
view-invariance reaches `z_v`. `b`'s head was never trained (`align_weight = 0`) and the instrument
returns its re-init control **to six decimals**, which independently validates the instrument;
`c`'s head collapses exactly as `a`'s.

**Mechanism gates.** Batch-0 `diff_loss` is **1.08283198 in all three cells**, bit-identical to
`m4base`'s derived 1.082832 (its `train_loss` 1.4437798 minus its `aux_loss` 0.360948) — the
RNG-lockstep claim read from the artifacts rather than asserted. `align_loss` falls from 0.0405 at
batch 0 to ~1e-09 by e200 in `a` and `c`; `cons_loss` falls from 0.0021 to ~1e-06 in `b` and `c`.
The in-training rollout does not collapse in `a` (0.00 → 0.60 → 0.76 → 0.76 → 0.82), and sits at
the floor in `b` (0.00 → 0.00 → 0.10 → 0.12 → 0.22) and `c` (0.00 → 0.04 → 0.06 → 0.08 → 0.08).
**The registered R2 instrument did not fire**: `diff_loss` at e200 reads 0.00229 / 0.00281 /
0.00238, at or *below* the family band's 0.0028 floor, so the pre-registered "a cons term that
shrinks the UNet's dependence on conditioning would show up here" test reported clear while the
encoder and the behaviour were collapsing. That insensitivity is itself a result: `diff_loss` can
stay low on the low-dim observations alone, so it is not a collapse detector.

**The reading.** Both mechanisms fail, each in its own way, and in each case the failure is in the
loss as specified rather than in the training.

*`L_align` (PLAN_Z §1) is degenerate.* An L2 between L2-normalised projections has a global minimum
at a constant, and the optimiser finds it: both trained heads collapse (`cos_across` 1.0000),
`L_align → 0` becomes trivial, and nothing about view-invariance reaches the encoder — cell `a`'s
screen is 12.2× (healthy), its `zv_pair_ratio` is 0.2634 against A's 0.2672, all four readouts are
in-range, and its behaviour is the baseline's to three decimals. This is not "alignment does not
help"; it is "**this objective cannot help**". PLAN_Z §1 lists InfoNCE as a later option, and this
round says that option is not optional: the standard remedy is the one that supplies *negative*
pairs and therefore keeps `cos_across` low.

*`L_cons` (PLAN_Z §2) destroys the representation.* At `W_CONS = 190.46` the cheapest way to make
two independent view draws agree is to make the encoder produce the same thing for every view, and
that is what happens: 1.6× / 1.7× random init, `z_g` uninformative on three readouts, behaviour at
the floor on every axis including untrained elevations. The weight is not an accident of the run:
it is the family's `0.33·diff0/loss0` convention applied to a target ~500× smaller than `diff0`,
and the driver's `W > 3` warning fired at 190. The pre-registration chose to record rather than
adjust it, which is precisely what makes this a clean, attributable result instead of an ambiguous
one.

*The two do not interact.* `a` = baseline, `b` = collapse, `c` = `b` — on the registered means, on
every instrument, and on the encoder health screen. Adding `z_v` alignment to the collapsed arm
changes nothing measurable, in either direction.

**Caveats.** One seed, 50 episodes per viewpoint; all contrasts are read on the 0.15 band whose
measured mean-scale spread is 0.05–0.08, so "c = b" is a bound, not a demonstrated equality. The
cells are instrument-comparable to `m4base` (every ring view trains), not ladder-comparable.
`W_ALIGN`/`W_CONS` were calibrated once (on cuda:1) and reused by `b` and `c` from
`data/z_calib_env.sh`, which is what preserves the `C = A + B` decomposition; both tripped the
`W > 3` warning and were registered rather than adjusted. The three runs are **wandb-offline**
(`logging.mode=offline`): the installed client (0.15.12) can no longer reach the wandb server, and
0.17.9 fails identically, so the `.wandb/offline-run-*` directories under `/data/zihan/runs/*/wandb/`
still need syncing after a client upgrade. Finally, the registered R2 instrument read **clear** at
e200 for all three cells despite the collapse; a future registration of this failure mode should
anchor on the collapse screen and the three moved readouts, which did fire, rather than on
`diff_loss`.

## The encoder ladder: B0/B1/S (2026-10-29)

**What it is.** The first two stages of `ENCODER_EXPERIMENT_PLAN.md`. **B** keeps the M3 policy
(no aux head) and changes only the EE-history protocol: B0 is the legacy
`eef_hist_window=sample` — the window-start clip that every committed M3/M4 run used, which with
To=2 gives `[B,B,B,B]` / `[B,B,B,B+1]` and therefore almost no history — and B1 is `episode`, the
true H most recent poses inside the same episode, clipped only at the episode start, which is
exactly what both rollout paths (`CamKeyImageWrapper`, `ViewpointImageWrapper`) serve. One CLI
flag differs; the data, K=11 slots over pool [1..11], N ∈ [1,11], the M3 policy, the latent
probe, seed 42 and all schedules are identical, and the view-draw/crop RNG is unchanged by the
flag, so B0/B1 is a clean one-variable pair. **S** replaces the encoder: pure-RGB ResNet18
layer3 → 1×1 conv 256→512 → 25 spatial tokens → one shared zero-init 512-d query, 8-head
intra-image cross-attention → one 512-d token per view → the same masked cross-view MHA fusion
as B. S is a **structural** comparison, not RNG-lockstep with B (module construction consumes
different global torch RNG), and its 5,016,640 encoder parameters are half B's 12,532,928, so
S−B is not capacity-matched either.

**Protocol.** Square PH, `_abs` actions; pool [1..11] = az −75…+75 every 15°, **all trained**
(unlike M2/M3's even/odd split — see *Setup and protocol*); N uniform in [1,11]; 201 epochs;
EMA epoch 200; 50 paired episodes per viewpoint; strict `success_rate`. Registered means:
in-distribution ring (11 vp, trained), interpolation ±7.5…±67.5 (10 vp, held out),
past-ring ±82.5/±90 (4 vp, outside the training range), elevation el_0/±15 (3 vp, descriptive).
Band **0.15** mean-over-viewpoints, pre-registered; direction not predicted.

**Results — behaviour.** Strict `success_rate`, 50 paired episodes per viewpoint:

| registered mean | B0 (`sample`) | B1 (`episode`) | S | m4base |
|---|---|---|---|---|
| in-dist ring (11 vp, trained) | 0.751 | **0.813** | 0.653 | 0.796 |
| interpolation ±7.5…±67.5 (10 vp) | 0.712 | **0.784** | 0.618 | 0.774 |
| past the ring ±82.5/±90 (4 vp) | 0.320 | **0.365** | 0.355 | 0.310 |
| elevation el_0 / m15 / p15 | 0.80 / 0.10 / 0.20 | 0.90 / 0.02 / 0.12 | 0.60 / 0.04 / 0.24 | 0.84 / 0.00 / 0.00 |

**B0 vs B1 is a null.** Every registered mean is inside the 0.15 band: in-distribution −0.062,
interpolation −0.072, past-ring −0.045, elevation +0.020 (descriptive). B1 is if anything
slightly better on the ring. The corrected history therefore changes nothing measurable. This is
the clean version of the M3/M4 "history is inert" result: those runs used `sample` and could not
test the true past; B0/B1 differs only in the flag, and the flag changes the history tensor
values without touching the view-draw/crop RNG.

**S vs B1 is negative on the ring.** In-distribution **−0.160** and interpolation **−0.166** are
both just outside the 0.15 band; past-ring is a null (−0.010 — both cells sit at the ~0.36
floor), and elevation is lower (−0.053; `el_0` 0.60 vs 0.90, `el_m15` 0.04 vs 0.02, `el_p15` 0.24
vs 0.12). The penalty is concentrated on the trained and interpolated ring, where B is strong,
and disappears at the past-ring floor. S's epoch-200 in-training score (0.62) is below its
epoch-150 topk (0.66) while B1's epoch 200 is its best (0.86); the registered protocol evaluates
epoch 200, and selecting the topk would be post-hoc checkpoint selection, which the plan forbids.

**B1 vs m4base validates the baseline.** +0.017 in-distribution and +0.010 interpolation, inside
the band; past-ring +0.055 and elevation +0.067 (descriptive). `m4base` carries the base-frame
aux head, so this is a reference check rather than a clean comparator; B1 is the encoder ladder's
internal baseline.

**Elevation is not fixed.** The downward-elevation failure persists in every cell: `el_m15` 0.02
(B1) / 0.10 (B0) / 0.04 (S), against `el_0` 0.90 / 0.80 / 0.60; `el_p15` is 0.12 / 0.20 / 0.24.
The elevation cells are descriptive, not registered, but no encoder change so far touches them.

**Caveats.** One seed (42) and 50 episodes per viewpoint. The S margins are only 0.010–0.016
over the threshold, so S is not yet a confirmed regression; the plan's next comparison is seed
43/44. The S/B runs are not RNG-lockstep (unlike B0/B1), so S−B is a structural and
capacity-confounded comparison, not a paired ablation. The S eval uses epoch 200 (`latest.ckpt`)
per protocol; its topk is epoch 150. The per-epoch latent probe ran in all three cells and the
npz series are on `miroc-server`; the mechanism-live reads (FiLM gamma/beta norms, probe series)
were not analysed here — the behavioural result is the headline.

**Operational note.** The first launch died at `wandb.init()` with the known client/server
incompatibility (`CommError: 'NoneType' object has no attribute 'get'`; wandb 0.15.12 cannot talk
to the current server, and 0.17.9 fails identically). Both run scripts now pass
`logging.mode="${LOG_MODE:-offline}"`, the same fix and reasoning as `data/z_run.sh`; the failed
first attempts are preserved as `*_ABORTED_wandb_20261028`. The campaign shared the machine with
another user's CPU-heavy RoboTwin evaluation, which stretched B0 and the B1 sweep; no run failed,
and the episode/seed protocol is unchanged, so the numbers are unaffected.

**Artifacts.** `data/outputs/run_square_enc_{b0,b1,s}_s42_200ep/logs.json.txt` (probe scalars per
epoch); `data/eval_{interp,heldout,el}_square_enc_{b0,b1,s}/eval_log.json`; the run dirs under
`/data/zihan/runs/run_square_enc_*` (~9 GB each) hold `latest.ckpt` and the latent-probe npz and
stay on `miroc-server`. **R is implemented and CPU-gated (2026-10-29) but untrained; RH and RHG
are not implemented.**

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

### The encoder ladder — B and S

**Method.** B0 vs B1 changes only `eef_hist_window` (`sample` vs `episode`) on the M3 recipe; S
replaces B's 9-channel RGB+Plücker conv1, per-stage FiLM and global pool with a pure-RGB
spatial-token readout (one shared 512-d query, 8-head intra-image attention) and B's cross-view
fusion. Registered means over 50 paired episodes; band 0.15.

**Result.** B0−B1 is null on every registered mean (in-dist −0.062, interp −0.072, past −0.045,
elevation +0.020 descriptive), so the corrected history is behaviourally inert. S−B1 is −0.160
in-distribution and −0.166 interpolation, both just outside the band; past-ring is a null
(−0.010 — both at the ~0.36 floor) and elevation is lower (−0.053, descriptive). B1 matches
`m4base` within the band.

**Conclusion.** The history correction does not matter behaviourally; the pure-RGB spatial
readout costs ~0.16 on the trained/interpolated ring but does not move the past-ring floor, so it
is not a free replacement for the view-conditioned encoder. S is one seed, just over the
threshold; a second seed is the registered confirmation step. R/RH/RHG remain untested.

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
  **Added 2026-10-28 (`zcons`/`zboth`).** The project now has its first **fully diagnosed** encoder
  collapse, which makes a useful reference signature for the unexplained ones: cells `b` and `c`
  of the z round collapsed *because* a supervision term at `W_CONS = 190.46` was cheapest to
  satisfy by making every view encode identically. The signature there is a matched-draw screen at
  **1.6×/1.7×** its own random init, `zv_pair_ratio` 0.135/0.154 (against A's 0.267), and — on
  `b` — a fused latent with `zg_norm_mean` **44.8** against 5.41 and `zg_rel_spread` **0.0298**
  against 0.1045. Note that `diff_loss` stayed *low* throughout (0.0028 at e200, the bottom of the
  healthy band), so the diffusion loss does **not** flag this kind of collapse; the screen and the
  readouts do. Whether the unexplained collapses (`[1,2]`, `[1,3]`, L1) share this signature is an
  open question this reference now makes testable.
- **M5's premise.** M3 already does N=1 novel-view inference well, so distillation is only
  worth it if the fused latent demonstrably carries something the single-view path cannot,
  which no result so far shows. The clear-run analysis is the first evidence *against* the
  premise rather than merely its absence: at inference the fused latent at N=1 is within ~0.5%
  (direction) of the 11-view one, so whatever the extra views contribute is not visible in `z_g`
  at test time. **Updated 2026-10-10 (`zgpose`).** That reading is now reversed at the
  representation — with the pose-set supervision `z_g` is ~2× the parents' N-dependence and the
  N=1 fused latent decodes the camera pose at 1.2° / 1.9 cm, so a student *would* have something
  to learn. The same run also shows the other side: making `z_g` view-aware bought no behaviour,
  so the premise is re-opened, not established — the deciding measurement (does a distilled
  student beat the plain N=1 path behaviourally?) is still unmade. **Extended 2026-10-14
  (`zvzg`).** The joint arm pushes the same axis further — `z_g`'s N-dependence 2.2–2.3×
  `zgpose`'s, the N=1 fused latent 2.17° / 3.93 cm pose-decodable — so a student would have
  still more to learn, while the held-out behaviour is the weakest of the four cells: the same
  caution again, at a stronger setting.
- **Does the S regression survive a second seed, and can R/RH/RHG recover the ring?** S is
  −0.160 in-distribution and −0.166 interpolation against B1, both just outside the 0.15 band,
  on one seed; the plan's next comparison is seed 43/44. **R (ray encoding on the attention
  keys) is implemented and CPU-gated as of 2026-10-29** — `ray_spatial_readout_obs_encoder.py`,
  the `enc_r` configs and `data/enc_r_{run,eval}.sh`; it is untrained, so whether the ray
  address recovers B1's ring performance is open. RH (full EE-trajectory-conditioned shared
  query) and RHG (point-to-ray attention bias) remain specified but unimplemented in
  `ENCODER_EXPERIMENT_PLAN.md`.

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
- **`ENCODER_EXPERIMENT_PLAN.md`** — the encoder ladder's stages, structures and pre-registered
  evaluation protocol; B0/B1/S are measured here, R/RH/RHG remain to implement.
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
| z_v supervision (`m4base`) | `config/task/m4_base_image_abs_multiview_az75.yaml`, `config/train_diffusion_unet_image_workspace_m4base_latent.yaml`, `data/m4base_{run,eval,analysis}.sh`; seam: `multiview_image_dataset.py` `aux_action_frame` (base-frame emission) |
| z_g supervision (`zgpose`) | `config/task/zgpose_image_abs_multiview_az75.yaml`, `config/train_diffusion_unet_image_workspace_zgpose_latent.yaml`, `model/vision/view_pose_set_head.py`, `policy/diffusion_unet_image_policy_zgpose.py`, `tests/test_zg_pose_supervision.py`, `data/zgpose_{run,eval,analysis}.sh`; seam: `multiview_image_dataset.py` `emit_aux_view_pose` (zero-filled pooled pose-set target, all rows scored) |
| joint supervision (`zvzg`) | `config/task/zvzg_base_image_abs_multiview_az75.yaml`, `config/train_diffusion_unet_image_workspace_zvzg_latent.yaml`, `policy/diffusion_unet_image_policy_zvzg.py`, `tests/test_zvzg_supervision.py`, `data/zvzg_{run,eval,analysis}.sh`; both dataset targets reuse the `m4base`/`zgpose` emissions unchanged; seam: `train_diffusion_unet_image_workspace.py` (per-term `aux_action_loss`/`aux_pose_loss` log lines) |
| probabilistic fusion (`prob_b/c/d`) | `model/vision/per_view_gaussian.py`, `model/vision/prob_view_conditioned_obs_encoder.py`, `policy/diffusion_unet_image_policy_prob.py`, `workspace/train_diffusion_unet_image_workspace_probprobe.py`, `config/task/prob_base_image_abs_multiview_az75.yaml`, `config/train_diffusion_unet_image_workspace_prob_latent.yaml`, `tests/test_prob_gaussian_fusion.py`, `data/prob_{run,eval}.sh`; seam: `train_diffusion_unet_image_workspace_latentprobe.py` (`_probe_extras` extension hook — the base returns nothing, so every existing run's npz/JSONL artifacts stay byte-identical) |
| encoder ladder (B/S) | `model/vision/spatial_readout_obs_encoder.py`, `config/task/enc_{b,s}_image_abs_multiview_az75.yaml`, `config/train_diffusion_unet_image_workspace_enc_{b,s}_latent.yaml`, `data/enc_{b,s}_{run,eval}.sh`, `tests/test_spatial_readout_obs_encoder.py` (plus the 32-line extension of `tests/test_view_conditioned_obs_encoder.py`), `ENCODER_EXPERIMENT_PLAN.md`, `ENCODER_REFERENCES.md`; seam: `multiview_image_dataset.py` `eef_hist_window` |
| encoder ladder (R) | `model/vision/ray_spatial_readout_obs_encoder.py`, `model/vision/plucker.py` (`plucker_ray_map_at`, `ray_dirs_cam_at`), `config/task/enc_r_image_abs_multiview_az75.yaml`, `config/train_diffusion_unet_image_workspace_enc_r_latent.yaml`, `data/enc_r_{run,eval}.sh`, `tests/test_ray_spatial_readout_obs_encoder.py`; same `eef_hist_window` seam as B/S, no new dataset/runner keys |

**Edited seams** — the only changes to files this fork did not itself add:
`multiview_image_dataset.py` (cam table, per-sample view draw, mask, camera-frame EE history;
the `eef_hist_window` selector added 2026-10-28 — `sample` default for committed runs, `episode`
for the corrected protocol; and the `view_count_range` guard removed 2026-09-24 so the ladder
can run `[1, 2]`),
`eval_novel_view.py` (`--m3-slots`, `--eef-hist-steps`, the elevation orbit),
`summarize_novel_view.py` (sort key for `el_*` names), and the `getattr`-guarded aux log lines
in the upstream `train_diffusion_unet_image_workspace.py` (`step_log['aux_loss']`, plus the
`zvzg` seam's per-term `aux_action_loss`/`aux_pose_loss`). No other upstream package file is
modified.
Note that `{square,lift}_image_single.yaml` (relative-action single-view variants) exist but are
unused — the baselines use the `_abs_single` variants.

Committed to git: every sweep's `eval_log.json`, the per-batch training logs, and the
probe/screen JSONs. Weights (4.6 GB per checkpoint) are **not** in git — rsync only, and they
live on `miroc-server`.
