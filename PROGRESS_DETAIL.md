# PROGRESS — detail and appendices

The parked detail behind `PROGRESS.md`. That file carries the module — the M1–M4 milestones
and their measured behaviour, which is what a new reader needs — plus one conclusion block per
follow-up experiment; this one carries the full protocol, the investigation log section by
section, the code map, and three appendices.

**Read Appendix C before quoting any number.** It records every correction and supersession,
dated, including numbers that must not be quoted or compared to any later one.

Read Part 2 by section, not front to back: each section states its own setting, gates and
limits, and the pointers in `PROGRESS.md` name them by title.

Sections in Part 2 and the appendices are reproduced **verbatim** from the pre-split
`PROGRESS.md` at git `c6ec793` (except where a section says otherwise), so every published
number is byte-exact and the commit history still lines up.

Last updated 2026-09-26.

## Part 1 — Protocol (full)

### Setup and protocol

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

#### Noise and resolution

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

## Part 2 — Investigation log

### Archived session log — 2026-09-25

Five things settled today. Two are negative results, and one is a correction of a diagnosis
made earlier in the same session — that is the honest shape of it.

**1. The balance hypothesis is dead, and it died by rule.** `[1,3]`'s floor was thought to come
from leaning too hard on proprioception — the n=8 reading said 1.75× harder than the working
cell. At n=64 across three seeds that gap is gone: the proprio arm reads **1.005×** between
`m3v13` and `m3off`, and does not vary across the ladder at all — the four rungs span **1.005×**
between their lowest and highest, and repeat runs of one cell agree to 1.001–1.009×. The
pre-registered gate in front of the proprioception-dropout run **failed** (min ratio 0.567;
2-of-3 seeds above the bar), so **the run was never launched**. The rule decided it before its
justification was examined.

**2. That leaves the open question with no candidate mechanism.** "Why does `[1,3]` fail while
its representation looks fine?" now has nothing proposed to answer it. What it does have is a
sharper specification — see 3.

**3. `image→action sensitivity` is not a scalar, and my own diagnosis of its noise was wrong.** I
said the screen's 2× instability came from an unseeded view draw and per-cell draw ranges. Both
were real defects and both are fixed (the tool is now bit-reproducible to 6 figures) — but the
matched-draw control left the spread **unchanged** (1.92–2.50×). The sharper truth: with the
input held identical across cells, the **rank order still flips across seeds**. So the missing
instrument cannot be built by tightening this statistic; it needs a verdict that does not depend
on which scenes are probed.

**4. The latent numbers point the wrong way across the whole ladder.** Filling `m3v15`'s missing
probe turned `[1,3]`'s anomaly into a monotone trend. As mean-N falls 4.0 → 3.0 → 2.0, `z_g` gets
*more* view-invariant and `z_v` *less* view-aware — toward what the proposal calls the goal —
while behaviour falls 0.764 → 0.456 → 0.080. Across four independently trained rungs, the
representation looks better and the policy works worse.

**5. The ladder's last question is answered, and M1 is exonerated.**
- **`[2,7]`** never trains at N=1 (min-N 2, mean-N 4.5) and is **indistinguishable from `[1,7]`**:
  Δ trained +0.104, inside the 0.15 band; Δ held-out +0.003. So the ingredient is *enough views
  on average* — not the presence of N>1, and not the availability of N=1. It also settles a
  design question: N=1 **inference** works without ever training at N=1, so M3's stated reason
  for randomising N is not necessary for the capability it protects.
- **M1's baseline was not collapsed** — 17.5× above its own random-init baseline. M1's
  view-tiedness is a *different* failure mode from the collapse that explains L1 square/can and
  `[1,2]`, so the collapse story does not run back to M1.

Housekeeping — disk freed, the recording gaps, and the duplicate checkpoint — is in
`NOTES.md`'s *Disk* and *Artifacts* sections; the research corrections are below in *Record of
corrections*.

Everything below is the evidence for the above, in the project's usual detail.

### L1 — view diversity only: does showing many views buy invariance?

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

### N>1 — the architectural confound, resolved

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

### Relational probe (step 1a) — the geometry is already in `z_v`

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

#### Conclusion

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

### N-diversity ladder — how much multi-view signal is needed?

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

#### Stage 1 — `[1,2]` lands at the floor, indistinguishable from `[1,1]`

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

#### Stage 2 — `[1,3]` (mean N 2.0) is also at the floor

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

#### Stage 3 — `[1,5]` (mean N 3.0) breaks the floor, and the ladder is graded after all

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

### Collapse is a failure mode

**Why this was measured at all.** Every evaluation here is N=1 inference, and at N=1 the
fusion softmax is over a *single* unmasked key, so the learnable query is inert. Large-N
training therefore cannot be teaching the fusion anything, which pointed at the **shared
backbone**: the hypothesis was that seven views of one scene press the encoder toward a
canonical, scene-level representation that two views do not. It predicted that a fusion-free
view-invariance statistic would separate the working cell from the floor cell.

#### `[1,2]`'s encoder has collapsed

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

#### It answers L1's task split — and it is *a* failure mode, not *the* one

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

#### The whole table, screened — and `m3n1gate` sharpens the mechanism

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

### The second failure mode — `[1,3]` fails with a healthy representation

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

### Proprioception dropout — the `[1,3]` balance test, retired at its own gate

**Why this was run.** *The second failure mode* left exactly one mechanism standing: **balance**.
`[1,3]`'s image→action sensitivity was identical to the working cell's (0.0067 vs 0.0068 at
n=8) and the only measured difference was that it leaned **~1.75× harder on proprioception**
(0.0674 vs 0.0386) — and proprioception cannot see where the nut is. A falsifying intervention
was coded and left ready: `DiffusionUnetImagePolicyPropDrop`, which drops proprioception for a
random half of training samples, in `compute_loss` only, so rollouts and every probe are
untouched. In front of it stood a **pre-registered gate**: the balance reading was n=1, so it
had to be re-measured at n=64 over three seeds before it was worth GPU-hours, with the rule
fixed in advance — *the hypothesis lives only if `m3v13`'s proprio/image ratio exceeds
`m3off`'s by ≥1.5× across all three seeds; at parity the intervention is off*.

**Setting.** Nine `screen_conditioning.py` runs, `--n-obs 64`, `m3v13`/`m3off`/`m3v12` × seeds
0/1/2. Artifacts: `data/screen_conditioning/square_*_n64_s*.json`. The formalisation
(`min_s R_m3v13/R_m3off >= 1.5`, and 2-of-3 is *not* support) was committed before any n=64
number existed.

#### Result 1 — the gate fails, and the intervention is off

| seed | `R_m3v13` | `R_m3off` | ratio | gate (≥1.5) |
|---|---|---|---|---|
| 0 | 38.53 | 22.59 | 1.705 | PASS |
| 1 | 40.05 | 12.60 | 3.179 | PASS |
| 2 | 16.20 | 28.56 | **0.567** | fail |

`min` over seeds = **0.567**, so the pre-registered *gate failed, reading ambiguous* branch
applies. **By pre-registration the intervention is off and the run was not launched** — a
decision taken before its justification was examined, which is the point of registering it.

#### Result 2 — the clean quantity refutes the balance hypothesis outright

The gate's own statistic turned out to be the wrong thing to lean on. `image_only` swings
**2.27×–2.47×** across the three seeds, so the ratio is not a measurement; but `proprio_only`
is **draw-independent** — the proprio keys are the same 64 low-dim rows whichever views are
live — and it is stable to **0.2–0.4%** within each cell across all three runs:

| cell | `proprio_only` (mean of 3) | across-seed spread |
|---|---|---|
| `m3v13` | 0.35003 | 1.002× |
| `m3off` | 0.34726 | 1.004× |
| `m3v12` | 0.34704 | 1.001× |

The proprio contrast is **1.008×**, so the n=8 claim of a 1.75× gap does not reproduce: **the
balance hypothesis as stated is refuted**, on the one arm the view draw cannot touch and
independently of the gate's instability. This *confirms* a suspicion already on record above
("what is solid here is the severed path, not the balance reading") rather than contradicting
anything. The control still works: `m3v12`'s image arm is 2.4–5.0e-05, ~256× below the others,
so the severed-path detector is intact at n=64.

#### Result 3 — the gate's instability is *not* the confound it was diagnosed as

`screen_conditioning.py` had two verified defects: it seeded `torch` only (so `np.random`, which
draws each sample's view subset, was never seeded), and it instantiated each cell's dataset from
that cell's own cfg — so `m3v13` drew `[1,3]` (mean 2.0) while `m3off` drew `[1,7]` (mean 4.0).
The second is *the same confound the project already corrected for `screen_collapse` on
2026-09-25*. Both were fixed (`np.random.seed(seed)`; an optional `--view-count-range`
mirroring `screen_collapse.py`'s, with the effective range recorded in the JSON), and the gate
was re-run at a **matched `[7,7]` draw** across all four ladder rungs × 3 seeds.

**The fix works and the diagnosis is wrong.** Same cell, same seed, same flags is now
*bit-reproducible* (0.00752622 twice, 6 s.f.) — but matching the draw **did not reduce the
spread at all**:

| cell | mean-N | image spread, unmatched | image spread, matched `[7,7]` |
|---|---|---|---|
| `m3v12` | 1.5 | — | 1.92× |
| `m3v13` | 2.0 | 2.47× | 2.09× |
| `m3off` | 4.0 | 2.27× | 2.50× |

So the unseeded draw was not the cause. What the matched control *did* establish is sharper
than "noisy": at a fixed `[7,7]` range `k` is always 7, so the RNG consumption is identical and
**every cell sees the same view order for a given seed** (verified directly) — the per-seed
comparison is genuinely matched, differing only in the weights. And under that matched
comparison **the rank order still flips**: the `m3v13`/`m3off` image ratio is 0.512 (s0),
0.272 (s1), 1.362 (s2). Same inputs, opposite conclusion. So *"image→action sensitivity" is not
a well-defined scalar for these policies* — it depends on the probe ensemble, and which cell
looks more image-driven depends on that ensemble. That is why this instrument cannot answer the
use question, and it is a stronger statement than a noise complaint.

The corresponding pre-registered prediction — `image_only` declines monotonically with mean-N
(`m3off` > `m3v15` > `m3v13` > `m3v12`) — is **falsified** on the `m3v15`/`m3off` pair
(measured 2.99e-02 / 1.84e-02 / 1.04e-02 / 3.5e-05). The named competing outcome, "flat across
cells", is **not** established either: 1.6× sits inside a 2× spread. The honest third outcome is
that the image arm is unresolvable at three seeds, and it is recorded as that rather than as
whichever of the two pre-registered branches reads better.

#### Result 4 — the balance question is dead ladder-wide, not just for `[1,3]`

The matched control was run on all four surviving rungs, and `proprio_only` is **flat across
the whole ladder**: means 0.347035 / 0.348878 / 0.347636 / 0.347262 for
`m3v12`/`m3v13`/`m3v15`/`m3off` — a **0.5% spread over four rungs**, with ≤0.4% across-seed
spread within each. Combined with Result 2, no rung leans harder on proprioception than any
other, so the axis the balance mechanism was proposed on does not vary across the ladder at all.

#### Conclusion

**The balance hypothesis is retired, and it was retired by a pre-registered gate rather than by
an argument.** The intervention it motivated was never launched. The measurement that killed it
is the one arm of the instrument that is draw-independent and reproducible to 0.3%.

**What survives, and what it costs the open question.** The session's negative results are
methodological and they narrow PLAN.md's rank-1 item rather than closing it. `image→action
sensitivity` — the quantity the "presence vs use" question most naturally reaches for — is
**not a scalar** for these policies: at matched inputs its rank order between cells flips across
probe ensembles. So the missing instrument cannot be built by tightening this statistic; it needs
a design whose verdict does not depend on which scenes are probed. That is now the concrete
specification of the open question, rather than a restatement of it.

**Limits.** One task (square), one seed per cell for the behaviour, three seeds for the screens,
one checkpoint each. The n=8 anchors are superseded as *method* (they came from the unseeded
path and are not reproducible under the fixed code) though their qualitative finding — `m3v12`'s
severed path — stands at n=64. The `image_only` spread is characterised on three seeds only;
a larger study could still bound it, but the rank flip at matched input says the problem is not
merely the size of the error bar.

### The `m3v15` latent gap, and the anti-correlation across the whole ladder

**Why.** `m3v15` (`[1,5]`, the rung that first broke the floor) was the **only** surviving cell
that had never been latent-probed — every latent table in this document is missing it. Free to
fill: the probe needs only the checkpoint. Artifacts: `data/probe_relpose_grid_square_m3v15/`,
`data/probe_zg_square_m3v15/`.

**Gates, all passing.** `draw_fingerprint` is **byte-identical** to all three existing grid
cells (`{'size_hist': {'1': 136, '2': 120}, 'first_views': [[2], [2, 12], [4, 8], [4, 8]]}`),
which is what makes the comparison valid rather than assumed; `mean_active` is **3.97**, so the
`[1,7]` override took (a silently-ignored one reads ~1.5).

**Result.** With the rung filled in, `[1,3]`'s anomaly is no longer an anomaly — it is the
middle of a **monotone trend**:

| cell | mean-N | `zg_across_view_subsets` `[1,7]` | `zv_pair_ratio` `[7,7]` | behaviour (trained) |
|---|---|---|---|---|
| `m3off` `[1,7]` | 4.0 | 0.1469 | 0.581 | **0.764** |
| `m3v15` `[1,5]` | 3.0 | 0.1131 | 0.491 | **0.456** |
| `m3v13` `[1,3]` | 2.0 | 0.0636 | 0.382 | 0.080 |
| `m3v12` `[1,2]` | 1.5 | 2.4e-06 *(collapsed)* | 1.277 *(degenerate)* | 0.028 |

**Conclusion.** As view diversity falls, `z_g` becomes **more view-invariant** and `z_v`
**less view-aware** — i.e. the latents move toward exactly what PROPOSAL §2.2 calls the goal —
while behaviour gets monotonically **worse**. So the project's central latent claim
*anti-correlates with success across the entire working range*, not merely in one anomalous
cell. That is a sharper framing of *the second failure mode*: the representational statistics
built here do not just fail to explain `[1,3]`, they point the wrong way across four
independently trained rungs.

**Limits.** Same instrument caveats as the rest of the probe section: decodability by a readout
is an upper bound on presence, the two latent columns are single statistics on one task, and
`m3v12`'s row is degenerate rather than informative. Read the **ordering** as the finding.

### `[2,7]` — min-N 2 matches `[1,7]`, so the availability of N=1 samples is not the ingredient

**Why this cell.** The ladder varied only the *upper* end of `view_count_range`, so every rung
kept N=1 in its range. That leaves "N>1 is needed" and "*variable* N is needed" inseparable —
and `[2,7]` is the only cell that separates them: **min-N 2, mean-N 4.5**, so it never trains at
N=1 at all. Its prediction was registered in advance: *if it works while `[1,2]`/`[1,3]`
collapse, the ingredient is mean view count rather than the presence of N=1.*

**Setting.** `m3_plucker_image_abs_multiview`, `use_plucker=false`, `use_eef_hist=false`, seed
42, 201 epochs — `m3off`'s exact configuration with one integer changed. Both presets, 50 paired
episodes. Artifacts: `data/eval_{interp,el}_square_m3v27/eval_log.json`,
`data/outputs/run_square_m3v27_s42_200ep/logs.json.txt`.

**Gates.** The 50 episode seed keys are **set-identical to `m3off`'s** (the mechanical proof the
episodes are paired); `|el_0 − az_0| = 0.080`, inside the band; the `[2,7]` override was verified
against 200 real draws (k = 4.31 per frame against an expected 4.5, with `[1,3]` reading 2.06
against 2.0 in the same check).

**Results.** Strict `success_rate`, 50 paired episodes:

| model | mean-N | trained | held-out | retention |
|---|---|---|---|---|
| `m3fixedn1` `[1,1]` | 1.0 | 0.036 | 0.047 | — |
| `m3v12` `[1,2]` | 1.5 | 0.028 | 0.043 | — |
| `m3v13` `[1,3]` | 2.0 | 0.080 | 0.073 | — |
| `m3v15` `[1,5]` | 3.0 | 0.456 | 0.373 | 82% |
| `m3off` `[1,7]` | 4.0 | 0.764 | 0.550 | 72% |
| **`m3v27` `[2,7]`** | **4.5** | **0.868** | **0.547** | **63%** |

`[2,7]` clears the floor cells by an order of magnitude, and against `[1,7]` it is
**indistinguishable on both axes**: Δ trained **+0.104**, inside the pre-registered 0.15 band,
and Δ held-out **+0.003**, dead even. The in-training curve agrees — az_0 `mean_score` reads
0.00 / 0.52 / 0.70 / 0.82 / 0.74 (mean 0.556) against `m3off`'s 0.496 and `m3v15`'s 0.244.
Elevation is the same shape too (0.800 / 0.220 / 0.040 against `m3off`'s 0.88 / 0.30 / 0.04),
including the holds-up-fails-down asymmetry.

**Conclusion 1 — the ladder's question is answered.** A model that **never sees N=1 during
training** performs the same as one that sees it with probability 1/7. So the ingredient is
having *enough views on average*, not the availability of single-view samples. Per the
pre-registered reading, "N>1 needed" is the losing branch.

**Conclusion 2 — and it corrects the project's own stated rationale.** `[2,7]` is evaluated at
**N=1 inference** like every other number here, and scores 0.547 held-out — the same as `[1,7]`.
So the N=1 inference path does not need N=1 training samples to work. M3's design rationale —
*"N is randomized so that the N=1 setting used by rollouts, evaluation, and M5 is in
distribution rather than a shift"* — is therefore **not necessary** for the capability it was
invoked to protect. That does not retroactively invalidate any measurement (every committed cell
did have N=1 in range, so nothing is confounded), but the *reason* given for the design is now
empirically refuted for the working regime. See *Record of corrections*.

**Limits.** One task, one seed, one checkpoint. And a confound this cell cannot remove:
mean-N 4.5 against 4.0 is ~12% more encoder compute per sample, so "more views" and "more
compute" are not fully separated — which means `[2,7]`*should* have been modestly ahead if view
count were the whole story, and it is not measurably ahead. The retention column also moves the
opposite way to the N=1 story: `[2,7]` retains **63%** against `[1,7]`'s 72%, extending the
pattern `[1,5]` started (82%) — more diversity buys absolute performance and costs
generalization.

### M1's baseline was **not** collapsed — view-tiedness and collapse are distinct failure modes

**Why.** `[1,2]`'s floor is an encoder collapse, and the same signature explains L1's task
split. That raised the question PLAN.md ranked second: **was the original single-view baseline
itself collapsed?** If so the collapse story would be *one* story running from M1 onward. M1's
weights were deleted on 2026-09-19, so the question needed a re-train (~3.3 h under two-run
contention), then a screen.

**Fidelity gate first, because the screen runs on a re-train and not the original.** The
re-train reproduces the original's *committed* rollout curve within **max |Δ| 0.06** (mean Δ
+0.000) across all five rollout epochs, with `val_loss` tracking to ~0.004 — well inside the
~0.14 per-viewpoint noise. So the re-train is a faithful reproduction, and the screen transfers
to the question about the original.

**Result.** `screen_collapse.py`, relative spread across 16 consecutive states, against a
random-init baseline **measured on this cell rather than borrowed**:

| | relative spread | vs its own random init |
|---|---|---|
| M1 re-train, random init | 6.97e-03 | — |
| **M1 re-train, trained** | **0.1221** | **17.5× ABOVE** |
| *L1 square (fails, collapsed)* | *1.49e-04* | *84× BELOW* |
| *L1 lift (works)* | *2.49e-02* | *2.0× above* |

**Conclusion.** M1's encoder is **emphatically not collapsed** — 17.5× above its own baseline,
and higher in absolute terms than even the working L1 lift. So M1's failure at ±15° is **not**
the collapse failure mode: it is view-tiedness with a healthy, richly-varying encoder.
**View-tiedness (M1) and collapse (L1 square/can, `[1,2]`) are distinct**, and the collapse story
is not "one story from M1 onward". PLAN.md's rank-2 item closes as a **negative** — which is the
useful direction, because it stops two unrelated failures being folded into one narrative.

**Limits.** One task, one seed; the screen is 16 frames and is correlation, not causation; and
the re-train is a reproduction rather than the original artifact, bounded by the fidelity gate
above rather than by identity. This M1's random-init baseline is 6.97e-03 against L1 square's
1.25e-02 — same encoder family, different data source (hdf5 against the multiview zarr), which
is precisely why the rule is to measure the baseline and never borrow it.

### Code map

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

## Appendix A — Index of results tables

Where every table in the two documents lives, and the committed artifact behind it. All sweep
artifacts are `eval_log.json` under the named `data/eval_*/` directory (plus `media/*.mp4` only
where `--n-test-vis > 0` kept videos). The **M1–M4 tables are in `PROGRESS.md`**; everything
else is in this file.

### In `PROGRESS.md`

- **M1** — `success_rate` and `mean_score` for square, can, lift at az 0/±15°/±30°
  (`azimuth_sweep5`): `data/eval_az5_{square,can,lift}_ep200/`.
- **M2** — the generation table (steps, images, gates, time, disk): from the generation logs;
  the **N=1 fidelity gate** on lift, five viewpoints against M1: `data/eval_az5_lift_n1gate/`.
- **M3** — the N=1 fidelity gate (single slot, `view_pool=[6]`):
  `data/eval_gate_n1gate/`; the square 14-column grid and the can grid, both presets:
  `data/eval_interp_square_m3{on,off,plucker,eef}/`, `data/eval_el_square_m3*/`,
  `data/eval_interp_can_m3{on,off}/`, `data/eval_el_can_m3{on,off}/`; the per-cell 3-viewpoint
  gates: `data/eval_gate_{square,can}_m3*/`; the M1/L1 elevation table:
  `data/eval_el_{square,can,lift}_{abs_single,randview}/`.
- **M4** — the m3/m4 trained–held-out–elevation table:
  `data/eval_{interp,el}_square_m4{on,off}/`. The `aux_loss` decay and the EMA weight-drift
  table are training-log and probe readouts, not sweeps: `data/probe_m4_square_logs.json.txt`.

### In this file

- **L1** — the 11-viewpoint table for lift, can, and square s42/s43:
  `data/eval_interp_{lift,can}_randview*`, `data/eval_interp_square_randview{,_s43}/`, and the
  `data/eval_el_*randview*/` elevation counterparts.
- **N>1** — the in-training az_0 `mean_score` curve (from the runs' `logs.json.txt`) and the
  strict sweep:
  `data/eval_interp_square_m3fixedn1/`, `data/eval_el_square_m3fixedn1/`.
- **Relational probe (step 1a)** — the six-cell readout table and the superseded
  `latent_stats`: `data/probe_relpose_square_{m3on,m3off}/probe_relpose.json`.
- **N-diversity ladder** — Stage 1's in-training curve and 11-viewpoint table, Stage 2's table,
  Stage 3's curve and the rung summary:
  `data/eval_{interp,el}_square_m3v{12,13,15}/`, with each rung's
  `data/outputs/run_square_m3v1*_s42_200ep/logs.json.txt`.
- **Collapse is a failure mode** — the matched `[7,7]` screen table:
  `data/screen_collapse/*.json` (trained) with the per-architecture controls in
  `*_RANDOM_INIT.json`.
- **The second failure mode** — all three stages:
  `data/probe_relpose_grid_square_*` (`z_v`), `data/probe_zg_square_*/probe_relpose.json` —
  the number is `latent_grid["1,1"].zg_abs_pose` (`z_g`) — and
  `data/screen_conditioning/square_*_latest.json` (the action head).
- **Proprioception dropout** — the gate table, the `proprio_only` table and the matched-draw
  spread table: `data/screen_conditioning/square_*_n64_s*.json` and
  `square_*_n64_matched77_s*.json`.
- **The `m3v15` latent gap** — the monotone trend table:
  `data/probe_relpose_grid_square_m3v15/`, `data/probe_zg_square_m3v15/`.
- **`[2,7]`** — the rung summary table: `data/eval_{interp,el}_square_m3v27/`, with
  `data/outputs/run_square_m3v27_s42_200ep/logs.json.txt`.
- **M1's baseline was not collapsed** — the screen table:
  `data/screen_collapse/square_abs_single_retrain_latest.json` (and `_RANDOM_INIT`), bounded by
  the re-train's fidelity gate against the committed
  `data/outputs/run_square_abs_single_*/logs.json.txt`.

**Provenance caveat, recorded because it is real.** These artifacts store no record of which
checkpoint produced them — `data/eval_*/` holds `eval_log.json` and nothing else — so which
checkpoint a committed number came from is recoverable only from the runbook convention
(`NOTES.md`) and the directory name. The one place this bit is written up is the ladder's
*Protocol note*, where a run's checkpoint directory held two different models.

## Appendix B — Parked investigations

Five lines of work that are not running and not planned. Each was parked by a measurement, not
by cost or by taste, and each records what would revive it — so the same idea is not
re-litigated from scratch, and a fix that has already been tried is not attempted twice.

### B1 — Relational supervision on `z_v` and `z_g` (steps 1b/1c)

**The claim.** A per-view relative-pose head (1b) and a predicted-relation attention bias into
fusion (1c) would *make geometry necessary* rather than better-injected — the lever `PLAN.md`
identified after M3/M4's conditioning null.

**What parked it, 2026-09-24.** Step 1a measured that the geometry 1b would supervise is
**already in `z_v`**: `m3off`, with no conditioning at all, recovers the relative pose between
two views to 9.34° / 15.36 cm against a 65.35° / 55.75 cm floor. So a head alone is a post-hoc
fit — the pre-registered decision rule fired. Structurally, the action head never sees `z_v`
(`forward` hands the UNet `z_g` alone), so per-view supervision must reach behaviour through
the fusion bottleneck, and at N=1 there is no pair to relate at all.

**What would revive it.** 1a also found headroom the rule did not anticipate: `m3on` is 4.6×
better at relative-pose rotation and 2.1× more view-discriminative, so a relational
*objective* has somewhere to act. 1c's attention bias is the live candidate, because each slot
is currently encoded alone — the model has no explicit view-to-view geometry anywhere.

**Reasoning:** `PLAN.md` *Next: relational supervision*; this file, *Relational probe (step 1a)*.

### B2 — The balance hypothesis, and its proprioception-dropout intervention

**The claim.** `[1,3]`'s floor came from leaning ~1.75× harder on proprioception than the
working cell (an n=8 reading), and proprioception cannot see where the nut is. The
falsifying intervention was coded and left ready (`DiffusionUnetImagePolicyPropDrop`, dropping
proprioception for a random half of training samples, `compute_loss` only).

**What parked it, 2026-09-25 — retired *unlaunched*, by its own gate.** Re-measured at
n=64 × 3 seeds: the gate failed (min ratio 0.567; 2-of-3 seeds above the 1.5× bar) and the
hypothesis was refuted outright on the draw-independent arm — `proprio_only` is 1.008× between
`m3v13` and `m3off` and flat across all four rungs (0.5% spread). By pre-registration the run
was never launched, which is the point of registering it.

**What would revive it.** A mechanism that predicts a proprio-usage *difference* between
cells — this one does not, ladder-wide.

**Reasoning:** this file, *Proprioception dropout*; launch trap for the `+policy.proprio_dropout`
override in `NOTES.md`.

### B3 — Tightening `image→action sensitivity`

**The claim.** The "presence vs use" question could be answered by a better-conditioned version
of the screen's image arm — the statistic the missing instrument most naturally reaches for.

**What parked it, 2026-09-25.** Both diagnosed defects were real and both were fixed (the
unseeded numpy view draw; the per-cell draw ranges). The tool is now bit-reproducible to six
figures — and the spread was **unchanged** (1.92–2.50×). At a matched `[7,7]` draw every cell
sees the same view order for a given seed, and under that matched comparison the **rank order
still flips across seeds** (image ratio 0.512 / 0.272 / 1.362). So `image_only` is not a scalar
for these policies; which cell looks more image-driven depends on the probe ensemble.

**What would revive it.** Nothing, as a statistic. It is parked permanently, and recorded so
the fix is not attempted twice: the missing instrument needs a design whose verdict does not
depend on which scenes are probed.

**Reasoning:** this file, *Proprioception dropout* Result 3, and the 2026-09-25 corrections.

### B4 — The invariance-pressure (canonicity) account of collapse

**The claim.** Seven views of one scene press the shared encoder toward a canonical,
scene-level representation that two views do not — and a fusion-free view-invariance statistic
(`zv_pair_ratio`) would separate the working cell from the floor cell.

**What parked it, 2026-09-25 — unreadable on `[1,2]`.** The pre-registered reading said a ratio
*above* the working cell's would confirm the hypothesis, and `m3v12` reads 1.277 against
`m3off`'s 0.581 — but both numerator and denominator are ~1e-5, because a constant
representation has neither view-distance nor state-distance to measure. The random-init control
is what caught it (a healthy encoder sits at 1.51, a working trained one at 0.58), so the
correct reading is the pre-registered **unreadable** branch, not confirmation.

**What replaced it.** "Large-N sampling *prevents collapse*" rather than "large-N supplies
canonicity pressure" — simpler, better supported, and it reframes the N>1 headline without
overturning it. **What would revive it:** a non-degenerate floor cell to test the statistic on.

**Reasoning:** this file, *Collapse is a failure mode* → *`[1,2]`'s encoder has collapsed*.

### B5 — M5's distillation stage

**The claim.** A single-view student encoder regressing the frozen teacher's multi-view fused
latent `z_g` — the optional second half of M5.

**What parked it.** The premise is unestablished. The capability lives in the *training signal*
— a model trained with one view per sample scores L1's floor — so a student would be distilling
a latent whose advantage comes from training-time sample diversity, not from inference-time
computation. M3 already performs N=1 novel-view inference at 0.55 held-out, so distillation
has to beat *that*, and no result so far shows the fused latent carries anything the N=1
encoder's own output does not.

**What would revive it.** A measurement showing `z_g` at N=1 predicts behaviour better than the
N=1 encoder's own output — a probe that can fail. The gate if it is built: final sweep tables
against M1's degradation curves.

**Reasoning:** `PLAN.md` *M5 — single-novel-view inference*; `PROGRESS.md` *Open questions*.

## Appendix C — Record of corrections and superseded numbers


Dated, newest first. These are kept because a document that quietly repairs its own headline
is worth less than one that shows the repair — read them before quoting any number they
touch.

- **2026-09-25 — M3's stated reason for randomising N is empirically refuted.** This document
  has said, since M3 and again in *N-diversity ladder* ("why every rung keeps N=1 in the
  range"), that randomising N keeps the N=1 corner *in distribution* rather than a shift.
  `[2,7]` never trains at N=1 and infers at N=1 as well as `[1,7]` does — 0.547 held-out against
  0.550 — so the N=1 inference path does not require N=1 training samples, and the rationale is
  not necessary for the capability it was invoked to protect. **No measurement is invalidated**
  (every committed cell did have N=1 in range, so nothing was confounded by the belief), but the
  *reason* is superseded. Note the related earlier step: `[1,2]`'s secondary result had already
  killed the out-of-distribution explanation for a `[2,2]`-style floor.
- **2026-09-25 — "the collapse story is one story from M1 onward" is wrong.** M1's re-trained
  encoder screens at 0.1221, **17.5× above** its own random-init baseline, so the original
  single-view baseline was *not* collapsed. M1's failure at ±15° is view-tiedness with a healthy
  encoder, which is a **different** failure mode from L1 square/can and `[1,2]`. Any sentence
  that folds M1 into the collapse account is superseded. The re-train's fidelity to the deleted
  original is bounded by the curve comparison in that section (max |Δ| 0.06), not by identity.
- **2026-09-25 — "N>1 needed" vs "*variable* N needed" is answered: neither the *presence* of
  N>1 nor the availability of N=1.** `[2,7]` (min-N 2, mean-N 4.5) is indistinguishable from
  `[1,7]` on both axes (Δ trained +0.104, inside the pre-registered 0.15 band; Δ held-out
  +0.003), while clearing the floor cells by an order of magnitude. The ingredient is having
  enough views on average. The section *N-diversity ladder* framed this as open; it is closed,
  with the confound that mean-N 4.5 against 4.0 also carries ~12% more compute per sample.
- **2026-09-25 — the balance hypothesis is refuted and its intervention retired at the gate.**
  The propdrop run was never launched. The n=8 reading (proprio 0.0674 vs 0.0386, a 1.75× gap)
  does not reproduce at n=64: the draw-independent `proprio_only` is **1.008×** between
  `m3v13` and `m3off`, and **flat across all four rungs** (0.5% spread). Anything in this
  document that treats "`[1,3]` leans harder on proprioception" as a live mechanism is
  superseded. See *Proprioception dropout — the `[1,3]` balance test*.
- **2026-09-25 — this session's own diagnosis of the screen's instability was wrong, and its
  own control is what caught it.** The n=64 gate's spread was diagnosed as the unseeded numpy
  view draw and the per-cell draw range, both recorded in `NOTES.md` and committed as
  pre-registration *before* the control existed. The matched `[7,7]` re-run left the spread
  **unchanged** (1.92–2.50× against 2.27–2.47×), so neither was the cause — the seeding fix
  made the tool bit-reproducible but did not buy comparability. The explanation that survives is
  that at matched inputs the *rank order between cells flips across seeds* (ratio 0.512 / 0.272
  / 1.362), i.e. `image_only` is ensemble-dependent rather than a scalar. The prediction
  registered alongside the fix (monotone decline with mean-N) is falsified, and its named
  competing outcome ("flat") is not established either.
- **2026-09-25 — the n=8 conditioning anchors are superseded *as method*.** They came from the
  unseeded code path, so they are not reproducible under the fixed tool even at identical flags.
  Their qualitative finding survives at n=64: `m3v12`'s image path is severed (2.4–5.0e-05,
  ~256× below the others). The absolute values 0.0067/0.0068/0.0674/0.0386/0.0686 should not be
  quoted or compared against any n=64 number — the arm samples `dataset[0..n-1]` and the reading
  scales with n (proprio moved 5.2× from n=8 to n=64).
- **2026-09-25 — `m3v15`'s absent latent reading is filled**, and it changes the `[1,3]` story
  from an anomaly into a monotone trend across four rungs: as mean-N falls 4.0 → 3.0 → 2.0,
  `z_g` becomes *more* view-invariant and `z_v` *less* view-aware while behaviour gets worse. Any
  earlier sentence framing the latent/behaviour mismatch as particular to `[1,3]` is superseded
  by the ladder-wide version.

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
