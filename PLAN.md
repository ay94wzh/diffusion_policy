# Plan: view-aware policy, milestones M1–M5

Direction and the original predictions: `PROPOSAL.md`. Results: `PROGRESS.md`.
Operational detail: `NOTES.md`. Last updated 2026-09-24.

## Constraints

- This is a fork of the original diffusion_policy repo — **modify it as little as
  possible**. Each milestone lands as **new files**; existing files are touched only at
  documented seams and only when unavoidable.
- Baselines first, then one part added at a time. Models are planned here, coded in their
  own milestone, and each milestone reads its result before the next one is committed to.
- Runs happen on the box, and **this checkout is the box** — verified 2026-09-24: 2× RTX
  5090, the `run_square_m3*` checkpoints and `data/multiview/*_ring13.zarr` present
  locally — so the runbooks in `NOTES.md` execute in place from the repo root. Disk is the
  binding constraint: check `df -h` first, it has sat at 97–99%.

## Status

| milestone | status | the number that decided it | detail |
|---|---|---|---|
| **M1** — single-view baseline + novel-view harness | ✅ done | 0.82/0.98/0.76 at az_0 → ≈0 at ±15° | PROGRESS *M1* |
| **M2** — multi-view data (13-pose ring) | ✅ done | 819k images, 4.5 GB, N=1 gate reproduces M1's curve | PROGRESS *M2* |
| **L1** — view diversity only | ✅ done | lift 0.76–0.96 to ±75°; square/can ≤0.08 | PROGRESS *L1* |
| **M3** — view-conditioned encoder + fusion | ✅ done | 0.50–0.57 / 0.74–0.77 held-out on square/can; conditioning inert | PROGRESS *M3* |
| **M4** — per-view aux action heads | ✅ done | `aux_loss` 52× down; `m4on − m4off` +0.02/+0.04 | PROGRESS *M4* |
| **architectural confound** | ✅ resolved | fixed-N=1 scores 0.04/0.05 → **N>1** is load-bearing | PROGRESS *N>1* |
| **M5** — single-novel-view inference | ⬜ not coded | — | below |
| **relational supervision** (`z_v`, `z_g`) | ⬜ retired by 1a's result — not cost | 1a measured the geometry 1b would supervise is *already* in `z_v` | *Next* above |
| **N-diversity ladder** | 🟡 Stage 1 `[1,2]` running | pre-registered: trained mean 0.15–0.40 | PROGRESS *N-diversity ladder* |

## Next: relational supervision on `z_v` (opened 2026-09-24, **retired 2026-09-24**)

> **Retired on reading 1a, not on cost.** 1a measured that the geometry 1b would supervise is
> *already* in `z_v` — the exact shape of intervention M4 found behaviourally null. And the
> action head never sees `z_v`: `forward` hands the UNet `z_g` alone, so any per-view
> supervision must reach behaviour through the fusion bottleneck. 1c is worse structurally:
> fusion is a single learnable query, so there is no view-to-view term to bias, and at N=1
> there is no pair at all. The section is kept for the reasoning, not as pending work. What
> replaced it is the *N-diversity ladder* (PROGRESS.md).

**The governing fact.** The scene is static, so the camera pose is recoverable from the
RGB itself and a better-injected Plücker map carries no information the image lacks. That
is the mechanism behind the measured conditioning null (KYC reports the same: static
scenes leak pose through background cues; randomizing appearance is what makes explicit
conditioning pay). So the lever is not *how* geometry is injected but **making geometry
necessary** — supervise each latent in the frame it is supposed to represent:

| latent | should be | supervised by |
|---|---|---|
| `z_v` | **view-aware** | a *relational* objective — predict `(R_ij, t_ij)` between two views |
| `z_g` | **view-invariant** | *agreement* — across views, and across the action produced |

Note the classical multi-view recipe (TCN) pulls simultaneous views *together*, i.e. it
pushes invariance: right for `z_g`, wrong for `z_v`, which would delete the geometry.

**Step 1 — relational supervision on `z_v`.**

| # | what | gate before it counts |
|---|---|---|
| 1a | **probe on frozen checkpoints** — is the geometry already decodable from `z_v`? | ✅ **done 2026-09-24** — beats both controls in all six cells |
| 1b | a relative-pose head + loss; the target is derived from the in-batch cam keys, so **no dataset change** | loss must reach well below the M4-style mean-collapse floor *before* any rollout |
| 1c | feed the **predicted** relation into fusion as an attention bias | stays permutation-invariant, degenerates correctly at N=1 |

**1a's answer, and what it does to 1b.** Yes — and *without any pose conditioning*. `m3off`
(no Plücker, no EE history) recovers the relative pose between two views to **9.34° / 15.4 cm**
against a 65.35° / 55.75 cm mean-predictor floor, and its own camera's absolute pose to
**4.13°** against 37.94°. So the pre-registered rule fires: **a head alone would be a
post-hoc fit**, and the objective is what has to change. But the probe also found the part
the rule did not anticipate — `m3on` is **4.6× better** (2.02°) and its `z_v` is **2.1× more
view-discriminative** (0.466 vs 0.224), so the Plücker path is **not inert in the latent**,
only in the behaviour. Two consequences for 1b/1c:

- The last cell of the original design — relpose-on + `use_plucker=False` — is no longer the
  "first mechanistic live-ness test the conditioning has had"; 1a already supplied that, and
  supplied it *positively*. 1c's attention bias is the more interesting target now, because
  the model still has no explicit **view-to-view** geometry (each slot is encoded alone).
- 1b's claimed value has to shift from "supplies information" to "organizes information the
  encoder already has into a frame the policy can use". A head that only re-derives what the
  MLP readout above already reads is the M4 result again at a different layer.

Cells: relpose-on/off (RNG-locked), and relpose-on + `use_plucker=False` to test whether the
objective can substitute for the ray map. Runbook: `NOTES.md`.

**Ranked behind it.**

- **Q1 — Plücker usage.** A separate ray stem, multi-scale/adapter injection, and
  Plücker-into-the-policy are cheap and *predicted null* in a static scene; the two with a
  real justification are the **pairwise relative-pose attention bias** (fixes the structural
  gap that the model currently has no view-to-view geometry at all — delivered by 1c) and
  **ray-conditioned view prediction** (makes the rays necessary), the strongest non-null
  candidate.
- **Q2 — camera pool.** Discrete ring → **dense continuous pose distribution**; InfiNoVA
  (2026) does exactly this and reports 5.4× VISTA augmentation and 1.7× better than five
  physical cameras. **Trap: N=1 per sample is L1, which already fails** — the pool must keep
  N>1. Probe it for free first with *contiguous-window* sampling from the existing ring; the
  full version holds out **regions** (elevation band / azimuth wedge), not poses. Costs:
  render time and disk (~8 GB at 65 poses for square), not IO.
- **Q3 — remaining constraints.** Epipolar/Plücker reprojection consistency (depth-free,
  calibration only) as the geometric companion to 1b; cross-view **action** consistency (the
  literature's version, with diffusion heads untested — but ±7-8pp is below the n=1 noise
  floor, so 2 seeds or a wider shift must be decided *before* running); counterfactual
  enforcement (MemCorr) as the antidote to ignored conditioning.

Depth stays out of scope for now (RGB-only): the cost is that simulator-verified pixel
correspondences are unavailable, leaving the epipolar form above.

## M5 — single-novel-view inference (+ optional distillation)

**Most of it already exists, and that is the thing to decide about.** Fusion accepts
N=1 and M3 was trained with randomized N precisely so that N=1 inference is *in
distribution*: `eval_novel_view.py --m3-slots K` publishes the perturbed camera pose
(read back from the simulator) into a single slot, and that path is already what produced
every M3 number — square infers at N=1 with 0.55 success on held-out azimuths. So the
capability M5 was written to deliver is measured, not pending.

What remains is the **optional distillation stage**: a single-view student encoder
regressing the frozen teacher's multi-view fused latent `z_g`.

**Premise, and why it should be re-examined first.** Distillation is only worth running
if the fused latent carries something the single-view path cannot. The N>1 result shows
the capability lives in the *training signal* — a model trained with one view per sample
scores L1's floor — so the student would be distilling a latent whose advantage comes from
training-time sample diversity, not from the inference-time computation. No result so far
demonstrates that a student would gain anything. Before coding it, the deciding
measurement is whether `z_g` at N=1 (single novel view) predicts behaviour better than the
N=1 encoder's own output — pick a probe that can fail.

**Gate if it is built:** final sweep tables against M1's degradation curves — the
reference the whole project exists to beat.

## Cheap open runs

Deferred deliberately, each one config line or one rerun. Listed in the order they would
add information per GPU-hour; full rationale in `PROGRESS.md`'s open questions.

| run | answers | cost |
|---|---|---|
| ~~`view_count_range=[2,2]`~~ | superseded — now a conditional Stage 2 *confound probe* (it never trains at N=1, so a floor is unreadable on its own) | ~3 h |
| `view_count_range=[2,7]` | "N>1 needed" vs "*variable* N needed" — the only cell that answers it directly | **~4 h, not ~2 h** (mean active N = 4.5 exceeds `m3off`'s 4.0) |
| `view_count_range=[1,2]` / `[1,3]` / `[1,4]` | how much diversity is enough (PROPOSAL §7), with N=1 in-distribution at every rung | ~2.8 / ~3.0 / ~3.2 h all-in |
| lift + `m3on` | do we regress the one task L1 already solves | ~1 h |
| second seed on one M3/M4 cell | whether the nulls hold at a resolution better than n=1 | ~1 h each |
| ±60° training pool | how much view *quality* alone buys | one render + run |
| can `m3plucker` / `m3eef` | the cheap way back into the conditioning question | ~2 h each |

## Out of scope for now

ACT. The mujoco pipeline — including the known normalizer bug at
`mujoco_image_dataset.py:64`, which fits `agent_pos` on `tcp_xyz_wxyz` concatenated with
*itself* (14 dims) instead of with `gripper_width`, and will fail against the 8-dim
`agent_pos` the dataset produces.
