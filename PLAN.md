# Plan: view-aware policy, milestones M1–M5

Direction and the original predictions: `PROPOSAL.md`. Results: `PROGRESS.md`.
Operational detail: `NOTES.md`. Last updated 2026-09-24.

## Constraints

- This is a fork of the original diffusion_policy repo — **modify it as little as
  possible**. Each milestone lands as **new files**; existing files are touched only at
  documented seams and only when unavoidable.
- Baselines first, then one part added at a time. Models are planned here, coded in their
  own milestone, and each milestone reads its result before the next one is committed to.
- Every training run happens on the remote box (see `NOTES.md`); this checkout is for
  code and documents.

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
| `view_count_range=[2,2]` | how much diversity is enough (PROPOSAL §7's own question) | ~2 h |
| `view_count_range=[2,7]` | "N>1 needed" vs "*variable* N needed" | ~2 h |
| lift + `m3on` | do we regress the one task L1 already solves | ~1 h |
| second seed on one M3/M4 cell | whether the nulls hold at a resolution better than n=1 | ~1 h each |
| ±60° training pool | how much view *quality* alone buys | one render + run |
| can `m3plucker` / `m3eef` | the cheap way back into the conditioning question | ~2 h each |

## Out of scope for now

ACT. The mujoco pipeline — including the known normalizer bug at
`mujoco_image_dataset.py:64`, which fits `agent_pos` on `tcp_xyz_wxyz` concatenated with
*itself* (14 dims) instead of with `gripper_width`, and will fail against the 8-dim
`agent_pos` the dataset produces.
