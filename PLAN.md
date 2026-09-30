# Plan: view-generalizable policy

Direction and the original predictions: `PROPOSAL.md`. The module, its measured behaviour and
the open questions: `PROGRESS.md`. Operational detail, runbooks and traps: `NOTES.md`.
Last updated 2026-09-30.

## Constraints

- This is a fork of the original diffusion_policy repo — **modify it as little as
  possible**. Each milestone lands as **new files**; existing files are touched only at
  documented seams and only when unavoidable.
- Baselines first, then one part added at a time. Models are planned here, coded in their
  own milestone, and each milestone reads its result before the next one is committed to.
- **Where things run.** This checkout is the **coding and documents** machine: code, documents
  and the committed records (`data/eval_*/eval_log.json`, `data/outputs/run_*/logs.json.txt`,
  `data/screen_*`, `data/probe_*`) live here. The demonstrations, the multi-view zarrs and every
  checkpoint live on the **remote training machine**. Runs are prepared here and launched there;
  the results come back as `eval_log.json` + `logs.json.txt` and are written into `PROGRESS.md`.
  Check disk on the training machine before a campaign — a checkpoint is 4.6 GB and `topk.k=1`
  roughly doubles a run's footprint.
- Every run uses **201 epochs, not 200**: checkpoint and rollout fire on `epoch % 50 == 0` and
  there is no save at the end of training, so 201 makes epoch 200 fire.

## Status

Numbers and their provenance live in `PROGRESS.md`; this table is verdicts only.

| milestone | status | verdict |
|---|---|---|
| **M1** — single-view baseline + novel-view harness | ✅ done | collapses to ≈0 at ±15° azimuth |
| **M2** — multi-view data (13-pose ring) | ✅ done | validated; ±75°/±90° views are low value |
| **L1** — view diversity only | ✅ done | solves lift, destroys square/can |
| **M3** — view-conditioned encoder + fusion | ✅ done | solves square/can at held-out views; conditioning inert |
| **M4** — per-view aux action heads | ✅ done | mechanism real, behaviourally null |
| **architectural confound** | ✅ resolved | multi-view sampling is the load-bearing ingredient |
| **N-diversity ladder** | ✅ closed at five rungs | a knee, then a graded rise; `[1,5]` is the more view-general working cell |
| **`[2,7]`** | ✅ done | enough views *on average* is the ingredient; N=1 inference needs no N=1 training samples |
| **M1 re-screen** | ✅ done | the single-view baseline was **not** collapsed — view-tiedness ≠ collapse |
| **M5** — single-novel-view inference | 🟡 capability **measured**, not pending | every M3 number is already an N=1 inference at a novel pose; only the optional distillation stage is uncoded |
| parked (encoder collapse, the `[1,3]` second failure mode, the balance refutation, the `m3v15` gap, instrument diagnostics) | ⬜ stopped, findings kept | reasoning and revival conditions at git tag `docs-full-20260930` |

## Main line — the three runs and the M5 write-up

Each run fills a hole the record itself names. All three are prepared here and launched on the
training machine; runbook commands, launch gates and epoch-time checks are in `NOTES.md`.

| # | run | the hole it fills | config delta | ~cost |
|---|---|---|---|---|
| a | **lift + `m3on`** | M3 was never swept on lift — the one task L1 already solves, so this is a "did we break it" question, not a result | `task=m3_plucker_image_abs_multiview task.task_name=lift`, `use_plucker=true use_eef_hist=true` (= `m3on`), K=7, `[1,7]`, seed 42 | 46 min |
| b | **second seed on `[1,5]`** | the working rung is n=1 and every behavioural null in the project is n=1 | `m3off` config with `view_count_range="[1,5]" training.seed=43` | 2.0 h |
| c | **±60° pool at `[1,5]`** | "how much does view *quality* alone buy" — never measured | **new task yaml** (below) | 2.0 h |

**Run (c), precisely.** Ring index 0 = az −90°, step 15°, so index 12 = +90°; dropping the two
±90° views (M2's low-value finding) means `view_pool=[2,4,6,8,10]` = (−60, −30, 0, +30, +60).
The dataset enforces `n_slots <= len(view_pool)`, so this **cannot** be a CLI override of the
7-slot config: it needs a new file
`diffusion_policy/config/task/m3_plucker_image_abs_multiview_pm60.yaml` keeping **five** rgb slot
keys, `view_count_range: [1,5]` and `env_runner.m3_slots: 5`. At `[1,5]` its mean active N is
**3.0 — identical to `m3v15`**, which is what makes it an isolation of view quality from view
count rather than a second diversity experiment. *(Likeliest mechanical error: dropping indices
10/12 instead of 0/12, i.e. removing +60/+90 instead of ±90.)*

**Reading the three, fixed in advance.** Run (a) fails if any held-out azimuth lands below both
L1's band (0.76–0.96) and its own M1 reference — it is a regression test with a named falsifier.
Run (b)'s read is the **spread across the two seeds of one config**; if the trained means differ
by more than the 0.15 band, the ladder's five-rung curve is re-read as two-population and the
knee claim is restated at the new resolution — a registered consequence, not an after-the-fact
rescue. Run (c) is compared against `m3v15` at matched mean-N.

**Then the M5 write-up** — assembled from committed `eval_log.json` alone (no GPU): the M1 vs L1
vs M3 degradation comparison across all three tasks, with lift filled in by run (a). See
`PROGRESS.md` *M3* and *`[2,7]`* for the two capability statements it rests on.

**Also open and cheap:** can `m3plucker` / `m3eef` — now a **re-train**, their checkpoints were
deleted 2026-09-24. The way back into the conditioning question on a second task, already
answered on square. ~2 h each.

## M5 — single-novel-view inference (+ optional distillation)

**Most of it already exists, and that is the thing to decide about.** Fusion accepts N=1 and the
N=1 inference path is already what produced every M3 number: `eval_novel_view.py --m3-slots K`
publishes the perturbed camera pose (read back from the simulator) into a single slot. Square
infers at N=1 with 0.55 success on held-out azimuths. So the capability M5 was written to deliver
is **measured, not pending**.

What remains is the **optional distillation stage**: a single-view student encoder regressing the
frozen teacher's multi-view fused latent `z_g`.

**Premise, and why it should be re-examined first.** Distillation is only worth running if the
fused latent carries something the single-view path cannot. The N>1 result shows the capability
lives in the *training signal* — a model trained with one view per sample scores L1's floor — and
`[2,7]` shows N=1 *inference* works without ever training at N=1. No result so far demonstrates
that a student would gain anything. Before coding it, the deciding measurement is whether `z_g` at
N=1 (single novel view) predicts behaviour better than the N=1 encoder's own output — pick a probe
that can fail.

**Gate if it is built:** final sweep tables against M1's degradation curves — the reference the
whole project exists to beat.

## Deferred — training distribution

The next model update is a **training-distribution** change, not an architecture one: the ladder
showed view count and view quality are what move behaviour, while the conditioning, the aux heads
and the encoder architecture at N=1 are all inert. Ranked, none scheduled.

1. **Dense / continuous camera-pose pool** — sample poses from a continuous distribution and
   hold out *regions* (an elevation band or azimuth wedge), not poses. InfiNoVA (2026) does this
   and reports 5.4× VISTA augmentation and 1.7× better than five physical cameras. Costs a render
   and disk (~8 GB at 65 poses for square), not IO. **Trap: N=1 per sample is L1, which already
   fails** — the pool must keep N>1.
2. **Contiguous-window probe first** — free, from the existing ring: *contiguous* azimuth windows
   instead of uniform subsets approximate a dense pool's local structure with no new render. The
   cheap test of whether (1) is worth its render.
3. **A dense-pool experiment proper** needs **two arms** (tight-window pool vs a matched baseline
   on the same rebuilt pipeline) — new-distribution numbers cannot be compared against committed
   ones without confounds.
4. **A run with per-epoch checkpoints** — the only way to order collapse against the behavioural
   failure; needs a light in-loop hook saving latents (~3.7 MB/epoch), not checkpoints (4.6 GB).
5. **An instrument sensitive to *correctness*, not presence** — the live specification of the
   open question: its verdict must not depend on which scenes are probed, because
   `image→action sensitivity` is not a scalar. Design work, no GPU.

## Out of scope for now

ACT. The mujoco pipeline — including the known normalizer bug at
`mujoco_image_dataset.py:64`, which fits `agent_pos` on `tcp_xyz_wxyz` concatenated with
*itself* (14 dims) instead of with `gripper_width`, and will fail against the 8-dim
`agent_pos` the dataset produces.
