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
- **Where things run.** **This box is the training machine** (2× RTX 5090); the code, the
  committed records, the multi-view zarrs and the weights are all here. Runs are launched here
  and written into `PROGRESS.md` here. Check disk before a campaign — a checkpoint is 4.6 GB and
  `topk.k=1` roughly doubles a run's footprint — and check the right volume: `/` is chronically
  near-full, so run dirs go to `/data/zihan/runs/` with only `logs.json.txt` copied back.
- Every run uses **201 epochs**: checkpoint and rollout fire on `epoch % 50 == 0` 

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
| **(a) lift + `m3on`** | ✅ done (recovered 2026-10-01) | **does not break lift** — held-out 0.870 vs L1's 0.873; elevation is where it gives ground |
| **(b) `[1,5]` seed 43** | ✅ done (recovered 2026-10-01) | **Δ 0.052 trained / 0.083 held-out, inside the band** — the knee stands at two seeds |
| **(c) ±60° pool** | ✅ done (recovered 2026-10-01) | **pool curation is worth +0.272 / +0.160 at fixed mean-N** — reaches `[1,7]` at 2/3 its mean-N; size vs content confounded |
| **clear run** (per-epoch latents) | ✅ done 2026-10-01 | first per-epoch latent series (201 epochs, clean); latent statistics saturate by ~epoch 50–75 while behaviour improves to ~150 — the encoder is decided early |
| **M5** — single-novel-view inference | 🟡 capability **measured**, not pending | every M3 number is already an N=1 inference at a novel pose; only the optional distillation stage is uncoded |
| parked (encoder collapse, the `[1,3]` second failure mode, the balance refutation, the `m3v15` gap, instrument diagnostics) | ⬜ stopped, findings kept | reasoning and revival conditions at git tag `docs-full-20260930` |

## Out of scope for now

ACT. The mujoco pipeline — including the known normalizer bug at
`mujoco_image_dataset.py:64`, which fits `agent_pos` on `tcp_xyz_wxyz` concatenated with
*itself* (14 dims) instead of with `gripper_width`, and will fail against the 8-dim
`agent_pos` the dataset produces.
