# Plan: view-generalizable policy

Direction and the original predictions: `PROPOSAL.md`. The module, its measured behaviour and
the open questions: `PROGRESS.md`. Operational detail, runbooks and traps: `NOTES.md`.
Last updated 2026-10-07.

## Constraints

- This is a fork of the original diffusion_policy repo — **modify it as little as
  possible**. Each milestone lands as **new files**; existing files are touched only at
  documented seams and only when unavoidable.
- Baselines first, then one part added at a time. Models are planned here, coded in their
  own milestone, and each milestone reads its result before the next one is committed to.
- **Where things run.** Code and documents on the **laptop**; all trainings and evals on
  **`miroc-server`** (2× RTX 5090), which also stores the ckpts and data — results are written
  into `PROGRESS.md` in this repo. Check disk on `miroc-server` before a campaign — a checkpoint
  is 4.6 GB and `topk.k=1` roughly doubles a run's footprint — and check the right volume: `/` is
  chronically near-full, so run dirs go to `/data/zihan/runs/` with only `logs.json.txt` copied back.
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

## Future work (candidates; added 2026-10-07)

Instruments first — both settle what the current model does before any method change:
- **Clear-run held-out eval** (`azimuth_offgrid` preset + `data/clear_heldout_eval.sh`): the
  7.5°-offset ring ±90, the one measurement the clear run lacks; its in-distribution ring
  (0.747) is committed.
- **Latent-distribution figures** (`visualize_latent_distribution.py` + `data/clear_latent_viz.sh`):
  z_v/z_g PCA from the per-epoch probe; committed coords JSON + PNGs.

Design facts the candidates build on (verified in the code, 2026-10-07):
- `z_v` has no loss of its own beyond (a) fusion → `z_g` → diffusion and (b) M4's camera-frame
  aux head (measured null). **`z_g` receives gradient only from the diffusion loss.**
- The Plücker map enters as 6 extra conv1 channels `[d_world | m_world]` (measured inert), and
  **fusion never sees the camera pose directly** — it attends over image-only tokens.

Candidates (one part at a time, per the constraints above):
1. **Fusion with pose tokens** — an explicit pose/Plücker embedding per view token: the one
   conditioning path not yet tried (the inert ones were conv1 channels and pre-fusion FiLM).
2. **`z_g` supervision** — the task loss already constructs `z_g` (cka to mean-pool 0.97 → 0.69
   over training; N=1 vs N=11 0.998), so view-consistency has little headroom; supervision with
   headroom would separate state from view (e.g. geometry prediction from `z_g`), or predict
   camera-frame actions from `z_g` rather than `z_v`.
3. **`z_v` supervision beyond the aux head** — the aux information is already in `z_v` (M4's
   null); the open question is what `z_v` should encode, not how much more to add.
4. **M5 distillation** (standing) — premise still weak: the N=1 fused latent is ~identical to
   the full-view one, so a student would reproduce something the single view already determines.

## Out of scope for now

ACT. The mujoco pipeline — including the known normalizer bug at
`mujoco_image_dataset.py:64`, which fits `agent_pos` on `tcp_xyz_wxyz` concatenated with
*itself* (14 dims) instead of with `gripper_width`, and will fail against the 8-dim
`agent_pos` the dataset produces.
