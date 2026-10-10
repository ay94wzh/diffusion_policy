# Plan: view-generalizable policy

Direction and the original predictions: `PROPOSAL.md`. The module, its measured behaviour and
the open questions: `PROGRESS.md`. Operational detail, runbooks and traps: `NOTES.md`.
Last updated 2026-10-10.

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
| **clear-run held-out eval** (instrument) | ✅ done 2026-10-07 | **interpolates**: off-grid ±7.5…±67.5 mean **0.716** vs 0.747 in-distribution, both inside the 0.15 threshold; ±82.5/±90 → 0.310 but view visibility is confounded there; **elevation 0 of 50 at both ±15°** — not a view-count problem (K=11, mean-N 6.0, the most views in the project) |
| **latent-distribution figures** (instrument) | ✅ done 2026-10-07 | the ring's ordering is **inherited**, not learned — corr(pairwise `z_v` distance, \|Δaz\|) ≥ 0.986 at *every* epoch, epoch 0 included; and the N-invariance is **non-monotone** (cos-dist 0.17% → 1.57% at e22 → 0.53% at e200) |
| **base-frame aux target** (`m4base`, §2.4's frame axis) | ✅ done 2026-10-08 | **behavioural null (pre-registered); the latent series re-organized** — `zv_pair_ratio` 0.485→0.239, settles e57 vs e126 |
| **`z_g` supervision** (`zgpose`, candidate 2) | ✅ done 2026-10-10 | **outcome (i) at the representation, behaviour inside band** — `z_g` became view-set-aware (both instruments ~2× the parents; N=1 pose decodable at 1.2° / 1.9 cm), registered means a third null, **−0.170 past the ring**; M5's distillation premise re-opened |
| **M5** — single-novel-view inference | 🟡 capability **measured**, not pending | every M3 number is already an N=1 inference at a novel pose; only the optional distillation stage is uncoded |
| parked (encoder collapse, the `[1,3]` second failure mode, the balance refutation, the `m3v15` gap, instrument diagnostics) | ⬜ stopped, findings kept | reasoning and revival conditions at git tag `docs-full-20260930` |

## Future work (candidates; added 2026-10-07)

**The two instruments this section opened with are done** (2026-10-07): the clear run's held-out
ring and its latent-distribution figures — numbers in `PROGRESS.md`, verdicts in the table above.
**Candidate 3 has now run too**: `m4base` (2026-10-08) executed its frame form — the aux target
in the base frame — and closed it: behavioural null (pre-registered), latent re-organized.
**And candidate 2 has run** (`zgpose`, 2026-10-10): the fused latent supervised on the live
views' pooled camera-pose set came back **outcome (i) at the representation** — `z_g` became
view-set-aware (both instruments ~2× the committed parents; the N=1 fused latent decodes the
camera pose at 1.2° / 1.9 cm) — with the **behaviour inside the 0.15 band on both registered
means** (0.735 in-dist / 0.714 off-grid) and **−0.170 past the ring**, a cost. That re-opens
M5's distillation premise; it does not establish it (see `PROGRESS.md`).
What remains are method changes. Design facts they build on (verified in the code, 2026-10-07;
m4base 2026-10-08):
- `z_v` has no loss of its own beyond (a) fusion → `z_g` → diffusion and (b) M4's camera-frame
  aux head (measured null). **`z_g` receives gradient only from the diffusion loss.**
- The Plücker map enters as 6 extra conv1 channels `[d_world | m_world]` (measured inert), and
  **fusion never sees the camera pose directly** — it attends over image-only tokens.
- **No training view varies elevation at all** (the ring is azimuth-only), and the cell with the
  most views is at the floor there (0 of 50 both ways) — so elevation is not bought with azimuth
  diversity, and any elevation claim needs elevation in the data or in the conditioning.
- **Supervision can re-shape `z_v` without moving behaviour.** `m4base` halved the view-vs-state
  ratio (`zv_pair_ratio` 0.485 → 0.239), settled it at e57 vs e126, and made the latents less
  pose-decodable — with the rollout unchanged; M4's encoder drift and M3's Plücker contrast are
  the same pattern in weaker form. The behavioural bottleneck is not the view/state balance.

Candidates (one part at a time, per the constraints above):
1. **Fusion with pose tokens** — an explicit pose/Plücker embedding per view token: the one
   conditioning path not yet tried (the inert ones were conv1 channels and pre-fusion FiLM).
2. **`z_g` supervision** — ✅ executed as `zgpose` (2026-10-10): the *geometry* form (pooled
   camera-pose set, all rows scored) landed outcome (i) at the representation with behaviour
   inside band; the **action** form — camera-frame actions predicted from `z_g` rather than
   `z_v` — remains untried, and is now the only variant of this candidate left.
3. **M5 distillation** (standing) — premise **re-opened** by `zgpose`: the N=1 fused latent is
   no longer ~identical to the full-view one (2× the parents' N-dependence; pose decodable at
   1.2°), so a student would have something to learn. The deciding measurement — whether a
   distilled student beats the plain N=1 path *behaviourally* — is still unmade, and `zgpose`'s
   own behavioural null is a caution that representation-level headroom need not become success.

## Out of scope for now

ACT. The mujoco pipeline — including the known normalizer bug at
`mujoco_image_dataset.py:64`, which fits `agent_pos` on `tcp_xyz_wxyz` concatenated with
*itself* (14 dims) instead of with `gripper_width`, and will fail against the 8-dim
`agent_pos` the dataset produces.
