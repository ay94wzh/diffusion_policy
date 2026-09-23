# PROGRESS

Last updated: 2026-09-23. See `PROPOSAL.md` (research direction) and `PLAN.md`
(milestones M1–M5).

**M1 is COMPLETE.** Single-view DP baselines were trained for can, lift, and
square (robomimic PH, agentview only, 201 epochs each) on the 2× RTX 5090
machine, and the novel-view degradation curves were measured at azimuth
0°, ±15°, ±30°. The headline result: success collapses essentially to zero at
±15° on all three tasks.

**M2 (multi-view data) is COMPLETE** (§7, 2026-09-15) and its data is validated
by the N=1 gate (§8.5). **L1 (the view-randomized baseline) is COMPLETE** across
all three tasks (§8, 2026-09-17): its effect is task-dependent in *both*
directions — it solves lift outright and destroys square/can — which is now M3's
motivation.

**M3 (view-conditioned encoder) is COMPLETE AND TRAINED** (§11, 2026-09-19).
Two headline findings, and they point in opposite directions:

1. **It solves square and can — the two tasks L1 destroyed.** At azimuths
   *excluded from the training pool*, M3 scores 0.50–0.57 (square) and 0.74–0.77
   (can) where M1 is 0.00–0.02 and L1 is at the noise floor *even on the poses it
   trained on*. Beating L1 at trained poses is the load-bearing half: L1's failure
   was never a failure to generalise, it was a failure to solve the task at all.
2. **The conditioning does nothing.** The `m3off` cell — `use_plucker=False`,
   `use_eef_hist=False` — matches or beats `m3on` on both tasks, at every
   viewpoint, inside the ±0.05 rollout noise. The gain is **architectural**
   (7 slots, per-sample N, attention fusion), not the pose conditioning this
   milestone existed to test. §10.6 predicted that `m3off` "carries the
   attribution"; it does, and the attribution is not the conditioning.

§10.5's unverified list is now closed — both never-executed paths run, after two
fatal bugs were found and fixed (§11.1). §10.6's planned 9-run matrix was
**deliberately not completed**: the square 2×2 plus both can endpoints were
enough to establish both findings, so the four single-signal cells and lift were
dropped as redundant (§11.6).

**M4 (per-view auxiliary heads) is COMPLETE** (§13, 2026-09-22). The pair is run and
the verdict has two halves that must be quoted together: the **mechanism is real** —
`aux_loss` falls 52× and the head provably uses per-view information (6.4% of a
mean-collapsing predictor's floor) — and the **behavioural effect is null**
(`m4on` − `m4off` = +0.02 trained / +0.04 held-out, inside the ±0.05 noise). M4 also
**did not** revive the conditioning: it sits inside M3's band on every axis.

**§6's four-way architectural confound is RESOLVED** (§14, 2026-09-23). M3's gain
comes from **multi-view sampling (N>1)**: M3-off with `view_count_range=[1,1]` —
one randomly drawn view per sample, everything else identical — scores 0.04 trained /
0.05 held-out, indistinguishable from L1 (0.02/0.02) and nowhere near `m3off`
(0.76/0.55). This session therefore closed all three candidates in the order they
looked most promising: pose conditioning (§11.3), aux pressure (§13), encoder
architecture (§14). PROPOSAL §2.2's fusion is what stands.

---

## 1. Results — novel-view degradation curves

`success_rate` at each viewpoint (50 paired test episodes, harness
`azimuth_sweep5`; strict `EnvRobosuite.is_success()` metric):

| viewpoint | square | can | lift |
|---|---|---|---|
| az_0 | 0.82 | 0.98 | 0.76 |
| az_m15 | 0.02 | 0.08 | 0.08 |
| az_p15 | 0.00 | 0.00 | 0.08 |
| az_m30 | 0.00 | 0.00 | 0.00 |
| az_p30 | 0.00 | 0.00 | 0.00 |

`mean_score` (max-reward, the stock runner's metric):

| viewpoint | square | can | lift |
|---|---|---|---|
| az_0 | 0.84 | 0.98 | 1.00 |
| az_m15 | 0.02 | 0.08 | 0.46 |
| az_p15 | 0.00 | 0.00 | 0.50 |
| az_m30 | 0.00 | 0.00 | 0.00 |
| az_p30 | 0.00 | 0.00 | 0.22 |

**The baseline is extremely view-tied**: on all three tasks success drops to
≈0 at the smallest perturbation tested (±15°), even though at az_0 the
policies are at or near the published performance band (paper: Lift ≈ 1.00,
Square ≈ 0.9–1.0; ours: 0.98 / 0.82). This sharp collapse is the reference
curve M3–M5 must beat.

Notes on the numbers:
- **Framing was verified, not assumed.** The displaced views keep their
  texture (frame std 66–69 vs az_0's 65) and differ from az_0 by only
  ~10–11% mean pixel value on same-seed first frames — the scene is fully
  visible at ±15°/±30°, so the collapse is a property of the policy, not a
  camera-looking-somewhere-useless artifact (see PROGRESS §4's caveat).
- **Scores are noisy ±~0.05**: diffusion sampling noise is unseeded, so
  repeated rollouts of the *same* checkpoint vary (square epoch-150 scored
  0.86 once and 0.94 on re-eval; epoch 100/200 both 0.88). Treat single-digit
  differences between checkpoints as noise.
- `success_rate` (is_success) is stricter than `mean_score` (max reward) —
  e.g. lift at az_0: success 0.76 while mean_score 1.00, and at ±15° the
  policy still gets near the cube (mean 0.46–0.50) but rarely closes the grip.

Artifacts (all under `data/`, gitignored):
```
data/outputs/run_{square,can,lift}_abs_single_s42_200ep/checkpoints/latest.ckpt   (epoch-200 models)
data/outputs/.../logs.json.txt                                                   (train + rollout history)
data/eval_az5_{square,can,lift}_ep200/eval_log.json + media/<viewpoint>/*.mp4    (sweeps)
```
Summarize any sweep with `python summarize_novel_view.py <dir>/eval_log.json`.

### 1.1 Weights and wandb

- **Trained weights are NOT in git** (4.6 GB/checkpoint exceeds GitHub's
  100 MB file limit) — only the numbers are. To move them between machines:
  ```bash
  rsync -av data/outputs <user>@<other>:/path/to/diffusion_policy/data/outputs
  ```
  or re-train from the runbook in §3 (~1–2.5 h per task).
- ⚠️ **The M1 (`*_abs_single`) weights were DELETED on 2026-09-19** to free disk
  for the M3 runs — the root volume was at 97–100% and nine M3 runs need ~83 GB.
  Their `logs.json.txt`, `media/` and `.hydra/` survive; only `checkpoints/`
  went. Every number derived from them is committed (the §1 sweeps) and their
  elevation curve was captured first (§11.2). Recovery is a ~1–2.5 h retrain per
  task from §3. The L1 weights and both M3 runs' weights are still present.
- **Training curves and rollout videos are on wandb**: project
  `diffusion_policy_view` (account zihan-wa23-tsinghua-university) — square
  `runs/1h4oj5p6`, can `runs/q95ylpsc`, lift `runs/poearmxq`.
- **Committed to git** (so a fresh clone has the numbers): the sweep
  `eval_log.json` files + all 66 viewpoint videos under
  `data/eval_az5_*_ep200/`, and the per-batch training logs at
  `data/outputs/run_*_abs_single_s42_200ep/logs.json.txt`.

## 2. Configs and scripts added for M1

| File | Purpose |
|---|---|
| `diffusion_policy/config/task/{square,lift,can}_image_abs_single.yaml` | single-view variants of the stock `_abs` configs (wrist camera dropped) |
| `eval_novel_view.py` | novel-view eval harness (from the M1 planning commit) |
| `summarize_novel_view.py` | aggregates `eval_log.json` files into the degradation table |

Note: the prepared `{square,lift}_image_single.yaml` (relative actions) are
unused — the baselines were trained with `_abs_single` variants, matching the
published robomimic setup (`image_abs.hdf5`, `abs_action: True`, 10-dim
actions). Verified single-view at runtime: the zarr conversion reads exactly
one camera's worth of images per task (e.g. square: 30154 = raw step count,
not 2×).

## 3. Runbook used on this machine (verified)

```bash
conda activate robodiff        # exists; torch 2.8.0+cu128 (sm_120), robosuite 1.2.0, robomimic 0.2.0
cd <repo root>

# train (201 epochs: checkpoint+rollout fire on epoch % 50 == 0, and there is
# NO save at the end of training — 201 makes epoch 200 fire; 200 would stop at 150)
python train.py --config-dir=. --config-name=train_diffusion_unet_image_workspace.yaml \
  task=<task>_image_abs_single training.seed=42 training.device=cuda:0 training.num_epochs=201 \
  dataloader.num_workers=8 checkpoint.topk.k=1 logging.project=diffusion_policy_view \
  hydra.run.dir=data/outputs/run_<task>_abs_single_s42_200ep

# novel-view sweep (az_0, ±15°, ±30°, paired seeds)
python eval_novel_view.py -c <run>/checkpoints/latest.ckpt -o data/eval_az5_<task>_ep200 \
  -d cuda:0 --preset azimuth_sweep5

# resume (e.g. after a kill): num_epochs = epochs still wanted; the loop runs
# that many epochs from the restored counter (checkpoint epoch + num_epochs − 1
# is the last epoch saved, iff num_epochs % 50 lands right — 51 from 150 → 200)
python train.py <same args, minus task switch> training.num_epochs=51 checkpoint.topk.k=1 \
  hydra.run.dir=<same run dir>
```

Timings on this box (idle; fp32, no AMP — the workspace has none):
- square ~22 s/epoch (440 batches @ bs 64), can ~16 s/epoch, lift ~7 s/epoch
- rollout ~2–3 min (28 envs, 56 episodes); epoch-0 rollout is the worst case
- full sweep ~18 min (5 viewpoints × 50 episodes)
- checkpoint = 4.62 GB (policy + EMA + Adam state); `topk.k=1` + latest ≈
  9.2 GB/run — so the 9 M3 runs need ~83 GB, which is why `df -h` is step 0 of
  the M3 preflight (§10.6). **At M1 time the root disk was 97–100% full**; as of
  2026-09-18 there is reported headroom, but that is a report, not a measurement —
  re-check before starting. `/data` (15 TB) exists but is owned by another user
  and not writable.

**M3 timings (measured 2026-09-19, §11.1)** — the M3 encoder runs a shared
resnet18 over N∈[1,7] views instead of 1, so it costs ~1.9× M1 per step:
- raw step cost at B=64: **94 ms** (min 92, max 98), 6.9 GB peak GPU
- square **43 s/epoch** (440 batches) at `num_workers=14` — **compute-bound**:
  8 workers still gave 41 s/epoch under 2-GPU contention, while 4 workers fell
  to **81 s/epoch** (decode-bound). So ≥8 workers, and ~2 concurrent runs is the
  practical ceiling on 24 cores.
- N=1 gate (K=1 ⇒ 1 view) **23 s/epoch**; can ~30 s/epoch (335 batches)
- rollout ~6 min (n_envs=14, 56 episodes) — 5 per run
- `azimuth_interp` (11 vp × 50 episodes) **~40 min**; ~46 min when two sweeps
  share the box. `azimuth_sweep3` / `elevation_az0` ~12 min.
- **2 GPUs give ~1.7×, not 2×**, on a 3-run tail: independent processes run at
  full speed in parallel (36–41 s/epoch each), but the last run has no partner.
  CPU, not GPU memory (7 GB of 32 GB), is the limit on concurrency.

## 4. Verified environment facts (all checked by running)

- `robodiff` env: torch **2.8.0+cu128** (the `conda_environment.yaml` pin
  `pytorch=1.12.1` is stale and unusable on RTX 5090 — do NOT recreate the
  env from it), robosuite 1.2.0, robomimic 0.2.0, mujoco_py 2.0.2.13.
  Offscreen rendering works with osmesa, EGL, and the default backend.
- Camera control at the mujoco_py level (no `CameraMover` in robosuite 1.2):
  `sim.model.cam_pos/cam_quat` + `sim.forward()`. `cam_quat` is **wxyz**;
  robosuite utils are **xyzw** — the harness converts explicitly.
- `hard_reset=False` (set by the stock runner) persists camera moves across
  soft resets; the harness re-applies the viewpoint at every reset, computed
  from the base pose captured on first use (never compounds).
- `eval_novel_view.py`'s checkpoint-loading path **has been exercised against
  real checkpoints** (all three sweeps above) — it mirrors `eval.py` and
  works, including the `${eval:...}` resolver dependency on importing the
  workspace class.
- The zarr cache (`<hdf5>.zarr.zip`) is keyed **only by hdf5 path**, not by
  `shape_meta` — a single-view run caches one camera; a later 2-camera run on
  the same path would `KeyError` on the wrist key. Delete the cache to switch.
- Dropping the wrist camera from `shape_meta` removes encoder FLOPs but not
  render cost: `camera_names` comes from the hdf5 `env_args`, so robosuite
  still renders both cameras every step.

## 5. Machine / operational notes

- This is the remote training machine: 2× RTX 5090 32 GB, 24 cores, 125 GB
  RAM, multiple users (load can spike from ~5 to ~25; expect 2× slowdowns).
- `n_envs: 28` is fine here (PROGRESS §5's old 6 GB-laptop warning is moot).
- Long jobs die when the Claude Code session ends (background tasks are killed
  with the session). For anything that must outlive the session, launch
  disowned: `setsid bash -c '<cmd> > log 2>&1' &` — used successfully for the
  can/lift sweeps.
- The epoch loop does not skip completed epochs on resume: it runs
  `num_epochs` iterations from the restored counter. LR on resume is the
  cosine schedule re-based to the new `num_epochs` (clamped ≥ 0, harmless in
  practice).
- `TopKCheckpointManager` state is in-memory only: on resume its map starts
  empty, so existing topk files on disk are never evicted by the new run (and
  delete stale ones yourself if space is tight).
- **The topk file is often a byte-identical duplicate of `latest.ckpt`.** The
  workspace calls `save_checkpoint()` (latest) and `save_checkpoint(path=topk)`
  back-to-back in the same block with the same in-memory state
  (`train_diffusion_unet_image_workspace.py:260-280`), so whenever the topk
  epoch equals the final epoch the two files are the same model. Freeing those
  bought 23 GB in the M3 session with zero information loss. Check
  `ls checkpoints/` — a topk named `epoch=0200-*` on a 201-epoch run is the
  duplicate case; `epoch=0150-*` is genuinely distinct.
- **To re-plan a running `setsid` batch without killing the job:** the driver
  is `bash -c 'for ...; do python train.py ...; done'`; `kill -TERM <bash pid>`
  stops the loop while the running `python` child survives (verified — the loop
  exits, the child is reparented to init and keeps training). Progress markers
  are appended to `data/m3_campaign.log` so a hand-over is auditable rather than
  looking like the loop finished normally.
- Two Claude-session lessons worth keeping: `preview_viewpoints.py`'s render
  path and both M3 runtime paths went from "import-checked, never executed" to
  working only by *running* them, and each failure was silent-degradation class
  (§11.1). Budget for a probe phase before any long run.

## 6. Not yet done / next

**The live question changed.** M3's conditioning is inert (§11.3), so the
proposal's central claim — that Plücker maps + camera-frame history buy view
generalization — is **not supported**. What *is* supported is that the M3
**architecture** fixes square and can. The next experiment should therefore
attack the architecture, not add more conditioning.

- **Which ingredient?** → ✅ **ANSWERED (§14, 2026-09-23): it is (b), multi-view
  sampling (N>1).** M3-off with `view_count_range=[1,1]` scores 0.04/0.05, i.e. L1's
  floor, where `m3off` scores 0.76/0.55. The encoder architecture alone reproduces
  nothing. Four things changed at once relative to L1: (a) 7 slots
  instead of 1, (b) per-sample N∈[1,7] instead of a fixed 1, (c) MHA fusion with
  a learnable query, (d) a shared backbone whose `conv1` is widened 3→9. The 2×2
  cannot separate them and neither can any run so far. The cheapest informative
  split is variable-N vs fixed-N=1 at 7 slots — that is one CLI line
  (`task.dataset.view_count_range=[1,1]`), and it tests the "N is randomized so
  N=1 is in distribution" design decision directly.
- **Lift is untested for M3.** §10.6 called it a regression check, and it was
  dropped along with the redundant cells (§11.6). It is the one task where L1
  already wins (0.76–0.96), so M3-on there is a genuine "did we break it"
  question, not a result. ~1 h: lift is 127 batches/epoch.
- **can's two single-signal cells** were skipped because both endpoints agreed;
  if the conditioning question is revisited, they are the cheap way back in.
- **A second seed** on one M3 cell would materially strengthen §11 — every M3
  number is n=1, 50 paired episodes, ±0.05 noise. The conditioning-null in
  particular is "indistinguishable at this resolution", not "proven identical".
- **M4 (per-view aux heads)** and **M5 (single-novel-view inference +
  distillation)** remain uncoded. Their hooks exist (the encoder keeps per-view
  latents separate before fusion; `eval_novel_view.py --m3-slots K` publishes the
  perturbed pose). **But M5's premise should be re-examined**: M3 already does
  N=1 novel-view inference well (§11.2), so distillation is only worth it if the
  fused latent is shown to carry something the single-view path cannot.
- **Restate the elevation axis in any follow-up.** M3 holds 0.18–0.34 at ±15°
  elevation where M1 is 0.00, but collapses at −15° (0.00–0.04 on square). The
  extrapolation claim currently holds on azimuth and half-holds on elevation;
  the `elevation_az0` preset (an orbit about the look-at point, not the pitch)
  is the tool.

## 7. M2 — multi-view data (DONE 2026-09-15)

Goal: for every demo timestep, N simultaneous renders of the same scene state
from N fixed camera poses, plus camera parameters, so M3 (Plücker conditioning),
M4 (camera-frame aux heads) and M5 (single-novel-view inference) have training
data. Decisions: all three tasks; azimuth ring every 15° to ±90° (13 views);
full data (every demo, every step).

### 7.1 Results — rendered 2026-09-15 (all three tasks)

Ran sequentially on this box with `--workers 4`, `setsid`-detached, one task at a
time. **Total 8h00m** at ~2.2 steps/s (square 3h50m, can 2h59m, lift 1h11m).

| task | steps | images (×13) | gate 1 mean\|diff\| | gate 2 in-frame | on disk |
|---|---|---|---|---|---|
| square | 30154 | 392k | 1.454/255 PASS | 33/39 | 1.6 GB |
| can | 23207 | 302k | 2.801/255 PASS | 39/39 | 2.4 GB |
| lift | 9666 | 126k | 1.502/255 PASS | 39/39 | 496 MB |

Total **819k images, 4.5 GB on disk** (square+can+lift).

Two PROGRESS estimates were wrong and are corrected here:
- **Image count is 819k, not ~400k** (the step counts were right; the ×13 was
  undercounted).
- **Size is 4.5 GB, not 6–9 GB** — but for a different reason than expected:
  Jpeg2k(50) measured **3365 B/img (6.3×)** on real frames, yet `du` is ~1.3–2×
  the compressed bytes because the zarr `DirectoryStore` writes **one file per
  (timestep, view) chunk** (chunks are `(1,84,84,3)`; **392k files** for square),
  and each ~3–8 KB file occupies a 4 KB-granularity block. Can is the worst case
  (2.4 GB) because its busier, darker scene compresses worse.
- ⇒ **The 84.75 GB `robomimic_image.zip` was NOT deleted and does not need to be.**

### 7.2 Verification actually performed (not assumed)

- **az_0 provenance** — the generated `view_06` (azimuth 0) vs the hdf5's stored
  `agentview_image`, at demos 0/100/199: square **1.28–1.30**, can **2.78–2.80**,
  lift **1.24–1.35** per 255. The generated az_0 view is provably the original
  dataset camera.
- The per-task spread is **scene-dependent re-render fidelity** (`reset_to` state
  replay), not compression: the zarr-side numbers (which *include* jpeg2k loss)
  match the in-RAM gate-1 numbers almost exactly. Can's cluttered dark shelf is
  simply more sensitive than square's white table. All are ~15× below the ~40
  signature of a flip bug (the thing gate 1 exists to catch), so all are sound.
- **No truncation** — a killed run is *silent* here (zarr returns fill-value 0 for
  unwritten chunks, and the gates never read the zarr back, so they still PASS).
  Checked explicitly: zero all-zero frames in the tail and mid-corpus of every
  view array. (Note the correct criterion is "no all-zero *frames*", not "no zero
  pixels" — square/can legitimately contain 3 and 1 pure-black pixels per 84,672.)
- Both montages (`data/multiview/{square,can,lift}_ring13.png`) inspected: the
  scene is visible at every azimuth and the gripper crosshair lands correctly.
- The three zarrs load through `MultiViewImageDataset` with the real task
  configs (13 views, 17 normalizer params, `(2,3,84,84)` obs / `(16,10)` action).

### 7.3 Fixes and facts from the first real execution

- **Bug fixed** (`generate_multiview_dataset.py`): the script ended with
  `env.close()`, but robomimic's `EnvRobosuite` has no `close()` — the robosuite
  env at `env.env` does. It crashed *after* the data and gates were written, so
  **every run exited non-zero**, making a crashed run indistinguishable from a
  clean one. Now `env.env.close()`, exit 0.
- Gates are computed only at the **very end** of a run, so a broken render loop
  surfaces after hours. The `--limit-demos 5` pilot (≈5 min) is the mitigation and
  is now proven: it reproduced the full run's gate values exactly (1.454, 33/39).
- The generator has **no resume**; `--overwrite` is the only recovery, and it
  wipes the store. Hence the driver wipes each output before starting.

### 7.4 Prerequisites for the N=1 gate (built, verified, not yet run)

> **Correction (2026-09-18, found while building M3):** the config below claimed
> rollouts were disabled by `training.rollout_every > num_epochs`. That is
> **false** — the workspace tests `epoch % rollout_every == 0`
> (`train_diffusion_unet_image_workspace.py:218`), which is true at epoch 0, so
> the first rollout fires and dies with `KeyError: 'view_06_image'` because the
> stock runner cannot serve that key. The config comment is now corrected and
> gives the overrides that actually disable rollouts
> (`training.rollout_every=1000000` **plus**
> `checkpoint.topk.monitor_key=val_loss checkpoint.topk.mode=min`, the second
> flag being necessary because `TopKCheckpointManager` does an unguarded
> `data[monitor_key]` lookup). This gate is in any case **superseded** by
> `config/task/m3_plucker_image_abs_n1.yaml`, which runs the same single-view
> check through the M3 encoder with a runner that can serve the slots.

- `diffusion_policy/config/task/single_view_image_abs_multiview.yaml` — one view
  (`view_06_image`) of the multi-view zarr. Because the multiview configs route
  `task_name` through `${task.task_name}`, **one file serves all three tasks**:
  `task=single_view_image_abs_multiview task.task_name=lift`.
- `eval_novel_view.py --serve-obs-key view_06_image` — needed because the live env
  emits `agentview_image` while a multi-view-trained policy expects
  `view_06_image`. **This is not a simple rename:** robomimic's
  `EnvRobosuite.get_observation` only emits rgb keys that are in the obs-modality
  mapping *and* present in robosuite's raw obs, so a multiview `shape_meta`
  produces **no image key at all**, and renaming at the `predict_action` call site
  is far too late. The fix is two-part: `main()` augments the env-side
  `shape_meta` with `render_obs_key`, and `ViewpointImageWrapper.get_observation`
  aliases the rendered frame to the policy's key. Verified: with the bridge
  `view_06_image` is pixel-identical to the rendered agentview frame; without it,
  KeyError. Defaults to a no-op, so M1's sweeps are unaffected. **M5 needs this
  same mechanism.**

### 7.5 Findings that change M3's design

- **Per-sample view subsampling saves encoder FLOPs but NOT IO.**
  `SequenceSampler` reads *every* zarr key regardless of `shape_meta`;
  `key_first_k` is built only from shape_meta keys
  (`multiview_image_dataset.py:125-126`). So a 1-view config still reads the other
  12 views — at full 26-frame length, making it **~6× slower per sample than the
  13-view config** (156 ms vs 24 ms, reproducible and order-independent). M3 must
  restrict the sampler's `keys`, not just `shape_meta`, if it subsamples views.
- 13 views costs ~24 ms/sample of IO (26 frame reads at ~0.9 ms). The whole square
  store is only 1.6 GB, so it fits in page cache — warm training IO will be much
  cheaper than these cold numbers suggest.

### Files added (all new; no upstream package file modified)

| File | Purpose |
|---|---|
| `generate_multiview_dataset.py` | offline re-renderer → ReplayBuffer-compatible zarr |
| `diffusion_policy/dataset/multiview_image_dataset.py` | `MultiViewImageDataset` + camera-math helpers |
| `tests/test_multiview_dataset.py` | CPU-only checks on a synthetic zarr (no simulator) |
| `config/task/{square,can,lift}_image_abs_multiview.yaml` | 13-view task configs |

### What was verified BEFORE rendering (CPU-only, synthetic data)

- `python tests/test_multiview_dataset.py` → **ALL MULTIVIEW DATASET CHECKS PASSED**:
  camera math (quaternion convention, intrinsics from fovy, projection sign and
  depth), zarr schema round-trip through `ReplayBuffer`/`SequenceSampler`,
  `__getitem__` shapes/dtypes/range, per-view content mapping, normalizer rules
  (image [0,1]→[-1,1], lowdim, invertible 10-dim abs action), `camera_params`
  round-trip, validation split, `get_all_actions`.
- `generate_multiview_dataset.py` imports cleanly and its pure helpers
  (`view_key`, `make_compressor`, `resolve_render_size`, `_draw_cross`) pass
  checks; the three task configs compose and resolve under Hydra with 13 rgb
  views, the right zarr/hdf5 paths and 10-dim abs actions.
- The render loop itself (`reset_to` + camera move + `sim.render`) **cannot be
  exercised here** — no robosuite env data on the dev laptop, and rendering is
  deliberately reserved for the GPU box.

### What must be on the render machine

Nothing new to download — the generator reads the hdf5 M1 already trained on:

| Needed | Status |
|---|---|
| `data/robomimic/datasets/{square,can,lift}/ph/image_abs.hdf5` | already there (M1) |
| `robodiff` env (robosuite 1.2.0, robomimic 0.2.0, torch 2.8) | already there |
| this code (`git pull`) | — |
| **~5 GB free disk** | it needed no cleanup at all — see below |

The hdf5 supplies everything the renderer consumes: `data/demo_*/states`
(replayed with `reset_to`), `data/demo_*/actions` (copied, converted to 10-dim
abs), `data/demo_*/obs/robot0_eef_{pos,quat}` + `robot0_gripper_qpos` (copied),
and the `env_args` used to build the env. No calibration files, depth or meshes
are needed — intrinsics come from `sim.model.cam_fovy`, extrinsics from the sim.

✅ **Resolved — no cleanup was needed.** The whole run consumed 4.5 GB and left
23 GB free. `data/robomimic_image.zip` (84.75 GB) is untouched; the earlier
"≥ 20 GB free, consider deleting the zip" concern came from an estimate that was
~2× too high on image count and ~1.5× too high on bytes (§7.1).

### Runbook

```bash
cd <repo root> && git pull
python tests/test_multiview_dataset.py          # CPU-only sanity, no render
df -h .                                         # need ~20 GB

# pilot: 5 demos of one task + montage, inspect before committing hours
python generate_multiview_dataset.py \
  --dataset data/robomimic/datasets/square/ph/image_abs.hdf5 \
  --output data/multiview/square_ph_ring13.zarr \
  --limit-demos 5 --montage /tmp/ring5.png

# full run: SEQUENTIAL (not the concurrent loop originally sketched here --
# one task at a time contains a failure and avoids GL contention), disowned so it
# survives the session ending, with a self-gating driver that aborts before the
# next task if gate 1 is not PASS and wipes each output first (no resume; a
# partial store is otherwise silently served as black frames).
setsid bash -c 'for t in square can lift; do
  rm -rf data/multiview/${t}_ph_ring13.zarr
  python generate_multiview_dataset.py \
    --dataset data/robomimic/datasets/$t/ph/image_abs.hdf5 \
    --output data/multiview/${t}_ph_ring13.zarr \
    --workers 4 --montage data/multiview/${t}_ring13.png \
    > data/gen_$t.log 2>&1 || break
  grep -aq "gate 1.*PASS" data/gen_$t.log || break
done' > data/gen_all.log 2>&1 &
```

`--workers 4`, not the default 24: `max_inflight = workers*5` and each in-flight
future pins a whole `(T,13,84,84,3)` buffer, so the default can reach ~8 GB of RAM.

The script prints three gates at the end of every run:

1. **az_0 re-render vs the hdf5's stored agentview image** — `mean|diff|` should
   be ≲ 3/255. A mean of ~40+ means the `[::-1]` flip is missing or doubled
   (mujoco's readPixels is bottom-up; robomimic flips it once, and the hdf5
   stores the flipped form). This is the check that catches an upside-down
   dataset, which otherwise looks plausible in a montage.
2. **projected gripper site** — how many (view, step) pairs the derived
   intrinsics/extrinsics put inside the frame. This validates the camera-parameter
   chain M3's Plücker maps depend on.
3. **ring montage PNG** — eyeball that the scene is visible at every angle and
   decide whether ±75°/±90° are worth keeping.

Measured cost (2026-09-15, shared box with other users active): **819k images
total** (30154 + 23207 + 9666 steps × 13 views), **8h00m** at ~2.2 steps/s,
**4.5 GB** on disk. Per-task: square 3h50m/1.6 GB, can 2h59m/2.4 GB,
lift 1h11m/496 MB. The jpeg2k encode is threaded over `--workers` and is not the
bottleneck (rendering is single-threaded and serial).

On gate 3: the montages confirm the scene is visible at every azimuth, but
**±75°/±90° are low-value** — dominated by the table edge, with the object small
or partly out of frame. They were kept (2/13 of the bytes) since M3 can accept a
subset of view slots, so no re-render is needed.

### Deliberately out of scope for M2

Elevation views, the wrist camera as an extra view, and a live multi-view
env_runner (the stock runner builds obs from `shape_meta`, so it cannot serve
`view_XX_image` keys; M3's encoder will accept a *subset* of view slots so the
existing single-camera novel-view harness can evaluate it). Plücker maps and
camera-frame actions are not stored — both are deterministic functions of the
stored camera params (+ base actions), so M3/M4 compute them at train time.

Sizing note for M3: 13 views ⇒ ~13× encoder FLOPs per sample; per-sample view
subsampling will likely be needed to keep epoch time near M1's.

---

## 8. L1 — the view-randomized baseline (COMPLETE, all three tasks, 2026-09-17)

**Naming.** "L1" is this fork's label, not a milestone and not an upstream term.
It names the second rung of the experiment ladder:

| rung | what it is |
|---|---|
| L0 | = M1. One camera, one pose (az_0). Collapses off-axis. |
| **L1** | **"View diversity only": M1's *exact* architecture, with its single camera slot filled per sample from a randomly drawn *training* view. No pose information anywhere.** |
| L2–L4 | = M3 (Plücker-conditioned encoder), M4 (aux heads), M5 (single-novel-view inference) |

L1 answers the question M3's design rests on: does merely *showing* a policy many
viewpoints buy view invariance, without telling it which viewpoint it is looking
from?

### 8.1 Training method

Config `diffusion_policy/config/task/randview_image_abs_multiview.yaml`; dataset
`MultiViewImageDataset` over `data/multiview/square_ph_ring13.zarr`.

- **Views.** The 15° ring's **even indices** are the training views:
  `view_subset = [0,2,4,6,8,10,12]` = az −90/−60/−30/0/+30/+60/+90 (every 30°).
  The **odd** indices (az ±15/±45/±75) are *never sampled*; holding them out by
  construction is what makes the "held out" column in §8.3 honest.
- **Sampling.** `__getitem__` draws one view uniformly from the subset. Both
  `n_obs_steps=2` frames come from *that same* view, so no viewpoint change
  occurs within a sample.
- **Slot naming — deliberate, not cosmetic.** The single rgb key is named
  `agentview_image`. In `view_subset` mode the shape_meta key is a *slot label*,
  not a zarr array name (`__getitem__` reads the array via `view_key(idx)`), so
  the name is free — and choosing the live camera's name keeps the pipeline
  stock: the rollout environment and `eval_novel_view.py` both render
  `agentview_image`, so training rollouts fire normally (in-distribution az_0
  metrics, and the topk monitor gets its key) and evaluation needs **no**
  `--serve-obs-key`.
- **Architecture: identical to M1** (`DiffusionUnetImagePolicy` +
  `MultiImageObsEncoder`, resnet18, one rgb key). L1 changes only the *training
  distribution of the image*, never the model.
- **Optimisation** (workspace defaults): horizon 16, `n_obs_steps=2`,
  `n_action_steps=8`, batch 64, 201 epochs, seed 42, AdamW + cosine LR + EMA.
- **In-training rollouts** every 50 epochs at the fixed default camera (az_0),
  `n_envs=14`, `n_test=50`. These give the az_0 curve in §8.3 for free.
- Exact CLI, per `data/outputs/run_square_randview_s42_200ep/.hydra/overrides.yaml`:
  `task=randview_image_abs_multiview training.seed=42 training.num_epochs=201
  training.rollout_every=50 dataloader.num_workers=10 checkpoint.topk.k=1
  logging.project=diffusion_policy_view`.

### 8.2 Evaluation method

```bash
python eval_novel_view.py -c <run>/checkpoints/latest.ckpt -o <out> \
  -d cuda:0 --preset azimuth_interp --n-envs 14
```

- Perturbs `agentview` at the mujoco_py level (`sim.model.cam_pos/cam_quat` +
  `sim.forward()`), azimuth about the base pose, recomputed from the base pose
  captured on first use at *every* reset so repeated resets never compound.
- **11 viewpoints**: az 0, ±15, ±30, ±45, ±60, ±75. **50 episodes each, with
  paired seeds** (`test_start_seed=100000`) — identical episode seeds across
  viewpoints, so a difference is attributable to the camera, not the episodes.
- Metrics: strict `EnvRobosuite.is_success()` (`success_rate`) and max-reward
  (`mean_score`). `success_rate` is the stricter of the two (M1 lift at az_0:
  0.76 vs 1.00).
- `az_0/±30` coincide with M1's `azimuth_sweep5`, so those columns are exact
  comparisons. `--preset azimuth_interp` was **added in this phase**
  (`eval_novel_view.py:66`); the earlier presets stopped at ±30.
- `eval_log.json` is a flat dict keyed `test/<viewpoint>/{mean_score,
  success_rate, sim_max_reward_<seed>}` — any number can be re-derived without
  re-running. `python summarize_novel_view.py <file>` prints the tables.

### 8.3 Results — all three tasks

**Final sweeps**, `azimuth_interp`, strict `success_rate`, 50 paired episodes per
viewpoint. All five runs, side by side:

| viewpoint | M1 lift | **L1 lift** | M1 can | L1 can | M1 square | L1 square (s42 / s43) |
|---|---|---|---|---|---|---|
| az_0 | 0.760 | **0.960** | 0.980 | 0.020 | 0.820 | 0.020 / 0.040 |
| az_p15 | 0.080 | **0.920** | 0.000 | 0.020 | 0.000 | 0.020 / 0.020 |
| az_m15 | 0.080 | **0.900** | 0.080 | 0.020 | 0.020 | 0.000 / 0.020 |
| az_p30 | 0.000 | **0.940** | 0.000 | 0.020 | 0.000 | 0.000 / 0.080 |
| az_m30 | 0.000 | **0.880** | 0.000 | 0.020 | 0.000 | 0.000 / 0.040 |
| az_p45 | — | 0.960 | — | 0.020 | — | 0.020 / 0.040 |
| az_m45 | — | 0.860 | — | 0.060 | — | 0.000 / 0.040 |
| az_p60 | — | 0.900 | — | 0.040 | — | 0.060 / 0.040 |
| az_m60 | — | 0.840 | — | 0.000 | — | 0.020 / 0.000 |
| az_p75 | — | 0.840 | — | 0.000 | — | 0.040 / 0.000 |
| az_m75 | — | 0.760 | — | 0.020 | — | 0.020 / 0.020 |

**Read the failing columns as a noise floor, not as low success.** With 50
episodes, L1's 0.000–0.080 on can and square is 0–4 episodes: it is not "weakly
succeeding", it is not solving the task at az_0 — the one pose it definitely
trained on, where M1 scores 0.98 and 0.82.

**In-training rollouts at az_0** (`mean_score`, for the record — note `mean_score`
saturates on lift and hides the sweep's result):

| task | M1 (1 pose) | L1 (7 poses) | L1 seeds |
|---|---|---|---|
| lift | 1.000 | **1.000** | 42 |
| square | 0.880 | 0.020 / 0.080 | 42, 43 |
| can | 0.980 | **0.020** | 42 |

Can confirms the square pattern exactly: uniform ~0 across all 11 viewpoints,
including trained poses. There is no third behaviour — the tasks split 2–1.

### 8.4 What this means

**The effect is task-dependent, and in opposite directions.**

- **On lift, 7-pose randomization *solves* the problem.** L1 holds 0.76–0.96 at
  every viewpoint out to ±75°, where M1 collapses to 0.08 (±15°) and 0.00 (±30°).
  A conditioning-free baseline already achieves the project's stated goal —
  single camera, novel pose — on lift.
- **On square and can, the same intervention is catastrophic.** L1 sits at ≤0.08
  everywhere, *including poses it trained on*, where M1 scores 0.82 / 0.98.

So "view diversity alone fails" is false as a general claim — wrong in both
directions, and worth stating that way rather than picking the flattering half.

**What separates lift from square/can is not identified.** The cleanest
structural difference is that lift is the only task requiring no goal-directed
placement — grasp-and-raise, where the target is wherever the object already is,
versus nut-onto-peg and can-into-bin. That would make the axis "how much precise
spatial localization from the image the task needs." That is a **hypothesis with
one task per side, not a finding.** A confound we cannot rule out: lift's scene
is visually far simpler (plain table + cube) than can's cluttered shelf, so it
could be scene regularity rather than task structure.

Limit that stands: on square, L1's val_loss is ~2× M1's (0.060 vs 0.029) — it
does fit worse — but a 2× loss gap does not explain a 44× rollout gap. The
mechanism is **inferred, not measured**; the probe that would test it was
deliberately dropped, so this hedge is permanent, not provisional.

### 8.5 The N=1 gate — why everything above is measured on the generated zarr

`run_lift_n1gate_s42_200ep` trains on the generated zarr using **only** its az_0
view (`task.dataset.view_subset=[6]`) — same single-camera-pose setup as M1, but
a different data source. It is a *fidelity control* on M2, not a new method.
Note it shares a task and seed with the §8.6 lift replication but is a different
experiment (`_n1gate_` = 1 view, `_randview_` = 7 views).

| viewpoint | M1 success | N=1 success | M1 mean | N=1 mean |
|---|---|---|---|---|
| az_0 | 0.760 | **0.840** | 1.000 | 1.000 |
| az_m15 | 0.080 | 0.040 | 0.460 | 0.280 |
| az_p15 | 0.080 | 0.160 | 0.500 | 0.620 |
| az_m30 | 0.000 | 0.000 | 0.000 | 0.000 |
| az_p30 | 0.000 | 0.000 | 0.220 | 0.240 |

Trained on the generated zarr, the model reproduces M1's **whole degradation
curve** — high at az_0, collapse at ±15°, zero at ±30° — not merely its az_0
score. Residual differences sit inside the ±0.05 rollout noise documented in §1.
This is the result that licenses measuring every later milestone on the
multi-view zarr instead of the hdf5.

### 8.6 Replication

Three runs beyond square/s42, all on the config default 7-view subset (no
`view_subset` override — that belonged to §8.5's gate):

| run | task | seed | az_0 rollout | sweep |
|---|---|---|---|---|
| `run_square_randview_s42_200ep` | square | 42 | 0.020 | ≤0.08, all 11 vp |
| `run_square_randview_s43_200ep` | square | 43 | 0.080 (best) | ≤0.08, all 11 vp |
| `run_lift_randview_s42_200ep` | lift | 42 | 1.000 | **0.76–0.96, all 11 vp** |
| `run_can_randview_s42_200ep` | can | 42 | 0.020 | running |

The replication did its job by **overturning** the original claim: square
reproduces across seeds, but lift inverts the story entirely — rather than merely
being "unaffected," it is far *better* than M1 off-axis.

**Read `success_rate`, not `mean_score`, on lift.** The improvement was initially
invisible: in-training rollouts log only `mean_score`, which saturates at 1.000
on lift (M1 lift: mean 1.00 but success 0.76). Only the strict sweep revealed it.

Naming trap: `run_lift_n1gate_*` (1 view, §8.5's fidelity control) and
`run_lift_randview_*` (7 views, this table) share a task and seed.

---

## 9. M3 — view-conditioned encoder (SPECIFIED, draft code unverified)

**Status: design written, code NOT working, nothing committed.** The full design
spec lives in `PLAN.md` ("M3 design spec"). This section records only the honest
state of what is on disk.

### 9.1 What exists

**No code.** The two drafts (`model/vision/plucker.py`,
`model/vision/view_conditioned_obs_encoder.py`) were **deleted**; the tree is
clean at `6aab4c6`. The durable artifact from this session is the **design spec in
`PLAN.md`**, not the code. §9.2 and §9.3 are kept as a record of what to avoid
and what to build when the encoder is actually written.

### 9.2 What was wrong with the deleted draft (keep as a warning)

- **`plucker.py` raises on the first real call.** `quat_wxyz_to_mat_torch`
  returns a **2-D** matrix when handed a **1-D** `(4,)` quaternion, so
  `einsum('bij,jhw->bhwi', ...)` fails with *"number of subscripts in the
  equation (3) does not match the number of dimensions (2)"*. Needs a leading
  batch dimension forced, or an `unsqueeze`.
- **The Plücker convention is UNVERIFIED.** A convention check was written but
  was **itself wrong**: it compared a *camera-frame* ray direction against a
  *world-frame* expected direction without applying the rotation `R`, so its
  output (dots of −0.85, −0.18, −0.60 instead of ≈1.0) says nothing about the
  code. A correct check must compare `R @ d_cam` against `(p_world − cam_pos)`
  normalised, at the pixel that `project_world_to_pixel` (the numpy path M2's
  gate 2 validated against the simulator) puts it at.

### 9.3 What is designed but unbuilt

- The dataset does **not** yet emit `view_XX_cam` keys, so the encoder has no
  pose input.
- `eval_novel_view.py` does **not** yet publish the perturbed pose into the obs
  dict, so a trained M3 model could not be evaluated at a novel view.
- No config references the new encoder; no training run has been attempted.

### 9.4 The one thing to preserve from the design

`fused_dim = 512` makes M3's `output_shape()` **521**, identical to M1's and
L1's. That is what lets M3 differ from the baselines in *training distribution and
conditioning* rather than in downstream capacity — and `use_plucker=False` is the
constructor flag that isolates conditioning at matched capacity. Losing that
property would make the headline comparison uninterpretable.

*(Superseded in detail — §10.3: `use_plucker=False` now *keeps* the widened
conv1 and feeds zeros rather than dropping the channels, which makes the
ablation parameter-**identical** rather than merely comparable. Same property,
stronger form.)*

---

## 10. M3 — view-conditioned encoder (implementation record)

**Status: implementation + CPU verification. Superseded by §11, which records
what happened when it was actually run.** This section is kept as the design
record: §10.1 what was built, §10.2 the 16-section CPU test and what it bought,
§10.3 the design decisions and *why*, §10.4 the eight silent-degradation bugs
caught before any run. §10.5 and §10.6 are the two sections the run overtook —
both are now annotated rather than deleted, because *why* the pre-run verification
was insufficient is the useful part.

### 10.1 What was built

| File | |
|---|---|
| `diffusion_policy/model/vision/plucker.py` | Plücker ray maps (torch only, no simulator import) |
| `diffusion_policy/model/vision/view_conditioned_obs_encoder.py` | `ViewConditionedObsEncoder` — drop-in for `MultiImageObsEncoder` |
| `diffusion_policy/env_runner/cam_key_image_runner.py` | `CamKeyImageRunner` + `CamKeyImageWrapper` for training rollouts |
| `diffusion_policy/config/task/m3_plucker_image_abs_multiview.yaml` | K=7 training task |
| `diffusion_policy/config/task/m3_plucker_image_abs_n1.yaml` | the N=1 fidelity gate |
| `diffusion_policy/config/train_diffusion_unet_image_workspace_m3.yaml` | workspace with `policy.obs_encoder._target_` swapped |
| `tests/test_view_conditioned_obs_encoder.py` | 16-section CPU test |
| `preview_viewpoints.py` | renders candidate eval viewpoints to a montage + coverage numbers |

Edited: `dataset/multiview_image_dataset.py` (M3 mode: cam table, per-sample view
draw, mask, camera-frame EE history, identity normalizers), `eval_novel_view.py`
(`--m3-slots`/`--eef-hist-steps`, serves the slot/cam/mask/history keys, an
elevation **orbit**, `elevation_az0` preset), `summarize_novel_view.py` (sort key
for `el_*` names), and the `single_view_image_abs_multiview.yaml` comment (§7.4).

**No upstream package file was modified.** All three edited files were themselves
added by this fork — `git log --diff-filter=A` gives `multiview_image_dataset.py`
ce94279 (M2), `eval_novel_view.py` fa2965f (M1), the M2 config 7205ad8.

### 10.2 What was verified on CPU (not assumed)

`python tests/test_view_conditioned_obs_encoder.py` → **ALL VIEW-CONDITIONED
ENCODER CHECKS PASSED**, 16 sections. In rough order of how much they buy:

- **Plücker convention to 6.66e-16**, over 360 pixel centres at three
  **non-identity** camera poses, with the moment at 2.22e-16 and grid alignment
  0.355° against a 1-pixel bound of 0.565°. Anchored to `project_world_to_pixel`,
  the numpy projector M2's gate 2 validated against the simulator — not to the
  torch code itself.
- **Five mutation power checks**, all caught (R.T / no-rotation 0.62, −R 1.98,
  flipped y row 1.44, xyzw-not-wxyz 1.59, +z-forward-not−z 1.80). This is §9.2's
  lesson made executable: *a convention check that cannot fail proves nothing*,
  and the deleted draft's check was itself wrong. A test with an **identity**
  camera would have been vacuous (`R == Rᵀ == I`), so all poses are non-identity.
- **Matched capacity, end to end**: `output_shape() == (521,)` for all four
  ablation-flag combinations, `global_cond_dim == 1042`, and — the strongest
  form — **all 148 `ConditionalUnet1D` parameter tensors shape-identical to a
  stock-encoder policy's**, by building both UNets and diffing `state_dict`.
  Exactly the 12 `cond_encoder.1.weight` matrices depend on this width.
- **Both ablations are exact, and parameter-identical**: the two flag
  combinations both instantiate **12,532,928** parameters, `use_plucker=False`
  drives the gradient into the 6 ray channels to *exactly* 0.0, and
  `use_eef_hist=False` provably ignores the history tensor.
- **Crop alignment**: image and Plücker map share one sampled window (verified
  exactly by spying on the offsets), and the eval path's `(4,4)` offset
  reproduces `torchvision.center_crop`.
- **N path**: K=1 and K=7 both forward; the fusion is permutation-invariant
  (3e-7), which the per-sample view shuffle depends on.
- **Dataset → encoder pairing**: each slot's camera vector and EE history belong
  to the *same view whose image is in that slot*, recovered from the
  deterministic render rather than assumed. A swapped cam key would otherwise
  train happily and show up only as a mysteriously weak result.
- Config consistency: both M3 configs resolve, and the slot count and
  `eef_hist_steps` agree across the dataset, the encoder and the runner.

Parameter budget (measured): stock encoder 11,176,512; M3 encoder 12,532,928
(**+12.14%**, of which the fusion MHA is 1,050,624 and the widened `conv1`'s 6
extra channels 18,816); UNet 277,632,138. So the M3 encoder is **0.49% of the
UNet**, and the whole model grows **+0.47%** over M1/L1.

### 10.3 Design decisions worth keeping

- **The cam keys, the mask, the EE history — and the extra view slots — all live
  OUTSIDE `shape_meta`.** Two independent reasons: `create_env` builds
  robomimic's obs-modality mapping from `shape_meta`, so a `cam` key makes
  `EnvRobosuite.get_observation` look for a sensor that does not exist and
  `RobomimicImageWrapper.__init__` raises on the key suffix; and with 7 rgb keys
  in `shape_meta` the rollout env would demand 7 cameras, which is exactly why
  the M2 13-view configs disable rollouts. Keeping M3's `shape_meta` at **one**
  rgb key (L1's trick) is what lets M3 keep training rollouts at all.
- **Non-image keys get IDENTITY normalizers, registered explicitly.** Mandatory —
  the policy normalizes every key present and `LinearNormalizer` hard-looks-up
  `params_dict[key]` — but they must not be *fitted*: under `mode='limits'` a
  constant dim maps to 0, which would silently destroy a novel camera pose at
  eval. The tables must be float32 (`_normalize` casts to the scale's dtype and
  `create_manual` does not cast).
- **N is randomized per sample** over `[1, K]`, with the slot count fixed at K
  and a `view_mask` marking the live ones, so the batch stays rectangular and the
  encoder gathers only active views. This makes the N=1 setting that M5, the
  N=1 gate and every rollout run at *in distribution* rather than a shift.
- **AdaGN walks the resnet's submodules explicitly** (`conv1, bn1, relu,
  maxpool, layer1..4, avgpool`) instead of calling torchvision's `forward`, which
  has no conditioning input and would force hooks or module-level state. The
  FiLM heads are zero-init, so training starts unconditioned and the
  conditioning-off ablation is exact.
- **`use_plucker=False` keeps the 9-channel `conv1` and feeds zeros.** Verified
  to give exactly-zero gradient into those channels and numerically identical
  conv output to a 3-channel `conv1` on the image part — so the two variants stay
  parameter-identical (above) and the ablation is exact rather than approximate.
- Matched capacity is **downstream** capacity. The encoder does grow by 12.14%
  (0.47% of the total model); what is held fixed is the width the UNet sees.

### 10.4 Bugs caught before the run (all silent-degradation class)

None of these crashed. Each would have produced a plausible-looking number.

1. **`keepdim=True` on the quaternion normalisation** produced a `(3,3,1)`
   rotation matrix instead of `(3,3)` — *the same bug family as §9.2*, caught by
   a shape assertion rather than by a wrong-looking result.
2. **`reshape(3,-1)` folded a leading view dim into the channel axis**, so
   per-view fovys silently mis-batched.
3. **A shadowed `rows` index** in the slot gather broke the row-major
   correspondence between the gathered views and their mask/history.
4. **The EE-history flatten dim** was allocated per-step instead of flattened,
   which would have thrown — but only at the first batch.
5. **Two crop-offset sharing mistakes**, i.e. cropping the image and the ray map
   with different windows: precisely the desynchronisation the design exists to
   avoid.
6. **`torch.cat` on a 0-dim tensor** for a scalar fovy.
7. **A normalisation asymmetry**: the numpy `quat_wxyz_to_mat` did not normalise
   while the batched and torch versions did. Found because a test fed a quaternion
   that was unit only to float precision and the look-at point landed 3.5e-6 px
   off centre. All three now normalise. (`_look_at_quat` itself returns a
   **float32** quaternion via robosuite's `mat2quat`, worth ~3e-8 rad ~ 3e-6 px
   of centring error; this is pre-existing, affects the committed M1/L1 sweeps
   identically, and was deliberately NOT changed — perturbing geometry that
   already produced committed results costs comparability for no measurable gain.)
8. **The M2 N=1 gate config's false claim** that rollouts were disabled (§7.4).

### 10.5 What was NOT verified before the run — and how each one turned out

**All four items below are now closed (§11.1). Kept because the list was
accurate and the outcome vindicates it: three of the four were hiding something,
and none of them could be caught without the simulator.**

- **The rollout runner (`cam_key_image_runner.py`) and the eval-harness serving
  path had never executed.** → **Two fatal bugs.** Both would have killed every
  M3 run at epoch 0 (§11.1). Import checks and `__mro__` assertions passed and
  proved nothing about the obs-key contract at runtime.
- **All dataset verification used the synthetic zarr**, not a real store. →
  **Held.** The real zarrs load, sample, and vary N per sample as designed; the
  only correction is that `numcodecs 0.10.2` has no `jpeg2k`, so anything
  opening these zarrs must import the dataset module first (`register_codecs()`).
- **`preview_viewpoints.py`'s render path was unexecuted.** → **Held** — it ran
  first try, and its montages are what licensed the elevation sweep.
- **The epoch cost was an estimate.** → **2.1× optimistic** in the doc's "~15 h
  for nine runs"; measured 43 s/epoch for square, ~1.9× M1 (§3).

The generalisable lesson: a CPU test can verify *shape* and *convention*
contracts (§10.2 did, thoroughly), but not the *wiring* between independently
written components. The obs-key mismatch was a wiring bug between three modules
each individually correct.

### 10.6 The training matrix and its gates

> **What was actually run (2026-09-19):** 7 of the 9 cells, then stopped
> deliberately. Square 2×2 (all four), can `m3off` + `m3on` (the two endpoints),
> and the N=1 gate. **Skipped: can `m3plucker`/`m3eef` and lift `m3on`.** The
> square 2×2 showed conditioning is inert and both can endpoints agreed, so the
> four single-signal cells could not change any conclusion (§11.6). Lift remains
> a genuine open question (§6). The gates below were followed as written; gate
> ordering is what made the stop cheap.

Nine runs, seed 42, batch 64, 201 epochs, identical shape/optimiser settings to
M1/L1. Cells differ only by CLI override, so the only variable is conditioning.
**M3-off carries the attribution** — M3-on vs L1 conflates "variable multi-view"
with "conditioning" and must not be quoted alone.

| run dir | task | Plücker | EE-hist FiLM | isolates |
|---|---|---|---|---|
| `run_{square,can}_m3off_s42_200ep` | square, can | off | off | the "more slots / variable N" effect |
| `run_{square,can}_m3plucker_s42_200ep` | square, can | **on** | off | the geometry effect |
| `run_{square,can}_m3eef_s42_200ep` | square, can | off | **on** | the camera-frame motion effect |
| `run_{square,can}_m3on_s42_200ep` | square, can | **on** | **on** | the headline |
| `run_lift_m3on_s42_200ep` | lift | **on** | **on** | regression: do not break what L1 solved |

Gates, in order:
0. **Preflight on the box** — both CPU suites pass there, the three zarrs exist,
   `df -h`, then `preview_viewpoints.py --preset elevation_az0` and *look at the
   montage*. A viewpoint set that cannot see the scene produces a
   plausible-looking curve, so this is checked before any sweep.
1. **1-epoch timing run** — replaces the §10.5 cost estimate with a measurement.
2. **N=1 gate** (`m3_plucker_image_abs_n1.yaml`, square) — must roughly
   reproduce M1/L1's az_0 result. If it does not, the Plücker or AdaGN path is
   corrupting the image path, and that must be found before ~15 h of runs.
3. **Square 2×2 → sweep → read it** before spending the remaining ~7 h.

Evaluation: `azimuth_interp` (11 viewpoints, ±75°, all 9 checkpoints — az_0/±30
coincide with M1, keeping the comparison exact) plus `elevation_az0` (3
viewpoints, 0/±15°). **Elevation, not azimuth, is how extrapolation is tested:
azimuth past ±90° was rejected because those views "can hardly see robot and
table", matching §7.1's montage finding.** The elevation viewpoints are an
**orbit** about the look-at point, not the harness's existing `elevation_deg`
pitch — the pitch swings the scene out of frame, which is the opposite of the
requirement. ±30° elevation was considered and dropped for the same reason.

`eval_novel_view.py --m3-slots 7` reduces the env-side `shape_meta` to the single
renderable key and serves the perturbed pose into slot 0, i.e. an **N=1**
inference — M5's setting, and in distribution because N was randomized during
training.

Stated follow-ups, deferred so the headline 2×2 stays L1-comparable: a **±60°
training pool** (how much does view *quality* alone buy?) and a **smaller K**
(PROPOSAL §7's "how much view diversity is needed", one config line).

---

## 11. M3 — RESULTS (2026-09-19, training box, seed 42)

**Headline: M3 solves square and can — the two tasks L1 destroyed — and its
conditioning contributes nothing measurable to that.** §10.6 predicted `m3off`
would carry the attribution; it does, and the attribution is **architectural**,
not the Plücker map and not the camera-frame history.

All numbers are strict `EnvRobosuite.is_success()` `success_rate`, 50 paired
episodes per viewpoint, seed 42, read from `latest.ckpt` (the epoch-200 model) —
the same convention as every M1/L1 number.

### 11.1 What the run cost, and what it caught first

Two bugs, both **fatal at epoch 0**, both in the paths §10.5 listed as
never-executed. Fixed in `44b975a`, before any long run — the probe phase is
what found them, and §10.5's ordering is why that cost ~35 min instead of ~15 h.

1. **Normalizer key mismatch** (rollout runner *and* eval harness). Both build
   an env-side `shape_meta` carrying one rgb key, `agentview_image`, so
   `create_env` can build robomimic's obs-modality mapping. That key reaches the
   obs dict via `MultiStepWrapper._get_obs`, and `predict_action` normalizes
   every key with no fallback — while an M3 normalizer has entries only for the
   slots, cam, mask, history and low-dim keys. Fix: drop the render key from the
   observation **space** (not the returned dict) once the M3 keys are registered;
   `render_cache` reads `raw_obs`, so video is unaffected. Guarded by
   `m3_slots > 0`, leaving M1/L1 (whose policy key *is* the render key) untouched.
2. **Struct-mode `DictConfig`** (eval harness only). `eval_novel_view.py` loads
   its cfg from the checkpoint's dill payload, where DictConfigs are in struct
   mode, so `del shape_meta['obs'][k]` raises `ConfigTypeError`. `deepcopy`
   preserves the flag. The runner escaped this because `hydra.utils.instantiate`
   hands it an unlocked copy — **which is exactly why probe 3a passed while 3b
   failed, and why running both was necessary.**

Verification after the fix: the obs key set matches the normalizer's **exactly**
— 19 keys at K=7, 7 at K=1, zero difference in either direction. The rollout
runner completed all 4 chunks and logged `test/mean_score` at epoch 0; the
`--m3-slots 7` eval wrote metrics for all three elevation viewpoints.

**Gate 2, the N=1 fidelity control** (`task=m3_plucker_image_abs_n1`, single
slot, `view_pool=[6]` = az_0) — a *fidelity* control, not a result:

| viewpoint | N=1 gate | M1 (hdf5, 1 view) | L1 (7 views) |
|---|---|---|---|
| az_0 | **0.94** | 0.82 | 0.02 |
| az_p30 | 0.00 | 0.00 | 0.00 |
| az_m30 | 0.00 | 0.00 | 0.00 |

It reproduces M1's **whole curve**, not just its az_0 point — §8.5's standard.
0.94 is *above* M1's 0.82, which §10.6 flagged as a possible leak; it is not one
(`view_pool=[6]` with `view_count_range=[1,1]` admits no other view) and the
same +0.08–0.12 pattern appeared in §8.5's lift gate. Treat the generated zarr as
a slightly cleaner source than the hdf5 pipeline.

### 11.2 The result: novel-view success on square and can

`*` = pose IS in the training pool (the ring's even indices, every 30°).
**Unmarked columns are never trained on** and are the actual test.

**Square** — `azimuth_interp` (11 vp) + `elevation_az0` (3 vp):

| cell | m75 | m60* | m45 | m30* | m15 | 0* | p15 | p30* | p45 | p60* | p75 | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m3off | 0.40 | 0.78* | 0.60 | 0.78* | 0.68 | 0.78* | 0.42 | 0.70* | 0.58 | 0.78* | 0.62 | 0.88 | 0.30 | 0.04 |
| m3plucker | 0.34 | 0.78* | 0.50 | 0.66* | 0.56 | 0.80* | 0.56 | 0.78* | 0.50 | 0.78* | 0.54 | 0.76 | 0.24 | 0.02 |
| m3eef | 0.48 | 0.84* | 0.66 | 0.66* | 0.68 | 0.80* | 0.48 | 0.74* | 0.50 | 0.80* | 0.64 | 0.82 | 0.18 | 0.02 |
| m3on | 0.50 | 0.72* | 0.56 | 0.72* | 0.66 | 0.76* | 0.52 | 0.74* | 0.58 | 0.74* | 0.46 | 0.78 | 0.18 | 0.00 |
| *L1 s42* | 0.02 | 0.02* | 0.00 | 0.00* | 0.00 | 0.02* | 0.02 | 0.00* | 0.02 | 0.06* | 0.04 | 0.02 | 0.02 | 0.00 |
| *M1* | — | — | — | 0.00 | 0.02 | 0.82 | 0.00 | 0.00 | — | — | — | 0.88 | 0.00 | 0.00 |

**Can**:

| cell | m75 | m60* | m45 | m30* | m15 | 0* | p15 | p30* | p45 | p60* | p75 | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m3off | 0.70 | 0.88* | 0.76 | 0.86* | 0.74 | 0.92* | 0.80 | 0.92* | 0.80 | 0.90* | 0.64 | 0.90 | 0.34 | 0.22 |
| m3on | 0.84 | 0.86* | 0.74 | 0.86* | 0.74 | 0.86* | 0.82 | 0.92* | 0.82 | 0.84* | 0.64 | 0.88 | 0.28 | 0.22 |
| *L1* | 0.02 | 0.00* | 0.06 | 0.02* | 0.02 | 0.02* | 0.02 | 0.02* | 0.02 | 0.04* | 0.00 | 0.02 | 0.00 | 0.06 |
| *M1* | — | — | — | 0.00 | 0.08 | 0.98 | 0.00 | 0.00 | — | — | — | 0.98 | 0.04 | 0.00 |

Means, split by whether the pose was trained:

| model | square trained | square **held out** | can trained | can **held out** |
|---|---|---|---|---|
| m3off | 0.764 | **0.550** | 0.896 | **0.740** |
| m3plucker | 0.760 | **0.500** | — | — |
| m3eef | 0.768 | **0.573** | — | — |
| m3on | 0.736 | **0.547** | 0.868 | **0.767** |
| *L1* | *0.02* | *0.00–0.06* | *0.02* | *0.00–0.06* |

**Read this carefully, because it is easy to overstate.** There *is* a real
trained/held-out gap (square 0.76 → 0.55, can 0.90 → 0.74). But the held-out
**floor is 0.40–0.84, not 0.00**, where M1 is 0.00–0.02 off-axis and L1 is at the
noise floor.

**The load-bearing comparison is L1 at trained poses.** L1 was trained on the
same pool and still scores ~0.02 at ±30° — poses it saw constantly. So L1's
failure was never a failure to *generalise*; it failed to solve the task at all.
"M3 generalises to novel views" is therefore the weaker of the two claims on
offer; the stronger one is that **per-slot fusion turns view diversity from
harmful into sufficient**.

### 11.3 The conditioning null

`m3off` — `use_plucker=False`, `use_eef_hist=False` — matches or beats `m3on` on
both tasks at every viewpoint, and `m3on` is never the best square cell:

- square held-out means span **0.500–0.573** across all four cells
- square trained means span **0.736–0.768**
- can: `m3off` 0.740 held-out vs `m3on` 0.767 — the *wrong* direction for the
  hypothesis, and inside ±0.05 noise either way

All inside the documented ±0.05 rollout noise, with no systematic sign. The
ablation is genuine, not a flag that failed to apply — verified in the EMA
weights: the six ray channels are non-zero **only** in the two `use_plucker=True`
cells, exactly zero in `m3off`/`m3eef` as the zero-gradient design predicts, and
all four checkpoints' `conv1` and `fusion_query` hashes are distinct.

**So the proposal's central claim is not supported.** PROPOSAL §7 asked whether
the aux heads need the camera-frame history *as well as* the Plücker map, or
whether one suffices; the answer here is that **neither** moved the number. What
moved it was 7 slots, per-sample N∈[1,7], and attention fusion — three changes
that came in as plumbing for the conditioning rather than as the hypothesis.

### 11.4 Elevation — a partial extrapolation result

The `elevation_az0` orbit is the off-manifold test (§10.6 chose it over azimuth
past ±90°, which §7.1's montages showed is dominated by the table edge).

M3 holds **0.18–0.34 at el_p15** on both tasks where M1 is 0.00–0.04 — and
**0.22 at el_m15 on can**, where M1 is 0.00 and L1 is 0.06. But on **square,
el_m15 collapses to 0.00–0.04** for every M3 cell. So extrapolation holds going
up and not down, on one of two tasks. That asymmetry is unexplained and is a
concrete follow-up, not a rounding error.

### 11.5 Limits — what these results do not show

- **n = 1 per cell, 50 paired episodes, ±0.05 noise.** The conditioning null is
  "indistinguishable at this resolution", **not** "proven identical". A real
  effect below ~0.1 would be invisible here, and §11.3 should be quoted with
  that bound attached.
- **The architecture is confounded four ways.** Relative to L1, four things
  changed at once (§6). Nothing here says *which* one matters.
- **Only square and can.** Lift is untested for M3 (§11.6) — and lift is the
  task where L1 already wins, so it is the one place M3 could regress.
- **`mean_score` in-training rollouts hide things.** All M3 cells sit at
  0.76–0.94 there while differing by up to 0.12 in the strict sweep; §8.6's
  warning holds. Worse, **`val_loss` does not track rollout behaviour at all**:
  all four square cells sit at 0.0568–0.0602 — essentially L1's 0.060 — while
  rolling out like M1. §8.4 kept "L1 fits worse, so it fails" as an *inferred*
  mechanism; this is direct evidence against it.
- **M3 costs a little in-distribution.** az_0 0.76–0.80 vs M1's 0.82. Small,
  but it is not a free lunch.

### 11.6 What was and was not run

Ran: N=1 gate; square `m3off`/`m3plucker`/`m3eef`/`m3on`; can `m3off`/`m3on`;
full `azimuth_interp` + `elevation_az0` on all six. Plus the M1/L1 elevation
baselines (§11.7). All 201 epochs, seed 42, batch 64.

**Not run, deliberately:** can `m3plucker`/`m3eef` and lift `m3on`. The square
2×2 established the conditioning null across the full 2×2, and both can
endpoints agreed with it, so the two remaining can cells could not have changed
a conclusion. Lift is a real open question and is listed in §6.

### 11.7 The elevation baselines (M1 and L1), captured before the M1 weights were deleted

§10.6 added `elevation_az0` to the evaluation protocol, but M1 and L1 had never
been swept on it, and the M1 checkpoints were deleted to make room for M3. These
were captured first. `el_0` is the same camera pose as `az_0`, which is the
internal check:

| task | model | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|
| square | M1 | 0.88 | 0.00 | 0.00 |
| square | L1 | 0.02 | 0.02 | 0.00 |
| can | M1 | 0.98 | 0.04 | 0.00 |
| can | L1 | 0.02 | 0.00 | 0.06 |
| lift | M1 | 0.74 | 0.00 | 0.02 |
| lift | L1 | **0.96** | **0.36** | **0.26** |

`el_0` reproduces each model's committed §1/§8.3 `az_0` value, which validates
the orbit's zero-point. The lift row is a genuine extension of §8.3's 2–1 task
split onto a second axis: **L1's lift advantage survives into elevation**
(0.96/0.36/0.26 vs M1's 0.74/0.00/0.02) — degraded from its azimuth
performance, but far above M1 everywhere. One more task-side datapoint for §8.4's
unresolved "what separates lift" question.

### 11.8 Artifacts

| what | where | in git |
|---|---|---|
| sweeps (all of the above) | `data/eval_{gate,el,interp}_*/eval_log.json` | ✅ `5ee979b`, `cc0696e` |
| training curves | `data/outputs/run_*m3*/logs.json.txt` | ✅ (this commit) |
| code fixes | `44b975a` | ✅ |
| weights (4.6 GB each) | `data/outputs/run_*m3*/checkpoints/latest.ckpt` | ❌ rsync only |
| campaign log (ordered, timestamped) | `data/m3_campaign.log` | ❌ on the box |

Reproduce any cell:

```bash
python train.py --config-name=train_diffusion_unet_image_workspace_m3 \
  task=m3_plucker_image_abs_multiview task.task_name=square \
  policy.obs_encoder.use_plucker=false policy.obs_encoder.use_eef_hist=false \
  training.seed=42 training.num_epochs=201 training.device=cuda:0 \
  dataloader.num_workers=14 val_dataloader.num_workers=2 \
  checkpoint.topk.k=1 logging.project=diffusion_policy_view \
  hydra.run.dir=data/outputs/run_square_m3off_s42_200ep

python eval_novel_view.py -c data/outputs/run_square_m3off_s42_200ep/checkpoints/latest.ckpt \
  -o data/eval_interp_square_m3off -d cuda:0 --preset azimuth_interp \
  --m3-slots 7 --eef-hist-steps 4 --n-envs 14 --n-test-vis 0
```

`--m3-slots` is **7 for all four cells including `m3off`** — the assert compares
against the checkpoint's own `shape_meta`, and the cam/mask/history normalizer
entries exist regardless of the flags.

---

## 12. M4 — per-view auxiliary heads (IMPLEMENTED, CPU-verified, NOT RUN)

**Status: code complete and passing on CPU; zero GPU time spent.** This section
is the implementation record, in the form of §10. There are no results yet.

### 12.1 What this is for

§11.3 measured M3's conditioning as inert. But **nothing in M3's objective ever
required the encoder to use the camera** — the only supervision was action
diffusion, and at a trained view the image alone predicts the action. PROPOSAL
§2.4 names the per-view auxiliary heads as what "forces `z_v` to be geometrically
meaningful rather than merely view-tagged": predicting an action *in camera k's
frame* from a single view requires `z_v` to encode where the scene sits relative
to that camera.

So M4 is not an add-on to an inert input; it is the missing pressure. The user's
framing for this milestone: **hold the M3 model fixed and add one part at a
time.** The encoder's forward path is unchanged.

### 12.2 Two design points that decide whether it works

1. **The chunk must be `n_action_steps` (8), not `horizon` (16).** Obs step `to`
   reads `action[to : to+C]`, so the last index touched is `C + n_obs_steps - 2`.
   At `To=2` a full-horizon target reaches index 16, which does not exist — it is
   `pad_after` **edge-repeat**, i.e. fake supervision the head would happily fit.
   The dataset ctor rejects `C + n_obs_steps - 1 > horizon`.
2. **The rot6d rotation must round-trip through the full 3×3 matrix.** The 6d
   encoding keeps only rows 0 and 1 of `R`, but `rows01(R_cᵀ R_b)` depends on all
   three rows, so rotating the 6d vector with a 6×6 linear map is **not** the
   transform — **and is exactly right when `R_c == I`**, which is why a test at
   the base camera would prove nothing. `action_to_cam` goes through
   `rot6d_to_mat`, which the test pins to pytorch3d (the code path that
   *produced* the stored actions).

### 12.3 Files (all new except one seam)

| File | |
|---|---|
| `diffusion_policy/policy/diffusion_unet_image_policy_aux.py` | `DiffusionUnetImagePolicyAux` + `masked_view_mean` |
| `diffusion_policy/model/vision/per_view_aux_head.py` | `PerViewAuxActionHead` (151,888 params, 0.052% of the policy) |
| `diffusion_policy/config/task/m4_aux_image_abs_multiview.yaml` | M3's task + `emit_aux_action`, `aux_n_steps` |
| `diffusion_policy/config/train_diffusion_unet_image_workspace_m4.yaml` | policy `_target_` swap + `aux_loss_weight: 1.0` |
| `tests/test_aux_action_heads.py` | the CPU suite |

Edited: `multiview_image_dataset.py` (adds `action_to_cam`/`rot6d_to_mat`, emits
the target as a **top-level** key, fits a normalizer entry),
`view_conditioned_obs_encoder.py` (adds `forward_full`), and **one documented
seam in the upstream workspace**
(`train_diffusion_unet_image_workspace.py`: a `getattr`-guarded
`step_log['aux_loss']`). `diffusion_unet_image_policy.py` is upstream and is
**unmodified** — the new policy subclasses it.

Design choices worth keeping:

- **The target is a top-level sample key, never an obs key.** It derives from
  demonstrated future actions, which do not exist at rollout, and the policy
  normalizes every obs key.
- **The aux head is built unconditionally**, including at weight 0, and consumes
  no RNG. Combined with the encoder running exactly once per step in both arms
  (`forward_full` vs `forward`, bit-identical), **the two ablation arms are
  RNG-locked**: every crop, noise and timestep draw is identical for the whole
  run, and they differ by exactly one scalar term and its gradient. That is
  stronger than "aux-off reproduces M3 modulo RNG".
- **Zero-init output layer** (this repo's pattern for added branches), so an
  aux-on run starts as M3. This costs one step, not a permanent delay: the
  head's own weights get gradient immediately, so the trunk sees the aux
  gradient from step 1.
- **One shared head for all slots** — a per-slot head could key off slot
  identity, which the dataset deliberately randomizes away, and could not train
  when a slot is inactive.

### 12.4 What was verified on CPU

`python tests/test_aux_action_heads.py` → **ALL AUX ACTION HEAD CHECKS PASSED**:

- `rot6d_to_mat` == `RotationTransformer('rotation_6d','matrix')` to 3.3e-16 over
  200 random rotations, and `action_to_cam` anchored to `quat_wxyz_to_mat` (the
  numpy camera math M2's gate 2 validated against the simulator) and **pinned to
  `eef_hist_to_cam` exactly (0.00e+00)** over 100 non-identity cameras.
- **Mutation power on 7 mutations**, all caught > 1e-3: `R_c` instead of `R_cᵀ`,
  dropped translation, rot6d or pos left in the base frame, first two *columns*
  instead of rows, `R_e R_cᵀ` order, and the 6×6 shortcut — with the executable
  assertion that the shortcut **passes at `R_c = I` and fails elsewhere**.
- Dataset: the target's shape/mask/pairing, where **each live slot's view is
  recovered from the renders** and its target recomputed independently (catches
  an off-by-one cam table, the failure that trains happily and reads as "aux
  didn't help"); inactive slots exactly zero; the flag consumes **no** RNG; the
  ctor's boundary checked from both sides (accepts 15, rejects 16).
- Policy: `forward == forward_full` **bit-for-bit**; `output_shape` still (521,);
  **`aux_loss_weight=0` is bit-identical to the parent's loss and all 179
  gradients** (the anti-drift pin on the copied body); `w=1` equals
  `diff + w·aux`; inactive slots cannot affect the aux loss; the head is exactly
  zero at init and, once non-zero, the aux gradient demonstrably reaches
  `conv1`'s image **and** ray channels.
- Configs: they compose, the encoder block is **byte-identical to M3's**, the
  policy instantiates through the real hydra path with `kwargs` empty (an aux key
  leaking into `**kwargs` would raise inside `scheduler.step` at **rollout**
  time), and `dataset.aux_n_steps == policy.aux_n_steps`.

**The suite caught one real bug**, and it is the exact failure mode §12.3's
"copied body" note predicts: `_compute_loss_with_aux` originally fed the UNet
`z_global` alone (512-wide) instead of the full `[z_g | low-dim]` concatenation
`forward` performs (521-wide), so `global_cond` came out 1024 instead of 1042.
The weight-0 arm could not catch it — that path calls the parent.

### 12.5 The run matrix (gates, not yet executed)

| # | run | why |
|---|---|---|
| 0 | the CPU suite **on the box** | cheap, and it is what caught the bug above |
| 1 | square `m4on` (`weight=1.0`) | headline |
| 2 | square `m4off` (`weight=0.0`) | the control |
| 3 | can `m4on` / `m4off` | only after the square pair is read |

```bash
python train.py --config-name=train_diffusion_unet_image_workspace_m4 \
  task=m4_aux_image_abs_multiview task.task_name=square training.seed=42 \
  training.num_epochs=201 training.device=cuda:0 dataloader.num_workers=14 \
  val_dataloader.num_workers=2 checkpoint.topk.k=1 \
  logging.project=diffusion_policy_view \
  hydra.run.dir=data/outputs/run_square_m4on_s42_200ep
# control: add policy.aux_loss_weight=0.0, run dir ..._m4off_...

python eval_novel_view.py -c data/outputs/run_square_m4on_s42_200ep/checkpoints/latest.ckpt \
  -o data/eval_interp_square_m4on -d cuda:0 --preset azimuth_interp \
  --m3-slots 7 --eef-hist-steps 4 --n-envs 14 --n-test-vis 0
```

`m4off` is a fresh run, not the committed `m3on`: the extra parameters shift the
RNG stream, so `m3on` is not a control for this pair — it is a *validation* that
`m4off` lands inside the ±0.05 noise band.

**The readout that M3 lacked.** §11.5's problem was measurement: rollout noise is
±0.05 and `val_loss` did not track rollout behaviour at all. The aux curve is a
directly measured training-time quantity that is sensitive to whether `z_v`
carries view information, and it now lands in `logs.json.txt` / wandb as
`aux_loss`. An aux head plateauing near its init value means the task is
unlearnable from one view and the mechanism is inert — M3's failure mode, caught
in the first 50 steps instead of after 200 epochs. Watch `aux/diff` too
(≈0.1–0.3 at init on real data) and check az_0 for regression before committing
hours: too small a weight is an inert mechanism reading as a null result, too
large over-writes `z_v`.

Optional third cell, scientifically the sharpest: **`m4on` with
`use_plucker=False`**. If the aux head only helps when the ray path is live, that
is PROPOSAL §2.4's claim directly — and it is the cell M3 never had.

### 12.6 Limits and risks, stated up front

- **n=1 per cell, 50 paired episodes, ±0.05 noise** — the same resolution §11.5
  flags, so a small effect remains invisible.
- **`train_loss` and `val_loss` now include the aux term** when the weight is
  non-zero, so they are not comparable to M3's; the separate `aux_loss` series is
  what makes the decomposition readable.
- **The weight is a guess** (1.0, PROPOSAL's literal "summed"). No sweep.
- **M4 only removes the "nothing requires the pose" objection.** It does not
  address §6's four-way architectural confound, the view-pool question (§7.1's
  ±75/±90 are still trained on while ±15/±45 are still held out), or the
  elevation asymmetry.

---

## 13. M4 — per-view auxiliary heads (RESULTS, 2026-09-22)

**Headline: the mechanism is real and the behavioural effect is null.** `aux_loss`
falls **52×** and the head provably uses per-view information — but `m4on` − `m4off`
is **+0.02 trained / +0.04 held-out**, inside the ±0.05 noise floor, and M4 sits
inside M3's band on every axis. It did **not** revive the conditioning.

The pair is the project's only **RNG-locked** A/B: same architecture (the aux head is
built unconditionally), same seed, batch order and worker count, one scalar term
different. M3's four cells were parameter-identical; these two are *draw*-identical.

### 13.1 What was run

| | |
|---|---|
| runs | `run_square_m4on_s42_200ep`, `run_square_m4off_s42_200ep` |
| cell | M3's model (`plucker`+`eef` on) + per-view camera-frame aux head |
| variable | `policy.aux_loss_weight` **1.0** vs **0.0** |
| protocol | 201 epochs, seed 42, batch 64, `num_workers` 10/2, `topk.k=1`, exit 0 both |
| records | 88,440 per run = 440 batches × 201 epochs (no truncation) |

**The weight was checked before committing hours, and the config's own comment was
wrong.** It predicted `aux/diff ≈ 0.1` at init; the measured batch-0 value is
**0.316** (diff 1.0816, aux 0.3415), and the CPU suite independently says 0.301 on
synthetic data. The reason is that `robomimic_abs_action_only_normalizer_from_stat`
range-normalizes only the 3 position dims — the rot6d (6) + gripper (1) dims sit at
scale 1, so the target is ~3× larger in normalized units than the comment assumed.
A rule was fixed in advance (proceed unless the ratio exceeded 1.0), so 1.0 stayed —
PROPOSAL §2.4's literal "summed", with no post-hoc tuning.

### 13.2 The A/B is a null

Strict `EnvRobosuite.is_success()`, 50 paired episodes, `azimuth_interp` + `elevation_az0`:

| model | trained\* | **held-out** | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|---|
| m3off | 0.76 | **0.55** | 0.88 | 0.30 | 0.04 |
| m3plucker | 0.76 | 0.50 | 0.76 | 0.24 | 0.02 |
| m3eef | 0.77 | 0.57 | 0.82 | 0.18 | 0.02 |
| m3on | 0.74 | 0.55 | 0.78 | 0.18 | 0.00 |
| **m4on** | **0.76** | **0.56** | 0.70 | 0.00 | 0.00 |
| **m4off** | **0.74** | **0.52** | 0.80 | 0.14 | 0.00 |
| *delta* | *+0.02* | *+0.04* | *−0.10* | *−0.14* | *0.00* |

Per-viewpoint deltas run −0.12 to +0.18 with the sign flipping freely: at ±15°, ±45°
and ±75° `m4on` is **behind**, at −45°/+45° well ahead. The means (+0.02, +0.04) are
the noise floor. **The aux term buys nothing measurable**, and M4 does not regress
either — it is inside M3's 0.50–0.57 held-out band.

The RNG lock was verified **exactly**, not statistically: `m4on`'s batch-0
`train_loss − aux_loss` = **1.081642270**, identical to `m4off`'s `train_loss` =
1.081642270, difference `0.000e+00`. The two arms saw identical batches, crops, noise
draws and timesteps.

### 13.3 The mechanism is alive — the part that is new

`aux_loss` collapses, which is the opposite of M3's failure mode:

| epoch | 0 (mean) | 5 | 15 | 25 | 50 | 200 |
|---|---|---|---|---|---|---|
| `aux_loss` | 0.2389 | 0.0304 | 0.0206 | 0.0173 | 0.0147 | **0.0066** |

A **52×** reduction from the 0.3415 batch-0 value. Flatness at the init value would
have meant "unlearnable from one view" — M3's failure mode caught in 50 steps. It is
not flat.

**And it is not a collapse to the mean.** The aux loss is mean-over-active-views then
mean-over-frames MSE in normalized camera-frame action space, so a **constant** head
scores the pooled target variance. Measured on 1024 real dataset samples
(frames × active views, 80 dims): that floor is **0.2692**, and the epoch-25 loss
(0.0173) is **6.4% of it** — the head is 15.6× below what a mean-collapsing predictor
can reach. Since the floor already pools over *rotated* per-view targets, beating it
that hard means the head reads view-relative geometry out of `z_v`. Pipeline check:
the same script's zero-predictor estimate (0.3469) matches the actual batch-0 value
(0.3415) to 1.6%, comparing 1024 samples against 64.

### 13.4 The trunk was reshaped as much as a re-seed

A weight-space diagnostic on the EMA weights (what rollouts use), relative L2:

| module | params | aux effect (on vs off @200) | **RNG-only reference** |
|---|---|---|---|
| `obs_encoder` | 12,532,928 | 0.596 | **0.471** |
| `model` (UNet) | 277,632,138 | 0.856 | **0.824** |
| `aux_head` | 151,888 | 1.167 | — |
| `normalizer` | 2,526 | **0.00000** | 0.00000 |

The reference column is `m4off` vs the committed `m3on`: same seed, same shapes,
**different RNG stream only** (§12.5's note that a fresh `m4off` is not an RNG
control for `m3on` — which is exactly what makes it a clean *scale* here).

The aux term moves the weights **about as much as re-seeding does**. So the null is
**not** explained by "the gradient was too weak to move anything" — that hypothesis
is dead. Two controls confirm the comparison is sound: `m4off`'s aux head drifts
1e-5 over 50 epochs (provably never touched, as the weight-0 path predicts), and
`normalizer` differs by *exactly* zero.

Caveat, stated because the metric invites over-reading: weight-space distance is
crude — SGD accumulates drift along directions that need not matter functionally. The
defensible claim is "a perturbation of the same order as run-to-run variation,
producing no behavioural change".

### 13.5 What the null means

The surviving explanation is the one §12.1 did not consider: **the information the aux
head needs is already in `z_v`.** M3 reaches 0.55 held-out with *no* aux supervision
at all, and a 151,888-param head (0.052% of the policy) can fit camera-frame targets
post hoc from a latent that already encodes the scene. The aux pressure then adds
nothing the diffusion objective had not already produced — which is why it can reshape
the weights substantially while changing no behaviour.

§12.6 flagged that the weight is a guess with no sweep. That stands as a limit, but it
is no longer the leading explanation.

### 13.6 Elevation — a caveat that cuts both ways

At face value M4 is *worse* than M3 at elevation (`m4on` 0.00 vs `m3on` 0.18 at
`el_p15`). **That reading does not survive its own internal check.** `el_0` is the
*same camera pose* as `az_0`; for `m4on` they read **0.70 and 0.84** — a **0.14**
spread on an identical pose from two separate sweeps. §1 documents unseeded diffusion
sampling giving ~0.08 spread on repeat evals of one checkpoint, so the effective noise
on these 50-episode sweeps **exceeds the ±0.05 the plan assumed**.

Both directions of over-claiming are blocked by the same measurement: the elevation
dip is not established, and §13.2's null is correspondingly less precise than
"+0.04 vs ±0.05" makes it sound.

One more caution from this session: `val_loss` **rises ~3×** after epoch ~50 for every
model (`m3off` 0.0190→0.0602, the N=1 gate 0.0189→0.0499, L1 0.0198→0.0599) —
overfitting on the 2% val split, and M3 and L1 end statistically identical while
rolling out 0.55 vs 0.02. **Compare `val_loss` only at matched epochs.** `m4on`'s
`val_loss` is not comparable to `m4off`'s at all: it includes the aux term
(0.0186 diff + 0.017 aux = 0.037 = observed 0.0372).

### 13.7 Limits

- **n = 1 per cell**, 50 paired episodes. The null is "indistinguishable at this
  resolution", and §13.6 shows the resolution is worse than ±0.05.
- **The aux weight is a guess** (1.0, unswept) — though §13.4 removes "too weak" as
  the explanation.
- **Square only.** Can and lift were not run; §12.5's gate said can only after the
  square pair was read, and the readout made the *mechanism* question the live one.
- The elevation arms differ by 0.14 — same order as the noise, so no attribution.

### 13.8 Artifacts

| what | where | in git |
|---|---|---|
| sweeps | `data/eval_{interp,el}_square_m4{on,off}/eval_log.json` | ✅ |
| training curves (+ the `aux_loss` series) | `data/outputs/run_square_m4{on,off}_s42_200ep/logs.json.txt` | ✅ |
| probe evidence | `data/probe_m4_square_logs.json.txt` | ✅ |
| campaign log (ordered, timestamped) | `data/m4_campaign.log` | ❌ on the box |
| weights (4.6 GB each) | `.../checkpoints/latest.ckpt` | ❌ rsync only |

Reproduce:

```bash
python train.py --config-name=train_diffusion_unet_image_workspace_m4 \
  task=m4_aux_image_abs_multiview task.task_name=square policy.aux_loss_weight=1.0 \
  training.seed=42 training.num_epochs=201 training.device=cuda:0 \
  dataloader.num_workers=10 val_dataloader.num_workers=2 \
  checkpoint.topk.k=1 logging.project=diffusion_policy_view \
  hydra.run.dir=data/outputs/run_square_m4on_s42_200ep
# control: policy.aux_loss_weight=0.0

python eval_novel_view.py -c .../latest.ckpt -o data/eval_interp_square_m4on \
  -d cuda:0 --preset azimuth_interp --m3-slots 7 --eef-hist-steps 4 --n-envs 14 --n-test-vis 0
```

Four config traps cost time if the CLI overrides are dropped: `checkpoint.topk.k` is
**5** (up to ~23 GB/run), `dataloader.num_workers` is **4** (measured 81 s/epoch vs 43
at 14), `logging.project` is `diffusion_policy_debug`, and `training.resume: True`
**silently resumes** an existing run dir.

---

## 14. §6's architectural confound — RESOLVED (2026-09-23)

**Headline: M3's gain comes from multi-view sampling (N>1).** §6 listed four things
that changed at once between L1 and M3 and called the variable-N-vs-fixed-N split "the
cheapest informative one". It is now run, and the answer is unambiguous.

### 14.1 The cell

`run_square_m3fixedn1_s42_200ep`: **M3-off's exact model**, with one CLI line changed —
`task.dataset.view_count_range=[1,1]`. In `_m3_slots` that forces `k_active = 1`, and
the single view is drawn `np.random.choice(view_pool, size=1)` from the **same 7-pose
pool**. So it is L1's training distribution routed through the M3 encoder (7 slots, 1
active, widened `conv1` fed zeros, MHA fusion degenerating on one token).

Not RNG-locked to `m3off` (the N draw consumes RNG differently) — a between-run
comparison, which is why the effect size matters more than the third decimal. Cost:
**21 s/epoch**, 201 epochs, exit 0.

### 14.2 The prediction, registered before the run

Written down while the run was at epoch 0: *this cell fails like L1, therefore N>1
fusion is the load-bearing ingredient*. The competing outcome — it still solves square
— would have meant the gain lives in the encoder path itself and invited a further
ablation. Stating this first is what keeps the result from being read post hoc.

### 14.3 Result

In-training rollouts (`az_0` `mean_score`), against the two committed references:

| epoch | m3off (N∈[1,7]) | L1 (1 view) | **fixed-N=1** |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 50 | 0.38 | 0.02 | 0.06 |
| 100 | 0.56 | 0.04 | 0.06 |
| 150 | 0.74 | 0.00 | 0.04 |
| 200 | 0.80 | 0.02 | 0.00 |

**Five of five checkpoints track L1's floor, not `m3off`.** The strict sweep agrees;
per-viewpoint, `m3fixedn1` never exceeds 0.06 at any of the eleven azimuths:

| model | trained\* | **held-out** | el_0 | el_p15 | el_m15 |
|---|---|---|---|---|---|
| m3off | 0.76 | **0.55** | 0.88 | 0.30 | 0.04 |
| L1 s42 | 0.02 | 0.02 | 0.02 | 0.02 | 0.00 |
| L1 s43 | 0.04 | 0.02 | — | — | — |
| **m3fixedn1** | **0.04** | **0.05** | 0.02 | 0.02 | 0.04 |

Gap to `m3off`: **0.72 trained / 0.50 held-out** — an order of magnitude beyond noise.
Gap to L1: **+0.02 / +0.03** — zero. The `el_0` = 0.02 is the sharpest single number:
on the *trained* camera pose this model cannot do the task at all.

### 14.4 What this establishes

**Multi-view training samples are necessary.** Of §6's four confounded changes, the one
that matters is N>1. Combined with §11.3 (conditioning inert) and §13 (aux pressure
inert), every other candidate is closed:

| candidate | verdict | where |
|---|---|---|
| Plücker rays + camera-frame history conditioning | inert | §11.3 |
| per-view auxiliary heads | learnable, behaviourally inert | §13 |
| encoder architecture at N=1 (widened conv1, fusion query, per-view path) | reproduces nothing | §14 |
| **multi-view sampling (N>1)** | **load-bearing** | §14 |

This is the first *positive* identification in the project. It also reframes the
story: PROPOSAL §2.2's fusion is what works, while §2.1's geometric conditioning and
§2.4's aux heads are both inert — so the honest summary is that a *plumbing change*
(variable-N fusion), not the proposal's stated mechanism, produced the gain.

### 14.5 What it does **not** establish

- **It does not say N=1 inference fails.** M3 trains with N∈[1,7] and infers at N=1
  well (0.55 held-out). The capability lives in the multi-view *training* signal and
  the inference path rides on it. That distinction is M5's premise, and this result
  does not settle it either way.
- **It does not separate "N>1 needed" from "variable N needed"** — the cell never sees
  N>1 at all, so both remain consistent. `view_count_range=[2,7]` would separate them.
- **It does not say how much diversity is enough.** `[2,2]` answers PROPOSAL §7's
  "2 demo views, or many poses?" directly — one config line, ~2 h.
- n=1, and the same noise caveats as §13.6.

### 14.6 Operational and doc corrections from this session

- **`CLAUDE.md` is wrong about `data/robomimic_image.zip`.** It claims the 84.75 GB
  archive "is present locally" (holds transport/tool_hang/all MH splits). It does not
  exist on this box; a filesystem search returns only source files. Anything sized
  against that archive needs re-checking.
- **§12.5's run matrix had no disk gate**, unlike §10.6's preflight. Disk was the
  binding constraint all session (98% used; 47.8 GB free at start).
- **The top-k duplicate rule is narrower than §5 implies.** A size heuristic is
  *wrong*: of 19 checkpoints, exactly **one** was byte-identical to its `latest.ckpt`
  (`run_square_randview_s43/epoch=0200-*`); six others were the same size and
  genuinely different models. Compare with `cmp`, never by size.
- **Epoch cost is load-dependent, so re-measure it every campaign.** The same M4
  config ran at **373 ms/step** while another user held both GPUs and **84 ms/step**
  once they finished — a 4× swing, bracketing M3's 94 ms. Two claude-session lessons
  from §5 are now three: budget for a probe phase, measure rather than estimate, and
  *re-measure when the box changes*.
- `wandb` is unattended-safe here via `~/.netrc` (v0.15.12, proven: detached runs
  reach `logging synced files` with no TTY).
