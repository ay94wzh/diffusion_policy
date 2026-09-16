# PROGRESS

Last updated: 2026-09-12. See `PROPOSAL.md` (research direction) and `PLAN.md`
(milestones M1–M5).

**M1 is COMPLETE.** Single-view DP baselines were trained for can, lift, and
square (robomimic PH, agentview only, 201 epochs each) on the 2× RTX 5090
machine, and the novel-view degradation curves were measured at azimuth
0°, ±15°, ±30°. The headline result: success collapses essentially to zero at
±15° on all three tasks.

**M2 (multi-view data) is COMPLETE.** All three multi-view zarrs were rendered
and verified on 2026-09-15 (§7). Next: the N=1 training gate, then M3.

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
  100 MB file limit). They exist only on the machine that trained them, at
  `data/outputs/run_<task>_abs_single_s42_200ep/checkpoints/` (`latest.ckpt`
  = epoch-200 model per task). To move them to another machine:
  ```bash
  rsync -av data/outputs <user>@<other>:/path/to/diffusion_policy/data/outputs
  ```
  or re-train from the runbook in §3 (~1–2.5 h per task).
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
  9.2 GB/run. Root disk is 97–100% full — check `df -h` before running more.
  `/data` (15 TB) exists but is owned by another user and not writable.

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

## 6. Not yet done / next

- **L1 replication** (§8.6): square/seed 43 and lift/seed 42 are training, to test
  whether the §8.3 result is seed- or task-specific. Results to be recorded in §8.6.
- **The ablation that separates two explanations of L1's failure** — no pose
  information vs only one slot to represent seven views. M3's Plücker
  conditioning tests the first; a K-slot *unconditioned* variant would test the
  second. Without it the M3 gain is not cleanly attributable to conditioning
  rather than to capacity.
- **Mechanism probe** (optional but cheap): view classification from L1's frozen
  encoder, to test the "viewpoint confusion" story directly rather than inferring
  it from the rollout numbers (§8.4).
- Open design questions to settle before M3: fusion module choice, and
  whether aux heads condition on camera-frame action history, Plücker map, or
  both (PROPOSAL.md §7).

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

## 8. L1 — the view-randomized baseline (square/seed 42 complete; replication in flight)

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

### 8.3 Results

**L1 square seed 42, final sweep** (`success_rate`; `mean_score` is identical to
3 dp at every viewpoint here):

| viewpoint | trained on? | L1 | M1 (comparison) |
|---|---|---|---|
| az_0 | **yes** | **0.020** | **0.820** |
| az_p15 | no | 0.020 | 0.000 |
| az_m15 | no | 0.000 | 0.020 |
| az_p30 | **yes** | 0.000 | 0.000 |
| az_m30 | **yes** | 0.000 | 0.000 |
| az_p45 | no | 0.020 | — |
| az_m45 | no | 0.000 | — |
| az_p60 | **yes** | 0.060 | — |
| az_m60 | **yes** | 0.020 | — |
| az_p75 | no | 0.040 | — |
| az_m75 | no | 0.020 | — |

**In-training rollouts at az_0** (a pose L1 trained on):

| epoch | 0 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|
| M1 | 0.000 | 0.840 | 0.880 | 0.860 | **0.880** |
| L1 | 0.000 | 0.020 | 0.040 | 0.000 | **0.020** |

Final val_loss: M1 0.0285, L1 0.0599.

### 8.4 Interpretation, and its limits

L1 is flat at ~0.00–0.06 **everywhere** — trained and held-out poses alike, with
no systematic difference between them. It is therefore **not** a failure to
generalize to novel viewpoints. Adding six extra training poses to M1's recipe
did not make the policy view-invariant; it removed the ability to act from *any*
pose, **including az_0, which it trained on and where M1 scores 0.82**.

Limits, stated plainly:
- L1's final val_loss is **2× M1's** (0.0599 vs 0.0285), so it does fit the
  demonstrations somewhat worse — a 7-pose problem is genuinely harder to fit.
  But a 2× loss gap does not explain a 44× rollout gap, and the failure is at a
  *trained* pose. Call this **well-supported, not proven**.
- **One seed, one task** (§8.6).
- **The mechanism is inferred, not measured.** The plausible story is that a
  shared encoder receiving seven mutually-inconsistent views of the same scene
  state, with no signal indicating which camera it is behind, converges to a
  representation confused at every pose. A targeted probe (e.g. view
  classification from the frozen encoder) would test this and has not been run.

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

### 8.6 Replication in flight

The §8.3/§8.4 claim currently rests on one seed of one task — too thin to carry
M3's motivation. Two runs were launched to test it from both directions:

| run | question |
|---|---|
| `run_square_randview_s43_200ep` | is the effect **seed**-specific? |
| `run_lift_randview_s42_200ep` | is it **task**-specific? |

Both use the config default 7-view subset (no `view_subset` override), then the
§8.2 sweep. **If either behaves differently, that is a finding to report, not a
failure to re-run** — it would mean the effect is task- or seed-dependent.
Results to be recorded here.
