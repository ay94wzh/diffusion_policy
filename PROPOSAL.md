# Project Proposal: View-Aware Temporal Encoder

## 1. Motivation

A visuomotor policy trained from images is typically tied to the camera pose it
was trained on: move the camera and performance collapses. The goal of this
project is a policy that generalizes across camera viewpoints, and that at
inference time can be driven by a **single camera placed at a novel pose**.

The key idea is to make the visual representation *view-aware*: instead of
hoping a CNN ignores viewpoint, we explicitly tell the encoder where the camera
is (Plücker ray maps) and how the end-effector has been moving **in that
camera's frame**, and we supervise each per-view latent with an auxiliary
action-prediction task expressed **in that camera's frame**.

## 2. Method: training with multiple views

A shared visual encoder processes images from N cameras. The camera pose is
part of the input, not a nuisance factor:

1. **View-conditioned encoder → per-view latent `z_v`.**
   One shared encoder is applied to each view's image. Each encoding is
   *modulated* by two conditions:
   - the camera's **Plücker ray map** (per-pixel ray origin/direction, which
     encodes both intrinsics and extrinsics),
   - the **history of recent actions expressed in that camera's coordinate
     frame**.
   The output is a per-view latent `z_v` that carries both the visual content
   and the viewpoint it was observed from.

2. **Fusion → global latent `z_g`.**
   The per-view latents are fused (e.g., by cross-attention over the view
   tokens) into a single fused latent `z_g`. The fusion must accept N = 1 at
   test time, so it cannot rely on pairwise view geometry.

3. **Base-frame action head (main objective).**
   `z_g` is the observation for a downstream action head — a diffusion policy —
   which predicts action chunks **in the robot base frame**. Training
   supervises these against the demonstrated actions.

4. **Per-view auxiliary heads (why `z_v` is trustworthy).**
   Each `z_v` additionally feeds a small auxiliary head that predicts actions
   **in that camera's frame** (base-frame actions transformed by that camera's
   extrinsics). These heads are what forces `z_v` to be geometrically
   meaningful rather than merely view-tagged: to predict camera-frame actions
   from a single view, `z_v` must encode where the scene is *relative to that
   camera*. Auxiliary losses are summed with the diffusion loss.

   Diagram (training):
   ```
   image_v ──┐
             ├─► [ shared encoder ] ─► z_v ──┬─► [ aux head v ] ─► actions in camera v frame
   plücker_v ┤        (modulated)            │
   action_h_v┘                               └─┐
                                               ├─► [ fusion ] ─► z_g ─► [ diffusion head ] ─► actions in base frame
   image_w ──┐                                 │
             ├─► [ shared encoder ] ─► z_w ──►─┘
   plücker_w ┤
   action_h_w┘
   ```

## 3. Inference: a single camera at a novel pose

At test time only **one** camera is available, and it is not one of the
training views:

- The novel camera's pose is known (it is placed by us in simulation), so its
  Plücker map and the camera-frame action history can be computed for it.
- The same shared encoder produces `z_v` for that single view; fusion with
  N = 1 yields `z_g`; the base-frame action head then emits actions.
- The hoped-for property: the *latent* is view-aware, while the *action output*
  is **view-invariant** — the same physical behaviour is recovered whether the
  scene was seen from the left, the right, or a new pose.

An optional **distillation** stage can follow training: a single-view student
encoder is trained to regress the multi-view fused latent `z_g` produced by the
frozen teacher, which can stabilize single-view inference.

## 4. Baselines

1. The ultimate goal is control from a novel viewpoint, so the natural baseline
   is **one camera for training, one camera for inference**.
2. Platform: **robomimic tasks with image demonstrations** (Square primary,
   Lift for smoke tests). The baseline uses the **agentview camera only**
   (wrist camera dropped), matching the single-view inference setting.
3. Baselines to run:
   - **Diffusion Policy (single view, agentview only)** — first, on Square/Lift
     with the PH dataset.
   - **ACT** — deferred; may be added later as a second baseline.
4. Evaluation: the same policy is rolled out at the training view and at
   increasingly displaced viewpoints (azimuth sweeps). The
   **degradation curve** is the quantity of interest: it quantifies how tied
   the baseline is to its training view, and it is the reference the proposed
   method must beat.

## 5. Expected contribution

Multi-view training with explicit geometric conditioning is a way to learn
view-invariant *control* while keeping the representation view-aware. If it
works, a policy can be deployed with a single camera placed somewhere new —
without re-collecting demonstrations for that camera.

## 6. Status and milestones

See `PLAN.md` for the concrete milestone plan and `PROGRESS.md` for the current
state of the work. In short:

- **M1 (current)** — single-view DP baseline + novel-view evaluation harness on
  robomimic (Square/Lift PH).
- **M2** — multi-view data: re-render demonstrations from many camera poses,
  storing per-view images, camera parameters, and camera-frame actions.
- **M3** — view-conditioned encoder + fusion (Plücker + action-history
  conditioning); drop-in replacement for the existing obs encoder.
- **M4** — per-view auxiliary action heads and their loss.
- **M5** — single-novel-view inference (+ optional distillation), evaluated
  against the M1 degradation curve.

## 7. Open questions

- How much view diversity is needed for generalization — 2 demo views, or
  re-rendered views at many poses?
- Do the per-view auxiliary heads need the camera-frame action history as
  *conditioning* as well as the Plücker map, or is one of the two sufficient?
- How well does the method extrapolate beyond the azimuth range seen in
  training?
