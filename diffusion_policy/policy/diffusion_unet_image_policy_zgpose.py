"""PLAN candidate 2: supervise the FUSED latent `z_g` with the pooled
camera-pose set of the live views.

`DiffusionUnetImagePolicyAux` (M4) plus ONE change: the auxiliary head reads
`enc['z_global']` -- the latent the diffusion head actually consumes, which
until now received gradient only from the diffusion loss -- and predicts the
live views' pooled camera-pose set instead of per-view action chunks. The
parent's `compute_loss` / `_compute_loss_with_aux` (the weight-0 delegation to
the base policy, `last_aux_loss` logging, the row-major (B, To) flatten) are
inherited unchanged; only `aux_loss` is overridden.

The scoring rule, and why it is load-bearing
--------------------------------------------
The loss scores ALL V rows against the zero-filled target -- dead rows are
scored AGAINST ZERO. The pose table itself is static (the cameras never move),
so liveness is the only sample-dependent content of the target; with every row
scored, no input-independent head can beat the per-row-constant floor, and any
improvement is sample variation read out of `z_g`. (REVISION 2, 2026-10-09:
the first version scored live rows only, which a constant head satisfied
exactly, taking the pre-registered floor to zero -- the run could not test its
own premise; see data/zgpose_run.sh's header for the record.)

The head swap
-------------
The parent builds its `PerViewAuxActionHead` LAST in `__init__`; this subclass
replaces it with `ViewPoseSetHead` immediately after `super().__init__` returns.
Every other module was constructed before that point, so all shared parameters
carry the parent's seed-42 init exactly. NOT claimed: draw-level RNG lockstep --
the discarded head's construction plus the new head's consume RNG, so the
training draws (shuffle order, crops, view draws, noise) differ from the
parents' runs. The driver header states this.
"""
from typing import Dict

import torch
import torch.nn.functional as F

from diffusion_policy.policy.diffusion_unet_image_policy_aux import (
    DiffusionUnetImagePolicyAux)
from diffusion_policy.model.vision.view_pose_set_head import ViewPoseSetHead
from diffusion_policy.dataset.multiview_image_dataset import (
    AUX_VIEW_POSE_KEY, AUX_VIEW_POSE_MASK_KEY, VIEW_POSE_DIM)


class DiffusionUnetImagePolicyZgPose(DiffusionUnetImagePolicyAux):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler,
            obs_encoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            aux_loss_weight: float = 0.0,
            aux_n_views: int = 11,
            aux_hidden_dim: int = 256,
            **kwargs):
        """
        NOTE: every aux parameter is a NAMED argument above and must stay that
        way -- the parent's warning, inherited: `conditional_sample` forwards
        `self.kwargs` into `scheduler.step`, so a stray key raises TypeError at
        ROLLOUT time, hours into a run. `aux_n_views` must equal the task's
        pool size; the loss cross-checks the emitted target against it, loudly.
        """
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            aux_loss_weight=aux_loss_weight,
            aux_hidden_dim=aux_hidden_dim,
            **kwargs)
        self.aux_n_views = int(aux_n_views)
        # the parent built its action head last; replace it (see the module
        # docstring: shared parameters are the parent's seed-42 stream, the
        # draw streams are not).
        self.aux_head = ViewPoseSetHead(
            in_dim=self.obs_encoder.fused_dim,
            n_views=self.aux_n_views,
            pose_dim=VIEW_POSE_DIM,
            hidden_dim=aux_hidden_dim)

    def aux_loss(self, enc: Dict[str, torch.Tensor], batch, batch_size: int):
        """All-rows MSE of the pooled pose set, read from the FUSED latent.

        Dead rows are scored AGAINST ZERO (the target is zero-filled there):
        that is what makes the loss sample-dependent, and therefore what
        requires the head to read which views are live out of `z_g` (a
        constant-per-row head can only reach the floor -- see the module
        docstring). Also usable standalone (tests, diagnostics).

        The mask is still consumed -- but only by the cross-checks below.
        """
        V, P = self.aux_n_views, VIEW_POSE_DIM
        tgt = batch[AUX_VIEW_POSE_KEY]
        tgt_mask = batch[AUX_VIEW_POSE_MASK_KEY]
        M = batch_size * self.n_obs_steps
        # loud, not silent: a mismatch here would broadcast or transpose into
        # a plausible-looking wrong number
        if tuple(tgt.shape) != (batch_size, self.n_obs_steps, V, P):
            raise ValueError(
                f'{AUX_VIEW_POSE_KEY} must be (B, To, V, {P}) = '
                f'({batch_size}, {self.n_obs_steps}, {V}, {P}), got '
                f'{tuple(tgt.shape)}')
        if tuple(tgt_mask.shape) != (batch_size, self.n_obs_steps, V):
            raise ValueError(
                f'{AUX_VIEW_POSE_MASK_KEY} must be (B, To, V) = '
                f'({batch_size}, {self.n_obs_steps}, {V}), got '
                f'{tuple(tgt_mask.shape)}')
        active = tgt_mask.reshape(M, V) > 0.5
        if not bool(active.any(dim=1).all()):
            raise ValueError('a frame has no live view in the pose-set mask')
        # the two masks are different objects (slot-indexed view_mask vs
        # pool-indexed pose mask), derived from the same draw; equal live
        # COUNTS per frame is the invariant that catches a collate/reshape
        # desync between the obs and the target
        enc_active = enc['view_active']
        if not torch.equal(active.sum(dim=1), enc_active.sum(dim=1)):
            raise ValueError(
                "pose-set mask and encoder activity disagree on a frame's "
                'live-view count')
        tgt = self.normalizer[AUX_VIEW_POSE_KEY].normalize(
            tgt.reshape(M, V, P))
        pred = self.aux_head(enc['z_global'])                    # (M, V, P)
        # ALL V rows are scored, dead ones against their zero targets: the
        # per-row-constant head is the floor, and beating it requires sample
        # variation = liveness from z_g (REVISION 2; see the module docstring).
        # Frames stay equally weighted; only WHICH entries within a frame are
        # scored changed vs revision 1.
        return F.mse_loss(pred, tgt, reduction='none').mean()
