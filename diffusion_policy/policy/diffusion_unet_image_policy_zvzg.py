"""The combined-supervision arm (`zvzg`): BOTH supervision lines on at once.

`DiffusionUnetImagePolicyAux` (M4) plus ONE change: the parent's per-view
BASE-frame action head on `z_v` stays exactly as it is, and a second head --
`ViewPoseSetHead` from the `zgpose` arm -- is added on the FUSED latent
`z_global`, predicting the live views' pooled camera-pose set. The two terms
are the two arms' measured designs, unchanged:

    total = diff + w_v * L_action(z_views, base-frame chunks)
                 + w_g * L_pose(z_global, pooled camera-pose set)

Why this arm exists
-------------------
`m4base` and `zgpose` were each run alone and each was behaviourally null, but
their measured latent effects are in TENSION: m4base's view-invariant target
halved the view-specific component of `z_v` (`zv_pair_ratio` 0.485 -> 0.239)
while zgpose's pose-set pressure made `z_g` view-set-aware and pushed back into
`z_v` (probe `zv_pair_ratio` 2.255 vs parents' 0.267/0.544). This is the first
cell whose instruments can see which pressure wins -- and whether either null
was an interaction effect.

Conventions, and the one that can bite
--------------------------------------
`last_aux_loss` is the WEIGHTED TOTAL `w_v*L_a + w_g*L_p` (that is what the
workspace's `aux_loss` log line means here), so `train_loss - aux_loss` is the
pure diffusion term at any weights. It is therefore NOT comparable to either
parent's `aux_loss`, which was an unweighted single term -- the per-term
attributes `last_aux_action_loss` / `last_aux_pose_loss` are (logged alongside
as `aux_action_loss` / `aux_pose_loss` by the workspace seam). At w_v=1.0 the
two conventions coincide for the action term.

Each term is GUARDED by its own weight: at w_v=0 no action-head graph is built
and `aux_head` gets no gradient (its .grad stays None, the idiom the parent's
tests pin); likewise for the pose term at w_g=0. The guards are also what keeps
the single-term arms bit-identical to their parents (see below).

RNG, claimed and not claimed
----------------------------
The parent's `_compute_loss_with_aux` copy and its draw order (noise, then
timesteps) are preserved, so at (w_v=1, w_g=0) this class is bit-identical to
`DiffusionUnetImagePolicyAux` and at (w_v=0, w_g=1) to
`DiffusionUnetImagePolicyZgPose` (tests/test_zvzg_supervision.py). NOT claimed:
draw-level lockstep with the parents' RUNS. The constructor builds one more
head than m4base does (consuming RNG), so the training streams differ from
m4base's; against zgpose the construction sequence is identical in shape
(action head 512->256->80, then pose head 512->256->99), so the shared trunk
AND the pose head init match zgpose's seed-42 stream -- stated as expected, not
registered (the drivers compare at the design level).
"""
from typing import Dict

import torch
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.policy.diffusion_unet_image_policy_aux import (
    DiffusionUnetImagePolicyAux)
from diffusion_policy.model.vision.view_pose_set_head import ViewPoseSetHead
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.multiview_image_dataset import (
    AUX_VIEW_POSE_KEY, AUX_VIEW_POSE_MASK_KEY, VIEW_POSE_DIM)


class DiffusionUnetImagePolicyZvZg(DiffusionUnetImagePolicyAux):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler,
            obs_encoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            aux_loss_weight: float = 0.0,        # w_v, the M4/BASE action term
            aux_n_steps: int = 8,
            aux_hidden_dim: int = 256,
            aux_pose_weight: float = 0.0,        # w_g, the zgpose pose-set term
            aux_n_views: int = 11,
            **kwargs):
        """
        NOTE: every aux parameter is a NAMED argument above and must stay that
        way -- the ancestor's warning, inherited twice over: `conditional_sample`
        forwards `self.kwargs` into `scheduler.step`, so a stray key raises
        TypeError at ROLLOUT time, hours into a run. `aux_pose_weight` and
        `aux_n_views` are consumed HERE and deliberately not forwarded to
        super().__init__ (which would swallow them into `self.kwargs`).
        `aux_n_views` must equal the task's pool size; the loss cross-checks the
        emitted target against it, loudly.
        """
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            aux_loss_weight=aux_loss_weight,
            aux_n_steps=aux_n_steps,
            aux_hidden_dim=aux_hidden_dim,
            **kwargs)
        self.aux_pose_weight = float(aux_pose_weight)
        self.aux_n_views = int(aux_n_views)
        # the parent built the action head as `self.aux_head`; add the pose-set
        # head beside it (built AFTER super(), so every parameter the parent
        # constructed carries the parent's seed stream -- cf. zgpose.py).
        self.pose_head = ViewPoseSetHead(
            in_dim=self.obs_encoder.fused_dim,
            n_views=self.aux_n_views,
            pose_dim=VIEW_POSE_DIM,
            hidden_dim=aux_hidden_dim)
        self.last_aux_action_loss = 0.0
        self.last_aux_pose_loss = 0.0

    # ========= training  ============
    def compute_loss(self, batch):
        if not (self.aux_loss_weight > 0.0 or self.aux_pose_weight > 0.0):
            # the unchanged M3 path, deliberately: same encoder entry point,
            # same draws, no aux graph. (The parent's compute_loss sees
            # aux_loss_weight <= 0 and delegates to the base policy itself.)
            loss = super().compute_loss(batch)
            self.last_aux_loss = 0.0
            self.last_aux_action_loss = 0.0
            self.last_aux_pose_loss = 0.0
            return loss
        return self._compute_loss_dual(batch)

    def _compute_loss_dual(self, batch):
        """The parent's `_compute_loss_with_aux` body, plus the pose term.

        Copied rather than refactored, the established pattern here -- and the
        copy IS the pin: at (1, 0) the guards make this evaluate exactly the
        parent's expression, asserted bit-for-bit (loss and all shared
        gradients) in tests/test_zvzg_supervision.py.
        """
        if not self.obs_as_global_cond:
            raise NotImplementedError(
                'zvzg implements the obs_as_global_cond path only; the '
                'inpainting path would need its own aux wiring')

        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        trajectory = nactions
        cond_data = trajectory

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs,
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        enc = self.obs_encoder.forward_full(this_nobs)
        # the SAME concatenation `forward` performs -- z_global alone is 512
        # wide, and the UNet was built for (fused_dim + low-dim) * n_obs_steps.
        nobs_features = torch.cat(
            [enc['z_global']]
            + [this_nobs[k] for k in self.obs_encoder.low_dim_keys], dim=-1)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps,
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        diff_loss = loss.mean()

        # ---- the two auxiliary terms, each guarded by its own weight -------
        # (an unguarded `w * term` would still contribute a zero gradient, but
        # it would ALSO build a useless graph and leave the inactive head with
        # zero grads instead of None -- the idiom the tests pin)
        if self.aux_loss_weight > 0.0:
            aux_action = self.aux_loss(enc, batch, batch_size)   # M4's, reused
            self.last_aux_action_loss = float(aux_action.detach())
        else:
            aux_action = None
            self.last_aux_action_loss = 0.0
        if self.aux_pose_weight > 0.0:
            aux_pose = self.aux_pose_loss(enc, batch, batch_size)  # zgpose's
            self.last_aux_pose_loss = float(aux_pose.detach())
        else:
            aux_pose = None
            self.last_aux_pose_loss = 0.0

        self.last_diff_loss = float(diff_loss.detach())
        # the weighted total is what the workspace's `aux_loss` line means here
        # (see the module docstring: NOT comparable to either parent's number)
        self.last_aux_loss = (self.aux_loss_weight * self.last_aux_action_loss
                              + self.aux_pose_weight * self.last_aux_pose_loss)

        total = diff_loss
        if aux_action is not None:
            total = total + self.aux_loss_weight * aux_action
        if aux_pose is not None:
            total = total + self.aux_pose_weight * aux_pose
        return total

    def aux_pose_loss(self, enc: Dict[str, torch.Tensor], batch,
                      batch_size: int):
        """zgpose's term, unchanged: all-rows MSE of the pooled pose set, read
        from the FUSED latent. Dead rows are scored AGAINST ZERO (the target is
        zero-filled there), which is what makes the loss sample-dependent and
        requires the head to read which views are live out of `z_g` (a
        constant-per-row head can only reach the floor -- see zgpose.py's
        REVISION 2 note). Copied rather than inherited because the class hier-
        archy makes this a separate head (`pose_head`), not the parent's
        `aux_head`. Also usable standalone (tests, diagnostics).
        """
        V, P = self.aux_n_views, VIEW_POSE_DIM
        tgt = batch[AUX_VIEW_POSE_KEY]
        tgt_mask = batch[AUX_VIEW_POSE_MASK_KEY]
        M = batch_size * self.n_obs_steps
        # loud, not silent: a mismatch here would broadcast or transpose into a
        # plausible-looking wrong number
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
        pred = self.pose_head(enc['z_global'])                   # (M, V, P)
        # ALL V rows are scored, dead ones against their zero targets
        return F.mse_loss(pred, tgt, reduction='none').mean()
