"""M4: `DiffusionUnetImagePolicy` + per-view camera-frame auxiliary heads.

`diffusion_unet_image_policy.py` is an UPSTREAM file (added in 4bf419a) and is
deliberately not modified -- this subclasses it. The M3 forward path is
untouched: `predict_action`, `forward`, `set_normalizer` and
`conditional_sample` are all inherited as-is, the aux head is a training-only
branch, and `aux_loss_weight=0.0` reproduces M3's loss exactly (asserted
bit-identical, gradients included, in tests/test_aux_action_heads.py).

Why this is the milestone that makes the conditioning matter
------------------------------------------------------------
M3's conditioning measured inert (`PROGRESS.md` section 11.3): the camera reaches
the model only through the Pluecker channels of `conv1`, and nothing in the
objective required the encoder to use it -- at a trained view the image alone
predicts the action. PROPOSAL.md section 2.4 names the per-view auxiliary heads
as what "forces z_v to be geometrically meaningful rather than merely
view-tagged": predicting an action in camera k's frame from a single view
requires z_v to encode where the scene sits relative to that camera.

The A/B is RNG-locked
---------------------
Both arms construct the head (same parameter count, same construction-time RNG
consumption, and `self.model` is built first either way), the head draws no RNG,
and the encoder runs exactly once per step in both arms. So every crop, noise
and timestep draw is identical between arms for the whole run: they differ by
exactly one scalar term and its gradient.
"""
from typing import Dict

import torch
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.policy.diffusion_unet_image_policy import (
    DiffusionUnetImagePolicy)
from diffusion_policy.model.vision.per_view_aux_head import PerViewAuxActionHead
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.multiview_image_dataset import AUX_ACTION_KEY


def masked_view_mean(pair_loss: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """``(M,)`` per-pair losses + ``(N, K)`` bool mask -> scalar.

    Mean over the ACTIVE views of each row, then mean over rows -- so every
    (frame) row contributes equally despite N being drawn uniformly over [1, K].
    Inactive slots are excluded, not merely zero-weighted: their `z_views` rows
    are exactly zero, so an unmasked head would be dragged toward the mean of
    all targets by a constant input -- plausible-looking and silently diluted.
    """
    rows, _ = active.nonzero(as_tuple=True)          # row-major, as in the encoder
    per_row = pair_loss.new_zeros(active.shape[0]).index_add_(0, rows, pair_loss)
    return (per_row / active.sum(dim=1)).mean()


class DiffusionUnetImagePolicyAux(DiffusionUnetImagePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler,
            obs_encoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            aux_loss_weight: float = 0.0,
            aux_n_steps: int = 8,
            aux_hidden_dim: int = 256,
            **kwargs):
        """
        NOTE: every aux parameter is a NAMED argument above and must stay that
        way. The parent stores `self.kwargs = kwargs` and `conditional_sample`
        forwards `**self.kwargs` into `scheduler.step(...)`, so a stray key would
        raise TypeError at ROLLOUT time -- hours into a run, not at construction.
        """
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            **kwargs)

        if not hasattr(obs_encoder, 'forward_full'):
            raise ValueError(
                f'M4 needs an encoder exposing per-view latents (forward_full); '
                f'{type(obs_encoder).__name__} does not. Use '
                f'ViewConditionedObsEncoder.')
        in_dim = getattr(obs_encoder, 'fused_dim', None)
        if in_dim is None:
            raise ValueError(
                f'{type(obs_encoder).__name__} has no `fused_dim`, so the aux '
                f'head input width is unknown')
        self.aux_loss_weight = float(aux_loss_weight)
        self.aux_n_steps = int(aux_n_steps)
        # built unconditionally -- including at weight 0 -- so the two ablation
        # arms consume an identical RNG stream and hold identical parameters
        self.aux_head = PerViewAuxActionHead(
            in_dim=in_dim,
            out_dim=self.aux_n_steps * self.action_dim,
            hidden_dim=aux_hidden_dim)
        self.last_aux_loss = 0.0
        self.last_diff_loss = 0.0

    # ========= training  ============
    def compute_loss(self, batch):
        if not self.aux_loss_weight > 0.0:
            # the unchanged M3 path, deliberately: same encoder entry point,
            # same draws, no aux graph
            loss = super().compute_loss(batch)
            self.last_aux_loss = 0.0
            self.last_diff_loss = float(loss.detach())
            return loss
        return self._compute_loss_with_aux(batch)

    def _compute_loss_with_aux(self, batch):
        """The parent's `compute_loss` body @4bf419a, plus the aux term.

        Copied rather than refactored because the parent is upstream and must
        not be edited. The copy is pinned: at weight 0 `compute_loss` delegates
        to `super()` and tests/test_aux_action_heads.py asserts the two agree
        bit-for-bit, gradients included, so drift cannot go unnoticed.

        Three changes: `forward_full` instead of `forward` (ONE encoder pass,
        reused for the UNet conditioning), and the aux term at the end. The
        draw order (noise, then timesteps) is deliberately the parent's, so the
        ablation arms stay RNG-locked.
        """
        if not self.obs_as_global_cond:
            raise NotImplementedError(
                'M4 implements the obs_as_global_cond path only; the inpainting '
                'path would need its own aux wiring')

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
        # Using z_global by itself silently feeds a narrower conditioning vector
        # and trips only at the UNet's first Linear (caught by
        # tests/test_aux_action_heads.py).
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

        # ---- M4: per-view auxiliary camera-frame action prediction ----------
        aux_loss = self.aux_loss(enc, batch, batch_size)
        self.last_aux_loss = float(aux_loss.detach())
        self.last_diff_loss = float(diff_loss.detach())
        return diff_loss + self.aux_loss_weight * aux_loss

    def aux_loss(self, enc: Dict[str, torch.Tensor], batch, batch_size: int):
        """Mean-over-views, then mean-over-frames MSE in camera-frame action space.

        Also usable standalone (tests, diagnostics) -- it needs only the
        encoder's output dict and the batch.
        """
        C = self.aux_n_steps
        K = enc['z_views'].shape[1]
        aux_tgt = batch[AUX_ACTION_KEY]
        # loud, not silent: a mismatch here would broadcast or transpose into a
        # plausible-looking wrong number
        if tuple(aux_tgt.shape[:3]) != (batch_size, self.n_obs_steps, K):
            raise ValueError(
                f'{AUX_ACTION_KEY} must be (B, To, K, C*Da) = '
                f'({batch_size}, {self.n_obs_steps}, {K}, '
                f'{C * self.action_dim}), got {tuple(aux_tgt.shape)}')
        if aux_tgt.shape[-1] != C * self.action_dim:
            raise ValueError(
                f'{AUX_ACTION_KEY} last dim {aux_tgt.shape[-1]} != aux_n_steps '
                f'({C}) * action_dim ({self.action_dim}); the dataset and the '
                f'policy disagree about aux_n_steps')

        active = enc['view_active']                       # (B*To, K)
        # row-major flatten (B, To) -> B*To, matching the obs flatten above, so
        # frame (b, to) keeps its own target. `active` indexing selects the same
        # (frame, view) pairs in the same order for z_views and the target.
        tgt = self.normalizer[AUX_ACTION_KEY].normalize(
            aux_tgt.reshape(batch_size * self.n_obs_steps, K, C * self.action_dim))
        aux_pred = self.aux_head(enc['z_views'][active])   # (M, C*Da)
        pair = F.mse_loss(aux_pred, tgt[active], reduction='none').mean(dim=-1)
        return masked_view_mean(pair, active)
