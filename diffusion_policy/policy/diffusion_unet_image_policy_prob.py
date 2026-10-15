"""The probabilistic-fusion policy (PLAN_G.md stage 1, cells B/C/D).

`DiffusionUnetImagePolicyAux` with exactly two changes:

1. the auxiliary action head can read the per-view **Gaussian means** instead of
   the raw per-view latents (`aux_input='mu_v'`), which is what moves the
   base-frame action supervision onto the quantity the PoE actually fuses
   (PLAN_G.md section 3);
2. the unweighted aux term is exposed as `last_aux_action_loss`, so the
   workspace's per-term logging seam records it.

Everything else is inherited, deliberately: `compute_loss` and
`_compute_loss_with_aux` call `self.aux_loss(...)` at a single site
(`diffusion_unet_image_policy_aux.py:197`), so overriding that one method is the
whole policy delta -- no copied loss body to drift. The parent's draw order
(noise, then timesteps) and its single encoder pass are untouched, so the cells
remain RNG-locked with the deterministic baseline (the Gaussian head itself is
built under a private stream inside the encoder; see
`prob_view_conditioned_obs_encoder.py`).

The switch is checked, not assumed: `aux_input='mu_v'` requires an encoder that
declares `emits_mu` (the probabilistic encoder), and a missing `mu_views` key at
loss time raises naming the encoder class rather than falling back silently.

`last_aux_action_loss` vs `last_aux_loss`: for this cell the parent's
`last_aux_loss` is the UNWEIGHTED aux term (it applies `aux_loss_weight` only in
the returned sum), so the two keys agree and `last_aux_action_loss` is the
quantity directly comparable to the baseline's logged `aux_loss`. In the zvzg
policy `last_aux_loss` means something different (the weighted total) -- do not
carry numbers between the two conventions.
"""
from typing import Dict

import torch
import torch.nn.functional as F

from diffusion_policy.policy.diffusion_unet_image_policy_aux import (
    DiffusionUnetImagePolicyAux, masked_view_mean)
from diffusion_policy.dataset.multiview_image_dataset import AUX_ACTION_KEY


class DiffusionUnetImagePolicyProb(DiffusionUnetImagePolicyAux):
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
            aux_input: str = 'z_v',
            **kwargs):
        """
        `aux_input`: `'z_v'` (the baseline's wiring, raw per-view latents) or
        `'mu_v'` (the Gaussian means). Named arguments throughout; `**kwargs`
        must stay empty (the parent forwards it to `scheduler.step`, which
        raises only at rollout time -- see the parent's docstring).
        """
        if aux_input not in ('z_v', 'mu_v'):
            raise ValueError(
                f"aux_input must be 'z_v' or 'mu_v', got {aux_input!r}")
        if (aux_input == 'mu_v'
                and not getattr(obs_encoder, 'emits_mu', False)):
            raise ValueError(
                f"aux_input='mu_v' needs an encoder exposing per-view "
                f"Gaussians (emits_mu); {type(obs_encoder).__name__} does "
                f"not. Use ProbViewConditionedObsEncoder.")
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
        self.aux_input = str(aux_input)
        self.aux_key = 'mu_views' if self.aux_input == 'mu_v' else 'z_views'
        # initialized here (not only on the weight>0 path) so the workspace's
        # per-term seam sees the key every epoch, as in the zvzg policy
        self.last_aux_action_loss = 0.0

    def aux_loss(self, enc: Dict[str, torch.Tensor], batch, batch_size: int):
        """The parent's `aux_loss` body with one change: which tensor is fed.

        Mean-over-views, then mean-over-frames MSE in base-frame action space,
        identical reduction and normalization to the baseline (`masked_view_mean`
        excludes inactive slots rather than zero-weighting them). The input is
        `enc[self.aux_key]`; with `aux_key == 'z_views'` this is bit-identical
        to the parent's method (pinned in tests/test_prob_gaussian_fusion.py).
        """
        if self.aux_key not in enc:
            raise ValueError(
                f"aux_input={self.aux_input!r} needs enc[{self.aux_key!r}], but "
                f"the encoder ({type(self.obs_encoder).__name__}) did not "
                f"return it; got keys {sorted(enc)}")
        C = self.aux_n_steps
        K = enc[self.aux_key].shape[1]
        aux_tgt = batch[AUX_ACTION_KEY]
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
        tgt = self.normalizer[AUX_ACTION_KEY].normalize(
            aux_tgt.reshape(batch_size * self.n_obs_steps, K, C * self.action_dim))
        aux_pred = self.aux_head(enc[self.aux_key][active])   # (M, C*Da)
        pair = F.mse_loss(aux_pred, tgt[active], reduction='none').mean(dim=-1)
        aux = masked_view_mean(pair, active)
        self.last_aux_action_loss = float(aux.detach())
        return aux
