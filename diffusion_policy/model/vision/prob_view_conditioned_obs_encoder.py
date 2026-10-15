"""The probabilistic-fusion obs encoder (PLAN_G.md stage 1, cells B/C/D).

Same per-slot backbone as ``ViewConditionedObsEncoder``; what changes is what
happens *after* the per-view latents exist. A shared ``PerViewGaussianHead``
maps each slot's ``z_v`` to ``(mu_v, logvar_v)``, and the fusion is chosen by
``fusion_mode``:

* ``'mha'``  -- the parent's attention fusion applied to the **means** (cell B,
  the deterministic-projection control: aux supervision and the UNet both see
  ``mu_v``, no variance anywhere);
* ``'poe'``  -- the Product-of-Experts over the live slots (cell C), with the
  fused log-variance exposed as ``z_global_logvar``;
* ``'mean'`` -- the unweighted mean of the means (cell D, the PoE-vs-plain-
  averaging control).

Design notes that the code relies on:

* ``forward_full`` **wraps** the parent instead of copying it: the parent's
  return already contains ``z_views`` = the pre-fusion token tensor (exactly
  the raw ``z_v``, asserted bit-identical to its own fusion input), so the
  Gaussian head is applied to that, and every parent validation (mask shape,
  all-False row guard, cam/image consistency) keeps applying. The parent's own
  MHA still runs and its output is discarded -- one wasted attention (~0.05% of
  the forward) in the non-``'mha'`` modes, against ~70 copied lines of drift
  risk. In those modes ``self.fusion``/``self.fusion_query`` end up with no
  gradient path; they stay constructed unconditionally so the cells remain
  RNG-lockstep (below).
* ``z_views`` in the returned dict is **unchanged**: still the raw ``z_v``. The
  per-view probe statistics (``zv_pair_ratio`` etc.) are registered cross-cell
  readouts and must keep describing the same quantity in every cell.
* The Gaussian head is built under a **private RNG stream** (save -> reseed a
  fresh CPU generator -> build -> restore), so constructing it consumes nothing
  from the global stream. The workspace seeds one stream and builds the whole
  model from it, so this is what makes the cells differ from the deterministic
  baseline by *only* the fusion branch and the aux head's input: every shared
  parameter is bit-identical to the baseline's, and so is every later draw
  (UNet init, auxiliary head init, DataLoader shuffle seed, crops, noise,
  timesteps). ``tests/test_prob_gaussian_fusion.py`` pins this with a
  ``torch.get_rng_state()`` equality check. Never ``torch.manual_seed`` here
  (it would also reseed CUDA, where the training noise is drawn) and never
  retain the Generator (the EMA model is a deepcopy).
* Head and fusion consume no RNG at forward time, so ``output_shape()`` and the
  probe hook stay RNG-neutral, as in the parent.
* Dead slots keep the "inactive rows are exactly zero" invariant in the new
  keys too (``mu_views``/``logvar_views``); ``masked_fill`` after the head is
  what the check below proves.
"""
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from diffusion_policy.model.vision.per_view_gaussian import (
    PerViewGaussianHead, poe_fuse, uniform_fuse)
from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    ViewConditionedObsEncoder)


class ProbViewConditionedObsEncoder(ViewConditionedObsEncoder):
    """``ViewConditionedObsEncoder`` + per-view Gaussians + selectable fusion.

    Extra keys in ``forward_full``'s return: ``mu_views`` ``(N,K,D)``,
    ``logvar_views`` ``(N,K,D)`` (both exactly zero on dead slots), and
    ``z_global_logvar`` -- the PoE-combined log-variance ``(N,D)`` in
    ``'poe'`` mode, ``None`` in the others (an MHA output has no meaningful
    log-variance; exposing a number there would be misread).
    """

    FUSION_MODES = ('mha', 'poe', 'mean')
    # what the policy checks before accepting aux_input='mu_v'
    emits_mu = True

    def __init__(self,
            shape_meta: dict,
            rgb_model: nn.Module,
            cam_keys: Optional[Sequence[str]] = None,
            view_mask_key: str = 'view_mask',
            eef_hist_key: str = 'view_eef_hist',
            eef_hist_steps: int = 4,
            fused_dim: int = 512,
            n_heads: int = 8,
            cond_dim: int = 128,
            use_plucker: bool = True,
            use_eef_hist: bool = True,
            crop_shape: Optional[Tuple[int, int]] = None,
            random_crop: bool = True,
            use_group_norm: bool = False,
            imagenet_norm: bool = False,
            resize_shape=None,
            share_rgb_model: bool = True,
            fusion_mode: str = 'mha',
            logvar_min: float = -6.0,
            logvar_max: float = 2.0,
            gauss_hidden: int = 256,
            head_init_seed: int = 42,
            ):
        # validate the new knobs BEFORE building the backbone, so a typo
        # fails without a resnet construction
        if fusion_mode not in self.FUSION_MODES:
            raise ValueError(
                f'fusion_mode must be one of {self.FUSION_MODES}, got '
                f'{fusion_mode!r}')
        if not (logvar_min < 0.0 < logvar_max):
            raise ValueError(
                f'logvar bounds must straddle 0 so the zero-init logvar '
                f'branch starts unclamped, got [{logvar_min}, {logvar_max}]')
        super().__init__(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            cam_keys=cam_keys,
            view_mask_key=view_mask_key,
            eef_hist_key=eef_hist_key,
            eef_hist_steps=eef_hist_steps,
            fused_dim=fused_dim,
            n_heads=n_heads,
            cond_dim=cond_dim,
            use_plucker=use_plucker,
            use_eef_hist=use_eef_hist,
            crop_shape=crop_shape,
            random_crop=random_crop,
            use_group_norm=use_group_norm,
            imagenet_norm=imagenet_norm,
            resize_shape=resize_shape,
            share_rgb_model=share_rgb_model,
        )
        # Private stream: the parent's construction above consumed exactly the
        # global stream the deterministic baseline would have consumed, and the
        # heads below consume none of it -- restore bit-exactly. See the module
        # docstring for what this buys (and do NOT keep the Generator around).
        cpu_state = torch.get_rng_state()
        try:
            generator = torch.Generator().manual_seed(int(head_init_seed))
            torch.set_rng_state(generator.get_state())
            self.gauss_head = PerViewGaussianHead(
                self.fused_dim, hidden_dim=gauss_hidden,
                logvar_min=logvar_min, logvar_max=logvar_max)
        finally:
            torch.set_rng_state(cpu_state)
        self.fusion_mode = str(fusion_mode)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def _check_gaussian(self, mu, logvar, active):
        """Loud invariants: finite on live rows, inside the clamp, dead rows zero.

        The dead-row check is not vacuous even though the masking happens right
        above: ``masked_fill`` on a NaN produces a NaN, not a zero.
        """
        live = active.unsqueeze(-1).expand_as(mu)     # (N, K, D)
        if not (bool(torch.isfinite(mu[live]).all())
                and bool(torch.isfinite(logvar[live]).all())):
            raise ValueError(
                'non-finite mu/logvar from the Gaussian head on live slots')
        lv_live = logvar[live]
        if (float(lv_live.min()) < self.logvar_min
                or float(lv_live.max()) > self.logvar_max):
            raise ValueError(
                f'live logvar outside [{self.logvar_min}, {self.logvar_max}]; '
                'the head must clamp before returning')
        if (torch.count_nonzero(mu[~live]) or torch.count_nonzero(logvar[~live])):
            raise ValueError(
                'dead-slot rows of mu_views/logvar_views must be exactly zero')

    def forward_full(self, obs_dict):
        out = super().forward_full(obs_dict)
        z = out['z_views']                       # raw z_v, (N, K, D)
        active = out['view_active']              # (N, K) bool
        n, k, d = z.shape
        mu, logvar = self.gauss_head(z.reshape(n * k, d))
        mu = mu.reshape(n, k, d).masked_fill(~active.unsqueeze(-1), 0.0)
        logvar = logvar.reshape(n, k, d).masked_fill(~active.unsqueeze(-1), 0.0)
        self._check_gaussian(mu, logvar, active)
        if self.fusion_mode == 'mha':
            # same call shape as the parent's fusion, over the MEANS
            query = self.fusion_query.expand(n, 1, self.fused_dim)
            z_g, _ = self.fusion(query, mu, mu,
                                 key_padding_mask=~active, need_weights=False)
            z_g = z_g.reshape(n, self.fused_dim)
            z_g_logvar = None
        elif self.fusion_mode == 'poe':
            z_g, z_g_logvar = poe_fuse(mu, logvar, active)
        else:                                    # 'mean'
            z_g = uniform_fuse(mu, active)
            z_g_logvar = None
        return {**out,
                'z_global': z_g,
                'mu_views': mu,
                'logvar_views': logvar,
                'z_global_logvar': z_g_logvar}
