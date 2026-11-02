"""B1 plus a residual token-interaction branch before the per-view pool.

The entire B1 path is kept: 9-channel RGB+Pluecker input, the shared ResNet18
with per-stage FiLM from the camera-frame EE history, the per-view average
pool, and the masked cross-view MHA fusion. The only addition is between the
layer4 FiLM output and the pool:

    F_v = layer4 map, (M, 512, 3, 3)          # G = 9 tokens per view
    z_v_base = avgpool(F_v)                   # B1's readout, unchanged
    U = one Pre-LN transformer over tokens
    delta_v = mean_spatial(U_v - F_v)
    z_v_new = z_v_base + W_out(delta_v)       # W_out zero-init

Modes (the A/B/C/D screen in ENCODER_B1_RESIDUAL_TOKEN_PLAN.md):
    none   -- B1 exactly; no branch, no extra parameters
    single -- one token per view (z_v_base), sequence length 1 per view
    intra  -- G spatial tokens per view, each view its own sequence
    cross  -- G spatial tokens per view, one padded joint sequence

`intra` needs no attention mask: batching each active view as its own sequence
is exactly "attention only within this view". `cross` uses only a
`key_padding_mask` over inactive slots. There is no positional encoding, no
camera slot id, no extra ray MLP and no extra history query -- the branch asks
whether the existing B1 features benefit from token interaction, not whether a
new tokenizer/geometry path helps.

The output projection is zero-initialized, so at init `z_v_new == z_v_base`
bit-for-bit and, with the same B1 state dict loaded, every mode reproduces B1's
`forward`/`forward_full` exactly. The transformer itself is normally
initialized; only the output gate is zero. Construction runs under a private
CPU RNG stream (the same convention as `ProbViewConditionedObsEncoder`), so
B/C/D branch parameters are bit-identical and constructing the subclass
consumes no global RNG. That keeps A/B/C/D's later data/crop/noise streams
aligned without relying on a reset.

Invalid views are never encoded by the parent, are excluded from the joint
sequence by the padding mask, and their `z_views` rows stay exactly zero (the
M4 invariant).
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn

from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    ViewConditionedObsEncoder)

TOKEN_MIXER_MODES = ('none', 'single', 'intra', 'cross')
# state-dict prefix of the branch inside the encoder module; the policy adds
# its own `obs_encoder.` prefix on top of this.
BRANCH_STATE_PREFIX = 'token_mixer.'


class ResidualTokenMixer(nn.Module):
    """One Pre-LN transformer layer plus a zero-initialized output gate.

    `forward` returns the transformer's residual-stream output; the caller
    subtracts the input and applies `out_proj` after the per-view mean.
    """

    def __init__(
            self,
            dim: int,
            n_heads: int = 8,
            ffn_dim: int = 1024,
            dropout: float = 0.0):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=float(dropout),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, seq, key_padding_mask=None):
        return self.layer(
            seq, src_key_padding_mask=key_padding_mask)


class ResidualTokenViewConditionedObsEncoder(ViewConditionedObsEncoder):
    """B1 with `none|single|intra|cross` residual token interaction."""

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
            crop_shape=None,
            random_crop: bool = True,
            use_group_norm: bool = False,
            imagenet_norm: bool = False,
            resize_shape=None,
            share_rgb_model: bool = True,
            token_mixer_mode: str = 'none',
            token_mixer_heads: int = 8,
            token_mixer_ffn_dim: int = 1024,
            token_mixer_dropout: float = 0.0,
            token_mixer_init_seed: int = 42,
            ):
        mode = str(token_mixer_mode)
        if mode not in TOKEN_MIXER_MODES:
            raise ValueError(
                f'token_mixer_mode must be one of {TOKEN_MIXER_MODES}, '
                f'got {token_mixer_mode!r}')

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

        self.token_mixer_mode = mode
        self.token_mixer = None
        if mode != 'none':
            # Private stream: the parent's construction consumed exactly the
            # global stream the B1 baseline would have consumed, and the branch
            # below consumes none of it. Do not keep the Generator around.
            cpu_state = torch.get_rng_state()
            try:
                generator = torch.Generator().manual_seed(
                    int(token_mixer_init_seed))
                torch.set_rng_state(generator.get_state())
                self.token_mixer = ResidualTokenMixer(
                    dim=self.fused_dim,
                    n_heads=int(token_mixer_heads),
                    ffn_dim=int(token_mixer_ffn_dim),
                    dropout=float(token_mixer_dropout),
                )
            finally:
                torch.set_rng_state(cpu_state)

    # ------------------------------------------------------------ readout seam
    def _encode_active_views(self, x, cond, rows, cols, active):
        """B1's pooled readout plus the optional residual token branch.

        `x` is `(M, 9, H, W)`, `cond` `(M, cond_dim)`, `rows`/`cols` the active
        `(sample, slot)` pairs in row-major order, and `active` the `(N, K)`
        bool mask. Returns `(z_active (M, D), extras)`, where the extras carry
        `z_views_base` for the latent-probe residual diagnostics.
        """
        if self.token_mixer is None:
            return super()._encode_active_views(
                x, cond, rows, cols, active)

        fmap = self._backbone_map(x, cond)             # (M, D, Gh, Gw)
        if fmap.shape[1] != self.fused_dim:
            raise RuntimeError(
                f'backbone map has {fmap.shape[1]} channels but fused_dim is '
                f'{self.fused_dim}')
        grid_h, grid_w = int(fmap.shape[-2]), int(fmap.shape[-1])
        g = grid_h * grid_w
        if g <= 0:
            raise RuntimeError(f'empty spatial grid {fmap.shape[-2:]}')

        z_base = torch.flatten(self.backbone.avgpool(fmap), 1)
        flat = fmap.flatten(2).transpose(1, 2)         # (M, G, D)

        if self.token_mixer_mode == 'single':
            # One token per view, sequence length 1. Attention over a single
            # token is the identity on values; the gain is the FFN/LayerNorm
            # capacity per view.
            u = self.token_mixer(z_base[:, None, :])   # (M, 1, D)
            delta = u[:, 0] - z_base
        elif self.token_mixer_mode == 'intra':
            # One sequence per active view: attention cannot cross views.
            u = self.token_mixer(flat)                 # (M, G, D)
            delta = (u - flat).mean(dim=1)
        else:                                          # 'cross'
            n, k = active.shape
            feat = fmap.new_zeros((n, k, g, self.fused_dim))
            feat[rows, cols] = flat
            pad = ~active.repeat_interleave(g, dim=1)  # (n, K*G), True=ignore
            u = self.token_mixer(
                feat.reshape(n, k * g, self.fused_dim),
                key_padding_mask=pad)
            delta = (u.reshape(n, k, g, self.fused_dim) - feat).mean(dim=2)
            delta = delta[rows, cols]

        z = z_base + self.token_mixer.out_proj(delta)

        tokens_base = z_base.new_zeros(
            (active.shape[0], self.n_slots, self.fused_dim))
        tokens_base[rows, cols] = z_base
        return z, {'z_views_base': tokens_base}
