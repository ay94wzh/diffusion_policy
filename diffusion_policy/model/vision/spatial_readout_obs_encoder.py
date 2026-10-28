"""Stage S: pure-RGB spatial-readout observation encoder.

One shared ResNet18 is applied to every view's image. The layer3 feature map
(5x5x256 for the 76x76 crop) is projected to 512 channels per pixel and read by
ONE learnable 512-d query per view through an 8-head intra-image
cross-attention, producing one 512-d token per view. The tokens then go through
the same masked cross-view MHA fusion as M3 and the 9 low-dim observations are
concatenated exactly as before, so ``output_shape()`` is 521 and the action
UNet is byte-identical to the M1/L1/B runs.

This is the S rung of ``ENCODER_EXPERIMENT_PLAN.md``: the ablation is against B
(the M3 encoder with the corrected EE-history protocol). S removes B's input
Pluecker channels, its per-stage FiLM from camera-frame EE history, and its
global average pool, and replaces the pool with a spatial query readout. There
is deliberately **no ray encoding, no EE-history conditioning and no extra 2D
positional encoding** in this round.

Why the camera/history keys are not touched
-------------------------------------------
The dataset and the rollout runner still emit ``view_slot_XX_cam`` and
``view_eef_hist`` because the shared latent-probe workspace stacks them
unconditionally and the normalizer has identity entries for them. S simply
ignores those keys: its constructor has no cam/history arguments, and its
``output_shape()`` dummy obs contains neither. A missing-key or shape error in
that path therefore fails loudly rather than silently changing the width.

Shapes
------
Per the policy's flattening, the leading axis of ``obs_dict`` here is ``B*To``:
each element is one frame with ``K`` view slots. ``view_mask`` is ``(N, K)``
(1.0 == live). Inactive slots are never encoded; their ``z_views`` rows are
exactly zero, and the fusion ignores them via ``key_padding_mask``.

Token order
-----------
``proj(fmap).flatten(2)`` enumerates the layer3 grid row-major: token
``t = i * W3 + j`` for feature cell ``(i, j)``. S does not consume this order
(no positional encoding), but R (the ray rung) must generate its feature-grid
Pluecker map in the same order.
"""
from typing import Dict, Optional, Tuple

import copy

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.crop_randomizer import (
    sample_random_image_crops, crop_image_from_indices)
from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    IMAGENET_MEAN, IMAGENET_STD)
from diffusion_policy.common.pytorch_util import replace_submodules


class SpatialReadoutObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: nn.Module,
            fused_dim: int = 512,
            n_heads: int = 8,
            crop_shape: Optional[Tuple[int, int]] = None,
            random_crop: bool = True,
            use_group_norm: bool = False,
            imagenet_norm: bool = False,
            share_rgb_model: bool = True,
            view_mask_key: str = 'view_mask',
            ):
        """S encoder: RGB-only backbone, spatial query readout, M3 fusion.

        Args are the B-encoder subset that S actually uses. There is no
        ``cam_keys`` / ``eef_hist_key`` / ``eef_hist_steps`` / ``cond_dim`` /
        ``use_plucker`` / ``use_eef_hist`` / ``resize_shape``: passing one is a
        TypeError, which is the guard that a copied B config cannot silently
        route camera or history data into S.
        """
        super().__init__()
        if not share_rgb_model:
            raise NotImplementedError(
                'S has exactly one shared backbone; share_rgb_model=False '
                'would silently build one resnet per view, which is not the '
                'intended architecture (the stock default is False, so set '
                'share_rgb_model: True in the config).')
        if not isinstance(rgb_model, nn.Module):
            raise TypeError(
                f'rgb_model must be an nn.Module, got {type(rgb_model)!r}')

        rgb_keys, low_dim_keys = list(), list()
        key_shape_map = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            typ = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if typ == 'rgb':
                rgb_keys.append(key)
            elif typ == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f'Unsupported obs type: {typ}')
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        if len(rgb_keys) == 0:
            raise ValueError('no rgb view slots in shape_meta')

        # A cam key in shape_meta would be concatenated as low-dim and change
        # the 521 width. The runner serves cam vectors out-of-band; they must
        # never appear here. view_mask and the EE-history key are likewise
        # extra obs keys, not low-dim inputs.
        for key in list(rgb_keys) + list(low_dim_keys):
            if key.endswith('_cam'):
                raise ValueError(
                    f'{key!r} looks like a camera key and must NOT be in '
                    'shape_meta -- the env cannot serve it as a normalizer '
                    'key. Pass it via the obs dict instead.')
        for key in (view_mask_key, 'view_eef_hist'):
            if key in low_dim_keys:
                raise ValueError(
                    f'{key!r} must not be a low_dim shape_meta key, or it '
                    'would be concatenated into the encoder output and change '
                    'the width the UNet sees')

        rgb_shapes = [key_shape_map[k] for k in rgb_keys]
        if len(set(rgb_shapes)) != 1:
            raise NotImplementedError(
                f'all view slots must share a shape (one shared backbone runs '
                f'on a stacked batch); got {rgb_shapes}')
        slot_shape = tuple(rgb_shapes[0])
        if slot_shape[0] != 3:
            raise ValueError(
                f'S is pure RGB: every view slot must have 3 channels, got '
                f'{slot_shape}')

        fused_dim = int(fused_dim)
        n_heads = int(n_heads)
        if fused_dim % n_heads != 0:
            raise ValueError(
                f'fused_dim {fused_dim} must be divisible by n_heads {n_heads}')
        crop_shape = None if crop_shape is None else tuple(crop_shape)
        if crop_shape is not None:
            ch, cw = crop_shape
            if not (0 < ch < slot_shape[1] and 0 < cw < slot_shape[2]):
                raise ValueError(
                    f'crop_shape {crop_shape} must be inside the '
                    f'{slot_shape[1]}x{slot_shape[2]} image')

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.slot_shape = slot_shape
        self.view_mask_key = str(view_mask_key)
        self.fused_dim = fused_dim
        self.n_heads = n_heads
        self.crop_shape = crop_shape
        self.random_crop = bool(random_crop)
        self.imagenet_norm = bool(imagenet_norm)
        self.n_slots = len(rgb_keys)

        # --- shared RGB backbone, truncated at layer3 ------------------------
        backbone = copy.deepcopy(rgb_model)
        if not hasattr(backbone, 'layer3'):
            raise TypeError(
                'rgb_model has no layer3; S expects a ResNet-style backbone')
        if not isinstance(backbone.conv1, nn.Conv2d):
            raise TypeError('rgb_model has no nn.Conv2d conv1')
        if backbone.conv1.in_channels != 3:
            raise ValueError(
                f'S is pure RGB: conv1 has {backbone.conv1.in_channels} input '
                'channels; expected 3 (no Pluecker channels)')
        # Drop layer4/avgpool/fc instead of leaving them unused: unused
        # parameters would still be copied into the EMA model, counted by the
        # optimizer and stored in every checkpoint.
        for name in ('layer4', 'avgpool', 'fc'):
            if hasattr(backbone, name):
                delattr(backbone, name)
        if use_group_norm:
            backbone = replace_submodules(
                root_module=backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features))
        self.backbone = backbone

        in_ch = backbone.layer3[-1].conv2.out_channels
        self.proj = nn.Conv2d(in_ch, self.fused_dim, kernel_size=1)
        self.intra_attn = nn.MultiheadAttention(
            embed_dim=self.fused_dim, num_heads=self.n_heads,
            batch_first=True)
        self.spatial_query = nn.Parameter(
            torch.zeros(1, 1, self.fused_dim))
        # Identical to M3's cross-view fusion: a separate learnable query, so
        # the plan's "one shared query" remains the spatial readout query.
        self.fusion = nn.MultiheadAttention(
            embed_dim=self.fused_dim, num_heads=self.n_heads,
            batch_first=True)
        self.fusion_query = nn.Parameter(
            torch.zeros(1, 1, self.fused_dim))

        # --- shape probe: record the feature grid, validate the width --------
        # This runs in eval mode so BatchNorm running statistics are not
        # mutated, and it consumes no RNG (the training RNG stream must stay
        # comparable to B's).
        h0, w0 = self.crop_shape if self.crop_shape is not None else slot_shape[1:]
        ref = next(self.backbone.parameters())
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                fmap = self._backbone_feature_map(torch.zeros(
                    1, 3, h0, w0, dtype=ref.dtype, device=ref.device))
        finally:
            self.backbone.train(was_training)
        if fmap.shape[1] != self.proj.in_channels:
            raise ValueError(
                f'backbone emits {fmap.shape[1]} channels but proj expects '
                f'{self.proj.in_channels}')
        self.token_grid = tuple(int(v) for v in fmap.shape[-2:])
        self.n_tokens = int(self.token_grid[0] * self.token_grid[1])

    # ------------------------------------------------------------------ views
    def _prepare_image(self, img: torch.Tensor, train: bool) -> torch.Tensor:
        """Crop one slot's images, then ImageNet-normalize.

        This is B's ``_prepare_slots`` minus the camera validation and the
        Pluecker map: one crop window is sampled per image, and the eval path
        uses the centre crop (the same offsets as ``ttf.center_crop`` for
        84 -> 76).
        """
        if img.dim() != 4 or img.shape[1] != 3:
            raise ValueError(
                f'expected an (N, 3, H, W) RGB batch, got {tuple(img.shape)}')
        n, _, h, w = img.shape
        if self.crop_shape is not None:
            ch, cw = self.crop_shape
            if train and self.random_crop:
                _, inds = sample_random_image_crops(
                    images=img, crop_height=ch, crop_width=cw, num_crops=1)
                inds = inds.reshape(n, 2)
            else:
                inds = img.new_full((n, 2), 0)
                inds[:, 0] = (h - ch) // 2
                inds[:, 1] = (w - cw) // 2
            inds = inds.long()
            img = crop_image_from_indices(img, inds, ch, cw)
        if self.imagenet_norm:
            mean = img.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = img.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
            img = (img - mean) / std
        return img

    def _backbone_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Walk the resnet explicitly up to layer3; no pool, no flatten."""
        b = self.backbone
        h = b.relu(b.bn1(b.conv1(x)))
        h = b.maxpool(h)
        h = b.layer1(h)
        h = b.layer2(h)
        h = b.layer3(h)
        return h

    def _encode_views(self, x: torch.Tensor) -> torch.Tensor:
        """(M, 3, ch, cw) -> (M, fused_dim): one shared query per view."""
        fmap = self._backbone_feature_map(x)                    # (M, C, h, w)
        tokens = self.proj(fmap).flatten(2).transpose(1, 2)     # (M, h*w, D)
        q = self.spatial_query.expand(tokens.shape[0], -1, -1)  # (M, 1, D)
        read, _ = self.intra_attn(q, tokens, tokens, need_weights=False)
        return read[:, 0]                                       # (M, D)

    # ---------------------------------------------------------------- forward
    def forward(self, obs_dict) -> torch.Tensor:
        """Drop-in contract, unchanged: ``(N, fused_dim + low-dim)``."""
        out = self.forward_full(obs_dict)
        low = [obs_dict[k] for k in self.low_dim_keys]
        return torch.cat([out['z_global'], *low], dim=-1)

    def forward_full(self, obs_dict) -> Dict[str, torch.Tensor]:
        """`forward`'s computation, plus the per-view tokens M4/probes need.

        Returns ``{'z_global': (N, D), 'z_views': (N, K, D),
        'view_active': (N, K) bool}``. Rows of inactive slots are exactly
        zero, which is the invariant the latent probe and M4's loss mask rely
        on. The intra-image attention never mixes views; the only cross-view
        path is the same masked MHA fusion as M3.
        """
        key0 = self.rgb_keys[0]
        if key0 not in obs_dict:
            raise KeyError(f'missing rgb slot {key0!r}')
        img0 = obs_dict[key0]
        n = img0.shape[0]
        if n == 0:
            raise ValueError('empty batch')
        for key in self.rgb_keys:
            if key not in obs_dict:
                raise KeyError(f'missing rgb slot {key!r}')
            img = obs_dict[key]
            if img.dim() != 4 or tuple(img.shape[1:]) != self.slot_shape:
                raise ValueError(
                    f'{key}: expected (N, {self.slot_shape}), got '
                    f'{tuple(img.shape)}')
            if img.shape[0] != n:
                raise ValueError(
                    f'{key}: batch size {img.shape[0]} != {n}; all view slots '
                    'must share the leading axis')
        if self.view_mask_key not in obs_dict:
            raise KeyError(f'missing {self.view_mask_key!r}')
        mask = obs_dict[self.view_mask_key].to(device=img0.device)
        if mask.dim() != 2 or tuple(mask.shape) != (n, self.n_slots):
            raise ValueError(
                f'{self.view_mask_key}: expected (N={n}, K={self.n_slots}), '
                f'got {tuple(mask.shape)}')
        mask_f = mask.to(torch.float32)
        if not bool(((mask_f == 0) | (mask_f == 1)).all()):
            raise ValueError(
                f'{self.view_mask_key} must be binary (0.0/1.0)')
        active = mask_f > 0.5
        if not bool(active.any(dim=1).all()):
            raise ValueError(
                'every sample needs at least one active view (an all-False '
                'row would leave the attention query fully masked -> NaN)')

        # Encode only the active slots; masked-out ones cost nothing.
        # nonzero() is row-major, so `sel` preserves that order within each
        # slot and `x[sel] = sub` scatters back into the same layout.
        rows, cols = active.nonzero(as_tuple=True)
        x = None
        for k in range(self.n_slots):
            sel = cols == k
            if not bool(sel.any()):
                continue
            sub = self._prepare_image(
                obs_dict[self.rgb_keys[k]][rows[sel]], self.training)
            if x is None:
                x = sub.new_empty((rows.numel(),) + tuple(sub.shape[1:]))
            x[sel] = sub
        readouts = self._encode_views(x)                        # (M, D)

        tokens = readouts.new_zeros((n, self.n_slots, self.fused_dim))
        tokens[rows, cols] = readouts
        z_g, _ = self.fusion(
            self.fusion_query.expand(n, 1, self.fused_dim),
            tokens, tokens,
            key_padding_mask=~active, need_weights=False)
        return {'z_global': z_g.reshape(n, self.fused_dim),
                'z_views': tokens,
                'view_active': active}

    @torch.no_grad()
    def output_shape(self):
        """Measured, like the stock encoder -- without any camera/history keys.

        Must stay eval-mode and RNG-free: `DiffusionUnetImagePolicy.__init__`
        calls this after the workspace seeds torch, so a train-mode dummy
        forward would draw random crops and shift the whole training stream.
        """
        was_training = self.training
        self.eval()
        try:
            example = dict()
            for key, attr in self.shape_meta['obs'].items():
                example[key] = torch.zeros(
                    (1,) + tuple(attr['shape']),
                    dtype=self.dtype, device=self.device)
            example[self.view_mask_key] = torch.ones(
                (1, self.n_slots), dtype=self.dtype, device=self.device)
            out = self.forward(example)
            expected = self.fused_dim + sum(
                self.key_shape_map[k][0] for k in self.low_dim_keys)
            if tuple(out.shape[1:]) != (expected,):
                raise RuntimeError(
                    f'encoder emits {tuple(out.shape[1:])}, expected '
                    f'({expected},) = fused_dim {self.fused_dim} + low-dim '
                    f'{expected - self.fused_dim}; the UNet conditioning width '
                    f'would differ from the baseline')
            if not bool(torch.isfinite(out).all()):
                raise RuntimeError(
                    'encoder output is not finite on dummy input')
            return out.shape[1:]
        finally:
            self.train(was_training)
