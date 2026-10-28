"""Stage R: spatial-readout observation encoder plus feature-level ray encoding.

R is S (``spatial_readout_obs_encoder.SpatialReadoutObsEncoder``) with ONE
structural addition, the feature-level geometric address of
``ENCODER_EXPERIMENT_PLAN.md`` section 4.2:

    k = W_K LN(f + e),    v = W_V LN(f),    e = E_ray([d, m / L])

where ``f`` is the 512-d visual token of a layer3 feature cell, ``(d, m)`` is
the Pluecker embedding of the ray through that cell's receptive-field centre
(unit direction; world-frame moment ``m = o x d``), and ``E_ray`` is a small
MLP. The ray is added to the **key** only; the value keeps the visual content.
S is the ``e = 0`` case. The shared 512-d query, the 8-head intra-image
attention, the masked cross-view MHA fusion and the 521-wide low-dim interface
are byte-identical to S, so the action UNet is unchanged.

Where the rays come from (crop synchronisation)
-----------------------------------------------
A resnet18 stem (conv1 s2, maxpool s2, layer2 s2, layer3 s2) has an effective
stride of 16 and symmetric padding, so layer3 cell ``(i, j)`` of a crop has its
receptive-field centre at crop pixel ``(S*i, S*j)``: in ``ray_dirs_cam``'s
continuous coordinates, ``u = off_w + S*j + 0.5`` and ``v = off_h + S*i + 0.5``
in the **original** image, where ``(off_h, off_w)`` is the sampled crop's
top-left. The intrinsics (``fovy`` and the principal point) are the original
image's, not the crop's -- computing them from the crop size would silently
shift every ray. ``_feature_grid_rays`` therefore evaluates
``plucker_ray_map_at`` at those coordinates directly, with the same crop
indices the image was cropped with; the map is never resized or interpolated.
Token order is ``proj(fmap).flatten(2)``'s row-major ``i * W3 + j``, which is
the order the coordinates are generated in.

The ``total_stride`` -- and the assert that the measured ``token_grid`` equals
``(crop - 1) // stride + 1`` -- is derived from the backbone's conv/maxpool
strides, so a different ResNet cannot silently misalign the rays. The formula
holds for symmetric padding (``p = (k-1)/2``), which torchvision ResNets use.

Initialisation order (important for reading early epochs)
---------------------------------------------------------
``spatial_query`` is zero-initialised exactly as in S, and the ray MLP's
output layer is zero-initialised as well. With a zero query the attention
logits are identically zero, so the **key path -- and therefore the ray MLP --
is inert at initialisation**: the output is a uniform attention over the
value path. The query's own gradient is not zero, so it moves on the first
optimizer step and the key path becomes live from then on; the ray MLP's
output layer then receives gradient, and its first layer follows one step
later (the usual zero-init-head chain). This is deliberate: R starts as close
to S as the added design allows, and the added path cannot inject noise before
it has been asked for. A run's first logged step is therefore expected to show
no ray effect; by the first checkpoint it is a live branch.

What R deliberately does not touch
----------------------------------
No EE-history conditioning (that is B / RH), no input-level Pluecker channels
(B's conv1), no 2D positional encoding, no depth. The dataset and runner still
emit ``view_eef_hist``: the shared latent-probe workspace stacks it
unconditionally and the normalizer has an identity entry, so R ignores the key
exactly as S does. ``cam_keys`` is the only extra input R reads. The history
constructor arguments are absent on purpose: passing one is a TypeError, the
guard that a copied B config cannot silently route history into R.

Shapes
------
Per the policy's flattening the leading axis of ``obs_dict`` here is ``B*To``:
each element is one frame with ``K`` view slots. ``view_mask`` is ``(N, K)``
(1.0 == live). Inactive slots are never encoded, so their (all-zero) camera
vector -- which is not a valid pose -- is never turned into rays; their
``z_views`` rows are exactly zero and the fusion ignores them.
"""
from typing import Dict, Optional, Sequence, Tuple

import copy

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.crop_randomizer import (
    sample_random_image_crops, crop_image_from_indices)
from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    IMAGENET_MEAN, IMAGENET_STD)
from diffusion_policy.model.vision.plucker import plucker_ray_map_at
from diffusion_policy.common.pytorch_util import replace_submodules

# [d(3) | m(3)] -- unit ray direction and world-frame Pluecker moment
RAY_FEAT_DIM = 6


class RaySpatialReadoutObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: nn.Module,
            cam_keys: Optional[Sequence[str]] = None,
            view_mask_key: str = 'view_mask',
            fused_dim: int = 512,
            n_heads: int = 8,
            ray_hidden_dim: int = 512,
            ray_scale: float = 1.0,
            use_ray_encoding: bool = True,
            crop_shape: Optional[Tuple[int, int]] = None,
            random_crop: bool = True,
            use_group_norm: bool = False,
            imagenet_norm: bool = False,
            share_rgb_model: bool = True,
            ):
        """R encoder: S's spatial readout with a ray-encoded key.

        ``ray_scale`` is the plan's ``L`` (metres): only the Pluecker moment is
        divided by it, for numerical normalisation. ``use_ray_encoding=False``
        is the diagnostic ablation -- ``e`` is exactly zero and the ray MLP is
        never called (so its gradient is exactly zero), while every parameter
        and every other path is identical. There is deliberately no
        ``eef_hist_key`` / ``eef_hist_steps`` / ``use_eef_hist`` / ``cond_dim``
        / ``use_plucker`` / ``resize_shape``: R has no history path, and
        passing one is a TypeError.
        """
        super().__init__()
        if not share_rgb_model:
            raise NotImplementedError(
                'R has exactly one shared backbone; share_rgb_model=False '
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

        # A cam key in shape_meta would be concatenated as low-dim (changing
        # the 521 width) and the rollout env raises on the key suffix. The
        # runner serves cam vectors out-of-band; they must never appear here.
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

        if cam_keys is None:
            cam_keys = [k[:-len('_image')] + '_cam' if k.endswith('_image')
                        else k + '_cam' for k in rgb_keys]
        cam_keys = list(cam_keys)
        if len(cam_keys) != len(rgb_keys):
            raise ValueError(
                f'cam_keys has {len(cam_keys)} entries but there are '
                f'{len(rgb_keys)} rgb view slots')

        rgb_shapes = [key_shape_map[k] for k in rgb_keys]
        if len(set(rgb_shapes)) != 1:
            raise NotImplementedError(
                f'all view slots must share a shape (one shared backbone runs '
                f'on a stacked batch); got {rgb_shapes}')
        slot_shape = tuple(rgb_shapes[0])
        if slot_shape[0] != 3:
            raise ValueError(
                f'R reads pure RGB; every view slot must have 3 channels, got '
                f'{slot_shape}')

        fused_dim = int(fused_dim)
        n_heads = int(n_heads)
        if fused_dim % n_heads != 0:
            raise ValueError(
                f'fused_dim {fused_dim} must be divisible by n_heads {n_heads}')
        ray_hidden_dim = int(ray_hidden_dim)
        if ray_hidden_dim <= 0:
            raise ValueError(f'ray_hidden_dim must be positive, got {ray_hidden_dim}')
        ray_scale = float(ray_scale)
        if not (ray_scale > 0.0) or ray_scale != ray_scale:
            raise ValueError(f'ray_scale (L) must be positive, got {ray_scale}')
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
        self.cam_keys = cam_keys
        self.key_shape_map = key_shape_map
        self.slot_shape = slot_shape
        self.view_mask_key = str(view_mask_key)
        self.fused_dim = fused_dim
        self.n_heads = n_heads
        self.ray_hidden_dim = ray_hidden_dim
        self.ray_scale = ray_scale
        self.use_ray_encoding = bool(use_ray_encoding)
        self.crop_shape = crop_shape
        self.random_crop = bool(random_crop)
        self.imagenet_norm = bool(imagenet_norm)
        self.n_slots = len(rgb_keys)

        # --- shared pure-RGB backbone, truncated at layer3 (same as S) -------
        backbone = copy.deepcopy(rgb_model)
        if not hasattr(backbone, 'layer3'):
            raise TypeError(
                'rgb_model has no layer3; R expects a ResNet-style backbone')
        if not isinstance(backbone.conv1, nn.Conv2d):
            raise TypeError('rgb_model has no nn.Conv2d conv1')
        if backbone.conv1.in_channels != 3:
            raise ValueError(
                f'R reads pure RGB: conv1 has {backbone.conv1.in_channels} '
                'input channels; expected 3 (the ray is added at the feature '
                'level, not the input level)')
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

        # --- feature-level ray encoding (section 4.2) ------------------------
        # e = E_ray([d, m/L]); the output layer is zero-init so e == 0 at
        # initialisation and the added path starts inert, the project's
        # convention for every added conditioning path (FiLM heads, aux head,
        # fusion queries).
        # Linear -> SiLU -> Linear with a zero-init output, the project's
        # head pattern (`per_view_aux_head` / `view_pose_set_head`).
        self.ray_mlp = nn.Sequential(
            nn.Linear(RAY_FEAT_DIM, self.ray_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.ray_hidden_dim, self.fused_dim),
        )
        nn.init.zeros_(self.ray_mlp[-1].weight)
        nn.init.zeros_(self.ray_mlp[-1].bias)
        # k = W_K LN(f + e), v = W_V LN(f): the LayerNorms are applied before
        # the attention module's own W_K / W_V projections
        self.key_norm = nn.LayerNorm(self.fused_dim)
        self.value_norm = nn.LayerNorm(self.fused_dim)

        self.intra_attn = nn.MultiheadAttention(
            embed_dim=self.fused_dim, num_heads=self.n_heads,
            batch_first=True)
        self.spatial_query = nn.Parameter(
            torch.zeros(1, 1, self.fused_dim))
        # Identical to S/M3's cross-view fusion: a separate learnable query.
        self.fusion = nn.MultiheadAttention(
            embed_dim=self.fused_dim, num_heads=self.n_heads,
            batch_first=True)
        self.fusion_query = nn.Parameter(
            torch.zeros(1, 1, self.fused_dim))

        # --- shape probe: feature grid, token count, ray-grid stride ---------
        # Eval mode so BatchNorm running statistics are not mutated; consumes
        # no RNG (the training RNG stream must stay comparable to S's draw
        # order).
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
        self.total_stride = self._backbone_stride()
        gh, gw = self.token_grid
        if ((h0 - 1) // self.total_stride + 1 != gh or
                (w0 - 1) // self.total_stride + 1 != gw):
            raise RuntimeError(
                f'feature grid {self.token_grid} does not follow the '
                f'symmetric-padding stride formula '
                f'floor((size-1)/stride)+1 with stride {self.total_stride}; '
                'the feature-cell -> pixel mapping that places the rays would '
                'be wrong. R supports standard torchvision ResNet stems.')

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _stage_stride(stage: nn.Module) -> int:
        """Largest downsample factor inside one stage (max, not product: a
        residual block's conv and its downsample conv both have stride 2)."""
        s = 1
        for m in stage.modules():
            if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
                st = m.stride if m.stride is not None else m.kernel_size
                st = st if isinstance(st, tuple) else (st, st)
                s = max(s, max(int(v) for v in st))
        return s

    def _backbone_stride(self) -> int:
        """Effective stride from the input image to the layer3 feature grid."""
        b = self.backbone
        s = self._stage_stride(b.conv1)
        s *= self._stage_stride(b.maxpool)
        for layer in (b.layer1, b.layer2, b.layer3):
            s *= self._stage_stride(layer)
        return s

    def _feature_grid_rays(self, cam: torch.Tensor, inds: torch.Tensor,
                           h: int, w: int) -> torch.Tensor:
        """``(N, G, 6)`` Pluecker embedding at the cropped feature-grid cells.

        ``inds`` is ``(N, 2)`` top-left crop offsets in the original image;
        ``h``/``w`` are the ORIGINAL image dimensions (the intrinsics' frame).
        Cell ``(i, j)`` sits at original pixel ``(inds_h + S*i,
        inds_w + S*j)``; the coordinates are the pixel centres. Row-major
        ``i * gw + j``, matching ``proj(fmap).flatten(2)``; the moment is
        already divided by ``ray_scale``.
        """
        n = cam.shape[0]
        gh, gw = self.token_grid
        stride = self.total_stride
        jj = torch.arange(gw, dtype=cam.dtype, device=cam.device) * stride
        ii = torch.arange(gh, dtype=cam.dtype, device=cam.device) * stride
        off = inds.to(dtype=cam.dtype, device=cam.device)
        # pixel centres (index + 0.5) of the receptive-field centres:
        #   u = off_w + S*j + 0.5,   v = off_h + S*i + 0.5
        u = off[:, 1:2] + jj[None, :] + 0.5            # (N, gw)
        v = off[:, 0:1] + ii[None, :] + 0.5            # (N, gh)
        # expand to the full grid in row-major (i, j) order
        u = u[:, None, :].expand(n, gh, gw).reshape(n, gh * gw)
        v = v[:, :, None].expand(n, gh, gw).reshape(n, gh * gw)
        rays = plucker_ray_map_at(
            cam[:, :3], cam[:, 3:7], cam[:, 7], h, w, u, v)  # (N, 6, G)
        rays = rays.transpose(1, 2).contiguous()             # (N, G, 6)
        rays[..., 3:] = rays[..., 3:] / self.ray_scale
        return rays

    # ------------------------------------------------------------------ views
    def _prepare_slot(self, img: torch.Tensor, cam: torch.Tensor,
                      train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crop one slot's images and place rays at the same crop window.

        The crop draw is S's exactly (one window per image; centre crop in
        eval), so S and R differ only in what they do with the camera vector.
        """
        if img.dim() != 4 or img.shape[1] != 3:
            raise ValueError(
                f'expected an (N, 3, H, W) RGB batch, got {tuple(img.shape)}')
        n, _, h, w = img.shape
        cam = cam.to(dtype=img.dtype, device=img.device)
        if cam.dim() != 2 or tuple(cam.shape) != (n, 10):
            raise ValueError(
                f'expected a (N={n}, 10) camera vector, got {tuple(cam.shape)}')
        for name, got, want in (('h', int(cam[0, 8]), h),
                                ('w', int(cam[0, 9]), w)):
            if got != want:
                raise RuntimeError(
                    f'camera vector says {name}={got} but the image is '
                    f'{name}={want}; the intrinsics must describe the image '
                    'the rays are placed in')

        if self.crop_shape is not None:
            ch, cw = self.crop_shape
            if train and self.random_crop:
                _, inds = sample_random_image_crops(
                    images=img, crop_height=ch, crop_width=cw, num_crops=1)
                inds = inds.reshape(n, 2)
            else:
                # == ttf.center_crop for 84 -> 76
                inds = img.new_full((n, 2), 0)
                inds[:, 0] = (h - ch) // 2
                inds[:, 1] = (w - cw) // 2
            inds = inds.long()
            img = crop_image_from_indices(img, inds, ch, cw)
        else:
            inds = torch.zeros((n, 2), dtype=torch.long, device=img.device)

        # rays use the ORIGINAL (h, w) intrinsics and the sampled crop offsets
        rays = self._feature_grid_rays(cam, inds, h, w)
        if self.imagenet_norm:
            mean = img.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = img.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
            img = (img - mean) / std
        return img, rays

    def _backbone_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Walk the resnet explicitly up to layer3; no pool, no flatten."""
        b = self.backbone
        h = b.relu(b.bn1(b.conv1(x)))
        h = b.maxpool(h)
        h = b.layer1(h)
        h = b.layer2(h)
        h = b.layer3(h)
        return h

    def _encode_views(self, x: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        """(M, 3, ch, cw) + (M, G, 6) -> (M, fused_dim): one shared query."""
        fmap = self._backbone_feature_map(x)                    # (M, C, h, w)
        tokens = self.proj(fmap).flatten(2).transpose(1, 2)     # (M, G, D)
        if tokens.shape[1] != rays.shape[1]:
            raise RuntimeError(
                f'feature grid has {tokens.shape[1]} tokens but the ray grid '
                f'has {rays.shape[1]}; the row-major orders must match')
        if self.use_ray_encoding:
            e = self.ray_mlp(rays.to(dtype=tokens.dtype))       # (M, G, D)
        else:
            e = torch.zeros_like(tokens)
        keys = self.key_norm(tokens + e)
        values = self.value_norm(tokens)
        q = self.spatial_query.expand(tokens.shape[0], -1, -1)  # (M, 1, D)
        read, _ = self.intra_attn(q, keys, values, need_weights=False)
        return read[:, 0]                                       # (M, D)

    # ---------------------------------------------------------------- forward
    def forward(self, obs_dict) -> torch.Tensor:
        """Drop-in contract, unchanged: ``(N, fused_dim + low-dim)``."""
        out = self.forward_full(obs_dict)
        low = [obs_dict[k] for k in self.low_dim_keys]
        return torch.cat([out['z_global'], *low], dim=-1)

    def forward_full(self, obs_dict) -> Dict[str, torch.Tensor]:
        """`forward`'s computation, plus the per-view tokens probes need.

        Returns ``{'z_global': (N, D), 'z_views': (N, K, D),
        'view_active': (N, K) bool}``. Rows of inactive slots are exactly
        zero, which is the invariant the latent probe and M4's loss mask rely
        on. The intra-image attention never mixes views; the only cross-view
        path is the same masked MHA fusion as S/B.
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
        for cam_key in self.cam_keys:
            if cam_key not in obs_dict:
                raise KeyError(f'missing camera vector {cam_key!r}')
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

        # Encode only the active slots; masked-out ones cost nothing and are
        # never turned into rays (their camera vector is all zeros, which is
        # not a valid pose). nonzero() is row-major, so `sel` preserves that
        # order within each slot and `x[sel] = sub` scatters back exactly.
        rows, cols = active.nonzero(as_tuple=True)
        x = None
        rays = None
        for k in range(self.n_slots):
            sel = cols == k
            if not bool(sel.any()):
                continue
            sub_img, sub_rays = self._prepare_slot(
                obs_dict[self.rgb_keys[k]][rows[sel]],
                obs_dict[self.cam_keys[k]][rows[sel]],
                self.training)
            if x is None:
                x = sub_img.new_empty((rows.numel(),) + tuple(sub_img.shape[1:]))
                rays = sub_rays.new_empty(
                    (rows.numel(),) + tuple(sub_rays.shape[1:]))
            x[sel] = sub_img
            rays[sel] = sub_rays
        readouts = self._encode_views(x, rays)                  # (M, D)

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
        """Measured, like the stock encoder. The dummy cameras must be VALID
        poses: R turns the camera vector into rays, and an all-zero quaternion
        yields NaN directions with a perfectly correct output shape."""
        was_training = self.training
        self.eval()
        try:
            example = dict()
            for key, attr in self.shape_meta['obs'].items():
                example[key] = torch.zeros(
                    (1,) + tuple(attr['shape']),
                    dtype=self.dtype, device=self.device)
            for k, cam_key in enumerate(self.cam_keys):
                cam = torch.zeros(10, dtype=self.dtype, device=self.device)
                cam[0] = 0.5                     # position
                cam[3] = 1.0                     # wxyz w = 1 -> identity rotation
                cam[7] = 45.0                    # fovy in degrees
                cam[8] = self.key_shape_map[self.rgb_keys[k]][1]
                cam[9] = self.key_shape_map[self.rgb_keys[k]][2]
                example[cam_key] = cam.unsqueeze(0)
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
