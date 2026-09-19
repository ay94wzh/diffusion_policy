"""M3's view-conditioned observation encoder (drop-in for MultiImageObsEncoder).

One shared resnet18 is applied to every view's image, concatenated with that
view's Pluecker ray map and *modulated* by that view's camera-frame
end-effector history (FiLM/AdaGN, per PROPOSAL.md section 2.1). The per-view
latents are fused by attention with a learnable query into a single global
latent, which is concatenated with the low-dim obs exactly as the stock
encoder does.

Drop-in contract (why the policy and the workspace are untouched)
----------------------------------------------------------------
``DiffusionUnetImagePolicy`` does ``obs_feature_dim = obs_encoder.output_shape()[0]``
and ``global_cond_dim = obs_feature_dim * n_obs_steps``, then feeds the encoder
the whole normalized obs dict. So this class must provide ``__init__``,
``forward(obs_dict) -> (B*To, D)`` and ``output_shape()``, and
``fused_dim=512`` makes ``D == 512 + 9 == 521`` -- identical to M1/L1, so the
UNet is byte-identical and any gain is attributable to conditioning rather
than capacity. ``AdaGN`` only scales and shifts features, so it can never
change that width; ``output_shape()`` asserts it against the stock encoder.

Where the extra keys live
-------------------------
The camera vectors, the view mask and the EE history are deliberately NOT in
``shape_meta``: the env builds its robomimic modality mapping from
``shape_meta['obs']`` and ``RobomimicImageWrapper.__init__`` raises on any key
not ending image/quat/qpos/pos, so a cam key there would break the rollout env
(and 7 rgb keys would demand 7 live cameras). They arrive as extra keys in the
obs dict -- which the policy passes through unfiltered -- and this class learns
their names from its constructor.

Shapes
------
Per the policy's flattening, the leading axis here is ``B*To``: each element is
one frame with ``K`` view slots. Extra keys must be tiled to ``(To, ...)`` per
sample: cam ``(To, 10)``, mask ``(To, K)``, EE history ``(To, K, 11*H_steps)``.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import copy

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.crop_randomizer import (
    sample_random_image_crops, crop_image_from_indices)
from diffusion_policy.model.vision.plucker import plucker_ray_map
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

# [pos(3), rot6d(6), gripper(2)] per history step, in the view's camera frame
EEF_HIST_STEP_DIM = 11
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ViewConditionedObsEncoder(ModuleAttrMixin):
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
            ):
        super().__init__()
        if not share_rgb_model:
            raise NotImplementedError(
                'M3 has exactly one shared backbone; share_rgb_model=False '
                'would silently build one resnet per view, which is not the '
                'intended architecture (the stock default is False, so set '
                'share_rgb_model: True in the config).')
        if resize_shape is not None:
            raise NotImplementedError(
                'resize_shape would have to be applied to the Pluecker map with '
                'matching interpolation to stay pixel-aligned; not supported.')

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

        # A cam key in shape_meta would break the env: RobomimicImageWrapper
        # raises "Unsupported type <key>". Fail loudly here instead.
        for key in list(rgb_keys) + list(low_dim_keys):
            if key.endswith('_cam'):
                raise ValueError(
                    f'{key!r} looks like a camera key and must NOT be in '
                    'shape_meta -- the env cannot serve it and '
                    'RobomimicImageWrapper raises on the key suffix. Pass it '
                    'via cam_keys instead.')
        # the low-dim pass-through is what the matched-width claim depends on;
        # the extra keys must never be bucketed here
        for key in [view_mask_key, eef_hist_key]:
            if key in low_dim_keys:
                raise ValueError(
                    f'{key!r} must not be a low_dim shape_meta key, or it would '
                    'be concatenated into the encoder output and change the '
                    'width the UNet sees')

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
                f'all view slots must share a shape (one shared backbone runs on '
                f'a stacked batch); got {rgb_shapes}')

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.cam_keys = cam_keys
        self.view_mask_key = view_mask_key
        self.eef_hist_key = eef_hist_key
        self.key_shape_map = key_shape_map
        self.fused_dim = int(fused_dim)
        self.use_plucker = bool(use_plucker)
        self.use_eef_hist = bool(use_eef_hist)
        self.imagenet_norm = bool(imagenet_norm)
        self.random_crop = bool(random_crop)
        self.crop_shape = None if crop_shape is None else tuple(crop_shape)
        self.n_slots = len(rgb_keys)
        self.eef_hist_steps = int(eef_hist_steps)
        self.eef_hist_dim = EEF_HIST_STEP_DIM * self.eef_hist_steps

        # --- shared backbone, conv1 widened 3 -> 3+6 -------------------------
        backbone = copy.deepcopy(rgb_model)
        old = backbone.conv1
        # always widen to 9: the plucker-off ablation feeds zeros for the 6 ray
        # channels rather than shrinking conv1, so the two variants stay
        # parameter-identical and the ablation is exact (the gradient into the
        # zeroed channels is exactly zero)
        in_ch = old.in_channels + 6
        new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_channels] = old.weight
            if old.bias is not None:
                new.bias.copy_(old.bias)
        backbone.conv1 = new
        if use_group_norm:
            backbone = replace_submodules(
                root_module=backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features))
        self.backbone = backbone

        # channel widths after each stage we modulate (resnet18: 64,64,128,256,512)
        widths = [backbone.conv1.out_channels,
                  backbone.layer1[-1].conv2.out_channels,
                  backbone.layer2[-1].conv2.out_channels,
                  backbone.layer3[-1].conv2.out_channels,
                  backbone.layer4[-1].conv2.out_channels]
        self.film_widths = widths

        # --- conditioning: EE history -> per-stage (gamma, beta) -------------
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.eef_hist_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )
        self.film = nn.ModuleList([nn.Linear(cond_dim, 2 * c) for c in widths])
        # zero-init the FiLM heads so training starts unconditioned and the
        # conditioning-off ablation is exact
        for layer in self.film:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # --- fusion over view tokens ----------------------------------------
        self.fusion = nn.MultiheadAttention(
            embed_dim=self.fused_dim, num_heads=n_heads, batch_first=True)
        self.fusion_query = nn.Parameter(torch.zeros(1, 1, self.fused_dim))

        # matched-width guard: the backbone's pooled width is the fused width
        probe = self.backbone(torch.zeros(1, in_ch, 8, 8))
        if probe.shape[1] != self.fused_dim:
            raise ValueError(
                f'backbone emits {probe.shape[1]} features but fused_dim is '
                f'{self.fused_dim}; they must match or output_shape() will not '
                f'equal the stock encoder\'s')

    # ------------------------------------------------------------------ views
    def _prepare_slots(self, obs_dict, k, train):
        """Crop + normalize slot k, and return its Pluecker map."""
        key = self.rgb_keys[k]
        img = obs_dict[key]
        if img.shape[1] != 3:
            raise ValueError(f'{key}: expected (N, 3, H, W), got {tuple(img.shape)}')
        n, _, h, w = img.shape
        cam = obs_dict[self.cam_keys[k]].to(dtype=img.dtype, device=img.device)
        if cam.shape[0] != n or cam.shape[-1] != 10:
            raise ValueError(
                f'{self.cam_keys[k]}: expected (N, 10) matching {key} (N={n}), '
                f'got {tuple(cam.shape)}')
        for name, got, want in (('h', int(cam[0, 8]), h),
                                ('w', int(cam[0, 9]), w)):
            if got != want:
                raise RuntimeError(
                    f'{self.cam_keys[k]} says {name}={got} but the image is '
                    f'{name}={want}; the camera vector must describe the image '
                    f'it is paired with')

        plk = plucker_ray_map(cam[:, :3], cam[:, 3:7], cam[:, 7], h, w)
        if not self.use_plucker:
            plk = torch.zeros_like(plk)

        # one crop window, applied to the image AND its ray map
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
            plk = crop_image_from_indices(plk, inds, ch, cw)

        if self.imagenet_norm:
            # image channels only -- the ray map is geometry, not intensity
            mean = img.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = img.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
            img = (img - mean) / std
        return torch.cat([img, plk], dim=1)          # (N, 9, ch, cw)

    def _backbone_features(self, x, cond):
        """Walk the resnet explicitly so FiLM can modulate each stage.

        Torchvision's forward has no conditioning input, and hooks or
        module-level state would be action-at-a-distance; walking the submodules
        is explicit and keeps the backbone a plain parameter container.
        """
        b = self.backbone
        h = b.relu(b.bn1(b.conv1(x)))
        h = self._film(h, 0, cond)
        h = b.maxpool(h)
        for i, layer in enumerate([b.layer1, b.layer2, b.layer3, b.layer4], start=1):
            h = layer(h)
            h = self._film(h, i, cond)
        return torch.flatten(b.avgpool(h), 1)

    def _film(self, h, i, cond):
        if cond is None:
            return h                                  # exact conditioning-off
        gamma, beta = self.film[i](cond).chunk(2, dim=-1)
        return h * (1.0 + gamma[..., None, None]) + beta[..., None, None]

    # ---------------------------------------------------------------- forward
    def forward(self, obs_dict):
        """Drop-in contract, unchanged: ``(N, fused_dim + low-dim)``."""
        out = self.forward_full(obs_dict)
        return torch.cat(
            [out['z_global']] + [obs_dict[k] for k in self.low_dim_keys], dim=-1)

    def forward_full(self, obs_dict) -> Dict[str, torch.Tensor]:
        """`forward`'s computation, plus the per-view latents M4 needs.

        Returns ``{'z_global': (N, D), 'z_views': (N, K, D),
        'view_active': (N, K) bool}``. `z_views` is the very tensor the fusion
        already builds (`tokens`), so this adds no compute and no allocation --
        `forward` is this function plus the final concatenation, and the two are
        asserted bit-identical (torch.equal) in
        tests/test_aux_action_heads.py. Rows of inactive slots are exactly zero,
        which is the invariant M4's loss mask relies on.
        """
        img_keys = self.rgb_keys
        n = obs_dict[img_keys[0]].shape[0]
        train = self.training

        # (N, K, ...) views, mask and history
        mask = obs_dict[self.view_mask_key].to(device=obs_dict[img_keys[0]].device)
        if mask.dim() != 2 or mask.shape[0] != n or mask.shape[1] != self.n_slots:
            raise ValueError(
                f'{self.view_mask_key}: expected (N={n}, K={self.n_slots}), '
                f'got {tuple(mask.shape)}')
        hist = obs_dict[self.eef_hist_key]
        if hist.dim() != 3 or hist.shape[:2] != (n, self.n_slots):
            raise ValueError(
                f'{self.eef_hist_key}: expected (N={n}, K={self.n_slots}, D), '
                f'got {tuple(hist.shape)}')
        if hist.shape[-1] != self.eef_hist_dim:
            raise ValueError(
                f'{self.eef_hist_key}: expected last dim {self.eef_hist_dim} '
                f'({EEF_HIST_STEP_DIM} per step x {self.eef_hist_steps} steps), '
                f'got {hist.shape[-1]}')

        active = mask > 0.5
        n_active = active.sum(dim=1)
        if bool((n_active == 0).any()):
            raise ValueError(
                'every sample needs at least one active view (an all-False row '
                'would leave the attention query fully masked -> NaN)')

        # encode only the active slots; masked-out ones cost nothing.
        # nonzero() is row-major, so `sel` preserves that order within each slot
        # and `x[sel] = sub` scatters back into the same layout.
        rows, cols = active.nonzero(as_tuple=True)
        x = None
        for k in range(self.n_slots):
            sel = cols == k
            if not bool(sel.any()):
                continue
            sub = self._prepare_slots(
                {**obs_dict,
                 self.rgb_keys[k]: obs_dict[self.rgb_keys[k]][rows[sel]],
                 self.cam_keys[k]: obs_dict[self.cam_keys[k]][rows[sel]]},
                k, train)
            if x is None:
                x = sub.new_empty((rows.numel(),) + tuple(sub.shape[1:]))
            x[sel] = sub
        cond = None
        if self.use_eef_hist:
            hist_a = hist.to(dtype=x.dtype, device=x.device)[rows, cols]
            cond = self.cond_mlp(hist_a)
        z = self._backbone_features(x, cond)          # (M, fused_dim)

        tokens = z.new_zeros((n, self.n_slots, self.fused_dim))
        tokens[rows, cols] = z
        pad_mask = ~active                             # True == ignore
        query = self.fusion_query.expand(n, 1, self.fused_dim)
        z_g, _ = self.fusion(query, tokens, tokens,
                             key_padding_mask=pad_mask, need_weights=False)
        z_g = z_g.reshape(n, self.fused_dim)

        return {'z_global': z_g, 'z_views': tokens, 'view_active': active}

    @torch.no_grad()
    def output_shape(self):
        """Measured, like the stock encoder -- but the extra keys must be
        injected by hand (they are not in shape_meta), and the dummy camera must
        be a VALID pose: an all-zero quaternion yields a zero rotation matrix and
        NaN ray directions, with a perfectly correct output *shape*."""
        was_training = self.training
        self.eval()                                    # deterministic centre crop
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
            example[self.eef_hist_key] = torch.zeros(
                (1, self.n_slots, self.eef_hist_dim),
                dtype=self.dtype, device=self.device)
            out = self.forward(example)
            expected = self.fused_dim + sum(
                self.key_shape_map[k][0] for k in self.low_dim_keys)
            if out.shape[1:] != (expected,):
                raise RuntimeError(
                    f'encoder emits {tuple(out.shape[1:])}, expected '
                    f'({expected},) = fused_dim {self.fused_dim} + low-dim '
                    f'{expected - self.fused_dim}; the UNet conditioning width '
                    f'would differ from the baseline')
            if not bool(torch.isfinite(out).all()):
                raise RuntimeError('encoder output is not finite on dummy input')
            return out.shape[1:]
        finally:
            self.train(was_training)
