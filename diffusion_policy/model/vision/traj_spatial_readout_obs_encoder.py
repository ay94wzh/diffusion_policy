"""Stage RH/RHG: trajectory-conditioned spatial readout (ray key) encoder.

RH is stage R (``ray_spatial_readout_obs_encoder.RaySpatialReadoutObsEncoder``)
with ONE structural addition: the ONE shared 512-d spatial query is conditioned
on the FULL end-effector trajectory window of
``ENCODER_EXPERIMENT_PLAN.md`` section 4.3. RHG adds, on top of RH, R07's
point-to-ray geometric bias on the intra-image attention logits (section 4.4),
on 4 of the 8 heads. Both are one class: ``use_geom_bias=False`` is exactly RH
(the bias is never computed and adds no parameter, so RH and RHG are
parameter-identical and differ only in the flag).

Information flow (per view, shared weights)
-------------------------------------------
::

    k = W_K LN(f + e),    v = W_V LN(f),    e = E_ray([d, m/L])      (as R)
    q_t = q_learned + W_h h_t                                        (RH)
    a_j = softmax_j( q_t^T k_j / sqrt(64) + b_j )  on heads 1..4     (RHG)

with ``h_t = E_traj({(p_i, R_i)}_{i=t-3..t})``: per-step ``[p_i/L,
rot6d(R_i), (p_i - p_t)/L, age]`` -> a small temporal Transformer -> the H
outputs concatenated in temporal order -> MLP -> 512-d. One ``q_t`` per image
time, reused by every valid view; the visual readout is a plain 1x512 token --
no raw query or history vector is residual-added to it. The cross-view MHA
fusion and the 521-wide low-dim interface are byte-identical to S/R/B, so the
action UNet is unchanged.

Where the trajectory comes from (frame, key, units)
---------------------------------------------------
The window is read from ``eef_hist_base`` ``(N, H, 9)`` -- per-sample and
view-INDEPENDENT, in the ROBOT BASE FRAME: ``[pos(3), rot6d(6)]`` per step,
oldest first, current last, no gripper (section 4.1). The per-slot
camera-frame ``view_eef_hist`` is still emitted and still stacked by the
shared latent-probe workspace, but this encoder ignores it, exactly as S/R do;
its own history argument names the base key, so a copied B config cannot
silently route the camera-frame history in. The dataset registers an IDENTITY
normalizer for the key: the encoder's geometry (and RHG's point-to-ray
distances) must see raw metres, never fitted-normalized coordinates.

``p_t`` -- the query's relative-position reference and RHG's bias anchor -- is
the LAST step of the window: the window ends at the observation frame, exactly
as ``MultiViewImageDataset._eef_history`` and both rollout deques
(``CamKeyImageWrapper``, ``ViewpointImageWrapper``) construct it.

Geometric bias (RHG only)
-------------------------
For active view ``v`` and spatial token ``j`` with Pluecker ray ``(d, m)``
(unit direction, world-frame moment in metres)::

    rho_{v,j} = || p_t x d_{v,j} - m_{v,j} ||          (point-to-line distance)
    b_{v,j}   = -alpha * min( rho^2 / (2 sigma^2), b_max )

``rho`` is computed in metres (the stored moment is divided by ``ray_scale`` L
for the KEY; it is multiplied back here). The bias is added to the attention
logits of heads 1..``geom_bias_heads`` only, then softmax -- heads 5..8 read
content with no geometry. ``sigma`` is in metres and is calibrated on the
training set pre-launch (~1 feature-grid neighbourhood) and LOCKED; the
constructor requires it whenever ``use_geom_bias`` is on, so a run cannot
silently use an uncalibrated value. Per the plan the bias is DISABLED for a
view whose current EE point is behind the camera or projects outside the
sampled crop (content attention is kept): ``b = 0`` for every token of that
view.

Initialisation order (important for reading early epochs)
---------------------------------------------------------
``spatial_query`` and ``query_proj`` (``W_h``, weight AND bias) are
zero-initialised, so at initialisation ``q_t = 0``: RH is bit-identical to a
zero-query R/S readout and the whole trajectory branch is inert. ``h_t`` is
NOT zero at init (the trajectory MLP is standard-initialised), so
``query_proj``'s gradient at the first step is non-zero and the branch is live
from the second optimizer step; the trajectory encoder itself follows one step
later (the usual zero-init-head chain). RHG's bias needs no training to act:
it is added to the logits even while ``q_t = 0``, so at init the attention
weights are ``softmax(b)`` over the tokens, not uniform.

What RH/RHG deliberately do not touch
-------------------------------------
No input-level Pluecker channels, no FiLM, no 2D positional encoding, no
depth, no per-view auxiliary heads. No residual of the query/history into the
visual readout. Only ONE query is added: the query COUNT stays 1 (pinned by
the CPU test), and ``q_learned`` remains a separate parameter from the
cross-view fusion query, as in S/R.

Shapes
------
Per the policy's flattening the leading axis of ``obs_dict`` here is ``B*To``:
each element is one frame with ``K`` view slots and one ``(H, 9)`` trajectory
window. ``view_mask`` is ``(N, K)`` (1.0 == live). Inactive slots are never
encoded, their ``z_views`` rows are exactly zero, and the fusion ignores them.
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
from diffusion_policy.model.vision.plucker import (
    plucker_ray_map_at, quat_wxyz_to_mat_torch)
from diffusion_policy.common.pytorch_util import replace_submodules

# [d(3) | m(3)] -- unit ray direction and world-frame Pluecker moment
RAY_FEAT_DIM = 6
# [pos(3) | rot6d(6)] per trajectory step, base frame (dataset's EEF_HIST_BASE)
TRAJ_STEP_DIM = 9
# per-step transformer input: [p/L(3) | rot6d(6) | (p - p_t)/L(3) | age(1)]
TRAJ_STEP_FEAT_DIM = 3 + 6 + 3 + 1


def point_to_ray_distances(p: torch.Tensor, rays: torch.Tensor,
                           ray_scale: float = 1.0) -> torch.Tensor:
    """Distance from a 3-D point to each Pluecker ray, in metres.

    ``p`` ``(M, 3)`` (robot base frame, metres); ``rays`` ``(M, G, 6)`` in the
    encoder's stored convention ``[d, m / L]`` (unit direction, moment divided
    by ``ray_scale`` L). Returns ``(M, G)``::

        || p x d - m || = || (p - o) x d ||

    i.e. the perpendicular distance from ``p`` to the line ``(o, d)`` (a
    Pluecker moment is invariant to sliding the origin along the ray, which is
    what makes the pair a line rather than a point). This is the plan's
    section 4.4 quantity ``rho_{v,j}``; the test pins it against a
    hand-computed offset. Shared by the encoder and ``calibrate_geom_bias.py``
    so the calibrated sigma and the bias cannot drift apart.
    """
    if p.dim() != 2 or p.shape[-1] != 3:
        raise ValueError(f'p must be (M, 3), got {tuple(p.shape)}')
    if rays.dim() != 3 or rays.shape[-1] != RAY_FEAT_DIM:
        raise ValueError(
            f'rays must be (M, G, 6), got {tuple(rays.shape)}')
    if p.shape[0] != rays.shape[0]:
        raise ValueError(
            f'p has {p.shape[0]} rows but rays have {rays.shape[0]}')
    d = rays[..., :3]
    m = rays[..., 3:] * float(ray_scale)          # back to world metres
    cross = torch.cross(p[:, None, :].expand_as(d), d, dim=-1)
    return (cross - m).pow(2).sum(dim=-1).sqrt()


def project_point_to_pixel(cam: torch.Tensor, p: torch.Tensor,
                           h: int, w: int
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a world point into the ORIGINAL image; ``(u, v, depth)``.

    ``cam`` ``(M, 10)`` = ``[pos(3), quat_wxyz(4), fovy_deg, h, w]``; ``p``
    ``(M, 3)``. Same pinhole convention as
    ``multiview_image_dataset.project_world_to_pixel`` (mujoco: -z forward,
    +x right, +y up; v is measured from the top of the stored, already-flipped
    image). ``depth`` is positive in front of the camera. Where ``depth <= 0``
    the u/v values are finite garbage and must be masked by the caller -- the
    denominator is clamped, never left to produce inf/NaN.

    RHG's validity gate uses this on the CURRENT EE position to decide whether
    the view's distance bias may act at all (section 4.4).
    """
    if cam.dim() != 2 or cam.shape[-1] != 10:
        raise ValueError(f'cam must be (M, 10), got {tuple(cam.shape)}')
    if p.shape != (cam.shape[0], 3):
        raise ValueError(
            f'p must be ({cam.shape[0]}, 3), got {tuple(p.shape)}')
    R = quat_wxyz_to_mat_torch(cam[:, 3:7])                  # camera -> world
    p_cam = torch.einsum('mij,mj->mi', R.transpose(1, 2), p - cam[:, :3])
    depth = -p_cam[:, 2]
    f = (h / 2.0) / torch.tan(torch.deg2rad(cam[:, 7]) / 2.0)
    denom = depth.clamp_min(1e-6)
    u = w / 2.0 + f * p_cam[:, 0] / denom
    v = h / 2.0 - f * p_cam[:, 1] / denom
    return u, v, depth


class TrajSpatialReadoutObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: nn.Module,
            cam_keys: Optional[Sequence[str]] = None,
            view_mask_key: str = 'view_mask',
            eef_hist_key: str = 'eef_hist_base',
            eef_hist_steps: int = 4,
            fused_dim: int = 512,
            n_heads: int = 8,
            ray_hidden_dim: int = 512,
            ray_scale: float = 1.0,
            use_ray_encoding: bool = True,
            traj_hidden_dim: int = 128,
            traj_layers: int = 1,
            traj_heads: int = 4,
            traj_ff_dim: int = 256,
            traj_mlp_hidden: int = 512,
            use_geom_bias: bool = False,
            geom_bias_heads: int = 4,
            geom_bias_alpha: float = 1.0,
            geom_bias_sigma_m: Optional[float] = None,
            geom_bias_max: float = 4.0,
            crop_shape: Optional[Tuple[int, int]] = None,
            random_crop: bool = True,
            use_group_norm: bool = False,
            imagenet_norm: bool = False,
            share_rgb_model: bool = True,
            ):
        """RH (``use_geom_bias=False``) / RHG (``use_geom_bias=True``).

        ``ray_scale`` is the plan's L (metres), used for the ray moment AND for
        the trajectory positions ("L=1 m, numerical normalisation only").
        ``use_ray_encoding=False`` is R's diagnostic "rays severed" ablation
        (parameters unchanged, ``e == 0``). ``geom_bias_sigma_m`` must be set
        whenever ``use_geom_bias`` is on -- it is the pre-launch calibrated,
        locked value, not a learned parameter; passing ``None`` raises rather
        than silently running uncalibrated.

        There is deliberately no ``cond_dim`` / ``use_plucker`` /
        ``use_eef_hist`` / ``resize_shape``: the trajectory is ALWAYS the base
        window named by ``eef_hist_key``.
        """
        super().__init__()
        if not share_rgb_model:
            raise NotImplementedError(
                'RH/RHG have exactly one shared backbone; '
                'share_rgb_model=False would silently build one resnet per '
                'view, which is not the intended architecture (the stock '
                'default is False, so set share_rgb_model: True in the config).')
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

        # Extra keys must never be bucketed as low-dim, or they would be
        # concatenated into the encoder output and change the 521 width.
        extra_keys = (view_mask_key, eef_hist_key, 'view_eef_hist')
        for key in list(rgb_keys) + list(low_dim_keys):
            if key.endswith('_cam'):
                raise ValueError(
                    f'{key!r} looks like a camera key and must NOT be in '
                    'shape_meta -- the env cannot serve it as a normalizer '
                    'key. Pass it via the obs dict instead.')
        for key in extra_keys:
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
                f'RH/RHG read pure RGB; every view slot must have 3 channels, '
                f'got {slot_shape}')

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

        eef_hist_steps = int(eef_hist_steps)
        if eef_hist_steps < 1:
            raise ValueError(
                f'eef_hist_steps must be >= 1, got {eef_hist_steps}')
        traj_hidden_dim = int(traj_hidden_dim)
        traj_layers = int(traj_layers)
        traj_heads = int(traj_heads)
        traj_ff_dim = int(traj_ff_dim)
        traj_mlp_hidden = int(traj_mlp_hidden)
        if traj_hidden_dim % traj_heads != 0:
            raise ValueError(
                f'traj_hidden_dim {traj_hidden_dim} must be divisible by '
                f'traj_heads {traj_heads}')
        if min(traj_layers, traj_ff_dim, traj_mlp_hidden) <= 0:
            raise ValueError('traj_layers / traj_ff_dim / traj_mlp_hidden '
                             'must be positive')

        use_geom_bias = bool(use_geom_bias)
        geom_bias_heads = int(geom_bias_heads)
        geom_bias_alpha = float(geom_bias_alpha)
        geom_bias_max = float(geom_bias_max)
        if not (0 <= geom_bias_heads <= n_heads):
            raise ValueError(
                f'geom_bias_heads {geom_bias_heads} must be in [0, {n_heads}]')
        if geom_bias_alpha < 0.0 or geom_bias_max <= 0.0:
            raise ValueError(
                'geom_bias_alpha must be >= 0 and geom_bias_max > 0, got '
                f'({geom_bias_alpha}, {geom_bias_max})')
        if use_geom_bias:
            if geom_bias_sigma_m is None:
                raise ValueError(
                    'use_geom_bias=True requires geom_bias_sigma_m, the '
                    'pre-launch calibrated value in metres (see '
                    'calibrate_geom_bias.py). It is deliberately NOT a '
                    'default: an uncalibrated sigma changes every bias term '
                    'by an unknown factor.')
            geom_bias_sigma_m = float(geom_bias_sigma_m)
            if not (geom_bias_sigma_m > 0.0):
                raise ValueError(
                    f'geom_bias_sigma_m must be positive, got {geom_bias_sigma_m}')
        else:
            geom_bias_sigma_m = None if geom_bias_sigma_m is None else float(
                geom_bias_sigma_m)

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
        self.eef_hist_key = str(eef_hist_key)
        self.eef_hist_steps = eef_hist_steps
        self.fused_dim = fused_dim
        self.n_heads = n_heads
        self.ray_hidden_dim = ray_hidden_dim
        self.ray_scale = ray_scale
        self.use_ray_encoding = bool(use_ray_encoding)
        self.traj_hidden_dim = traj_hidden_dim
        self.traj_layers = traj_layers
        self.traj_heads = traj_heads
        self.traj_ff_dim = traj_ff_dim
        self.traj_mlp_hidden = traj_mlp_hidden
        self.use_geom_bias = use_geom_bias
        self.geom_bias_heads = geom_bias_heads
        self.geom_bias_alpha = geom_bias_alpha
        self.geom_bias_sigma_m = geom_bias_sigma_m
        self.geom_bias_max = geom_bias_max
        self.crop_shape = crop_shape
        self.random_crop = bool(random_crop)
        self.imagenet_norm = bool(imagenet_norm)
        self.n_slots = len(rgb_keys)

        # --- shared pure-RGB backbone, truncated at layer3 (as S/R) ---------
        backbone = copy.deepcopy(rgb_model)
        if not hasattr(backbone, 'layer3'):
            raise TypeError(
                'rgb_model has no layer3; RH expects a ResNet-style backbone')
        if not isinstance(backbone.conv1, nn.Conv2d):
            raise TypeError('rgb_model has no nn.Conv2d conv1')
        if backbone.conv1.in_channels != 3:
            raise ValueError(
                f'RH reads pure RGB: conv1 has {backbone.conv1.in_channels} '
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

        # --- feature-level ray encoding (R, section 4.2) ---------------------
        self.ray_mlp = nn.Sequential(
            nn.Linear(RAY_FEAT_DIM, self.ray_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.ray_hidden_dim, self.fused_dim),
        )
        nn.init.zeros_(self.ray_mlp[-1].weight)
        nn.init.zeros_(self.ray_mlp[-1].bias)
        self.key_norm = nn.LayerNorm(self.fused_dim)
        self.value_norm = nn.LayerNorm(self.fused_dim)

        # --- trajectory condition (RH, section 4.3) --------------------------
        self.traj_in = nn.Linear(TRAJ_STEP_FEAT_DIM, traj_hidden_dim)
        # learned step-order embedding (the window is small and ordered)
        self.traj_pos = nn.Parameter(torch.zeros(1, eef_hist_steps, traj_hidden_dim))
        nn.init.normal_(self.traj_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=traj_hidden_dim,
            nhead=traj_heads,
            dim_feedforward=traj_ff_dim,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True)
        self.traj_encoder = nn.TransformerEncoder(
            layer, num_layers=traj_layers, enable_nested_tensor=False)
        self.traj_out = nn.Sequential(
            nn.Linear(eef_hist_steps * traj_hidden_dim, traj_mlp_hidden),
            nn.SiLU(),
            nn.Linear(traj_mlp_hidden, self.fused_dim),
        )
        # W_h: zero-init weight AND bias, so q_t = q_learned exactly at
        # initialisation. h_t is NOT zero at init (traj_out is standard-init),
        # so this layer's gradient at the first step is non-zero -- the branch
        # activates on step 2, and the trajectory encoder on step 3 (the usual
        # zero-init-head chain). Zero-initialising BOTH sides would deadlock
        # the branch (dq/dW_h = h = 0 and dh/dtheta = W_h^T dL/dq = 0).
        self.query_proj = nn.Linear(self.fused_dim, self.fused_dim)
        nn.init.zeros_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

        self.intra_attn = nn.MultiheadAttention(
            embed_dim=self.fused_dim, num_heads=self.n_heads,
            batch_first=True)
        self.spatial_query = nn.Parameter(
            torch.zeros(1, 1, self.fused_dim))
        # Identical to S/R's cross-view fusion: a separate learnable query.
        self.fusion = nn.MultiheadAttention(
            embed_dim=self.fused_dim, num_heads=self.n_heads,
            batch_first=True)
        self.fusion_query = nn.Parameter(
            torch.zeros(1, 1, self.fused_dim))

        # --- shape probe: feature grid, token count, ray-grid stride ---------
        # Eval mode (BatchNorm stats untouched), no RNG consumed.
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
                'be wrong. RH supports standard torchvision ResNet stems.')

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _stage_stride(stage: nn.Module) -> int:
        """Largest downsample factor inside one stage (max, not product)."""
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

        Identical to R's: cell ``(i, j)`` sits at original pixel
        ``(off_h + S*i, off_w + S*j)`` (pixel centres), row-major
        ``i * gw + j``, and the moment is divided by ``ray_scale``.
        """
        n = cam.shape[0]
        gh, gw = self.token_grid
        stride = self.total_stride
        jj = torch.arange(gw, dtype=cam.dtype, device=cam.device) * stride
        ii = torch.arange(gh, dtype=cam.dtype, device=cam.device) * stride
        off = inds.to(dtype=cam.dtype, device=cam.device)
        u = off[:, 1:2] + jj[None, :] + 0.5            # (N, gw)
        v = off[:, 0:1] + ii[None, :] + 0.5            # (N, gh)
        u = u[:, None, :].expand(n, gh, gw).reshape(n, gh * gw)
        v = v[:, :, None].expand(n, gh, gw).reshape(n, gh * gw)
        rays = plucker_ray_map_at(
            cam[:, :3], cam[:, 3:7], cam[:, 7], h, w, u, v)  # (N, 6, G)
        rays = rays.transpose(1, 2).contiguous()             # (N, G, 6)
        rays[..., 3:] = rays[..., 3:] / self.ray_scale
        return rays

    def _geom_bias(self, cam: torch.Tensor, inds: torch.Tensor,
                   p: torch.Tensor, rays: torch.Tensor, h: int, w: int,
                   crop_hw: Tuple[int, int]) -> torch.Tensor:
        """RHG's additive attention bias ``(M, G)`` for one batch of views.

        ``p`` ``(M, 3)`` is the CURRENT EE point (base frame, metres); ``cam``
        and ``inds`` are the view's camera vector and crop offsets; ``rays``
        the view's ``[d, m/L]`` grid. Zero for a view whose EE point is behind
        the camera or projects outside the crop window (the plan's gate).
        """
        rho = point_to_ray_distances(p, rays, self.ray_scale)     # (M, G) m
        sigma = self.geom_bias_sigma_m
        # b = -alpha * min(rho^2 / (2 sigma^2), b_max)
        b = -self.geom_bias_alpha * torch.clamp(
            rho.pow(2) / (2.0 * sigma * sigma), max=self.geom_bias_max)
        u, v, depth = project_point_to_pixel(
            cam.to(dtype=b.dtype, device=b.device), p, h, w)
        off = inds.to(dtype=b.dtype, device=b.device)
        ch, cw = crop_hw
        valid = ((depth > 1e-6)
                 & (u >= off[:, 1]) & (u < off[:, 1] + cw)
                 & (v >= off[:, 0]) & (v < off[:, 0] + ch))
        return b * valid.to(dtype=b.dtype)[:, None]

    # ------------------------------------------------------------------ views
    def _prepare_slot(self, img: torch.Tensor, cam: torch.Tensor,
                      train: bool
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Crop one slot's images; rays and crop offsets use the SAME window.

        Returns ``(img, rays, inds)``; ``inds`` ``(N, 2)`` (top-left, original
        pixels) is needed by RHG's projection-validity gate.
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
        return img, rays, inds

    def _backbone_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Walk the resnet explicitly up to layer3; no pool, no flatten."""
        b = self.backbone
        h = b.relu(b.bn1(b.conv1(x)))
        h = b.maxpool(h)
        h = b.layer1(h)
        h = b.layer2(h)
        h = b.layer3(h)
        return h

    # ------------------------------------------------------------ trajectory
    def _traj_step_features(self, hist: torch.Tensor) -> torch.Tensor:
        """``(M, H, 9)`` base window -> ``(M, H, 13)`` per-step features.

        ``[p_i/L (3) | rot6d_i (6) | (p_i - p_t)/L (3) | age (1)]`` with
        ``age`` linear from -1 (oldest) to 0 (current) and ``p_t`` the
        window's LAST step -- the observation frame's own EE position. Split
        out so the test can pin the content and order without reading weights.
        """
        if hist.dim() != 3 or hist.shape[1:] != (
                self.eef_hist_steps, TRAJ_STEP_DIM):
            raise ValueError(
                f'{self.eef_hist_key}: expected (M, {self.eef_hist_steps}, '
                f'{TRAJ_STEP_DIM}), got {tuple(hist.shape)}')
        if not bool(torch.isfinite(hist).all()):
            raise ValueError(f'{self.eef_hist_key} contains non-finite values')
        m = hist.shape[0]
        L = self.ray_scale
        p = hist[..., :3] / L
        rot6 = hist[..., 3:]
        p_t = hist[:, -1, :3] / L
        rel = p - p_t[:, None, :]
        age = torch.linspace(
            -1.0, 0.0, self.eef_hist_steps,
            dtype=hist.dtype, device=hist.device).expand(m, -1)[..., None]
        return torch.cat([p, rot6, rel, age], dim=-1)          # (M, H, 13)

    def _traj_vector(self, hist: torch.Tensor) -> torch.Tensor:
        """``(M, H, 9)`` base-frame window -> ``(M, fused_dim)`` trajectory h_t.

        Per-step features -> a learned step-order embedding -> the small
        temporal Transformer -> the H outputs concatenated in temporal order
        -> MLP. See `_traj_step_features` for the feature contract.
        """
        feat = self._traj_step_features(hist)
        x = self.traj_in(feat) + self.traj_pos
        x = self.traj_encoder(x)                                # (M, H, d)
        return self.traj_out(x.reshape(hist.shape[0], -1))      # (M, D)

    def _query(self, hist_rows: torch.Tensor) -> torch.Tensor:
        """``(M, 1, D)`` shared query: q_learned + W_h h_t."""
        h = self._traj_vector(hist_rows)
        q = self.spatial_query.expand(hist_rows.shape[0], -1, -1)
        return q + self.query_proj(h)[:, None, :]

    def _encode_views(self, x: torch.Tensor, rays: torch.Tensor,
                      q: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(M, 3, ch, cw) + (M, G, 6) + (M, 1, D) -> (M, fused_dim).

        ``bias`` ``(M, G)`` (RHG) is added to the attention logits of the
        first ``geom_bias_heads`` heads only; ``None``/zero is RH.
        """
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

        attn_mask = None
        if bias is not None and self.geom_bias_heads > 0:
            n, g = tokens.shape[0], tokens.shape[1]
            # (N*H, 1, G): the MHA math path views a 3-D mask as
            # (batch, head, L, S), so head h < geom_bias_heads gets b and the
            # rest get 0 -- the plan's "4/8 heads".
            mask = bias.new_zeros((n, self.n_heads, 1, g))
            mask[:, :self.geom_bias_heads] = bias[:, None, None, :]
            attn_mask = mask.reshape(n * self.n_heads, 1, g)
        read, _ = self.intra_attn(q, keys, values,
                                  attn_mask=attn_mask, need_weights=False)
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
        zero. The intra-image attention never mixes views; the only cross-view
        path is the same masked MHA fusion as S/R/B.
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
        if self.eef_hist_key not in obs_dict:
            raise KeyError(
                f'missing {self.eef_hist_key!r}: RH/RHG need the per-sample '
                'base-frame trajectory window (set emit_eef_hist_base: True '
                'on the dataset and serve_eef_hist_base: True on the runner)')
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
        hist = obs_dict[self.eef_hist_key]
        if hist.dim() != 3 or tuple(hist.shape[1:]) != (
                self.eef_hist_steps, TRAJ_STEP_DIM):
            raise ValueError(
                f'{self.eef_hist_key}: expected (N={n}, {self.eef_hist_steps}, '
                f'{TRAJ_STEP_DIM}), got {tuple(hist.shape)}')
        if hist.shape[0] != n:
            raise ValueError(
                f'{self.eef_hist_key}: batch size {hist.shape[0]} != {n}')

        # Encode only the active slots; masked-out ones cost nothing and are
        # never turned into rays (their camera vector is all zeros, which is
        # not a valid pose). nonzero() is row-major, so `sel` preserves that
        # order within each slot and `x[sel] = sub` scatters back exactly.
        rows, cols = active.nonzero(as_tuple=True)
        x = None
        rays = None
        cams = None
        inds_all = None
        for k in range(self.n_slots):
            sel = cols == k
            if not bool(sel.any()):
                continue
            sub_img, sub_rays, sub_inds = self._prepare_slot(
                obs_dict[self.rgb_keys[k]][rows[sel]],
                obs_dict[self.cam_keys[k]][rows[sel]],
                self.training)
            if x is None:
                x = sub_img.new_empty((rows.numel(),) + tuple(sub_img.shape[1:]))
                rays = sub_rays.new_empty(
                    (rows.numel(),) + tuple(sub_rays.shape[1:]))
                cams = sub_img.new_empty((rows.numel(), 10))
                inds_all = sub_inds.new_empty((rows.numel(), 2))
            x[sel] = sub_img
            rays[sel] = sub_rays
            cams[sel] = obs_dict[self.cam_keys[k]][rows[sel]].to(
                dtype=sub_img.dtype, device=sub_img.device)
            inds_all[sel] = sub_inds

        # q_t: one trajectory-conditioned query per frame, shared by ALL views
        hist_rows = hist.to(dtype=x.dtype, device=x.device)[rows]
        q = self._query(hist_rows)                              # (M, 1, D)

        bias = None
        if self.use_geom_bias:
            # p_t = the current window's last step (the observation frame)
            p_t = hist_rows[:, -1, :3]
            h_orig, w_orig = int(img0.shape[2]), int(img0.shape[3])
            bias = self._geom_bias(
                cams, inds_all, p_t, rays, h_orig, w_orig,
                self.crop_shape if self.crop_shape is not None
                else (self.slot_shape[1], self.slot_shape[2]))
        readouts = self._encode_views(x, rays, q, bias=bias)    # (M, D)

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
        """Measured, like the stock encoder. Dummy cameras must be VALID
        poses, and the dummy trajectory must be a finite (H, 9) window: RHG
        turns them into point-to-ray distances and a projection."""
        was_training = self.training
        self.eval()
        try:
            example = dict()
            for key, attr in self.shape_meta['obs'].items():
                example[key] = torch.zeros(
                    (1,) + tuple(attr['shape']),
                    dtype=self.dtype, device=self.device)
            for k, cam_key in enumerate(self.cam_keys):
                hw = self.key_shape_map[self.rgb_keys[k]]
                cam = torch.zeros(10, dtype=self.dtype, device=self.device)
                cam[0], cam[2] = 0.5, 1.2         # position (in front of the scene)
                cam[3] = 1.0                     # wxyz w = 1 -> identity rotation
                cam[7] = 45.0                    # fovy in degrees
                cam[8] = hw[1]
                cam[9] = hw[2]
                example[cam_key] = cam.unsqueeze(0)
            example[self.view_mask_key] = torch.ones(
                (1, self.n_slots), dtype=self.dtype, device=self.device)
            hist = torch.zeros(
                (1, self.eef_hist_steps, TRAJ_STEP_DIM),
                dtype=self.dtype, device=self.device)
            hist[..., :3] = torch.tensor(
                [0.5, 0.0, 0.6], dtype=self.dtype, device=self.device)
            hist[..., 3], hist[..., 7] = 1.0, 1.0   # rot6d of the identity
            example[self.eef_hist_key] = hist
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
