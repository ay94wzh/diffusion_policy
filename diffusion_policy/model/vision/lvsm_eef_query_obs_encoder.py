"""Stage LQ: LVSM-style RGB+Pluecker patchify with EEF-history query tokens.

This is the first encoder in the ladder that does NOT use a CNN backbone and
does NOT read out per view. One bidirectional transformer jointly processes

    [all valid views' spatial visual tokens X, EEF-history query tokens Q]

and only the ``Q``-position outputs are kept. They are projected 256 -> 128
each, flattened oldest -> current, and concatenated with the current 9-d robot
state to form the same 521-wide per-frame condition as B/S/R/RH (two frames ->
1042 -> the unchanged ``ConditionalUnet1D``). See
``ENCODER_LVSM_EEF_QUERY_V1.md`` (v0.2) for the design.

Tokenizer (user decision, 2026-10-29)
------------------------------------
LVSM's original tokenizer, not the design draft's ResNet-layer3 default:
each valid view's 84x84 RGB is cropped to 72x72, a per-pixel Pluecker map
``[d(3) | m/L(3)]`` is computed for the SAME crop window (original intrinsics),
and the 9 channels ``[RGB | d | m/L]`` are patchified by one shared
``Conv2d(9, D, kernel_size=8, stride=8)``. 72/8 = 9, so G = 81 tokens/view.
76 is not divisible by 8, hence the 84 -> 72 crop (the B/S/R/RH ladder uses
76; this is a structural comparison with a crop confound, and a same-crop
control is future work).

Information flow (per observation frame, shared weights)
--------------------------------------------------------
::

    X_v = LN_x(patch_embed(cat[RGB_crop, Pluecker_crop]) + e_visual)  (81, D)
    Q_i = LN_q(E_h([p_i/L, rot6d_i, (p_i - p_t)/L, age_i]) + e_i^time + e_eef)
    S   = [X_1 ... X_K, Q_0 ... Q_{H-1}]          (K*G + H, D)
    Z   = transformer_layers(S)[:, K*G:]          only the H query positions
    c   = flatten(out_proj(LN_o(Z)))              (H*128 = 512)
    out = [c, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos]  (521)

There is no slot-ID embedding and no global-index embedding: geometry enters
only through the Pluecker channels, and permuting the view slots must permute
the visual block without changing the query outputs (the CPU test pins this).

Masks
-----
Dead view slots are encoded as finite zeros (never ``new_empty``: a NaN in a
zero-weight masked key poisons the softmax through 0 * NaN) and are marked in
``src_key_padding_mask`` after expanding each slot's flag G times. The H query
rows are never padded, so no attention row is fully masked. Invalid slots are
never cropped and never turned into rays (their all-zero camera is not a valid
pose).

Controls (design doc section 8)
-------------------------------
``history_query=False`` replaces the 13-d query features by zeros before the
SAME ``query_mlp`` (parameters and construction RNG unchanged) - the learned /
constant-query control. ``visual_reads_history=False`` adds a bool ``(S, S)``
``src_mask`` blocking every visual row from every query column (all other
blocks stay open); query rows still read the history-derived query keys
directly, so the control isolates the history -> visual -> query feedback path,
not the direct history -> query path.

What LQ deliberately does not do
--------------------------------
No ResNet, no ray MLP, no feature-grid-centre approximation, no per-view
readout, no cross-view MHA, no slot ID, no zero-init chain (all new branches
use standard initialisation, unlike S/R/RH). The four transformer layers are
constructed independently (torch 1.12's ``TransformerEncoder`` deep-copies one
layer, so its clones start bit-identical and cannot be re-initialised).

Frames
------
Both the camera poses and ``eef_hist_base`` come from the same stored robomimic
world frame (the dataset calls the stored pose "base"; it applies no transform,
and neither does this encoder). Rays use world-frame Pluecker moments, as in
``plucker.py``; applying a base correction to only one side would break the
correspondence.

Shapes
------
The policy flattens ``(B, To, ...)`` to ``B*To`` before calling the encoder, so
the leading axis here is ``B*To``. ``view_mask`` is ``(N, K)`` (1.0 == live).
``forward`` returns ``(N, 521)``; ``forward_full`` additionally exposes the
probe/training readouts.
"""
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.crop_randomizer import (
    sample_random_image_crops, crop_image_from_indices)
from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    IMAGENET_MEAN, IMAGENET_STD)
from diffusion_policy.model.vision.plucker import plucker_ray_map_at

# per history step: [pos(3) | rot6d(6)] from the dataset's EEF_HIST_BASE
EEF_STEP_DIM = 9
# query input: [p/L(3) | rot6d(6) | (p - p_t)/L(3) | age(1)]
QUERY_FEAT_DIM = 13
# patch input: [RGB(3) | d(3) | m/L(3)]
PATCH_IN_CHANNELS = 9
# the UNet's per-frame condition minus the 9-d state; pinned by the design
CONDITION_WIDTH = 512


class LvsmEefQueryObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            cam_keys: Optional[Sequence[str]] = None,
            view_mask_key: str = 'view_mask',
            eef_hist_key: str = 'eef_hist_base',
            eef_hist_steps: int = 4,
            d_model: int = 256,
            n_heads: int = 8,
            n_layers: int = 4,
            ffn_dim: int = 1024,
            dropout: float = 0.0,
            out_token_dim: int = 128,
            patch_size: int = 8,
            crop_shape: Optional[Tuple[int, int]] = (72, 72),
            random_crop: bool = True,
            imagenet_norm: bool = False,
            ray_scale: float = 1.0,
            history_query: bool = True,
            visual_reads_history: bool = True,
            ):
        """LVSM-tokenised joint visual/history transformer encoder.

        ``patch_size`` and ``crop_shape`` must be compatible: both crop sides
        divisible by the patch size (72/8 = 9 -> 81 tokens/view). ``ray_scale``
        is the design's L0 in metres: it divides the Pluecker moment AND the
        query positions, so ``p/L`` and ``m/L`` share one unit. ``history_query``
        and ``visual_reads_history`` are the two registered controls; both keep
        the parameter set and construction RNG identical.
        """
        super().__init__()

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
                f'all view slots must share a shape (one shared patch embed '
                f'runs on a stacked batch); got {rgb_shapes}')
        slot_shape = tuple(rgb_shapes[0])
        if slot_shape[0] != 3:
            raise ValueError(
                f'LQ reads pure RGB; every view slot must have 3 channels, '
                f'got {slot_shape}')

        d_model = int(d_model)
        n_heads = int(n_heads)
        n_layers = int(n_layers)
        ffn_dim = int(ffn_dim)
        dropout = float(dropout)
        out_token_dim = int(out_token_dim)
        patch_size = int(patch_size)
        eef_hist_steps = int(eef_hist_steps)
        if d_model <= 0 or n_heads <= 0 or n_layers <= 0 or ffn_dim <= 0:
            raise ValueError(
                'd_model / n_heads / n_layers / ffn_dim must be positive')
        if d_model % n_heads != 0:
            raise ValueError(
                f'd_model {d_model} must be divisible by n_heads {n_heads}')
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f'dropout must be in [0, 1), got {dropout}')
        if out_token_dim <= 0:
            raise ValueError(
                f'out_token_dim must be positive, got {out_token_dim}')
        if eef_hist_steps < 1:
            raise ValueError(
                f'eef_hist_steps must be >= 1, got {eef_hist_steps}')
        if eef_hist_steps * out_token_dim != CONDITION_WIDTH:
            raise ValueError(
                f'eef_hist_steps * out_token_dim must equal '
                f'{CONDITION_WIDTH} (the UNet condition width before the 9-d '
                f'state); got {eef_hist_steps} * {out_token_dim} = '
                f'{eef_hist_steps * out_token_dim}. Changing it would silently '
                'rebuild global_cond_dim.')
        if patch_size <= 0:
            raise ValueError(f'patch_size must be positive, got {patch_size}')
        ray_scale = float(ray_scale)
        if not (ray_scale > 0.0) or ray_scale != ray_scale:
            raise ValueError(f'ray_scale (L0) must be positive, got {ray_scale}')

        if crop_shape is None:
            crop_hw = (int(slot_shape[1]), int(slot_shape[2]))
        else:
            crop_hw = tuple(int(v) for v in crop_shape)
            if len(crop_hw) != 2:
                raise ValueError(f'crop_shape must be 2-D, got {crop_shape}')
            ch, cw = crop_hw
            if not (0 < ch <= slot_shape[1] and 0 < cw <= slot_shape[2]):
                raise ValueError(
                    f'crop_shape {crop_hw} must be inside the '
                    f'{slot_shape[1]}x{slot_shape[2]} image')
        ch, cw = crop_hw
        if ch % patch_size != 0 or cw % patch_size != 0:
            raise ValueError(
                f'crop_shape {crop_hw} must be divisible by patch_size '
                f'{patch_size} on both sides; use e.g. [72, 72] for p=8 '
                '(76 is not divisible by 8)')
        gh, gw = ch // patch_size, cw // patch_size

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.cam_keys = cam_keys
        self.key_shape_map = key_shape_map
        self.slot_shape = slot_shape
        self.view_mask_key = str(view_mask_key)
        self.eef_hist_key = str(eef_hist_key)
        self.eef_hist_steps = eef_hist_steps
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.out_token_dim = out_token_dim
        self.patch_size = patch_size
        self.crop_shape = crop_hw
        self.random_crop = bool(random_crop)
        self.imagenet_norm = bool(imagenet_norm)
        self.ray_scale = ray_scale
        self.history_query = bool(history_query)
        self.visual_reads_history = bool(visual_reads_history)
        self.n_slots = len(rgb_keys)
        self.token_grid = (gh, gw)
        self.n_tokens = int(gh * gw)
        self.seq_visual = int(self.n_slots * self.n_tokens)
        self.seq_len = int(self.seq_visual + eef_hist_steps)

        # --- LVSM tokenizer: [RGB(3) | d(3) | m/L(3)] -> patches ------------
        self.patch_embed = nn.Conv2d(
            PATCH_IN_CHANNELS, d_model,
            kernel_size=patch_size, stride=patch_size)

        # --- token-type / history-step embeddings ---------------------------
        self.type_emb = nn.Embedding(2, d_model)   # 0 = visual, 1 = EEF query
        self.step_emb = nn.Embedding(eef_hist_steps, d_model)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        nn.init.normal_(self.step_emb.weight, std=0.02)

        # --- EEF history -> query tokens (design section 4.3) ---------------
        self.query_mlp = nn.Sequential(
            nn.Linear(QUERY_FEAT_DIM, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.ln_x = nn.LayerNorm(d_model)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)

        # --- joint bidirectional transformer (design section 4.4) ----------
        # Built as independent layers: nn.TransformerEncoder's _get_clones
        # deep-copies ONE layer, so its clones begin bit-identical and torch
        # 1.12 offers no _reset_parameters to break the symmetry. The explicit
        # loop below consumes RNG per layer and is version-stable.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True)
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Linear(d_model, out_token_dim)

    # ------------------------------------------------------------- geometry
    def _crop_plucker_map(self, cam: torch.Tensor, inds: torch.Tensor,
                          h: int, w: int) -> torch.Tensor:
        """``(M, 6, ch, cw)`` Pluecker map at the crop's pixel centres.

        ``cam`` ``(M, 10)`` = ``[pos(3), quat_wxyz(4), fovy_deg, h, w]``;
        ``inds`` ``(M, 2)`` = ``(top, left)`` crop offsets in the ORIGINAL
        image. The map is computed at every pixel of the crop with the
        ORIGINAL intrinsics, exactly where the RGB crop samples; row-major
        ``i * cw + j`` so it lines up with the patch-embed output. The moment
        is divided by ``ray_scale`` (L0), matching the query's ``p/L``.
        """
        ch, cw = self.crop_shape
        m = cam.shape[0]
        jj = torch.arange(cw, dtype=cam.dtype, device=cam.device) + 0.5
        ii = torch.arange(ch, dtype=cam.dtype, device=cam.device) + 0.5
        off = inds.to(dtype=cam.dtype, device=cam.device)
        u = (off[:, 1:2] + jj[None, :])                    # (M, cw)
        v = (off[:, 0:1] + ii[None, :])                    # (M, ch)
        u = u[:, None, :].expand(m, ch, cw).reshape(m, ch * cw)
        v = v[:, :, None].expand(m, ch, cw).reshape(m, ch * cw)
        rays = plucker_ray_map_at(
            cam[:, :3], cam[:, 3:7], cam[:, 7], h, w, u, v)  # (M, 6, G)
        rays = rays.reshape(m, 6, ch, cw)
        rays[:, 3:] = rays[:, 3:] / self.ray_scale
        return rays

    def _prepare_slot(self, img: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
        """Crop one slot's images; RGB and rays share the SAME window.

        Returns ``(M, 9, ch, cw)`` = ``cat([RGB, d, m/L], dim=1)``. The RGB is
        cropped randomly in training (one crop per image, matching the ladder)
        and centre-cropped in eval; ImageNet normalisation is applied to the
        RGB channels only -- the Pluecker channels stay raw.
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

        ch, cw = self.crop_shape
        if self.training and self.random_crop:
            _, inds = sample_random_image_crops(
                images=img, crop_height=ch, crop_width=cw, num_crops=1)
            inds = inds.reshape(n, 2)
        else:
            # == ttf.center_crop for 84 -> 72
            inds = img.new_full((n, 2), 0)
            inds[:, 0] = (h - ch) // 2
            inds[:, 1] = (w - cw) // 2
        inds = inds.long()
        img = crop_image_from_indices(img, inds, ch, cw)
        if self.imagenet_norm:
            mean = img.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = img.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
            img = (img - mean) / std
        rays = self._crop_plucker_map(cam, inds, h, w).to(dtype=img.dtype)
        return torch.cat([img, rays], dim=1)

    # ------------------------------------------------------------- history
    def _query_features(self, hist: torch.Tensor) -> torch.Tensor:
        """``(M, H, 9)`` base window -> ``(M, H, 13)`` per-step features.

        ``[p_i/L (3) | rot6d_i (6) | (p_i - p_t)/L (3) | age (1)]`` with
        ``age`` linear from -1 (oldest) to 0 (current) and ``p_t`` the
        window's LAST step (the observation frame's own EE position). Split
        out so the test can pin the content and order without reading weights.
        """
        if hist.dim() != 3 or hist.shape[1:] != (
                self.eef_hist_steps, EEF_STEP_DIM):
            raise ValueError(
                f'{self.eef_hist_key}: expected (M, {self.eef_hist_steps}, '
                f'{EEF_STEP_DIM}), got {tuple(hist.shape)}')
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

    def _queries(self, hist: torch.Tensor) -> torch.Tensor:
        """``(M, H, 9)`` base window -> ``(M, H, D)`` query tokens.

        ``history_query=False`` routes ZEROS through the same MLP, so the
        learned/constant-query control keeps every parameter and the
        construction RNG identical (the design section 8 control).
        """
        feat = self._query_features(hist)
        if not self.history_query:
            feat = torch.zeros_like(feat)
        x = self.query_mlp(feat)
        x = x + self.step_emb.weight[None, :, :]
        x = x + self.type_emb.weight[1][None, None, :]
        return self.ln_q(x)

    # ---------------------------------------------------------------- forward
    def forward(self, obs_dict) -> torch.Tensor:
        """Drop-in contract, unchanged: ``(N, 512 + low-dim)``."""
        out = self.forward_full(obs_dict)
        low = [obs_dict[k] for k in self.low_dim_keys]
        return torch.cat([out['z_global'], *low], dim=-1)

    def forward_full(self, obs_dict) -> Dict[str, torch.Tensor]:
        """`forward`'s computation, plus the probe readouts.

        Returns ``{'z_global': (N, 512), 'z_views': (N, K, D),
        'view_active': (N, K) bool, 'query_tokens': (N, H, D),
        'condition_512': (N, 512), 'z_visual_out': (N, K, D)}``.

        ``z_views`` is the per-view mean of the PRE-transformer visual tokens
        (the only view-local quantity; it does not depend on other views or on
        the history). ``z_visual_out`` is the per-view mean of the final
        visual-position outputs and IS context-dependent; rows of inactive
        slots are exactly zero in both. ``query_tokens`` are the final
        query-position outputs before the output projection.
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
                f'missing {self.eef_hist_key!r}: LQ needs the per-sample '
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
                self.eef_hist_steps, EEF_STEP_DIM):
            raise ValueError(
                f'{self.eef_hist_key}: expected (N={n}, {self.eef_hist_steps}, '
                f'{EEF_STEP_DIM}), got {tuple(hist.shape)}')
        if hist.shape[0] != n:
            raise ValueError(
                f'{self.eef_hist_key}: batch size {hist.shape[0]} != {n}')

        # Encode only the active slots. Dead slots are never cropped or turned
        # into rays (their camera vector is all zeros, not a valid pose); their
        # visual rows stay finite zeros so a masked key cannot leak NaN.
        rows, cols = active.nonzero(as_tuple=True)
        x = None
        for k in range(self.n_slots):
            sel = cols == k
            if not bool(sel.any()):
                continue
            sub = self._prepare_slot(
                obs_dict[self.rgb_keys[k]][rows[sel]],
                obs_dict[self.cam_keys[k]][rows[sel]])
            sub_tokens = self.patch_embed(sub).flatten(2).transpose(1, 2)
            sub_tokens = self.ln_x(
                sub_tokens + self.type_emb.weight[0][None, None, :])
            if x is None:
                x = sub_tokens.new_zeros(
                    (n, self.n_slots, self.n_tokens, self.d_model))
            x[rows[sel], k] = sub_tokens
        if x is None:                                  # unreachable: mask gate
            raise RuntimeError('no active views to encode')
        z_views = x.mean(dim=2)                        # (N, K, D) pre-joint
        z_views = z_views.masked_fill(~active[..., None], 0.0)
        x = x.reshape(n, self.seq_visual, self.d_model)
        if x.shape[1] != self.seq_visual:
            raise RuntimeError(
                f'visual block is {x.shape[1]} tokens, expected '
                f'{self.seq_visual}; the query slice K*G would be wrong')

        q = self._queries(hist.to(dtype=x.dtype, device=x.device))
        seq = torch.cat([x, q], dim=1)                 # (N, K*G+H, D)

        pad_vis = (~active).repeat_interleave(self.n_tokens, dim=1)
        pad_q = torch.zeros(
            (n, self.eef_hist_steps), dtype=torch.bool, device=x.device)
        padding = torch.cat([pad_vis, pad_q], dim=1)

        src_mask = None
        if not self.visual_reads_history:
            # bool True == blocked. Visual rows may not read query columns;
            # every other block stays open (the design section 8 control).
            src_mask = torch.zeros(
                (self.seq_len, self.seq_len),
                dtype=torch.bool, device=x.device)
            src_mask[:self.seq_visual, self.seq_visual:] = True

        out = seq
        for layer in self.layers:
            out = layer(out, src_mask=src_mask,
                        src_key_padding_mask=padding)

        z = out[:, self.seq_visual:, :]                # (N, H, D) queries only
        condition = self.out_proj(self.ln_out(z))
        condition = condition.reshape(n, -1)           # (N, H*out_token_dim)
        z_visual_out = out[:, :self.seq_visual].reshape(
            n, self.n_slots, self.n_tokens, self.d_model).mean(dim=2)
        z_visual_out = z_visual_out.masked_fill(~active[..., None], 0.0)
        return {'z_global': condition,
                'z_views': z_views,
                'view_active': active,
                'query_tokens': z,
                'condition_512': condition,
                'z_visual_out': z_visual_out}

    @torch.no_grad()
    def output_shape(self):
        """Measured, like the stock encoder. Dummy cameras and history are
        valid poses; eval mode gives the centre crop and consumes no RNG."""
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
                (1, self.eef_hist_steps, EEF_STEP_DIM),
                dtype=self.dtype, device=self.device)
            hist[..., :3] = torch.tensor(
                [0.5, 0.0, 0.6], dtype=self.dtype, device=self.device)
            hist[..., 3], hist[..., 7] = 1.0, 1.0   # rot6d of the identity
            example[self.eef_hist_key] = hist
            out = self.forward(example)
            expected = CONDITION_WIDTH + sum(
                self.key_shape_map[k][0] for k in self.low_dim_keys)
            if tuple(out.shape[1:]) != (expected,):
                raise RuntimeError(
                    f'encoder emits {tuple(out.shape[1:])}, expected '
                    f'({expected},) = {CONDITION_WIDTH} + low-dim '
                    f'{expected - CONDITION_WIDTH}; the UNet conditioning width '
                    f'would differ from the baseline')
            if not bool(torch.isfinite(out).all()):
                raise RuntimeError(
                    'encoder output is not finite on dummy input')
            return out.shape[1:]
        finally:
            self.train(was_training)
