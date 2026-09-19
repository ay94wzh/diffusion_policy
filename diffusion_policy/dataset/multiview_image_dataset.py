"""
Dataset over a multi-view image zarr built by `generate_multiview_dataset.py`.

Each timestep holds N simultaneous renders of the same scene state from N fixed
camera poses (named `view_00_image` ... `view_{N-1:02d}_image`, zero-padded so
alphabetical order == index order, which is what MultiImageObsEncoder sorts by),
plus the low-dim obs and base-frame actions copied verbatim from the source
hdf5, plus per-view camera parameters in `meta` (used by M3 for Plücker maps and
M4 for camera-frame actions).

Mirrors `RobomimicReplayImageDataset.__getitem__` / `get_normalizer` so policies
train identically; the only difference is that the zarr is pre-built (no hdf5
conversion step, no cache) and actions are already in the 10-dim absolute form.
"""
from typing import Dict, List, Optional
import copy
import os

import numpy as np
import torch
import zarr

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()

# meta keys the generator writes (see generate_multiview_dataset.py)
META_VIEW_KEY = 'view_keys'          # unused placeholder, kept for clarity
VIEW_AZIMUTH_KEY = 'view_azimuth_deg'
VIEW_POS_KEY = 'view_cam_pos'
VIEW_QUAT_KEY = 'view_cam_quat_wxyz'
VIEW_FOVY_KEY = 'view_fovy_deg'
BASE_POS_KEY = 'base_cam_pos'
BASE_QUAT_KEY = 'base_cam_quat_wxyz'
RENDER_HW_KEY = 'render_hw'


# ---------------------------------------------------------------------------
# camera geometry helpers (numpy only -- no robosuite/mujoco dependency, so this
# module stays importable on machines without a simulator)
# ---------------------------------------------------------------------------

def quat_wxyz_to_mat(q):
    """(w,x,y,z) quaternion -> 3x3 rotation matrix (camera frame -> world).

    Normalizes first, matching `quat_wxyz_to_mat_batch` and the torch
    `plucker.quat_wxyz_to_mat_torch`. All three implement one convention and
    must not disagree on this: a quaternion that is unit only to float
    precision otherwise yields a matrix that is not quite orthogonal, which
    shows up as a few 1e-6 px of projection error. For unit inputs this is a
    no-op, so M2's gate-2 calibration is unaffected.
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_wxyz_to_mat_batch(q):
    """(..., 4) wxyz quaternions -> (..., 3, 3) rotation matrices.

    Batched form of `quat_wxyz_to_mat`; the two are checked against each other
    in tests/test_view_conditioned_obs_encoder.py.
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


EEF_HIST_STEP_DIM = 11   # [pos(3), rot6d(6), gripper(2)] per step, camera frame

ACTION_DIM = 10          # [pos(3), rot6d(6), gripper(1)] absolute, base frame
# M4's camera-frame action target. Deliberately NOT a shape_meta key: it is
# derived from the demonstrated future actions, which do not exist at rollout
# time, so it must never be an obs key the policy normalizes / the env serves.
# It rides at the TOP LEVEL of the sample dict, next to 'action'.
AUX_ACTION_KEY = 'aux_action'


def _identity_normalizer(dim: int):
    """An exact identity normalizer of width `dim`.

    `get_identity_normalizer_from_stat` needs min/max/mean/std all of the target
    shape (`create_manual` asserts it), and the dtype must be float32:
    `_normalize` casts the input to the scale's dtype while `create_manual` does
    not cast, so a float64 stat here would promote the tensor to float64 and
    break the encoder's matmul against float32 rotation matrices.
    """
    z = np.zeros(dim, dtype=np.float32)
    o = np.ones(dim, dtype=np.float32)
    return get_identity_normalizer_from_stat(
        {'min': z, 'max': o, 'mean': z, 'std': o})


def eef_hist_to_cam(eef_pos, eef_quat_wxyz, gripper_qpos, cam_pos, cam_quat_wxyz):
    """Express a base-frame EE history in one camera's frame.

    Per step::

        [ R_c^T (p - cam_pos) (3) , R_c^T R_eef as rot6d (6) , gripper (2) ]

    `eef_pos` ``(..., 3)``, `eef_quat_wxyz` ``(..., 4)``, `gripper_qpos`
    ``(..., 2)``, camera as ``(3,)``/``(4,)``. Returns ``(..., 11)`` float32.

    The rot6d is the first two **rows** of the relative rotation matrix, matching
    pytorch3d's ``matrix_to_rotation_6d`` that the rest of this repo reaches via
    ``RotationTransformer``.

    This lives here, next to the numpy camera math, so the dataset (training) and
    `eval_novel_view.py` (rollout) share ONE implementation. They must agree
    exactly -- a mismatch would be invisible during training and wrong at eval.
    """
    R = quat_wxyz_to_mat(cam_quat_wxyz)                    # camera -> world
    Rt = R.T
    p = np.asarray(eef_pos, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    pos_cam = np.einsum('ij,...j->...i', Rt, p)
    R_eef = quat_wxyz_to_mat_batch(eef_quat_wxyz)
    R_rel = np.einsum('ij,...jk->...ik', Rt, R_eef)
    rot6 = np.concatenate([R_rel[..., 0, :], R_rel[..., 1, :]], axis=-1)
    out = np.concatenate([pos_cam, rot6,
                          np.asarray(gripper_qpos, dtype=np.float64)], axis=-1)
    return out.astype(np.float32)


def rot6d_to_mat(d6):
    """``(..., 6)`` -> ``(..., 3, 3)``, the inverse of pytorch3d's
    ``matrix_to_rotation_6d`` (rows 0 and 1 of R, concatenated).

    Gram-Schmidt exactly as pytorch3d's ``rotation_6d_to_matrix`` implements it.
    Pinned to ``RotationTransformer('rotation_6d', 'matrix')`` in
    tests/test_aux_action_heads.py, so this numpy copy cannot drift from the
    convention that PRODUCED the stored actions (`_convert_actions`).
    """
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def action_to_cam(action, cam_pos, cam_quat_wxyz):
    """Express a base-frame absolute action chunk in ONE camera's frame.

    Per step::

        [ R_c^T (p - cam_pos) (3) , R_c^T R_b as rot6d (6) , gripper (1) ]

    `action` is ``(..., 10)`` = [pos(3), rot6d(6), gripper(1)]; the camera is
    ``(3,)``/``(4,)``. Returns ``(..., 10)`` float32. Only EXTRINSICS enter --
    fovy/intrinsics must never appear here.

    The rotation MUST round-trip through the full 3x3 matrix. The 6d encoding
    keeps only rows 0 and 1 of R, but ``rows01(R_c^T R_b)`` is a function of all
    three rows of R_b -- so rotating the 6d vector with a 6x6 linear map is NOT
    the transform. That shortcut is exactly right when ``R_c == I``, which is
    why a test at an identity camera proves nothing
    (tests/test_aux_action_heads.py asserts the shortcut fails elsewhere).

    Sibling of `eef_hist_to_cam` (same convention, EE pose instead of action,
    rot6d in instead of a quaternion); the two are pinned to each other in the
    test.
    """
    a = np.asarray(action, dtype=np.float64)
    R = quat_wxyz_to_mat(cam_quat_wxyz)                    # camera -> world
    Rt = R.T
    p = a[..., :3] - np.asarray(cam_pos, dtype=np.float64)
    pos_cam = np.einsum('ij,...j->...i', Rt, p)
    R_rel = np.einsum('ij,...jk->...ik', Rt, rot6d_to_mat(a[..., 3:9]))
    rot6 = np.concatenate([R_rel[..., 0, :], R_rel[..., 1, :]], axis=-1)
    out = np.concatenate([pos_cam, rot6, a[..., 9:]], axis=-1)
    return out.astype(np.float32)


def fovy_to_intrinsics(fovy_deg, height, width):
    """
    Pinhole intrinsics for a square-pixel render of (height, width) with vertical
    field of view `fovy_deg` (degrees, as stored in mujoco's cam_fovy).
    robosuite 1.2 / mujoco_py 2.1.2 have no get_camera_intrinsic_matrix.
    """
    f = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    return {'fx': f, 'fy': f, 'cx': width / 2.0, 'cy': height / 2.0}


def project_world_to_pixel(p_world, cam_pos, cam_quat_wxyz, intrinsics):
    """
    Project a world point into a stored image pixel. Returns (u, v, depth) where
    u is the column and v the row of the *stored* image (top-down, i.e. after the
    [::-1] flip robomimic applies). depth is along the camera's viewing axis and
    is positive in front of the camera.

    Mujoco cameras look along -z with +x right and +y up in the camera frame.
    """
    R = quat_wxyz_to_mat(cam_quat_wxyz)
    p_cam = R.T @ (np.asarray(p_world, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64))
    depth = -p_cam[2]
    u = intrinsics['cx'] + intrinsics['fx'] * p_cam[0] / depth
    v = intrinsics['cy'] - intrinsics['fy'] * p_cam[1] / depth
    return u, v, depth


class MultiViewImageDataset(BaseImageDataset):
    """
    Reads a pre-built multi-view zarr (DirectoryStore) directly -- no conversion,
    no in-memory copy, so a multi-GB dataset is memory-mapped rather than loaded.
    """

    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=True,
            use_legacy_normalizer=False,
            view_subset=None,
            view_pool=None,
            view_count_range=None,
            eef_hist_steps=4,
            emit_aux_action=False,
            aux_n_steps=8,
            view_mask_key='view_mask',
            eef_hist_key='view_eef_hist',
            seed=42,
            val_ratio=0.0):
        replay_buffer = ReplayBuffer.create_from_path(
            os.path.expanduser(dataset_path), mode='r')

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        n_views = len(np.asarray(replay_buffer.meta[VIEW_AZIMUTH_KEY][:]))
        if view_subset is not None:
            # Random-view mode: one rgb slot, filled per sample with a view drawn
            # from `view_subset`. The shape_meta rgb key is then a slot label,
            # not a zarr array name -- the array is read via view_key(idx) in
            # __getitem__. Views outside the subset are never trained on, which
            # is what makes held-out-viewpoint evaluation honest.
            assert len(rgb_keys) == 1, (
                f'view_subset needs exactly 1 rgb key, got {rgb_keys}')
            view_subset = [int(v) for v in view_subset]
            assert len(view_subset) > 0, 'view_subset is empty'
            assert min(view_subset) >= 0 and max(view_subset) < n_views, (
                f'view_subset {view_subset} out of range for {n_views} views')
        self.view_subset = view_subset
        self.n_views = n_views

        # ---- M3 mode: K view SLOTS, each filled per sample from a random
        # subset of `view_pool`, plus per-slot camera vectors, an activity mask
        # and the camera-frame EE history. Independent of `view_subset` (L1's
        # fixed single-slot mode), which must keep behaving exactly as before.
        self.view_pool = None
        self.cam_keys = []
        self.view_mask_key = view_mask_key
        self.eef_hist_key = eef_hist_key
        self.eef_hist_steps = int(eef_hist_steps)
        self.view_count_range = None
        self.cam_table = None
        # M4: emit the per-slot camera-frame action chunk alongside 'action'.
        # Defaults off, so every pre-M4 config and dataset test is unchanged.
        self.emit_aux_action = False
        self.aux_n_steps = int(aux_n_steps)
        if view_pool is not None:
            view_pool = [int(v) for v in view_pool]
            if len(view_pool) == 0:
                raise ValueError('view_pool is empty')
            if min(view_pool) < 0 or max(view_pool) >= n_views:
                raise ValueError(
                    f'view_pool {view_pool} out of range for {n_views} views')
            if len(set(view_pool)) != len(view_pool):
                raise ValueError(f'view_pool {view_pool} has duplicates')
            if view_subset is not None:
                raise ValueError('view_pool and view_subset are mutually exclusive')
            n_slots = len(rgb_keys)
            if n_slots < 1:
                raise ValueError('M3 mode needs at least one rgb slot')
            if n_slots > len(view_pool):
                raise ValueError(
                    f'{n_slots} view slots but the pool has only {len(view_pool)} '
                    f'views; slots are filled without replacement')
            if view_count_range is None:
                view_count_range = (1, n_slots)
            lo, hi = (int(view_count_range[0]), int(view_count_range[1]))
            if not (1 <= lo <= hi <= n_slots):
                raise ValueError(
                    f'view_count_range {view_count_range} must satisfy '
                    f'1 <= lo <= hi <= n_slots ({n_slots})')
            if lo != hi and hi != n_slots:
                raise ValueError(
                    'an active count drawn from [lo, hi] can only fill slots '
                    'without replacement when hi == n_slots')
            # camera key per slot: `view_slot_03_image` -> `view_slot_03_cam`
            self.cam_keys = [
                k[:-len('_image')] + '_cam' if k.endswith('_image') else k + '_cam'
                for k in rgb_keys]
            for k in self.cam_keys + [view_mask_key, eef_hist_key]:
                if k in obs_shape_meta:
                    raise ValueError(
                        f'{k!r} must NOT be in shape_meta: the env builds its '
                        f'robomimic modality mapping from shape_meta and '
                        f'RobomimicImageWrapper raises on the key suffix')
            # camera_params reads self.replay_buffer; bind it now (the same
            # object is assigned again below) rather than duplicating the
            # meta-parsing here
            self.replay_buffer = replay_buffer
            cp = self.camera_params
            h, w = (int(x) for x in cp['render_hw'])
            # [pos(3), quat_wxyz(4), fovy_deg(1), h(1), w(1)], float32 so the
            # normalizer's cast does not promote it to float64 and break matmul
            self.cam_table = np.concatenate([
                cp['cam_pos'], cp['cam_quat_wxyz'], cp['fovy_deg'][:, None],
                np.tile(np.array([h, w], dtype=np.float64), (n_views, 1)),
            ], axis=-1).astype(np.float32)
            assert self.cam_table.shape == (n_views, 10), self.cam_table.shape
            self.view_pool = view_pool
            self.view_count_range = (lo, hi)
            self.n_slots = n_slots
            self.slot_shape = tuple(obs_shape_meta[rgb_keys[0]]['shape'])

            # M4's camera-frame target. Validated here rather than in
            # __getitem__ so a bad combination fails at construction.
            self.emit_aux_action = bool(emit_aux_action)
            if self.emit_aux_action:
                if not abs_action:
                    raise ValueError(
                        'emit_aux_action requires abs_action: the camera-frame '
                        'transform assumes the 10-dim absolute action layout')
                if n_obs_steps is None:
                    raise ValueError(
                        'emit_aux_action requires n_obs_steps (the target is '
                        'per obs step)')
                # Obs step `to` is supervised on action[to : to+C], so the last
                # index touched is C + n_obs_steps - 2. Past `horizon - 1` the
                # sampler's pad_after edge-repeats the final action -- fake
                # supervision the head would happily fit.
                span = self.aux_n_steps + int(n_obs_steps) - 1
                if span > horizon:
                    raise ValueError(
                        f'aux_n_steps {self.aux_n_steps} with n_obs_steps '
                        f'{n_obs_steps} reaches action index {span - 1}, past '
                        f'horizon {horizon}: the tail of the chunk would be '
                        f'pad_after edge-repeat, not real data. Need '
                        f'aux_n_steps + n_obs_steps - 1 <= horizon.')

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        # Restrict the sampler to the keys this dataset actually consumes: its
        # default is every array in the buffer, which decodes all 13 views per
        # sample (~22 ms/sample) even when the config uses one. In random-view
        # mode no view is read here at all -- the view is drawn per sample, so
        # __getitem__ reads just the chosen one (~1.7 ms/sample).
        if view_subset is None and self.view_pool is None:
            sampler_keys = list(rgb_keys) + list(lowdim_keys) + ['action']
        else:
            # slot modes: the rgb keys are slot LABELS, not zarr array names --
            # the arrays are read per sample via _view_obs_frames()
            sampler_keys = list(lowdim_keys) + ['action']

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            keys=sampler_keys,
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.sampler_keys = sampler_keys
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self.dataset_path = dataset_path

    # ------------------------------------------------------------------
    # camera parameters (for M3 Plücker conditioning / M4 camera-frame actions)
    # ------------------------------------------------------------------
    @property
    def camera_params(self) -> Dict[str, np.ndarray]:
        """
        Per-view camera parameters as stored by the generator:
            azimuth_deg (V,), cam_pos (V,3), cam_quat_wxyz (V,4), fovy_deg (V,),
            intrinsics (dict of scalars, identical for all views), render_hw (2,),
            base_cam_pos (3,), base_cam_quat_wxyz (4,).
        View index i corresponds to obs key `view_{i:02d}_image`.
        """
        meta = self.replay_buffer.meta
        azimuth = np.asarray(meta[VIEW_AZIMUTH_KEY][:], dtype=np.float64)
        pos = np.asarray(meta[VIEW_POS_KEY][:], dtype=np.float64)
        quat = np.asarray(meta[VIEW_QUAT_KEY][:], dtype=np.float64)
        fovy = np.asarray(meta[VIEW_FOVY_KEY][:], dtype=np.float64)
        h, w = (int(x) for x in np.asarray(meta[RENDER_HW_KEY][:]))
        assert pos.shape == (len(azimuth), 3), f'bad view_cam_pos shape {pos.shape}'
        return {
            'azimuth_deg': azimuth,
            'cam_pos': pos,
            'cam_quat_wxyz': quat,
            'fovy_deg': fovy,
            'intrinsics': fovy_to_intrinsics(fovy[0], h, w),
            'render_hw': np.array([h, w], dtype=np.int64),
            'base_cam_pos': np.asarray(meta[BASE_POS_KEY][:], dtype=np.float64),
            'base_cam_quat_wxyz': np.asarray(meta[BASE_QUAT_KEY][:], dtype=np.float64),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def view_key(view_idx: int) -> str:
        return f'view_{view_idx:02d}_image'

    def aux_action_pooled(self) -> np.ndarray:
        """``(V*T, 10)`` camera-frame actions, pooled over the training pool.

        The statistics behind ``normalizer[AUX_ACTION_KEY]``. Built with the
        SAME `action_to_cam` that `_m3_slots` emits, so the fit cannot drift
        from the data it normalizes. Reads only the low-dim action array, so it
        costs milliseconds even though it covers every view and timestep.
        """
        if not self.emit_aux_action:
            raise RuntimeError('aux_action_pooled needs emit_aux_action=True')
        actions = np.asarray(self.replay_buffer['action'], dtype=np.float64)
        return np.concatenate([
            action_to_cam(actions, self.cam_table[v][:3], self.cam_table[v][3:7])
            for v in self.view_pool], axis=0)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            keys=self.sampler_keys,
            # without this the val sampler decodes the full `horizon` of every
            # view instead of only n_obs_steps (~8x the images per sample)
            key_first_k=self.sampler.key_first_k
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        # M3's extra keys: the policy normalizes EVERY key present in the obs
        # dict (LinearNormalizer hard-looks-up params_dict[key]), so these must
        # be registered -- but as IDENTITY. Fitting them with mode='limits'
        # would map any constant dim to 0 (scale=1, offset=-input_min), which
        # would silently destroy a novel camera pose at eval.
        if self.view_pool is not None:
            for k in self.cam_keys:
                normalizer[k] = _identity_normalizer(10)
            normalizer[self.view_mask_key] = _identity_normalizer(self.n_slots)
            normalizer[self.eef_hist_key] = _identity_normalizer(
                self.n_slots * EEF_HIST_STEP_DIM * self.eef_hist_steps)
            if self.emit_aux_action:
                # FITTED, unlike the identity keys above: this is the aux
                # head's regression target, and camera-frame pos axes have
                # different extents from the base-frame action's, so reusing
                # the action key's scale would be axis-mismatched. Sharing the
                # abs-action normalizer keeps the aux term in the same
                # normalized units as the diffusion target, which is what makes
                # aux_loss_weight interpretable.
                normalizer[AUX_ACTION_KEY] = \
                    robomimic_abs_action_only_normalizer_from_stat(
                        array_to_stats(self.aux_action_pooled()))
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        # the buffer is a lazy zarr store here (not an in-memory numpy backend),
        # so materialize the array before handing it to torch
        return torch.from_numpy(np.asarray(self.replay_buffer['action']))

    def __len__(self):
        return len(self.sampler)

    def _view_obs_frames(self, idx: int, view_idx: int) -> np.ndarray:
        """First n_obs_steps frames of one view for sample `idx` (T,H,W,C).

        Mirrors SequenceSampler.sample_sequence's pad/repeat semantics exactly:
        obs step j takes sample[j - sample_start_idx] inside the window,
        sample[0] before it and sample[-1] past its end, so episode-edge samples
        repeat a frame instead of reading out of range. Reading only this view
        is the point -- the generic sampler would decode all 13.
        """
        buffer_start, _, sample_start, sample_end = self.sampler.indices[idx]
        n_obs = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        n_sample = sample_end - sample_start
        rel = np.clip(np.arange(n_obs) - sample_start, 0, n_sample - 1)
        arr = self.replay_buffer[self.view_key(view_idx)]
        return np.asarray(arr.oindex[buffer_start + rel])

    def _eef_history(self, idx: int):
        """(To, H, ...) base-frame EE history ending at each obs step.

        The H most recent steps at each obs step, with the SAME pad rule the obs
        frames use (`_view_obs_frames`): indices before the sample window repeat
        the window's first frame rather than reading out of range.

        `eval_novel_view.py` reproduces this rule by seeding its history buffer
        with H copies of the first observed pose -- if the two ever disagree the
        history would be wrong only at episode starts, which is nearly invisible
        in training and wrong at rollout, so they share one transform
        (`eef_hist_to_cam`) and this documented pad rule.
        """
        buffer_start, _, sample_start, sample_end = self.sampler.indices[idx]
        n_obs = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        n_sample = sample_end - sample_start
        rel = np.clip(np.arange(n_obs) - sample_start, 0, n_sample - 1)   # (To,)
        h = self.eef_hist_steps
        hist_rel = np.clip(rel[:, None] - (h - 1) + np.arange(h)[None, :],
                           0, n_sample - 1)                                # (To,H)
        flat = (buffer_start + hist_rel).reshape(-1)
        out = []
        for key in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'):
            arr = np.asarray(self.replay_buffer[key].oindex[flat])
            out.append(arr.reshape(n_obs, h, -1).astype(np.float32))
        return out

    def _m3_slots(self, idx: int, action_chunk: np.ndarray):
        """All K view slots + camera vectors + activity mask + EE history.

        Every slot is always emitted: a ragged schema (only the drawn views)
        would not collate. `view_mask` marks which slots are live for this
        sample, and the encoder fuses only those. Masked slots are filled with
        zeros, so if the mask were ever ignored the result would be loudly wrong
        rather than subtly wrong.

        Returns ``(obs_dict, aux_action)``. `aux_action` is ``(To, K, C*10)``
        (M4), or None when `emit_aux_action` is off -- it is returned SEPARATELY
        rather than placed in `obs_dict`, because it derives from demonstrated
        future actions and must never reach the policy's obs normalizer. The
        per-slot camera-frame chunk for obs step `to` is built from the SAME
        view draw as that slot's image, so the pairing cannot drift.
        """
        n_slots = self.n_slots
        view_pool = self.view_pool
        view_count_range = self.view_count_range
        if view_pool is None or view_count_range is None:
            raise RuntimeError('_m3_slots requires view_pool and view_count_range')
        lo, hi = view_count_range
        k_active = int(np.random.randint(lo, hi + 1))
        # without replacement: when k_active == n_slots this is a permutation of
        # the drawn subset, so the encoder cannot key off slot identity (and the
        # fusion is permutation-invariant)
        chosen = np.random.choice(view_pool, size=k_active, replace=False)

        n_obs = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        pos, quat, grip = self._eef_history(idx)   # (To,H,3),(To,H,4),(To,H,2)

        obs_dict = dict()
        mask = np.zeros((n_obs, n_slots), dtype=np.float32)
        hist = np.zeros(
            (n_obs, n_slots, EEF_HIST_STEP_DIM * self.eef_hist_steps),
            dtype=np.float32)
        C = self.aux_n_steps
        aux = (np.zeros((n_obs, n_slots, C * ACTION_DIM), dtype=np.float32)
               if self.emit_aux_action else None)
        # (n_obs, C) window starts: obs step `to` is supervised on action[to:to+C]
        win = np.arange(n_obs)[:, None] + np.arange(C)[None, :]
        for slot in range(n_slots):
            key, cam_key = self.rgb_keys[slot], self.cam_keys[slot]
            if slot < k_active:
                v = int(chosen[slot])
                # T,H,W,C -> T,C,H,W and uint8 -> float32 [0,1]
                obs_dict[key] = np.moveaxis(
                    self._view_obs_frames(idx, v), -1, 1
                    ).astype(np.float32) / 255.
                mask[:, slot] = 1.0
                cam = self.cam_table[v]            # (10,) float32
                # flatten (H, 11) -> (11*H,): step-major, and the eval wrapper
                # must flatten in this same order
                hist[:, slot] = eef_hist_to_cam(
                    pos, quat, grip, cam[:3], cam[3:7]).reshape(n_obs, -1)
                if aux is not None:
                    # raw stored action -> this slot's camera frame; the same
                    # view `v` that supplied the image above
                    aux[:, slot] = action_to_cam(
                        action_chunk[win], cam[:3], cam[3:7]
                        ).reshape(n_obs, C * ACTION_DIM)
            else:
                obs_dict[key] = np.zeros(
                    (n_obs,) + self.slot_shape, dtype=np.float32)
                cam = np.zeros(10, dtype=np.float32)
            # tiled over obs steps, matching the image's leading T axis
            obs_dict[cam_key] = np.tile(cam, (n_obs, 1))
        obs_dict[self.view_mask_key] = mask
        obs_dict[self.eef_hist_key] = hist
        return obs_dict, aux

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        aux_action = None
        if self.view_pool is not None:
            # M3: all slots, their camera vectors, the mask and the EE history.
            # M4: plus the per-slot camera-frame action chunk, which comes back
            # separately -- it must not enter `obs_dict` (see AUX_ACTION_KEY).
            m3_obs, aux_action = self._m3_slots(idx, data['action'])
            obs_dict.update(m3_obs)
        else:
            for key in self.rgb_keys:
                if self.view_subset is None:
                    # move channel last to channel first
                    # T,H,W,C
                    # convert uint8 image to float32
                    obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1
                        ).astype(np.float32) / 255.
                    # T,C,H,W
                    del data[key]
                else:
                    # random-view mode: fill this sample's slot from one view
                    # drawn uniformly over the training subset (never a held-out
                    # view). This view is not one of the sampler's keys, so read
                    # it here.
                    view_idx = self.view_subset[
                        np.random.randint(len(self.view_subset))]
                    obs_dict[key] = np.moveaxis(
                        self._view_obs_frames(idx, view_idx), -1, 1
                        ).astype(np.float32) / 255.
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        if aux_action is not None:
            torch_data[AUX_ACTION_KEY] = torch.from_numpy(aux_action)
        return torch_data
