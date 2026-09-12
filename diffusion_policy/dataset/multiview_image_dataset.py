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
    """(w,x,y,z) unit quaternion -> 3x3 rotation matrix (camera frame -> world)."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


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

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

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
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
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

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
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
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        # the buffer is a lazy zarr store here (not an in-memory numpy backend),
        # so materialize the array before handing it to torch
        return torch.from_numpy(np.asarray(self.replay_buffer['action']))

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data
