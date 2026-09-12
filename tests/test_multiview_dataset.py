"""
CPU-only test for the multi-view dataset contract (no simulator, no rendering).

Builds a synthetic zarr with exactly the schema `generate_multiview_dataset.py`
writes, then checks that it loads through ReplayBuffer/SequenceSampler and that
MultiViewImageDataset serves it like the stock robomimic image dataset.

Run from the repo root:
    python tests/test_multiview_dataset.py

This is what lets the generator/dataset be verified before shipping them to the
render machine -- schema or dtype mistakes fail here instead of after hours of
rendering.
"""
import os
import sys
import shutil
import tempfile

# repo convention: tests run from the repo root
this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(this_dir)
os.chdir(repo_root)
sys.path.insert(0, repo_root)

import numpy as np
import torch
import zarr

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.multiview_image_dataset import (
    MultiViewImageDataset, quat_wxyz_to_mat, fovy_to_intrinsics,
    project_world_to_pixel,
)
register_codecs()

N_VIEWS = 3
H = W = 8
C = 3
EPISODE_LENGTHS = (5, 7, 4)
AZIMUTHS = [-30.0, 0.0, 30.0]


def build_synthetic_zarr(path, codec='jpeg2k', val_ratio=0.34):
    """Write the exact schema the generator produces, from random data."""
    rng = np.random.default_rng(0)
    n_steps = int(sum(EPISODE_LENGTHS))
    episode_ends = np.cumsum(EPISODE_LENGTHS).astype(np.int64)

    store = zarr.DirectoryStore(path)
    root = zarr.group(store, overwrite=True)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    meta_group.array('episode_ends', episode_ends, dtype=np.int64,
        compressor=None, overwrite=True)
    meta_group.array('view_azimuth_deg', np.array(AZIMUTHS), dtype=np.float64,
        compressor=None, overwrite=True)
    meta_group.array('view_cam_pos',
        np.array([[1.0, 0.0, 1.0], [0.5, 0.0, 1.35], [1.0, 0.0, 1.0]]),
        dtype=np.float64, compressor=None, overwrite=True)
    meta_group.array('view_cam_quat_wxyz',
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N_VIEWS, 1)),
        dtype=np.float64, compressor=None, overwrite=True)
    meta_group.array('view_fovy_deg', np.full(N_VIEWS, 45.0), dtype=np.float64,
        compressor=None, overwrite=True)
    meta_group.array('base_cam_pos', np.array([0.5, 0.0, 1.35]),
        dtype=np.float64, compressor=None, overwrite=True)
    meta_group.array('base_cam_quat_wxyz', np.array([1.0, 0.0, 0.0, 0.0]),
        dtype=np.float64, compressor=None, overwrite=True)
    meta_group.array('render_hw', np.array([H, W], dtype=np.int64),
        dtype=np.int64, compressor=None, overwrite=True)

    # low-dim obs + action (abs, 10-dim = pos3 + rot6d + gripper)
    for key, dim in (('robot0_eef_pos', 3), ('robot0_eef_quat', 4),
                     ('robot0_gripper_qpos', 2)):
        arr = rng.normal(size=(n_steps, dim)).astype(np.float32)
        if key.endswith('quat'):
            arr /= np.linalg.norm(arr, axis=-1, keepdims=True)
        data_group.array(name=key, data=arr, shape=arr.shape, chunks=arr.shape,
            compressor=None, dtype=arr.dtype)
    action = rng.normal(size=(n_steps, 10)).astype(np.float32)
    data_group.array(name='action', data=action, shape=action.shape,
        chunks=action.shape, compressor=None, dtype=action.dtype)

    # images: uint8 (T,H,W,C), one chunk per frame, jpeg2k like the real thing
    compressor = Jpeg2k(level=50) if codec == 'jpeg2k' else None
    for v in range(N_VIEWS):
        arr = data_group.require_dataset(
            name=f'view_{v:02d}_image', shape=(n_steps, H, W, C),
            chunks=(1, H, W, C), compressor=compressor, dtype=np.uint8)
        # deterministic, view-dependent content so view mixups are detectable
        base = np.arange(n_steps * H * W * C).reshape(n_steps, H, W, C)
        arr[:] = ((base + v * 17) % 256).astype(np.uint8)

    shape_meta = {
        'obs': {f'view_{v:02d}_image': {'shape': [C, H, W], 'type': 'rgb'}
                for v in range(N_VIEWS)},
        'action': {'shape': [10]},
    }
    shape_meta['obs']['robot0_eef_pos'] = {'shape': [3]}
    shape_meta['obs']['robot0_eef_quat'] = {'shape': [4]}
    shape_meta['obs']['robot0_gripper_qpos'] = {'shape': [2]}
    return shape_meta, episode_ends


def test_camera_math():
    # quat_wxyz_to_mat: +90 deg about z must map x -> y
    half = np.pi / 4
    R = quat_wxyz_to_mat([np.cos(half), 0.0, 0.0, np.sin(half)])
    assert np.allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-9), 'yaw convention'
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), 'not orthonormal'
    assert np.isclose(np.linalg.det(R), 1.0), 'det != 1'

    intr = fovy_to_intrinsics(45.0, H, W)
    assert np.isclose(intr['cx'], W / 2) and np.isclose(intr['cy'], H / 2)
    assert intr['fx'] > 0 and np.isclose(intr['fx'], intr['fy'])
    # 45deg vertical fov: f = (H/2)/tan(22.5deg)
    assert np.isclose(intr['fy'], (H / 2) / np.tan(np.radians(22.5)), atol=1e-9)

    # a camera at the origin with identity rotation looks along -z
    eye = np.zeros(3)
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    u, v, d = project_world_to_pixel([0, 0, -2], eye, ident, intr)
    assert np.isclose(u, intr['cx']) and np.isclose(v, intr['cy'])
    assert np.isclose(d, 2.0)
    # +x world is to the right of the image, +y world is up (smaller row index)
    u, v, _ = project_world_to_pixel([0.5, 0, -2], eye, ident, intr)
    assert u > intr['cx'], 'x should map to the right'
    u, v, _ = project_world_to_pixel([0, 0.5, -2], eye, ident, intr)
    assert v < intr['cy'], 'y should map above the image centre'
    # a point behind the camera has negative depth
    _, _, d = project_world_to_pixel([0, 0, 2], eye, ident, intr)
    assert d < 0
    print('camera math OK (quat convention, intrinsics, projection)')


def test_dataset(path, shape_meta, episode_ends):
    # 1. loads through ReplayBuffer with the expected schema
    rb = ReplayBuffer.create_from_path(path, mode='r')
    assert rb.n_episodes == len(EPISODE_LENGTHS), rb.n_episodes
    assert rb.n_steps == int(episode_ends[-1])

    # 2. dataset construction + sampling
    ds = MultiViewImageDataset(
        shape_meta=shape_meta, dataset_path=path, horizon=4,
        pad_before=1, pad_after=7, n_obs_steps=2, abs_action=True,
        val_ratio=0.34, seed=0)
    assert len(ds) > 0, 'no training samples'
    assert sorted(ds.rgb_keys) == [f'view_{v:02d}_image' for v in range(N_VIEWS)]
    assert sorted(ds.lowdim_keys) == [
        'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

    batch = ds[0]
    assert set(batch.keys()) == {'obs', 'action'}, batch.keys()
    assert batch['action'].shape == (4, 10), batch['action'].shape
    assert batch['action'].dtype == torch.float32
    for key in ds.rgb_keys:
        x = batch['obs'][key]
        assert x.shape == (2, C, H, W), (key, x.shape)   # n_obs_steps, C, H, W
        assert x.dtype == torch.float32, (key, x.dtype)
        assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0, 'images not in [0,1]'
    for key in ds.lowdim_keys:
        assert batch['obs'][key].dtype == torch.float32

    # views must differ from each other (catches a key/index mixup)
    v0 = batch['obs']['view_00_image']
    v1 = batch['obs']['view_01_image']
    assert not torch.allclose(v0, v1), 'views are identical -- key mapping is wrong'

    # 3. views carry the content written for that view index (jpeg2k is lossy,
    #    so compare with tolerance -- the point is to catch key/index mixups)
    px0 = int(np.asarray(rb['view_00_image'][0]).reshape(-1)[0])
    px1 = int(np.asarray(rb['view_01_image'][0]).reshape(-1)[0])
    assert abs(px0 - 0) <= 8, f'view_00 first pixel {px0}, expected ~0'
    assert abs(px1 - 17) <= 8, f'view_01 first pixel {px1}, expected ~17'
    print('dataset sampling OK (shapes, dtypes, range, per-view content)')

    # 4. normalizer rules
    normalizer = ds.get_normalizer()
    n_keys = set(normalizer.params_dict.keys())
    for key in ds.rgb_keys + ds.lowdim_keys + ['action']:
        assert key in n_keys, f'{key} missing from normalizer {sorted(n_keys)}'
    img_norm = normalizer['view_00_image']
    lo = float(img_norm.normalize(torch.tensor(0.0)))
    hi = float(img_norm.normalize(torch.tensor(1.0)))
    assert np.isclose(lo, -1.0) and np.isclose(hi, 1.0), (lo, hi)
    assert normalizer['robot0_eef_quat'].params_dict['scale'].shape[-1] == 4
    assert normalizer['robot0_eef_pos'].params_dict['scale'].shape[-1] == 3
    # action normalizer round-trips
    a = batch['action']
    back = normalizer['action'].unnormalize(normalizer['action'].normalize(a))
    assert torch.allclose(a, back, atol=1e-5), 'action normalizer not invertible'
    print('normalizer OK (image range, lowdim, invertible abs action)')

    # 5. validation split + actions
    val = ds.get_validation_dataset()
    assert len(val) >= 0
    all_actions = ds.get_all_actions()
    assert all_actions.shape == (int(episode_ends[-1]), 10), all_actions.shape
    print('validation split + get_all_actions OK')

    # 6. camera params round-trip
    cp = ds.camera_params
    assert np.allclose(cp['azimuth_deg'], AZIMUTHS)
    assert cp['cam_pos'].shape == (N_VIEWS, 3)
    assert cp['cam_quat_wxyz'].shape == (N_VIEWS, 4)
    assert np.allclose(cp['render_hw'], [H, W])
    assert np.isclose(cp['intrinsics']['fy'], (H / 2) / np.tan(np.radians(22.5)))
    assert np.allclose(cp['base_cam_pos'], [0.5, 0.0, 1.35])
    print('camera_params OK (azimuths, poses, fovy, intrinsics, base pose)')


def test():
    tmp = tempfile.mkdtemp(prefix='multiview_test_')
    try:
        path = os.path.join(tmp, 'synthetic.zarr')
        shape_meta, episode_ends = build_synthetic_zarr(path)
        test_camera_math()
        test_dataset(path, shape_meta, episode_ends)
        print('ALL MULTIVIEW DATASET CHECKS PASSED')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test()
