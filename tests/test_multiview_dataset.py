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
    MultiViewImageDataset, quat_wxyz_to_mat, quat_wxyz_to_mat_batch,
    fovy_to_intrinsics, project_world_to_pixel, eef_hist_to_cam,
)
register_codecs()

N_VIEWS = 3
H = W = 8
C = 3
EPISODE_LENGTHS = (5, 7, 4)
AZIMUTHS = [-30.0, 0.0, 30.0]


def build_synthetic_zarr(path, codec='jpeg2k', val_ratio=0.34, h=H, w=W):
    """Write the exact schema the generator produces, from random data.

    ``h``/``w`` are overridable so a test that needs a feature grid larger
    than 1x1 (the default 8x8 gives stride-16 1x1) can ask for one; every
    existing caller keeps the 8x8 fixture.
    """
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
    meta_group.array('render_hw', np.array([h, w], dtype=np.int64),
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
            name=f'view_{v:02d}_image', shape=(n_steps, h, w, C),
            chunks=(1, h, w, C), compressor=compressor, dtype=np.uint8)
        # deterministic, view-dependent content so view mixups are detectable
        base = np.arange(n_steps * h * w * C).reshape(n_steps, h, w, C)
        arr[:] = ((base + v * 17) % 256).astype(np.uint8)

    shape_meta = {
        'obs': {f'view_{v:02d}_image': {'shape': [C, h, w], 'type': 'rgb'}
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


def _expected_history(frames, ep_start, h):
    frames = np.asarray(frames, dtype=np.int64)
    return np.clip(frames[:, None] - (h - 1) + np.arange(h)[None, :],
                   int(ep_start), frames[:, None])


def _find_row(ds, buffer_start=None, sample_start=None, tail=False):
    for i, row in enumerate(ds.sampler.indices):
        b, e, ss, se = (int(x) for x in row)
        if buffer_start is not None and b != buffer_start:
            continue
        if sample_start is not None and ss != sample_start:
            continue
        if tail and (se - ss) >= ds.horizon:
            continue
        return i, b, e, ss, se
    raise AssertionError(
        f'no sample with buffer_start={buffer_start}, '
        f'sample_start={sample_start}, tail={tail}')


def test_eef_history(path, shape_meta):
    """Pin both EE-history window rules and the training/inference match.

    'sample' is the legacy sample-window clip (kept for committed runs);
    'episode' is the true per-observation past-four that both rollout paths
    serve. The last check simulates the rollout deque and is the regression
    that would have caught the original mismatch.
    """
    rb = ReplayBuffer.create_from_path(path, mode='r')
    ends = np.asarray(rb.episode_ends[:], dtype=np.int64)
    pos_all = np.asarray(rb['robot0_eef_pos'][:])

    common = dict(
        shape_meta=shape_meta, dataset_path=path, horizon=4,
        pad_before=1, pad_after=7, n_obs_steps=2, abs_action=True,
        view_pool=[0, 1, 2], view_count_range=[1, 3],
        eef_hist_steps=4, seed=0, val_ratio=0.34)
    ds = MultiViewImageDataset(eef_hist_window='episode', **common)
    assert ds.eef_hist_window == 'episode'
    h = ds.eef_hist_steps

    # ordinary mid-episode sample in episode 1 (episode start is frame 5)
    idx, b, _, _, _ = _find_row(ds, buffer_start=8)
    assert b == 8
    pos, quat, grip = ds._eef_history(idx)
    np.testing.assert_array_equal(
        pos, pos_all[_expected_history([8, 9], 5, h)])

    # the legacy 'sample' rule on the same sample: only B and B+1 appear
    ds.eef_hist_window = 'sample'
    pos_s, _, _ = ds._eef_history(idx)
    np.testing.assert_array_equal(
        pos_s, pos_all[np.array([[8, 8, 8, 8], [8, 8, 8, 9]])])
    ds.eef_hist_window = 'episode'

    # episode-start padded sample: H repeats of the first pose
    idx0, b0, _, ss0, _ = _find_row(ds, buffer_start=0, sample_start=1)
    assert b0 == 0 and ss0 == 1
    pos0, _, _ = ds._eef_history(idx0)
    np.testing.assert_array_equal(
        pos0, pos_all[np.zeros((2, h), dtype=np.int64)])

    # first sample of a later episode: no bleed from the previous episode
    idx1, b1, _, _, _ = _find_row(ds, buffer_start=5, sample_start=0)
    pos1, _, _ = ds._eef_history(idx1)
    np.testing.assert_array_equal(
        pos1, pos_all[_expected_history([5, 6], 5, h)])

    # episode-end tail sample (n_sample < horizon)
    idxt, bt, et, sst, set_ = _find_row(ds, tail=True)
    n_sample = set_ - sst
    frames_t = bt + np.clip(np.arange(2) - sst, 0, n_sample - 1)
    ep_t = int(np.searchsorted(ends, bt, side='right'))
    start_t = 0 if ep_t == 0 else int(ends[ep_t - 1])
    post, _, _ = ds._eef_history(idxt)
    np.testing.assert_array_equal(
        post, pos_all[_expected_history(frames_t, start_t, h)])

    # validation-dataset sample uses the same episode-anchored rule
    val = ds.get_validation_dataset()
    assert val.eef_hist_window == 'episode'
    vb, _, vss, vse = (int(x) for x in val.sampler.indices[0])
    vn = vse - vss
    vframes = vb + np.clip(np.arange(2) - vss, 0, vn - 1)
    vep = int(np.searchsorted(ends, vb, side='right'))
    vstart = 0 if vep == 0 else int(ends[vep - 1])
    pv, _, _ = val._eef_history(0)
    np.testing.assert_array_equal(
        pv, pos_all[_expected_history(vframes, vstart, h)])

    # H longer than the episode: repeat the episode's first frame
    ds8 = MultiViewImageDataset(
        eef_hist_window='episode', **{**common, 'eef_hist_steps': 8})
    i8, _, _, _, _ = _find_row(ds8, buffer_start=4, sample_start=0)
    p8, _, _ = ds8._eef_history(i8)
    np.testing.assert_array_equal(
        p8, pos_all[_expected_history([4, 4], 0, 8)])

    # To=3: one window per observation frame, not one window per sample
    ds3 = MultiViewImageDataset(
        eef_hist_window='episode',
        **{**common, 'n_obs_steps': 3, 'pad_before': 2})
    i3, _, _, _, _ = _find_row(ds3, buffer_start=8)
    p3, _, _ = ds3._eef_history(i3)
    np.testing.assert_array_equal(
        p3, pos_all[_expected_history([8, 9, 10], 5, 4)])

    # the default stays the legacy rule
    dsd = MultiViewImageDataset(**common)
    assert dsd.eef_hist_window == 'sample'
    pd, _, _ = dsd._eef_history(idx)
    np.testing.assert_array_equal(
        pd, pos_all[np.array([[8, 8, 8, 8], [8, 8, 8, 9]])])

    # emitted history is the camera-frame transform of `_eef_history`,
    # flattened step-major, and paired with the view that supplied the image
    dsc = MultiViewImageDataset(
        eef_hist_window='episode',
        **{**common, 'view_count_range': [1, 1]})
    ci, _, _, _, _ = _find_row(dsc, buffer_start=8)
    batch = dsc[ci]['obs']
    hist = batch['view_eef_hist'].numpy()[:, 0]       # (To, 44)
    cam = batch[dsc.cam_keys[0]].numpy()[0]
    v = next(v for v in range(N_VIEWS)
             if np.allclose(cam, dsc.cam_table[v], atol=1e-6))
    p, q, g = dsc._eef_history(ci)
    want_hist = eef_hist_to_cam(
        p, q, g, dsc.cam_table[v][:3], dsc.cam_table[v][3:7]).reshape(2, -1)
    np.testing.assert_allclose(hist, want_hist, atol=1e-6)

    # rollout-deque equivalence at every frame of the first episode. This is
    # the check the original dataset would fail from frame 1 onward.
    buf = [0] * h
    for f in range(h + 1):
        if f > 0:
            buf = (buf + [f])[-h:]
        j, _, _, _, _ = _find_row(ds, buffer_start=f, sample_start=0)
        pj, _, _ = ds._eef_history(j)
        np.testing.assert_array_equal(pj[0], pos_all[np.array(buf)])

    print('EE-history windows OK (sample vs episode, start/tail/val, '
          'H>episode, To>2, camera frame, rollout deque)')


def test_eef_hist_base(path, shape_meta):
    """RH/RHG's base-frame trajectory key: frame, content, ordering, flags.

    The per-slot camera-frame history and this per-sample base-frame window
    must be the SAME window (only the frame differs), the key must be
    view-independent, its last step must be the observation frame's own EE
    pose (RHG's bias anchor and RH's relative-position reference), and the
    flag must default off so every committed run's obs dict is unchanged.
    """
    from diffusion_policy.dataset.multiview_image_dataset import eef_hist_base

    common = dict(
        shape_meta=shape_meta, dataset_path=path, horizon=4,
        pad_before=1, pad_after=7, n_obs_steps=2, abs_action=True,
        view_pool=[0, 1, 2], view_count_range=[1, 3],
        eef_hist_steps=4, eef_hist_window='episode', seed=0, val_ratio=0.34)

    # default off: the key is absent and nothing else changes
    dso = MultiViewImageDataset(**common)
    assert dso.emit_eef_hist_base is False
    assert 'eef_hist_base' not in dso[0]['obs']

    ds = MultiViewImageDataset(emit_eef_hist_base=True, **common)
    assert ds.emit_eef_hist_base is True
    assert ds.eef_hist_base_key == 'eef_hist_base'
    idx, b, _, _, _ = _find_row(ds, buffer_start=8)
    obs = ds[idx]['obs']
    base = obs['eef_hist_base']
    assert tuple(base.shape) == (2, 4, 9), base.shape

    # same window as _eef_history, base frame, no gripper
    pos, quat, grip = ds._eef_history(idx)
    want = eef_hist_base(pos, quat)
    np.testing.assert_allclose(base.numpy(), want, atol=1e-6)

    # the rot6d is rows 0,1 of R_eef (the shared convention), not columns
    R = quat_wxyz_to_mat_batch(quat)
    np.testing.assert_allclose(
        base.numpy()[..., 3:], np.concatenate([R[..., 0, :], R[..., 1, :]],
                                              axis=-1), atol=1e-6)
    # a non-identity rotation is what makes rows-vs-columns visible
    assert not np.allclose(R[..., 0, :], R[..., :, 0], atol=1e-3)

    # ordering: oldest first, current last, and the last step IS the obs
    # frame's own EE position (the window `_eef_history` returns)
    np.testing.assert_allclose(
        base.numpy()[:, -1, :3], obs['robot0_eef_pos'].numpy(), atol=1e-6)
    pos_all = np.asarray(ReplayBuffer.create_from_path(path, mode='r')
                         ['robot0_eef_pos'][:])
    np.testing.assert_allclose(base.numpy()[0, -1, :3], pos_all[b], atol=1e-6)

    # view-independent: redrawing the view subset must not change it
    np.random.seed(0)
    a = ds[idx]['obs']['eef_hist_base']
    np.random.seed(999)
    c = ds[idx]['obs']['eef_hist_base']
    assert torch.equal(a, c)

    # identity normalizer (raw metres / raw rot6d reach the encoder)
    params = ds.get_normalizer()['eef_hist_base'].params_dict
    assert bool((params['scale'] == 1).all())
    assert bool((params['offset'] == 0).all())
    assert tuple(params['scale'].shape) == (9,)

    # the validation dataset inherits the flag and the key
    val = ds.get_validation_dataset()
    assert val.emit_eef_hist_base is True
    assert 'eef_hist_base' in val[0]['obs']

    # guards
    try:
        MultiViewImageDataset(
            emit_eef_hist_base=True,
            **{**common, 'eef_hist_base_key': 'view_eef_hist'})
        raise AssertionError('sharing the camera-frame key should raise')
    except ValueError:
        pass
    sm = {**shape_meta, 'obs': {**shape_meta['obs'],
                                'eef_hist_base': {'shape': [36]}}}
    try:
        MultiViewImageDataset(emit_eef_hist_base=True, **{**common,
                                                          'shape_meta': sm})
        raise AssertionError('a shape_meta base key should raise')
    except ValueError:
        pass
    try:
        MultiViewImageDataset(
            emit_eef_hist_base=True,
            **{k: v for k, v in common.items() if k != 'view_pool'})
        raise AssertionError('emit without view_pool should raise')
    except ValueError:
        pass

    print('EE-history base window OK (frame, rot6d rows, ordering, '
          'view independence, identity normalizer, val, guards)')


def test():
    tmp = tempfile.mkdtemp(prefix='multiview_test_')
    try:
        path = os.path.join(tmp, 'synthetic.zarr')
        shape_meta, episode_ends = build_synthetic_zarr(path)
        test_camera_math()
        test_dataset(path, shape_meta, episode_ends)
        test_eef_history(path, shape_meta)
        test_eef_hist_base(path, shape_meta)
        print('ALL MULTIVIEW DATASET CHECKS PASSED')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test()
