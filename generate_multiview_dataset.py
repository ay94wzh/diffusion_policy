"""
Generate a multi-view image dataset from a robomimic image hdf5 (Milestone 2).

For every demo timestep the scene state is restored (via the hdf5's stored
`states` + `env.reset_to`) and re-rendered from N fixed camera poses obtained by
orbiting the robot's fixed camera. Images, low-dim obs and base-frame actions are
written to a ReplayBuffer-compatible zarr; per-view camera parameters (pose,
fovy) go into `meta` so downstream code can build Plücker maps (M3) and
camera-frame actions (M4) without re-rendering.

Design notes (each verified against the installed source -- see PROGRESS.md):
- Only images are re-rendered; low-dim obs and actions are copied verbatim from
  the hdf5, with actions converted to the 10-dim absolute form (pos + rotation_6D
  + gripper) exactly as the stock robomimic dataset does for `abs_action=True`.
- Rendering uses `sim.render(camera_name=...)` directly and applies **exactly one
  `[::-1]`**: mujoco's readPixels is bottom-up and robomimic's EnvRobosuite flips
  it (env_robosuite.py:172-207), which is the orientation stored in the hdf5.
  Skipping the flip trains on upside-down images; the az_0 gate below catches it.
- `camera_names` is restricted to the one camera we drive, so robosuite builds a
  single camera sensor instead of all of them (robot_env.py:155-161).
- Camera poses are computed from the pristine base pose with the same helper the
  M1 novel-view harness uses, so view "+30" here is exactly M1's `az_p30`.

Usage:
python generate_multiview_dataset.py \
  --dataset data/robomimic/datasets/square/ph/image_abs.hdf5 \
  --output data/multiview/square_ph_ring13.zarr \
  --montage /tmp/ring13.png
python generate_multiview_dataset.py --dataset ... --output ... --limit-demos 5
"""

import os
import sys
import pathlib
import multiprocessing
import concurrent.futures

# bootstrap: make repo root importable regardless of cwd
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import click
import numpy as np
import h5py
import zarr
import numcodecs
import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.robomimic_replay_image_dataset import _convert_actions
from diffusion_policy.dataset.multiview_image_dataset import (
    project_world_to_pixel, fovy_to_intrinsics)
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from eval_novel_view import _compute_perturbed_pose
import robomimic.utils.file_utils as FileUtils

register_codecs()

# every 15 degrees out to +-90 (13 views), sorted ascending so index order is
# stable; azimuth 0 is the original dataset camera pose
DEFAULT_VIEW_AZIMUTHS = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
DEFAULT_LOWDIM_KEYS = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']


def view_key(view_idx):
    """Zero-padded so alphabetical order (what MultiImageObsEncoder sorts by)
    equals numeric view order."""
    return f'view_{view_idx:02d}_image'


def make_compressor(codec):
    if codec == 'jpeg2k':
        return Jpeg2k(level=50)          # matches the stock hdf5->zarr cache
    elif codec == 'blosc':
        return numcodecs.Blosc(cname='zstd', clevel=5,
            shuffle=numcodecs.Blosc.BITSHUFFLE)
    elif codec == 'none':
        return None
    raise ValueError(f'unknown codec {codec}')


def resolve_render_size(env_meta, camera):
    """Render resolution the env uses for `camera` (heights/widths may be ints or
    per-camera lists)."""
    kwargs = env_meta['env_kwargs']
    heights = kwargs.get('camera_heights', 84)
    widths = kwargs.get('camera_widths', 84)
    names = kwargs.get('camera_names', ['agentview'])
    if isinstance(names, str):
        names = [names]
    names = list(names)
    idx = names.index(camera) if camera in names else 0
    if isinstance(heights, (list, tuple)):
        heights = heights[idx]
    if isinstance(widths, (list, tuple)):
        widths = widths[idx]
    return int(heights), int(widths)


def build_env(dataset_path, camera, shape_meta):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False
    # render only the camera we drive
    env_meta['env_kwargs']['camera_names'] = [camera]
    env = create_env(env_meta=env_meta, shape_meta=shape_meta)
    env.env.hard_reset = False   # keep camera model fields across soft resets
    return env


def find_eef_site(sim):
    """Best-effort name of the gripper/eef site, for the projection sanity check."""
    try:
        names = [sim.model.site_id2name(i) for i in range(sim.model.nsite)]
    except Exception:
        return None
    for wanted in ('grip_site', 'grip', 'eef', 'hand'):
        for name in names:
            if wanted in name:
                return name
    return None


def _write_block(zarr_arr, idx_slice, data):
    """Encode one (view, demo) block; read it back to make sure it decodes."""
    try:
        zarr_arr[idx_slice] = data
        _ = zarr_arr[idx_slice]
        return True
    except Exception:
        return False


def _apply_view(sim, cid, pos, quat_wxyz):
    sim.model.cam_pos[cid] = pos
    sim.model.cam_quat[cid] = quat_wxyz
    sim.forward()


def _render_views(sim, cid, view_poses, camera_name, h, w):
    """Render every view of the *current* sim state -> (n_views, h, w, 3) uint8."""
    out = np.empty((len(view_poses), h, w, 3), dtype=np.uint8)
    for v_idx, (pos, quat) in enumerate(view_poses):
        _apply_view(sim, cid, pos, quat)
        img = sim.render(camera_name=camera_name, width=w, height=h, depth=False)
        out[v_idx] = img[::-1]   # match robomimic's vertical flip, exactly once
    return out


def _draw_cross(img, u, v):
    """Red crosshair at (u, v) if it lands inside the image."""
    h, w = img.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return img
    img = img.copy()
    for d in range(-3, 4):
        if 0 <= ui + d < w:
            img[vi, ui + d] = (255, 0, 0)
        if 0 <= vi + d < h:
            img[vi + d, ui] = (255, 0, 0)
    return img


def write_gates(env, sim, cid, camera, view_poses, azimuths, demo_states,
        dataset_path, demo_idx, step_idxs, h, w, fovy, montage_path):
    """
    Post-generation checks, run while the env is still alive:
      1. az_0 re-render vs the hdf5's stored agentview image (orientation/resolution)
      2. geometry: project the gripper site through our intrinsics/extrinsics
      3. a ring montage with the projected point drawn on it, for eyeballing
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_views = len(view_poses)
    az0_idx = int(np.argmin(np.abs(np.asarray(azimuths, dtype=float))))
    intrinsics = fovy_to_intrinsics(fovy, h, w)
    site = find_eef_site(sim)

    # reference images (may be absent for low-dim-only hdf5s)
    hdf5_images = None
    with h5py.File(dataset_path, 'r') as f:
        demo = f['data'][f'demo_{demo_idx}']
        if 'obs/agentview_image' in demo:
            hdf5_images = demo['obs/agentview_image'][:]

    grid = np.zeros((len(step_idxs), n_views, h, w, 3), dtype=np.uint8)
    ok_in_frame = 0
    ok_total = 0
    az0_diffs = []
    for row, t in enumerate(step_idxs):
        env.reset_to({'states': demo_states[t]})
        grid[row] = _render_views(sim, cid, view_poses, camera, h, w)

        if hdf5_images is not None:
            a = grid[row, az0_idx].astype(np.float32)
            b = hdf5_images[t].astype(np.float32)
            if a.shape == b.shape:
                az0_diffs.append(float(np.abs(a - b).mean()))

        if site is not None:
            try:
                p_world = np.array(sim.data.get_site_xpos(site), dtype=np.float64)
            except Exception:
                site = None
                p_world = None
            if p_world is not None:
                for v_idx, (pos, quat) in enumerate(view_poses):
                    u, v, depth = project_world_to_pixel(p_world, pos, quat, intrinsics)
                    ok_total += 1
                    if depth > 0 and 0 <= u < w and 0 <= v < h:
                        ok_in_frame += 1
                    if v_idx == az0_idx:
                        grid[row, v_idx] = _draw_cross(grid[row, v_idx], u, v)

    if montage_path is not None:
        pathlib.Path(montage_path).parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(len(step_idxs), n_views,
            figsize=(1.15 * n_views, 1.15 * len(step_idxs)), squeeze=False)
        for row, t in enumerate(step_idxs):
            for v_idx in range(n_views):
                ax = axes[row][v_idx]
                ax.imshow(grid[row, v_idx])
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(f'{int(azimuths[v_idx]):+d}', fontsize=6)
                if v_idx == 0:
                    ax.set_ylabel(f't={t}', fontsize=6)
        fig.suptitle('ring views (crosshair = projected gripper site, drawn on the '
                     '0-deg view of row 1)', fontsize=8)
        fig.tight_layout()
        fig.savefig(montage_path, dpi=110)
        plt.close(fig)
        print(f'[gate 3] wrote montage {montage_path}')

    if az0_diffs:
        mean_d = float(np.mean(az0_diffs))
        verdict = 'PASS' if mean_d < 3.0 else 'FAIL'
        print(f'[gate 1] az_0 re-render vs hdf5 stored agentview: '
              f'mean|diff| = {mean_d:.3f}/255 over {len(az0_diffs)} steps -> {verdict}')
        if verdict == 'FAIL':
            print('         expected < 3. A mean of ~40+ usually means the '
                  '[::-1] flip is missing or doubled.')
    else:
        print('[gate 1] skipped: source hdf5 has no obs/agentview_image')

    if ok_total:
        print(f'[gate 2] projected gripper site in frame for '
              f'{ok_in_frame}/{ok_total} (view, step) pairs (site={site!r})')
    else:
        print('[gate 2] skipped: no gripper/eef site found in the model')


@click.command()
@click.option('--dataset', required=True, help='source robomimic image hdf5')
@click.option('--output', required=True, help='output zarr directory store')
@click.option('--camera', default='agentview', help='fixed camera to orbit')
@click.option('--views', default=None,
    help='comma-separated azimuths in degrees (default: ring every 15 deg to +-90)')
@click.option('--lowdim-keys', default=','.join(DEFAULT_LOWDIM_KEYS),
    help='low-dim obs keys copied verbatim from the hdf5')
@click.option('--limit-demos', type=int, default=None)
@click.option('--workers', type=int, default=None)
@click.option('--codec', type=click.Choice(['jpeg2k', 'blosc', 'none']), default='jpeg2k')
@click.option('--montage', default=None, help='write a gate montage PNG here')
@click.option('--overwrite', is_flag=True, help='allow writing into an existing output')
def main(dataset, output, camera, views, lowdim_keys, limit_demos, workers, codec,
        montage, overwrite):
    dataset = os.path.expanduser(dataset)
    output = os.path.expanduser(output)
    lowdim_keys = [k.strip() for k in lowdim_keys.split(',') if k.strip()]
    azimuths = ([float(v) for v in views.split(',')] if views is not None
                else list(DEFAULT_VIEW_AZIMUTHS))
    n_views = len(azimuths)

    if os.path.exists(output) and not overwrite:
        raise click.ClickException(f'{output} exists; pass --overwrite to replace it')
    if workers is None:
        workers = multiprocessing.cpu_count()

    # minimal shape_meta: only tells robomimic which obs keys are images
    probe_meta = FileUtils.get_env_metadata_from_dataset(dataset)
    render_h, render_w = resolve_render_size(probe_meta, camera)
    shape_meta = {
        'obs': {f'{camera}_image': {'shape': [3, render_h, render_w], 'type': 'rgb'}},
        'action': {'shape': [10]},
    }

    print(f'source     : {dataset}')
    print(f'output     : {output}')
    print(f'camera     : {camera}  render {render_h}x{render_w}  codec={codec}')
    print(f'views ({n_views}) : {[int(a) for a in azimuths]}')

    env = build_env(dataset, camera, shape_meta)
    sim = env.env.sim
    cid = sim.model.camera_name2id(camera)
    base_pos = np.array(sim.model.cam_pos[cid], dtype=np.float64)
    base_quat = np.array(sim.model.cam_quat[cid], dtype=np.float64)
    fovy = float(sim.model.cam_fovy[cid])
    assert abs(float(np.linalg.norm(base_quat)) - 1.0) < 1e-3, 'base cam quat not unit'
    print(f'base cam_pos : {np.round(base_pos, 4)}  fovy={fovy:.1f}deg')

    # poses are a pure function of the base pose -> compute once, never compound
    view_poses = [_compute_perturbed_pose(base_pos, base_quat, {'azimuth_deg': a})
                  for a in azimuths]
    if 0.0 in azimuths:   # az_0 must be a no-op
        z = azimuths.index(0.0)
        assert np.allclose(view_poses[z][0], base_pos)
        assert np.allclose(view_poses[z][1], base_quat)

    env.reset()   # renders are only correct after a full reset

    # ------------------------------------------------------------------
    # write the zarr
    # ------------------------------------------------------------------
    compressor = make_compressor(codec)
    store = zarr.DirectoryStore(output)
    root = zarr.group(store, overwrite=True)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset, 'r') as f:
        demos = f['data']
        n_demos = len(demos) if limit_demos is None else min(limit_demos, len(demos))
        lengths = [demos[f'demo_{i}']['actions'].shape[0] for i in range(n_demos)]
        episode_ends = np.cumsum(lengths).astype(np.int64)
        episode_starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
        n_steps = int(episode_ends[-1])
        meta_group.array('episode_ends', episode_ends, dtype=np.int64,
            compressor=None, overwrite=True)

        # camera parameters (constant per view) + provenance
        meta_group.array('view_azimuth_deg', np.asarray(azimuths, dtype=np.float64),
            dtype=np.float64, compressor=None, overwrite=True)
        meta_group.array('view_cam_pos', np.stack([p for p, _ in view_poses]),
            dtype=np.float64, compressor=None, overwrite=True)
        meta_group.array('view_cam_quat_wxyz', np.stack([q for _, q in view_poses]),
            dtype=np.float64, compressor=None, overwrite=True)
        meta_group.array('view_fovy_deg', np.full(n_views, fovy, dtype=np.float64),
            dtype=np.float64, compressor=None, overwrite=True)
        meta_group.array('base_cam_pos', base_pos, dtype=np.float64,
            compressor=None, overwrite=True)
        meta_group.array('base_cam_quat_wxyz', base_quat, dtype=np.float64,
            compressor=None, overwrite=True)
        meta_group.array('render_hw', np.array([render_h, render_w], dtype=np.int64),
            dtype=np.int64, compressor=None, overwrite=True)

        # low-dim obs + action (copied / converted, not re-derived)
        for key in tqdm.tqdm(lowdim_keys + ['action'], desc='lowdim'):
            data_key = 'obs/' + key if key != 'action' else 'actions'
            this_data = np.concatenate(
                [demos[f'demo_{i}'][data_key][:].astype(np.float32)
                 for i in range(n_demos)], axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=True,
                    rotation_transformer=RotationTransformer(
                        from_rep='axis_angle', to_rep='rotation_6d'))
                print(f'  action: {this_data.shape} (abs, rotation_6d)')
            else:
                print(f'  {key}: {this_data.shape}')
            data_group.array(
                name=key, data=this_data, shape=this_data.shape,
                chunks=this_data.shape, compressor=None, dtype=this_data.dtype)

        # images: one pre-allocated array per view, filled a (demo, view) block at
        # a time on worker threads (the sim itself is not thread safe)
        img_arrs = [data_group.require_dataset(
                        name=view_key(v), shape=(n_steps, render_h, render_w, 3),
                        chunks=(1, render_h, render_w, 3),
                        compressor=compressor, dtype=np.uint8)
                    for v in range(n_views)]

        gate_demo = 0
        gate_len = int(lengths[gate_demo])
        gate_steps = sorted(set([0, gate_len // 2, gate_len - 1]))

        max_inflight = workers * 5
        futures = set()

        def drain(blocking_all=False):
            nonlocal futures
            if blocking_all:
                done, futures = concurrent.futures.wait(futures)
            else:
                done, futures = concurrent.futures.wait(futures,
                    return_when=concurrent.futures.FIRST_COMPLETED)
            for fu in done:
                if not fu.result():
                    raise RuntimeError('failed to encode an image block')

        with tqdm.tqdm(total=n_steps, desc='render + write', mininterval=2.0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                for demo_idx in range(n_demos):
                    states = demos[f'demo_{demo_idx}']['states'][:]
                    T = states.shape[0]
                    start = int(episode_starts[demo_idx])
                    buf = np.empty((T, n_views, render_h, render_w, 3), dtype=np.uint8)
                    for t in range(T):
                        env.reset_to({'states': states[t]})
                        buf[t] = _render_views(sim, cid, view_poses, camera,
                            render_h, render_w)
                    for v_idx in range(n_views):
                        if len(futures) >= max_inflight:
                            drain()
                        futures.add(ex.submit(_write_block, img_arrs[v_idx],
                            slice(start, start + T), buf[:, v_idx]))
                    pbar.update(T)
                drain(blocking_all=True)

    replay_buffer = ReplayBuffer(root)   # validates the schema
    print(f'wrote {replay_buffer.n_episodes} episodes, {replay_buffer.n_steps} '
          f'steps, {n_views} views -> {output}')

    # ------------------------------------------------------------------
    # gates (env still alive)
    # ------------------------------------------------------------------
    with h5py.File(dataset, 'r') as f:
        gate_states = f['data'][f'demo_{gate_demo}']['states'][:]
    write_gates(env, sim, cid, camera, view_poses, azimuths, gate_states,
        dataset, gate_demo, gate_steps, render_h, render_w, fovy, montage)

    # robomimic's EnvRobosuite wrapper has no close(); the robosuite env it
    # holds does. Calling this matters because a clean exit status is how the
    # full-run driver distinguishes a finished run from a crashed one.
    env.env.close()


if __name__ == '__main__':
    main()
