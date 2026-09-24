"""Render a candidate viewpoint set to a montage, BEFORE running any sweep.

Why this exists
---------------
A bad viewpoint set does not crash -- it produces a plausible-looking degradation
curve. `NOTES.md` records that M2's gates only ran at the *end* of an 8-hour job,
and that the `--limit-demos 5` pilot became the mitigation. This is
the same idea for evaluation viewpoints, which are cheap to render and expensive
to get wrong: it takes about a minute instead of an evening.

It also answers the requirement that drove the elevation preset: **every view must
contain relatively complete information** (robot AND table in frame). That is
checked two ways -- a montage to look at, and two numbers per viewpoint.

The numbers
-----------
* `gripper in frame` -- the projection check M2's gate 2 already validated against
  the simulator. Directly answers "can you see the robot".
* `coverage` -- the fraction of pixels differing from the frame's dominant colour.
  This is a *proxy* for "how much scene is in frame", not ground truth: it assumes
  the background occupies a large share of the frame and is fairly uniform. Treat
  the montage as the evidence and this as a sorting aid.

Usage
-----
    # preview the elevation eval preset (the +-15 orbit)
    python preview_viewpoints.py --dataset data/robomimic/datasets/square/ph/image_abs.hdf5 \
        --preset elevation_az0 --montage data/preview_elevation.png

    # preview a training ring, to document the +-90 weakness PROGRESS 7.1 found
    python preview_viewpoints.py --dataset <hdf5> --azimuths -90,-60,-30,0,30,60,90 \
        --montage data/preview_ring7.png
"""
import os
import sys
import pathlib

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import click
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eval_novel_view import PRESETS, _compute_perturbed_pose
from generate_multiview_dataset import (
    build_env, find_eef_site, resolve_render_size, _draw_cross)
from diffusion_policy.dataset.multiview_image_dataset import (
    fovy_to_intrinsics, project_world_to_pixel)
import robomimic.utils.file_utils as FileUtils


def az_name(deg):
    """az_0 / az_p15 / az_m15 -- matching the eval harness's viewpoint names."""
    d = int(deg)
    if d == 0:
        return 'az_0'
    return f'az_p{d}' if d > 0 else f'az_m{-d}'


def scene_coverage(img, thresh=20):
    """Fraction of pixels differing from the frame's dominant colour.

    A proxy for how much scene is in frame (see the module docstring): the mode
    per channel estimates the background, and pixels further than `thresh` (summed
    over channels, out of 765) count as scene.
    """
    flat = img.reshape(-1, 3).astype(np.int16)
    bg = np.array([np.bincount(flat[:, c], minlength=256).argmax()
                   for c in range(3)], dtype=np.int16)
    return float((np.abs(flat - bg).sum(axis=1) > thresh).mean())


def resolve_viewpoints(preset, azimuths):
    """-> [(name, spec)], from an eval preset or an explicit azimuth ring."""
    if preset is not None:
        p = PRESETS[preset]
        out = []
        for vp in p['viewpoints']:
            spec = dict(vp)
            spec['camera'] = p['camera']
            out.append((vp['name'], spec))
        return out
    return [(az_name(a), {'azimuth_deg': float(a)}) for a in azimuths]


@click.command()
@click.option('--dataset', required=True, help='source robomimic image hdf5 (env + states)')
@click.option('--preset', default=None, type=click.Choice(list(PRESETS.keys())),
    help='preview an eval preset viewpoint set')
@click.option('--azimuths', default=None,
    help='comma-separated azimuths, to preview a ring (instead of --preset)')
@click.option('--camera', default='agentview', help='fixed camera to move')
@click.option('--demo', type=int, default=0, help='which demo to sample states from')
@click.option('--n-states', type=int, default=3, help='states to render per viewpoint')
@click.option('--montage', default='data/preview_viewpoints.png')
@click.option('--min-coverage', type=float, default=0.15,
    help='flag a viewpoint whose scene coverage falls below this')
def main(dataset, preset, azimuths, camera, demo, n_states, montage, min_coverage):
    if preset is None and azimuths is None:
        raise click.UsageError('pass either --preset or --azimuths')
    if preset is not None and azimuths is not None:
        raise click.UsageError('--preset and --azimuths are mutually exclusive')
    az_list = ([float(v) for v in azimuths.split(',')] if azimuths is not None else None)
    views = resolve_viewpoints(preset, az_list)

    dataset = os.path.expanduser(dataset)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset)
    render_h, render_w = resolve_render_size(env_meta, camera)
    # minimal shape_meta: it only tells robomimic which obs keys are images
    shape_meta = {
        'obs': {f'{camera}_image': {'shape': [3, render_h, render_w], 'type': 'rgb'}},
        'action': {'shape': [10]},
    }

    label = preset if preset is not None else f'ring({len(views)})'
    print(f'source : {dataset}')
    print(f'camera : {camera}  render {render_h}x{render_w}  demo={demo}')
    print(f'views  : {label} -> {[n for n, _ in views]}')

    env = build_env(dataset, camera, shape_meta)
    sim = env.env.sim
    cid = sim.model.camera_name2id(camera)
    base_pos = np.array(sim.model.cam_pos[cid], dtype=np.float64)
    base_quat = np.array(sim.model.cam_quat[cid], dtype=np.float64)
    fovy = float(sim.model.cam_fovy[cid])
    intrinsics = fovy_to_intrinsics(fovy, render_h, render_w)
    site = find_eef_site(sim)
    print(f'base pos {np.round(base_pos, 4)}  fovy {fovy:.1f}  eef site {site!r}')

    poses = []
    for name, spec in views:
        pos, quat = _compute_perturbed_pose(base_pos, base_quat, spec)
        poses.append((name, np.asarray(pos, dtype=np.float64),
                      np.asarray(quat, dtype=np.float64)))
        # a viewpoint carrying no offset must actually be a no-op
        if not any(k in spec for k in
                   ('azimuth_deg', 'elevation_deg', 'elevation_orbit_deg',
                    'pos_delta', 'euler_delta_deg')):
            assert np.allclose(pos, base_pos) and np.allclose(quat, base_quat), \
                f'{name} is the identity viewpoint but moved the camera'

    with h5py.File(dataset, 'r') as f:
        demo_grp = f['data'][f'demo_{demo}']
        states = demo_grp['states'][:]
    n_states = max(1, min(int(n_states), len(states)))
    step_idxs = np.linspace(0, len(states) - 1, n_states).astype(int)

    grid = np.zeros((n_states, len(poses), render_h, render_w, 3), dtype=np.uint8)
    coverage = np.zeros(len(poses))
    in_frame = np.zeros(len(poses), dtype=int)
    for row, t in enumerate(step_idxs):
        env.reset_to({'states': states[t]})
        for col, (name, pos, quat) in enumerate(poses):
            sim.model.cam_pos[cid] = pos
            sim.model.cam_quat[cid] = quat
            sim.forward()
            img = sim.render(camera_name=camera, width=render_w,
                             height=render_h, depth=False)[::-1]
            p_world = None
            if site is not None:
                try:
                    p_world = np.array(sim.data.get_site_xpos(site), dtype=np.float64)
                except Exception:
                    p_world = None
            if p_world is not None:
                u, v, depth = project_world_to_pixel(p_world, pos, quat, intrinsics)
                if depth > 0 and 0 <= u < render_w and 0 <= v < render_h:
                    in_frame[col] += 1
                img = _draw_cross(img, u, v)
            grid[row, col] = img
            if row == 0:
                coverage[col] = scene_coverage(img)

    print()
    print(f'{"viewpoint":<12}{"coverage":>10}{"gripper in frame":>18}   verdict')
    bad = []
    for col, (name, _, _) in enumerate(poses):
        ok = coverage[col] >= min_coverage
        if not ok:
            bad.append(name)
        print(f'{name:<12}{coverage[col]:>10.3f}{in_frame[col]:>10}/{n_states:<7}'
              f'   {"ok" if ok else "LOW COVERAGE"}')
    if site is None:
        print('note: no gripper/eef site found, so the in-frame column is all zeros')

    pathlib.Path(montage).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n_states, len(poses),
        figsize=(1.35 * len(poses), 1.35 * n_states), squeeze=False)
    for row, t in enumerate(step_idxs):
        for col, (name, _, _) in enumerate(poses):
            ax = axes[row][col]
            ax.imshow(grid[row, col])
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f'{name}\ncov {coverage[col]:.2f}', fontsize=7)
            if col == 0:
                ax.set_ylabel(f't={t}', fontsize=7)
    fig.suptitle(f'{" ".join([label])} viewpoints (crosshair = projected gripper)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(montage, dpi=120)
    plt.close(fig)
    print(f'\nwrote {montage}')

    if bad:
        print(f'WARNING: low scene coverage at {bad}; those views likely cannot '
              f'support the task (see the M2 section of PROGRESS.md on the +-90 ring views)')
    else:
        print('all viewpoints clear the coverage floor')


if __name__ == '__main__':
    main()
