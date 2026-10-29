"""Calibrate RHG's locked geometric-bias sigma on the training zarr.

RHG (``ENCODER_EXPERIMENT_PLAN.md`` section 4.4) adds, on the first
``geom_bias_heads`` of the intra-image attention's heads::

    b_{v,j} = -alpha * min( rho_{v,j}^2 / (2 sigma^2), b_max )

where ``rho`` is the distance from the CURRENT EE point ``p_t`` to token
``j``'s Pluecker ray, in metres. ``alpha = 1`` and ``b_max = 4`` are fixed by
the plan; **sigma is not learned** -- it is calibrated here on the training set
and then LOCKED into the run as a config override, so that the bias's spatial
reach is set once, before any result exists, and is identical for every epoch
and seed.

What sigma is calibrated to
---------------------------
"About 1-2 feature-grid neighbourhoods": for every (state, observation step,
live view) whose EE point projects INSIDE the sampled crop -- the views the
bias is allowed to act on at run time -- find the token nearest the projection
and measure the point-to-ray distance to its 1-cell and 2-cell neighbours
(the 4-neighbourhood; 2-cell neighbours only along the row/column). **sigma is
the median 1-cell distance.** At that sigma the nearest token's bias is ~0,
a 1-cell neighbour sits near ``-alpha/2``, and a 2-cell neighbour near
``-2 alpha`` (capped at ``-b_max``), i.e. the bias is a soft, local focus on
the rays near the current EE -- not a hard mask.

The crop used here is the deterministic CENTRE crop (the eval-mode window).
Training draws random crops, but a crop only shifts which window the tokens
sample; the point-to-token spacing that sigma measures is a property of the
image scale and the 16-px feature stride, not of the offset.

The geometry is the encoder's OWN code (``_feature_grid_rays``,
``point_to_ray_distances``, ``project_point_to_pixel``), so the calibrated
value and the run-time bias cannot drift apart. No checkpoint is needed: the
rays and the projection depend on cameras and pose only, never on weights.

Usage (on miroc-server -- needs the generated multi-view zarr)::

    python calibrate_geom_bias.py -c train_diffusion_unet_image_workspace_enc_rh_latent \\
        --n-samples 256 --out data/geom_bias_calib_square.json

then pass the printed value to the RHG run::

    policy.obs_encoder.geom_bias_sigma_m=<GEOM_BIAS_SIGMA>

Outputs a JSON with the statistic, its quantiles and the implied bias at 1/2
cells, plus the shell line ``GEOM_BIAS_SIGMA=<value>`` for the driver log.
"""
import os
import sys
import json
import pathlib

if __name__ == '__main__':
    sys.path.append(str(pathlib.Path(__file__).parent))

import click
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.model.vision.traj_spatial_readout_obs_encoder import (
    point_to_ray_distances, project_point_to_pixel)


def _quantiles(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return None
    return {
        'n': int(v.size),
        'mean': float(v.mean()),
        'median': float(np.median(v)),
        'p10': float(np.percentile(v, 10)),
        'p90': float(np.percentile(v, 90)),
        'min': float(v.min()),
        'max': float(v.max()),
    }


def _implied_bias(rho_median, sigma, alpha, b_max):
    if rho_median is None:
        return None
    return float(-alpha * min((rho_median ** 2) / (2.0 * sigma ** 2), b_max))


def calibrate(cfg, n_samples=256, seed=42, num_threads=4, config_name=None):
    """Run the calibration on a composed config; returns the summary dict.

    Split out of the CLI so a CPU test can drive the same code path on a
    synthetic zarr (the geometry, not the CLI, is what must be verified).
    ``cfg`` needs ``task.dataset`` and ``policy.obs_encoder``, as any of the
    RH/RHG workspace configs provides.
    """
    torch.set_num_threads(int(num_threads))

    # --- the dataset, forced to a full draw so every state sees all views --
    ds_conf = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=True))
    pool = [int(v) for v in ds_conf.view_pool]
    ds_conf.view_count_range = [len(pool), len(pool)]
    ds_conf.val_ratio = 0.0
    dataset = hydra.utils.instantiate(ds_conf)
    if ds_conf.get('emit_eef_hist_base', False) is not True:
        raise click.ClickException(
            'the dataset config does not emit eef_hist_base; RHG calibration '
            'needs the base-frame trajectory window (emit_eef_hist_base: true)')

    # --- the encoder's geometry, without any bias (sigma is what we compute) --
    enc_conf = OmegaConf.create(OmegaConf.to_container(cfg.policy.obs_encoder, resolve=True))
    enc_conf.use_geom_bias = False
    enc_conf.pop('geom_bias_sigma_m', None)
    enc = hydra.utils.instantiate(enc_conf)
    enc.eval()

    base_key = enc.eef_hist_key
    crop = enc.crop_shape
    if crop is None:
        raise click.ClickException(
            'this calibration assumes a crop (crop_shape); the rays are placed '
            'on the crop window and the validity gate tests the crop bounds')
    ch, cw = int(crop[0]), int(crop[1])
    stride = int(enc.total_stride)
    gh, gw = enc.token_grid
    alpha = float(enc_conf.get('geom_bias_alpha', 1.0))
    b_max = float(enc_conf.get('geom_bias_max', 4.0))
    if int(enc_conf.get('geom_bias_heads', 4)) <= 0:
        raise click.ClickException(
            'geom_bias_heads is 0: the bias acts on no head, sigma is moot')

    rng = np.random.RandomState(int(seed))
    n = int(min(n_samples, len(dataset)))
    state_idx = np.sort(rng.choice(len(dataset), size=n, replace=False))

    rho_1, rho_2, rho_min = [], [], []
    n_valid = n_skipped = 0
    n_frames = 0
    for i in state_idx.tolist():
        obs = dataset[i]['obs']
        hist = obs[base_key].numpy().astype(np.float64)     # (To, H, 9)
        mask = obs[enc.view_mask_key].numpy()
        to_n = hist.shape[0]
        for to in range(to_n):
            p = hist[to, -1, :3]                             # current EE, metres
            for k in range(enc.n_slots):
                if mask[to, k] <= 0.5:
                    continue
                n_frames += 1
                cam = obs[enc.cam_keys[k]][to].numpy().astype(np.float64)
                img_h = int(cam[8])
                img_w = int(cam[9])
                off_h, off_w = (img_h - ch) // 2, (img_w - cw) // 2
                inds = torch.tensor([[off_h, off_w]], dtype=torch.long)
                cam_t = torch.from_numpy(cam[None]).float()
                p_t = torch.from_numpy(p[None]).float()
                with torch.no_grad():
                    rays = enc._feature_grid_rays(cam_t, inds, img_h, img_w)  # (1,G,6)
                    u, v, depth = project_point_to_pixel(cam_t, p_t, img_h, img_w)
                uu, vv, dd = (float(u[0]), float(v[0]), float(depth[0]))
                inside = (dd > 1e-6 and off_w <= uu < off_w + cw
                          and off_h <= vv < off_h + ch)
                if not inside:
                    # the run-time gate disables the bias for this view; it
                    # contributes nothing to the calibrated scale either
                    n_skipped += 1
                    continue
                n_valid += 1
                with torch.no_grad():
                    rho = point_to_ray_distances(p_t, rays)[0].numpy().astype(np.float64)
                rho = rho.reshape(gh, gw)
                rho_min.append(float(rho.min()))
                ci = int(np.clip(np.rint((vv - off_h) / stride), 0, gh - 1))
                cj = int(np.clip(np.rint((uu - off_w) / stride), 0, gw - 1))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ii, jj = ci + di, cj + dj
                    if 0 <= ii < gh and 0 <= jj < gw:
                        rho_1.append(float(rho[ii, jj]))
                for di, dj in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                    ii, jj = ci + di, cj + dj
                    if 0 <= ii < gh and 0 <= jj < gw:
                        rho_2.append(float(rho[ii, jj]))

    if len(rho_1) == 0:
        raise click.ClickException(
            'no valid (state, view) contributed -- every EE point projected '
            'outside the crop or behind the camera; sigma cannot be calibrated')
    sigma = float(np.median(np.asarray(rho_1, dtype=np.float64)))
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise click.ClickException(f'degenerate sigma {sigma!r}')

    summary = {
        'config_name': config_name or str(cfg.get('name', '')),
        'dataset_path': str(ds_conf.dataset_path),
        'task': str(cfg.task.task_name),
        'seed': int(seed),
        'n_states': n,
        'n_view_frames': int(n_frames),
        'n_valid_view_frames': int(n_valid),
        'n_skipped_view_frames': int(n_skipped),
        'valid_fraction': float(n_valid / max(n_frames, 1)),
        'crop_shape': [ch, cw],
        'token_grid': [gh, gw],
        'feature_stride': stride,
        'alpha': alpha,
        'b_max': b_max,
        'sigma_m': sigma,
        'rho_1cell': _quantiles(rho_1),
        'rho_2cell': _quantiles(rho_2),
        'rho_nearest': _quantiles(rho_min),
        'implied_bias_1cell': _implied_bias(
            float(np.median(rho_1)), sigma, alpha, b_max),
        'implied_bias_2cell': _implied_bias(
            _quantiles(rho_2)['median'] if rho_2 else None, sigma, alpha, b_max),
        'note': (
            'sigma = median point-to-ray distance at a 1-cell feature-grid '
            'neighbour of the EE projection, metres; calibrated once on the '
            'training zarr (centre crop) and LOCKED. The bias acts only on '
            'heads 1..geom_bias_heads and only for views whose EE projection '
            'is inside the crop (this script skips the others in the same way '
            'the encoder gates them).'),
    }
    return summary


@click.command()
@click.option('-c', '--config-name',
              default='train_diffusion_unet_image_workspace_enc_rhg_latent',
              help='workspace config whose task.dataset and obs_encoder '
                   'geometry to use. The RHG config carries the real '
                   'geom_bias_alpha / geom_bias_heads / geom_bias_max; the RH '
                   'config works too (its encoder geometry is identical, with '
                   'the plan defaults).')
@click.option('-o', '--out', default='data/geom_bias_calib_square.json')
@click.option('--n-samples', default=256, help='dataset states to read')
@click.option('--seed', default=42, help='state-selection seed (reported)')
@click.option('--num-threads', default=4)
@click.option('--dataset-path', default=None,
              help='override task.dataset.dataset_path (e.g. another task zarr)')
@click.option('--task-name', default=None,
              help='override task.task_name (metadata; the zarr is what is read)')
def main(config_name, out, n_samples, seed, num_threads,
         dataset_path, task_name):
    repo_root = str(pathlib.Path(__file__).parent)
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    overrides = []
    if dataset_path is not None:
        overrides.append(f'task.dataset.dataset_path={dataset_path}')
    if task_name is not None:
        overrides.append(f'task.task_name={task_name}')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    summary = calibrate(cfg, n_samples=n_samples, seed=seed,
                        num_threads=num_threads, config_name=config_name)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"GEOM_BIAS_SIGMA={summary['sigma_m']:.6f}")
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
