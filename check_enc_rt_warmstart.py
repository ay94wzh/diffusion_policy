"""Server-side CPU gate: the real B1 checkpoint loads into the new encoder.

This is the strongest behavior-preservation check before the 30-epoch screen.
It builds the B1 encoder from the checkpoint's own config, loads the
`obs_encoder.*` slice strictly, then builds the new encoder in `none` and
`cross` modes from the same config and loads the parent state dict into both.
On a fixed fake batch:

  - mode `none` must be bit-identical to the parent (z_views and z_global);
  - mode `cross` must be bit-identical to mode `none` at init (W_out == 0);
  - the cross missing keys must be exactly the 13 branch keys.

This script needs the B1 checkpoint (which lives on miroc-server), but no GPU
and no zarr. It runs on CPU and exits non-zero on any failure.

Usage:
    python check_enc_rt_warmstart.py \
        --checkpoint /data/zihan/runs/run_square_enc_b1_s42_200ep/checkpoints/latest.ckpt
"""
import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import dill
import hydra
import torch
from omegaconf import OmegaConf

from diffusion_policy.model.vision.residual_token_view_conditioned_obs_encoder import (
    BRANCH_STATE_PREFIX)


def _fake_obs(enc, n=2):
    """A valid, deterministic batch for the encoder's own schema."""
    obs = dict()
    for key in enc.rgb_keys:
        shape = enc.key_shape_map[key]
        obs[key] = torch.zeros((n,) + tuple(shape))
    for k, cam_key in enumerate(enc.cam_keys):
        h, w = enc.key_shape_map[enc.rgb_keys[k]][1:]
        cam = torch.zeros(n, 10)
        cam[:, 0] = 0.5
        cam[:, 3] = 1.0
        cam[:, 7] = 45.0
        cam[:, 8] = h
        cam[:, 9] = w
        obs[cam_key] = cam
    obs[enc.view_mask_key] = torch.ones(n, enc.n_slots)
    obs[enc.eef_hist_key] = torch.zeros(n, enc.n_slots, enc.eef_hist_dim)
    for key in enc.low_dim_keys:
        obs[key] = torch.zeros((n,) + tuple(enc.key_shape_map[key]))
    return obs


def _build_new(enc_cfg, mode):
    cfg = OmegaConf.to_container(enc_cfg, resolve=True)
    cfg = dict(cfg)
    cfg['_target_'] = (
        'diffusion_policy.model.vision.'
        'residual_token_view_conditioned_obs_encoder.'
        'ResidualTokenViewConditionedObsEncoder')
    for key in list(cfg):
        if key.startswith('token_mixer_'):
            del cfg[key]
    cfg.update(token_mixer_mode=mode, token_mixer_heads=8,
               token_mixer_ffn_dim=1024, token_mixer_dropout=0.0,
               token_mixer_init_seed=42)
    return hydra.utils.instantiate(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    payload = torch.load(
        args.checkpoint, pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    enc_cfg = cfg.policy.obs_encoder
    state_dict = payload['state_dicts']['model']
    enc_state = {
        k[len('obs_encoder.'):]: v
        for k, v in state_dict.items() if k.startswith('obs_encoder.')
    }

    parent = hydra.utils.instantiate(enc_cfg)
    parent.load_state_dict(enc_state, strict=True)
    parent.eval()

    none = _build_new(enc_cfg, 'none')
    none.load_state_dict(parent.state_dict(), strict=True)
    none.eval()

    cross = _build_new(enc_cfg, 'cross')
    result = cross.load_state_dict(parent.state_dict(), strict=False)
    branch_keys = {k for k in cross.state_dict()
                   if k.startswith(BRANCH_STATE_PREFIX)}
    if set(result.missing_keys) != branch_keys:
        raise RuntimeError(
            'cross missing keys are not exactly the branch keys: '
            f'{sorted(set(result.missing_keys) ^ branch_keys)[:10]}')
    if result.unexpected_keys:
        raise RuntimeError(
            f'cross has unexpected keys: {sorted(result.unexpected_keys)[:10]}')
    cross.eval()

    obs = _fake_obs(parent)
    with torch.no_grad():
        a = parent.forward_full(obs)
        b = none.forward_full(obs)
        c = cross.forward_full(obs)

    if not torch.equal(a['z_views'], b['z_views']):
        raise RuntimeError('mode none z_views differ from the B1 parent')
    if not torch.equal(a['z_global'], b['z_global']):
        raise RuntimeError('mode none z_global differs from the B1 parent')
    if not torch.equal(b['z_views'], c['z_views']):
        raise RuntimeError('mode cross differs from B1 at init (z_views)')
    if not torch.equal(b['z_global'], c['z_global']):
        raise RuntimeError('mode cross differs from B1 at init (z_global)')
    if tuple(cross.output_shape()) != (521,):
        raise RuntimeError(f'cross output_shape {cross.output_shape()} != (521,)')

    print(
        f'WARMSTART_CHECK_PASS ckpt={args.checkpoint} '
        f'branch_keys={len(branch_keys)} '
        f'parent_shape={tuple(parent.output_shape())}',
        flush=True)


if __name__ == '__main__':
    main()
