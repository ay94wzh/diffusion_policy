"""CPU-only test for the per-epoch latent-snapshot hook (no simulator, no GPU).

Builds the synthetic multi-view zarr (tests/test_multiview_dataset.py's fixture),
wires a minimal shim of the workspace + policy around a real
ViewConditionedObsEncoder, and checks the three properties the "clear" run's
snapshots depend on:

1. the probe draws are FIXED -- two epochs produce bit-identical raw z_v/z_g;
2. the hook is RNG-NEUTRAL -- the numpy/torch streams after it are exactly what
   they would have been without it (this is what keeps the training draws and
   the DataLoader's per-epoch base_seed untouched);
3. the artifact layout is what the runbook documents (meta.json, one npz per
   epoch, snapshots.jsonl).

It also composes the new task + workspace configs and asserts the clear run's
fixed choices (11 slots, pool 1..11, range [1,11], every module on).

Run from the repo root:
    python tests/test_latent_probe_hook.py
"""
import json
import os
import shutil
import sys
import tempfile

# repo convention: tests run from the repo root
this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(this_dir)
os.chdir(repo_root)
sys.path.insert(0, repo_root)

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from tests.test_multiview_dataset import build_synthetic_zarr, N_VIEWS, H, W, C
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    ViewConditionedObsEncoder)
from diffusion_policy.workspace.train_diffusion_unet_image_workspace_latentprobe import (
    TrainDiffusionUnetImageWorkspaceLatentProbe)
register_codecs()

# the fixture's ring: az [-30, 0, 30], so ring index 1 is az_0 (the N=1 view)
AZ0_VIEW = 1


def _dataset_cfg(zarr_path):
    shape_meta = {
        'obs': {f'view_slot_{k:02d}_image': {'shape': [C, H, W], 'type': 'rgb'}
                for k in range(N_VIEWS)},
        'action': {'shape': [10]},
    }
    for key, dim in (('robot0_eef_pos', 3), ('robot0_eef_quat', 4),
                     ('robot0_gripper_qpos', 2)):
        shape_meta['obs'][key] = {'shape': [dim]}
    return OmegaConf.create(dict(
        _target_='diffusion_policy.dataset.multiview_image_dataset.'
                 'MultiViewImageDataset',
        shape_meta=shape_meta,
        dataset_path=str(zarr_path),
        horizon=2, pad_before=0, pad_after=1, n_obs_steps=1,
        abs_action=True,
        view_pool=[0, 1, 2], view_count_range=[1, 3],
        eef_hist_steps=2, emit_aux_action=True, aux_n_steps=2,
        seed=0, val_ratio=0.0,
    ))


class _FakePolicy:
    """Minimal stand-in for the EMA policy: the hook only uses .parameters()
    (for the device), .obs_encoder and .normalizer."""

    def __init__(self, encoder, normalizer):
        self.obs_encoder = encoder
        self.normalizer = normalizer

    def parameters(self):
        return self.obs_encoder.parameters()


def _make_shim(ds_cfg, out_dir, n_states=4, seed=777):
    """A workspace instance with __init__ bypassed (no hydra/wandb/GPU): the
    hook only reads cfg.task.dataset, latent_probe cfg, output_dir, epoch."""
    cls = TrainDiffusionUnetImageWorkspaceLatentProbe
    ws = cls.__new__(cls)
    ws.cfg = OmegaConf.create(
        {'task': {'dataset': OmegaConf.to_container(ds_cfg)}})
    ws._latent_probe_cfg = OmegaConf.create(dict(
        n_states=n_states, seed=seed, az0_view=AZ0_VIEW, save_raw=True,
        out_dir='latent_snapshots'))
    ws._output_dir = out_dir        # `output_dir` is a read-only property
    ws.epoch = 0
    ws._probe_cache = None
    ws._latent_probe_hook = ws._run_latent_probe
    return ws


def test_hook(tmp):
    zarr_path = os.path.join(tmp, 'synthetic.zarr')
    build_synthetic_zarr(zarr_path)
    # The fixture gives views 0 and 2 the SAME camera pose ([[1,0,1]] twice),
    # which makes any slot->view map ambiguous by construction. The real ring
    # has 13 distinct poses; give every view a distinct pose before using it.
    root = zarr.open(zarr_path, mode='a')
    root['meta/view_cam_pos'][:] = np.array(
        [[1.0, 0.0, 1.0], [0.5, 0.0, 1.35], [1.2, 0.1, 0.9]])
    ds_cfg = _dataset_cfg(zarr_path)

    dataset = hydra.utils.instantiate(ds_cfg)
    normalizer = dataset.get_normalizer()
    encoder = ViewConditionedObsEncoder(
        shape_meta=OmegaConf.to_container(ds_cfg.shape_meta, resolve=True),
        rgb_model=get_resnet(name='resnet18', weights=None),
        use_plucker=True, use_eef_hist=True, eef_hist_steps=2,
        fused_dim=512, n_heads=8, cond_dim=128,
        crop_shape=None, random_crop=False, use_group_norm=True,
        imagenet_norm=True, share_rgb_model=True)
    encoder.eval()
    policy = _FakePolicy(encoder, normalizer)

    out_dir = os.path.join(tmp, 'run')
    os.makedirs(out_dir, exist_ok=True)
    ws = _make_shim(ds_cfg, out_dir)

    # ---- 2. RNG neutrality (the load-bearing property) --------------------
    # The sequence must be EXACTLY what it would have been with no hook in
    # between: same np.random and torch CPU streams.
    got = []
    np.random.seed(123)
    torch.manual_seed(123)
    got.append(float(np.random.rand()))
    ws.epoch = 0
    stats0 = ws._latent_probe_hook(policy)          # builds the cache inside
    got.append(float(np.random.rand()))
    got.append(float(torch.rand(1).item()))
    ws.epoch = 1
    stats1 = ws._latent_probe_hook(policy)
    got.append(float(np.random.rand()))
    got.append(float(torch.rand(1).item()))

    ref = []
    np.random.seed(123)
    torch.manual_seed(123)
    ref.append(float(np.random.rand()))
    ref.append(float(np.random.rand()))
    ref.append(float(torch.rand(1).item()))
    ref.append(float(np.random.rand()))
    ref.append(float(torch.rand(1).item()))
    assert got == ref, f'hook consumed RNG: got {got} != reference {ref}'
    print('RNG neutrality OK (numpy + torch streams identical with the hook)')

    # ---- summary scalars ---------------------------------------------------
    assert set(stats0.keys()) == set(stats1.keys())
    for k, v in stats0.items():
        assert k.startswith('latent_probe/'), k
        assert np.isfinite(v), (k, v)
    assert stats0['latent_probe/zv_norm_mean'] > 0
    assert stats0['latent_probe/zg_n1_norm_mean'] > 0
    assert 0.0 < stats0['latent_probe/zv_rel_spread_mean']
    print('summary scalars OK:', ', '.join(sorted(stats0)))

    # ---- 1. fixed draws: two epochs are bit-identical ----------------------
    snap_dir = os.path.join(out_dir, 'latent_snapshots')
    a = np.load(os.path.join(snap_dir, 'epoch_0000.npz'))
    b = np.load(os.path.join(snap_dir, 'epoch_0001.npz'))
    for key in ('zv', 'zg', 'zg_n1', 'state_idx'):
        assert np.array_equal(a[key], b[key]), f'{key} moved between epochs'
    assert a['zv'].shape == (4, N_VIEWS, 512), a['zv'].shape
    assert a['zv'].dtype == np.float16 and a['state_idx'].dtype == np.int64
    # fusing 3 views must differ from the single az_0 slot
    assert not np.array_equal(a['zg'], a['zg_n1'])
    print('fixed draws OK (bit-identical z_v/z_g across two epochs)')

    # ---- 3. artifact layout -------------------------------------------------
    meta = json.load(open(os.path.join(snap_dir, 'meta.json')))
    assert meta['pool_ring'] == [0, 1, 2], meta['pool_ring']
    assert meta['az0_ring'] == AZ0_VIEW
    assert meta['az0_pos'] == meta['pool_ring'].index(AZ0_VIEW)
    assert meta['conditions'] == ['full', 'n1_az0']
    lines = open(os.path.join(snap_dir, 'snapshots.jsonl')).read().strip().split('\n')
    assert len(lines) == 2, lines
    assert json.loads(lines[0])['epoch'] == 0
    print('artifact layout OK (meta.json, per-epoch npz, snapshots.jsonl)')

    # ---- a check that can fail: the snapshot tracks the MODEL --------------
    # zero the image input channels of conv1; the encoding must move.
    with torch.no_grad():
        encoder.backbone.conv1.weight[:, :3].zero_()
    ws.epoch = 2
    stats2 = ws._latent_probe_hook(policy)
    moved = (stats2['latent_probe/zv_norm_mean'] !=
             stats0['latent_probe/zv_norm_mean'])
    assert moved, 'zeroing the image channels did not change the snapshot'
    print('mutation check OK (zeroing conv1 image channels moves the stats)')


def test_config_composes():
    from hydra import compose, initialize_config_dir
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name='train_diffusion_unet_image_workspace_m4latent',
            overrides=['task.task_name=square'])
    task = cfg.task
    rgb = [k for k, v in task.shape_meta.obs.items()
           if v.get('type', 'low_dim') == 'rgb']
    assert rgb == [f'view_slot_{k:02d}_image' for k in range(11)], rgb
    assert list(task.train_view_pool) == list(range(1, 12))
    assert list(task.view_count_range) == [1, 11]
    assert task.env_runner.m3_slots == 11
    assert task.dataset.emit_aux_action is True
    assert task.dataset.aux_n_steps == 8
    assert cfg.policy.aux_loss_weight == 1.0
    assert cfg.policy.obs_encoder.use_plucker is True
    assert cfg.policy.obs_encoder.use_eef_hist is True
    assert 'TrainDiffusionUnetImageWorkspaceLatentProbe' in cfg._target_
    assert cfg.latent_probe.n_states == 128
    assert cfg.latent_probe.az0_view == 6      # ring index 6 == az_0
    # interpolations resolve the way the run's paths depend on
    OmegaConf.resolve(cfg)
    assert cfg.task.dataset.pad_after == 7
    assert cfg.task.dataset.dataset_path == \
        'data/multiview/square_ph_ring13.zarr', cfg.task.dataset.dataset_path
    print('config compose OK (11 slots, pool 1..11, range [1,11], all modules on)')


def test():
    tmp = tempfile.mkdtemp(prefix='latent_probe_test_')
    try:
        test_config_composes()
        test_hook(tmp)
        print('ALL LATENT-PROBE HOOK CHECKS PASSED')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test()
