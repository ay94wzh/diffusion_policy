"""CPU-only test for stage S's pure-RGB spatial-readout encoder.

Verifies, in order:
  1. the architecture: 3-channel conv1, layer3 only (no layer4/avgpool/fc),
     256->512 projection, two 8-head 512-wide attentions, zero-init queries;
  2. the 76x76 crop -> layer3 5x5 -> 25-token grid, independent of K;
  3. output_shape (521,) for K=1 and K=11, from train and eval, RNG-neutral;
  4. the drop-in contract (forward == cat(z_global, low-dims); B*To slicing);
  5. masking: inactive slots stay exactly zero and cannot affect the output;
  6. permutation invariance of the view tokens and the fusion;
  7. N=1 and slot-agnosticism;
  8. crop sampling in train vs centre crop in eval, with no eval RNG;
  9. the RGB-only contract: camera/history keys are ignored and optional;
 10. gradients reach the backbone, projection, both attentions and both
     queries;
 11. the exact parameter count (5,016,640) and that S is smaller than B;
 12. the 521/1042 width interface shared with the stock encoder;
 13. the real S configs compose, resolve and instantiate the encoder.

Run from the repo root:
    python tests/test_spatial_readout_obs_encoder.py
"""
import os
import sys

# repo convention: tests run from the repo root
this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(this_dir)
os.chdir(repo_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, this_dir)

import torch
import torchvision.transforms.functional as ttf

from diffusion_policy.model.vision.spatial_readout_obs_encoder import (
    SpatialReadoutObsEncoder, IMAGENET_MEAN, IMAGENET_STD,
)
from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    ViewConditionedObsEncoder,
)
from diffusion_policy.model.vision.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)
from diffusion_policy.model.vision.crop_randomizer import crop_image_from_indices
from diffusion_policy.model.vision.model_getter import get_resnet
import diffusion_policy.model.vision.spatial_readout_obs_encoder as sroe

H = W = 84
CROP = 76
HIST_STEPS = 4
HIST_DIM = 11 * HIST_STEPS
S_PARAMS = 5_016_640


def _shape_meta(k_slots):
    obs = {f'view_slot_{k:02d}_image': {'shape': [3, H, W], 'type': 'rgb'}
           for k in range(k_slots)}
    obs['robot0_eef_pos'] = {'shape': [3]}
    obs['robot0_eef_quat'] = {'shape': [4]}
    obs['robot0_gripper_qpos'] = {'shape': [2]}
    return {'obs': obs, 'action': {'shape': [10]}}


def _build_encoder(k_slots, **kw):
    kw.setdefault('crop_shape', [CROP, CROP])
    kw.setdefault('random_crop', True)
    kw.setdefault('use_group_norm', True)
    kw.setdefault('imagenet_norm', True)
    kw.setdefault('share_rgb_model', True)
    return SpatialReadoutObsEncoder(
        shape_meta=_shape_meta(k_slots),
        rgb_model=get_resnet('resnet18', weights=None), **kw)


def _fake_obs(k_slots, n, seed=0, mask=None, with_extra=True):
    """A batch shaped as the policy delivers it: leading dim is B*To.

    Camera vectors are per-slot `(n, 10)` (the real schema), and the EE history
    is `(n, K, 44)`. S never reads either; `with_extra=False` builds the
    RGB-only dict the contract must also accept.
    """
    g = torch.Generator().manual_seed(seed)
    obs = {}
    for k in range(k_slots):
        obs[f'view_slot_{k:02d}_image'] = torch.rand(n, 3, H, W, generator=g)
        if with_extra:
            cam = torch.zeros(n, 10)
            cam[:, :3] = torch.randn(n, 3, generator=g) * 0.3
            q = torch.randn(n, 4, generator=g)
            q = q / q.norm(dim=-1, keepdim=True)
            cam[:, 3:7] = q
            cam[:, 7] = 45.0
            cam[:, 8] = H
            cam[:, 9] = W
            obs[f'view_slot_{k:02d}_cam'] = cam
    obs['view_mask'] = torch.ones(n, k_slots) if mask is None else mask
    if with_extra:
        obs['view_eef_hist'] = torch.randn(n, k_slots, HIST_DIM, generator=g)
    obs['robot0_eef_pos'] = torch.randn(n, 3, generator=g)
    obs['robot0_eef_quat'] = torch.randn(n, 4, generator=g)
    obs['robot0_gripper_qpos'] = torch.randn(n, 2, generator=g)
    return obs


def test_architecture():
    enc = _build_encoder(5)
    assert tuple(enc.backbone.conv1.weight.shape) == (64, 3, 7, 7), (
        enc.backbone.conv1.weight.shape)
    for name in ('layer4', 'avgpool', 'fc'):
        assert not hasattr(enc.backbone, name), name
    assert not any(n.startswith('backbone.layer4')
                   for n, _ in enc.named_parameters())
    assert enc.proj.in_channels == 256
    assert enc.proj.out_channels == 512
    assert tuple(enc.proj.kernel_size) == (1, 1)
    mhas = [m for m in enc.modules()
            if isinstance(m, torch.nn.MultiheadAttention)]
    assert len(mhas) == 2, len(mhas)
    for m in mhas:
        assert m.embed_dim == 512 and m.num_heads == 8, (m.embed_dim, m.num_heads)
    assert tuple(enc.spatial_query.shape) == (1, 1, 512)
    assert tuple(enc.fusion_query.shape) == (1, 1, 512)
    assert bool((enc.spatial_query == 0).all())
    assert bool((enc.fusion_query == 0).all())
    assert enc.spatial_query is not enc.fusion_query
    assert enc.token_grid == (5, 5), enc.token_grid
    assert enc.n_tokens == 25, enc.n_tokens
    try:
        _build_encoder(2, share_rgb_model=False)
        raise AssertionError('share_rgb_model=False should raise')
    except NotImplementedError:
        pass
    print('architecture OK (RGB conv1, layer3 only, 256->512, 2x MHA(512,8), '
          'zero queries, 5x5 grid)')


def test_token_grid():
    for k, n in ((11, 3), (3, 3)):
        enc = _build_encoder(k)
        enc.eval()
        obs = _fake_obs(k, n, seed=1)
        seen = {}

        def hook(module, inp, out):
            seen['in'] = tuple(inp[0].shape)
            seen['out'] = tuple(out.shape)

        handle = enc.proj.register_forward_hook(hook)
        try:
            enc(obs)
        finally:
            handle.remove()
        assert seen['in'] == (k * n, 256, 5, 5), (k, seen['in'])
        assert seen['out'] == (k * n, 512, 5, 5), (k, seen['out'])
    print('token grid OK (K=11 and K=3: proj sees (M,256,5,5), emits '
          '(M,512,5,5))')


def test_output_shape():
    for k in (1, 11):
        enc = _build_encoder(k)
        for train in (True, False):
            enc.train(train)
            assert tuple(enc.output_shape()) == (521,), k
            assert enc.training == train
    # output_shape is called inside the policy constructor after the workspace
    # seeds torch: it must not consume the training RNG.
    enc = _build_encoder(11)
    enc.train()
    torch.manual_seed(11)
    r0 = torch.rand(1)
    torch.manual_seed(11)
    enc.output_shape()
    r1 = torch.rand(1)
    assert torch.equal(r0, r1), (r0, r1)
    print('output_shape OK (521 for K=1/K=11 from train/eval; RNG-neutral)')


def test_forward_contract():
    k = 11
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, 4, seed=2)
    out = enc(obs)
    assert tuple(out.shape) == (4, 521), out.shape
    assert bool(torch.isfinite(out).all())
    full = enc.forward_full(obs)
    assert set(full) == {'z_global', 'z_views', 'view_active'}
    assert tuple(full['z_global'].shape) == (4, 512)
    assert tuple(full['z_views'].shape) == (4, k, 512)
    assert tuple(full['view_active'].shape) == (4, k)
    want = torch.cat([
        full['z_global'],
        obs['robot0_eef_pos'],
        obs['robot0_eef_quat'],
        obs['robot0_gripper_qpos'],
    ], dim=-1)
    assert torch.equal(out, want), (out - want).abs().max()

    # reproduce the policy's own B,To slicing without building the 277M UNet
    B, To = 3, 2
    flat = _fake_obs(k, B * To, seed=3)
    obs2 = {key: val.reshape(B, To, *val.shape[1:])
            for key, val in flat.items()}
    this = {key: val[:, :To].reshape(-1, *val.shape[2:])
            for key, val in obs2.items()}
    feats = enc(this)
    assert tuple(feats.shape) == (B * To, 521), feats.shape
    assert tuple(feats.reshape(B, -1).shape) == (B, 1042)
    print('forward contract OK (521-wide, forward == cat(z_global, low-dims), '
          'global_cond 1042)')


def test_inactive_and_independence():
    k = 7
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, 3, seed=4)
    mask = torch.ones(3, k)
    mask[0, [1, 4, 5]] = 0
    mask[1, [0, 2, 3]] = 0
    mask[2, [6]] = 0
    obs['view_mask'] = mask
    out = enc(obs)
    full = enc.forward_full(obs)
    active = full['view_active']
    assert bool((full['z_views'][~active] == 0).all())
    assert bool(torch.isfinite(out).all())

    # garbage in inactive slots must not reach the encoder
    obs2 = {key: val.clone() for key, val in obs.items()}
    for row in range(3):
        for slot in range(k):
            if not bool(active[row, slot]):
                obs2[f'view_slot_{slot:02d}_image'][row] = float('nan')
                obs2[f'view_slot_{slot:02d}_cam'][row] = float('nan')
                obs2['view_eef_hist'][row, slot] = float('nan')
    out2 = enc(obs2)
    full2 = enc.forward_full(obs2)
    assert torch.equal(out, out2)
    assert torch.equal(full['z_views'], full2['z_views'])

    # changing one active view changes only that view's z_v (before fusion)
    # and z_g; the other views are computed independently
    obs3 = {key: val.clone() for key, val in obs.items()}
    obs3['view_slot_02_image'][0] = torch.rand_like(
        obs3['view_slot_02_image'][0])
    full3 = enc.forward_full(obs3)
    assert not torch.allclose(full['z_views'][0, 2], full3['z_views'][0, 2])
    assert torch.equal(full['z_views'][0, 0], full3['z_views'][0, 0])
    assert not torch.allclose(full['z_global'][0], full3['z_global'][0])

    # all-False row -> the fusion query would be fully masked (NaN)
    bad = _fake_obs(k, 1, seed=5, mask=torch.zeros(1, k))
    try:
        enc(bad)
        raise AssertionError('an all-masked row should raise')
    except ValueError:
        pass
    # a soft mask is not a liveness mask
    bad2 = _fake_obs(k, 1, seed=6, mask=torch.full((1, k), 0.5))
    try:
        enc(bad2)
        raise AssertionError('a non-binary mask should raise')
    except ValueError:
        pass
    print('masking OK (inactive rows exactly zero and ignored; all-False and '
          'non-binary masks raise)')


def test_permutation():
    k = 7
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, 2, seed=7)
    perm = [3, 0, 6, 1, 5, 2, 4]
    shuffled = {}
    for dst, src in enumerate(perm):
        shuffled[f'view_slot_{dst:02d}_image'] = obs[
            f'view_slot_{src:02d}_image']
        shuffled[f'view_slot_{dst:02d}_cam'] = obs[f'view_slot_{src:02d}_cam']
    shuffled['view_mask'] = obs['view_mask'][:, perm]
    shuffled['view_eef_hist'] = obs['view_eef_hist'][:, perm]
    for key in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'):
        shuffled[key] = obs[key]
    a = enc.forward_full(obs)
    b = enc.forward_full(shuffled)
    assert torch.allclose(a['z_global'], b['z_global'], atol=1e-5), (
        (a['z_global'] - b['z_global']).abs().max())
    perm_t = torch.tensor(perm)
    assert torch.allclose(a['z_views'][:, perm_t], b['z_views'], atol=1e-5)
    print('permutation OK (slot order is not information; z_g and z_views '
          'permute together)')


def test_n1():
    k = 11
    enc = _build_encoder(k)
    enc.eval()
    base = _fake_obs(k, 2, seed=8)
    img = base['view_slot_00_image'].clone()

    def one(slot):
        o = {key: torch.zeros_like(val) for key, val in base.items()}
        for key in ('robot0_eef_pos', 'robot0_eef_quat',
                    'robot0_gripper_qpos'):
            o[key] = base[key].clone()
        o[f'view_slot_{slot:02d}_image'] = img.clone()
        o['view_mask'] = torch.zeros(2, k)
        o['view_mask'][:, slot] = 1.0
        return o

    o0, o5 = one(0), one(5)
    out0 = enc(o0)
    assert tuple(out0.shape) == (2, 521)
    assert bool(torch.isfinite(out0).all())
    f0 = enc.forward_full(o0)
    f5 = enc.forward_full(o5)
    # one unmasked key: the fusion output must not depend on the slot index
    assert torch.equal(f0['z_global'], f5['z_global'])
    assert bool((f0['z_views'][:, 1:] == 0).all())
    assert not bool((f0['z_views'][:, 0] == 0).all())
    o0b = one(0)
    o0b['view_slot_00_image'] = torch.rand_like(img)
    assert not torch.allclose(enc.forward_full(o0b)['z_global'],
                              f0['z_global'])

    enc1 = _build_encoder(1)
    enc1.eval()
    assert tuple(enc1(_fake_obs(1, 2, seed=9)).shape) == (2, 521)
    print('N=1 OK (one live view, slot-agnostic z_g; K=1 shape_meta works)')


def test_crop():
    enc = _build_encoder(1)
    enc.train()
    img = torch.rand(3, 3, H, W)
    recorded = {}
    real = sroe.sample_random_image_crops

    def spy(images, crop_height, crop_width, num_crops, pos_enc=False):
        crops, inds = real(images=images, crop_height=crop_height,
                           crop_width=crop_width, num_crops=num_crops,
                           pos_enc=pos_enc)
        recorded['inds'] = inds.detach().clone()
        return crops, inds

    sroe.sample_random_image_crops = spy
    try:
        got = enc._prepare_image(img, train=True)
    finally:
        sroe.sample_random_image_crops = real
    inds = recorded['inds'].reshape(-1, 2).long()
    assert len({tuple(t) for t in inds.tolist()}) > 1, (
        'crop offsets are not random')
    want = crop_image_from_indices(img, inds, CROP, CROP)
    mean = want.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = want.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    want = (want - mean) / std
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()

    # a fixed seed reproduces the sampled crops
    torch.manual_seed(21)
    a = enc._prepare_image(img, train=True)
    torch.manual_seed(21)
    b = enc._prepare_image(img, train=True)
    assert torch.equal(a, b)

    # eval: the centre crop, identical to torchvision's center_crop
    enc.eval()
    got = enc._prepare_image(img, train=False)
    img_cc = ttf.center_crop(img, CROP)
    want_cc = (img_cc - mean) / std
    assert torch.allclose(got, want_cc, atol=1e-6)
    enc_raw = _build_encoder(1, imagenet_norm=False)
    enc_raw.eval()
    assert torch.allclose(enc_raw._prepare_image(img, train=False), img_cc,
                          atol=1e-6)

    # eval consumes no RNG: reset the seed before the call, then compare the
    # next draw with the no-call branch
    torch.manual_seed(22)
    r0 = torch.rand(1)
    torch.manual_seed(22)
    enc._prepare_image(img, train=False)
    r1 = torch.rand(1)
    assert torch.equal(r0, r1), (r0, r1)
    print('crop OK (sampled window in train, centre in eval, ImageNet norm, '
          'no eval RNG)')


def test_ignores_cam_hist():
    k = 4
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, 3, seed=10)
    out = enc(obs)
    pert = {key: val.clone() for key, val in obs.items()}
    for j in range(k):
        pert[f'view_slot_{j:02d}_cam'] = torch.randn(3, 10) * 9.0
    pert['view_eef_hist'] = torch.randn_like(obs['view_eef_hist']) * 9.0
    assert torch.equal(out, enc(pert))
    no = {key: val for key, val in obs.items()
          if not key.endswith('_cam') and key != 'view_eef_hist'}
    assert torch.equal(out, enc(no))
    print('RGB-only contract OK (camera/history keys ignored and optional)')


def test_gradients():
    enc = _build_encoder(4)
    enc.train()
    obs = _fake_obs(4, 2, seed=11)
    out = enc(obs).sum()
    enc.zero_grad(set_to_none=True)
    out.backward()
    tensors = {
        'conv1': enc.backbone.conv1.weight,
        'layer3': enc.backbone.layer3[0].conv1.weight,
        'proj': enc.proj.weight,
        'intra_attn': enc.intra_attn.in_proj_weight,
        'spatial_query': enc.spatial_query,
        'fusion': enc.fusion.in_proj_weight,
        'fusion_query': enc.fusion_query,
    }
    for name, p in tensors.items():
        assert p.grad is not None, name
        assert bool(torch.isfinite(p.grad).all()), name
        assert float(p.grad.abs().sum()) > 0, name
    print('gradients OK (backbone, projection, both attentions, both queries)')


def test_parameter_count():
    counts = []
    for k in (1, 11):
        enc = _build_encoder(k)
        n = sum(p.numel() for p in enc.parameters())
        counts.append(n)
        assert n == S_PARAMS, (k, n)
    b = ViewConditionedObsEncoder(
        shape_meta=_shape_meta(1),
        rgb_model=get_resnet('resnet18', weights=None),
        fused_dim=512, n_heads=8, cond_dim=128,
        use_plucker=True, use_eef_hist=True, eef_hist_steps=HIST_STEPS,
        crop_shape=[CROP, CROP], random_crop=True, use_group_norm=True,
        share_rgb_model=True, imagenet_norm=True)
    n_b = sum(p.numel() for p in b.parameters())
    assert counts[0] < n_b, (counts[0], n_b)
    print(f'parameter count OK (S {counts[0]:,}, B {n_b:,}; S is smaller)')


def test_width_interface():
    stock = MultiImageObsEncoder(
        shape_meta=_shape_meta(1),
        rgb_model=get_resnet('resnet18', weights=None),
        resize_shape=None, crop_shape=[CROP, CROP], random_crop=True,
        use_group_norm=True, share_rgb_model=False, imagenet_norm=True)
    assert tuple(stock.output_shape()) == (521,)
    for k in (1, 11):
        enc = _build_encoder(k)
        assert tuple(enc.output_shape()) == (521,)
    assert 521 * 2 == 1042
    print('width interface OK (521 per frame, 1042 for To=2, same as stock)')


def test_config():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_enc_s_latent')
    assert cfg.task.name == 'enc_s_image_abs_multiview_az75'
    assert 'DiffusionUnetImagePolicy' in cfg.policy._target_
    assert 'LatentProbe' in cfg._target_
    assert sorted(cfg.task.dataset.view_pool) == list(range(1, 12))
    assert tuple(cfg.task.dataset.view_count_range) == (1, 11)
    assert cfg.task.env_runner.m3_slots == 11
    assert 'cam_key_image_runner' in cfg.task.env_runner._target_
    assert bool(cfg.task.dataset.emit_aux_action) is False
    assert cfg.task.dataset.eef_hist_window == 'episode'
    assert cfg.task.dataset.eef_hist_steps == 4
    assert cfg.task.env_runner.eef_hist_steps == 4
    assert cfg.latent_probe.n_states == 128
    assert cfg.latent_probe.seed == 12345
    assert cfg.latent_probe.az0_view == 6
    obs_cfg = cfg.policy.obs_encoder
    assert obs_cfg._target_.endswith(
        'spatial_readout_obs_encoder.SpatialReadoutObsEncoder'), (
        obs_cfg._target_)
    assert obs_cfg.fused_dim == 512
    assert obs_cfg.n_heads == 8
    assert tuple(obs_cfg.crop_shape) == (76, 76)
    assert bool(obs_cfg.share_rgb_model) is True
    assert bool(obs_cfg.use_group_norm) is True
    assert bool(obs_cfg.imagenet_norm) is True
    for key in ('cond_dim', 'use_plucker', 'use_eef_hist', 'eef_hist_steps',
                'cam_keys', 'eef_hist_key', 'resize_shape'):
        assert key not in obs_cfg, key
    enc = hydra.utils.instantiate(obs_cfg)
    assert tuple(enc.output_shape()) == (521,)
    assert enc.n_slots == 11
    assert enc.token_grid == (5, 5)
    OmegaConf.resolve(cfg)
    assert cfg.task.dataset.dataset_path == (
        'data/multiview/square_ph_ring13.zarr')
    print('S config OK (RGB-only encoder, K=11 pool [1..11], probe 128/az0=6)')


def test():
    test_architecture()
    test_token_grid()
    test_output_shape()
    test_forward_contract()
    test_inactive_and_independence()
    test_permutation()
    test_n1()
    test_crop()
    test_ignores_cam_hist()
    test_gradients()
    test_parameter_count()
    test_width_interface()
    test_config()
    print('ALL SPATIAL-READOUT ENCODER CHECKS PASSED')


if __name__ == '__main__':
    test()
