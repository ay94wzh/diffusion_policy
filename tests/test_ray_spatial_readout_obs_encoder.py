"""CPU-only test for stage R's ray-encoded spatial-readout encoder.

Verifies, in order:
  1. the architecture: 3-channel conv1, layer3 only, 256->512 projection, two
     8-head 512-wide attentions, a 6->512->512 ray MLP with a zero-init output,
     two 512-d LayerNorms, zero-init queries, the 5x5 grid and stride 16;
  2. the geometric convention: the per-cell rays equal the full-resolution
     Pluecker map sampled at the receptive-field centres (offset + 16*i), are
     unit length, and split into a rotation-only direction and a
     translation-carrying moment;
  3. crop synchronisation: a random crop's rays use the SAME window as the
     image, and a fixed seed reproduces them;
  4. output_shape (521,) for K=1 and K=11, from train and eval, RNG-neutral;
  5. the drop-in contract (forward == cat(z_global, low-dims); B*To slicing);
  6. masking: inactive slots stay exactly zero, their all-zero camera is never
     ray-encoded, and they cannot affect the output;
  7. permutation invariance of the view tokens and the fusion;
  8. N=1 and slot-agnosticism;
  9. camera validation: missing keys, size mismatches and a degenerate
     quaternion on a LIVE slot all fail loudly;
 10. the ray branch is inert at initialisation (e == 0, bit-identical to
     use_ray_encoding=False) and live afterwards;
 11. gradients reach the backbone, projection, both attentions, both queries,
     both LayerNorms and the ray MLP's output layer (the first layer's
     gradient is exactly zero at init by the zero-init design, then live);
 12. the exact parameter count (5,284,928) between S and B;
 13. the 521/1042 width interface shared with the stock encoder;
 14. the real R configs compose, resolve and instantiate the encoder.

Run from the repo root:
    python tests/test_ray_spatial_readout_obs_encoder.py
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

from diffusion_policy.model.vision.ray_spatial_readout_obs_encoder import (
    RaySpatialReadoutObsEncoder, RAY_FEAT_DIM,
)
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
from diffusion_policy.model.vision.plucker import plucker_ray_map
from diffusion_policy.model.vision.model_getter import get_resnet
import diffusion_policy.model.vision.ray_spatial_readout_obs_encoder as rroe

H = W = 84
CROP = 76
STRIDE = 16
GRID = 5
HIST_STEPS = 4
HIST_DIM = 11 * HIST_STEPS
R_PARAMS = 5_284_928
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
    return RaySpatialReadoutObsEncoder(
        shape_meta=_shape_meta(k_slots),
        rgb_model=get_resnet('resnet18', weights=None), **kw)


def _cam(n, g, seed=0, fovy=45.0):
    """A batch of valid camera vectors [pos(3), quat_wxyz(4), fovy, h, w]."""
    cam = torch.zeros(n, 10)
    cam[:, :3] = torch.randn(n, 3, generator=g) * 0.3
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    cam[:, 3:7] = q
    cam[:, 7] = fovy
    cam[:, 8] = H
    cam[:, 9] = W
    return cam


def _fake_obs(k_slots, n, seed=0, mask=None):
    """A batch shaped as the policy delivers it: leading dim is B*To.

    Camera vectors are per-slot `(n, 10)` (the real schema); the EE history is
    `(n, K, 44)`. R reads the cameras and ignores the history -- the key is
    kept because the shared probe/workspace stack it.
    """
    g = torch.Generator().manual_seed(seed)
    obs = {}
    for k in range(k_slots):
        obs[f'view_slot_{k:02d}_image'] = torch.rand(n, 3, H, W, generator=g)
        obs[f'view_slot_{k:02d}_cam'] = _cam(n, g)
    obs['view_mask'] = torch.ones(n, k_slots) if mask is None else mask
    obs['view_eef_hist'] = torch.randn(n, k_slots, HIST_DIM, generator=g)
    obs['robot0_eef_pos'] = torch.randn(n, 3, generator=g)
    obs['robot0_eef_quat'] = torch.randn(n, 4, generator=g)
    obs['robot0_gripper_qpos'] = torch.randn(n, 2, generator=g)
    return obs


def _expected_rays(cam, inds, ray_scale=1.0):
    """Full map sampled at (off + stride*i, off + stride*j), row-major."""
    n = cam.shape[0]
    full = plucker_ray_map(cam[:, :3], cam[:, 3:7], cam[:, 7], H, W)  # (n,6,H,W)
    step = torch.arange(GRID) * STRIDE
    rows = []
    for r in range(n):
        m = full[r][:, inds[r, 0] + step, :][:, :, inds[r, 1] + step]  # (6,G,G)
        m = m.reshape(6, GRID * GRID).transpose(0, 1).clone()          # (G,6)
        m[:, 3:] = m[:, 3:] / ray_scale
        rows.append(m)
    return torch.stack(rows)


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

    assert isinstance(enc.ray_mlp[0], torch.nn.Linear)
    assert enc.ray_mlp[0].in_features == RAY_FEAT_DIM == 6
    assert enc.ray_mlp[0].out_features == 512
    assert isinstance(enc.ray_mlp[-1], torch.nn.Linear)
    assert enc.ray_mlp[-1].in_features == 512
    assert enc.ray_mlp[-1].out_features == 512
    assert bool((enc.ray_mlp[-1].weight == 0).all())
    assert bool((enc.ray_mlp[-1].bias == 0).all())
    assert isinstance(enc.key_norm, torch.nn.LayerNorm)
    assert isinstance(enc.value_norm, torch.nn.LayerNorm)
    assert enc.key_norm.normalized_shape == (512,)
    assert enc.value_norm.normalized_shape == (512,)
    assert enc.key_norm is not enc.value_norm

    assert enc.token_grid == (GRID, GRID), enc.token_grid
    assert enc.n_tokens == GRID * GRID
    assert enc.total_stride == STRIDE, enc.total_stride

    try:
        _build_encoder(2, share_rgb_model=False)
        raise AssertionError('share_rgb_model=False should raise')
    except NotImplementedError:
        pass
    # history arguments do not exist on R: a copied B block must fail loudly
    for bad in ('eef_hist_key', 'eef_hist_steps', 'use_eef_hist',
                'cond_dim', 'use_plucker', 'resize_shape'):
        try:
            _build_encoder(2, **{bad: 1})
            raise AssertionError(f'{bad} should be rejected by R')
        except TypeError:
            pass
    print('architecture OK (RGB conv1, layer3 only, 256->512, 2x MHA(512,8), '
          '6->512->512 ray MLP zero-init, 2 LayerNorms, stride 16)')


def test_ray_convention():
    enc = _build_encoder(1)
    enc.eval()
    n = 4
    g = torch.Generator().manual_seed(7)
    cam = _cam(n, g)
    img = torch.rand(n, 3, H, W, generator=g)
    _, rays = enc._prepare_slot(img, cam, train=False)

    off = (H - CROP) // 2
    inds = torch.full((n, 2), off, dtype=torch.long)
    want = _expected_rays(cam, inds)
    assert rays.shape == (n, GRID * GRID, RAY_FEAT_DIM), rays.shape
    err = (rays - want).abs().max()
    assert float(err) < 1e-6, float(err)
    assert float((rays[..., :3].norm(dim=-1) - 1).abs().max()) < 1e-6

    # d depends on rotation only: translating the camera leaves d bit-identical
    cam_t = cam.clone()
    cam_t[:, :3] += 1.0
    _, rays_t = enc._prepare_slot(img, cam_t, train=False)
    assert torch.equal(rays[..., :3], rays_t[..., :3])
    assert float((rays[..., 3:] - rays_t[..., 3:]).abs().max()) > 0.1

    # fovy moves the directions (and therefore the moments)
    cam_f = cam.clone()
    cam_f[:, 7] += 10.0
    _, rays_f = enc._prepare_slot(img, cam_f, train=False)
    assert float((rays[..., :3] - rays_f[..., :3]).abs().max()) > 1e-3

    # ray_scale divides the moment only
    enc5 = _build_encoder(1, ray_scale=5.0, imagenet_norm=False)
    enc5.eval()
    _, rays5 = enc5._prepare_slot(img, cam, train=False)
    assert torch.allclose(rays5[..., :3], rays[..., :3], atol=1e-6)
    assert torch.allclose(rays5[..., 3:], rays[..., 3:] / 5.0, atol=1e-6)
    print('ray convention OK (full-map sampling at off+16*i, unit d, '
          'd translation-invariant, m translation-carrying, L scales m only)')


def test_crop_ray_alignment():
    enc = _build_encoder(1)
    enc.train()
    n = 6
    g = torch.Generator().manual_seed(11)
    img = torch.rand(n, 3, H, W, generator=g)
    cam = _cam(n, g)

    real = rroe.sample_random_image_crops
    recorded = {}

    def spy(images, crop_height, crop_width, num_crops, pos_enc=False):
        crops, inds = real(images=images, crop_height=crop_height,
                           crop_width=crop_width, num_crops=num_crops,
                           pos_enc=pos_enc)
        recorded['inds'] = inds.detach().clone()
        return crops, inds

    rroe.sample_random_image_crops = spy
    try:
        img_c, rays = enc._prepare_slot(img, cam, train=True)
    finally:
        rroe.sample_random_image_crops = real
    inds = recorded['inds'].reshape(n, 2).long()
    assert len({tuple(t) for t in inds.tolist()}) > 1, (
        'crop offsets are not random')
    want = _expected_rays(cam, inds)
    err = float((rays - want).abs().max())
    assert err < 1e-6, err
    # the image is cropped at the SAME window as the rays
    want_img = crop_image_from_indices(img, inds, CROP, CROP)
    mean = want_img.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = want_img.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    want_img = (want_img - mean) / std
    assert torch.allclose(img_c, want_img, atol=1e-6)

    torch.manual_seed(21)
    a = enc._prepare_slot(img, cam, train=True)
    torch.manual_seed(21)
    b = enc._prepare_slot(img, cam, train=True)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    # eval: centre crop, identical to torchvision's center_crop, and no RNG
    enc.eval()
    img_c, _ = enc._prepare_slot(img, cam, train=False)
    img_cc = ttf.center_crop(img, CROP)
    assert torch.allclose(img_c, (img_cc - mean) / std, atol=1e-6)
    torch.manual_seed(22)
    r0 = torch.rand(1)
    torch.manual_seed(22)
    enc._prepare_slot(img, cam, train=False)
    r1 = torch.rand(1)
    assert torch.equal(r0, r1), (r0, r1)
    print('crop alignment OK (image and rays share one window; centre crop in '
          'eval; no eval RNG)')


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
        assert seen['in'] == (k * n, 256, GRID, GRID), (k, seen['in'])
        assert seen['out'] == (k * n, 512, GRID, GRID), (k, seen['out'])
    print('token grid OK (K=11 and K=3: proj sees (M,256,5,5), emits '
          '(M,512,5,5))')


def test_output_shape():
    for k in (1, 11):
        enc = _build_encoder(k)
        for train in (True, False):
            enc.train(train)
            assert tuple(enc.output_shape()) == (521,), k
            assert enc.training == train
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

    # garbage in inactive slots -- including the ALL-ZERO camera, which is not
    # a valid pose -- must not reach the encoder
    obs2 = {key: val.clone() for key, val in obs.items()}
    for row in range(3):
        for slot in range(k):
            if not bool(active[row, slot]):
                obs2[f'view_slot_{slot:02d}_image'][row] = float('nan')
                obs2[f'view_slot_{slot:02d}_cam'][row] = 0.0
    out2 = enc(obs2)
    full2 = enc.forward_full(obs2)
    assert torch.equal(out, out2)
    assert torch.equal(full['z_views'], full2['z_views'])

    # changing one active view changes only that view's z_v and z_g
    obs3 = {key: val.clone() for key, val in obs.items()}
    obs3['view_slot_02_image'][0] = torch.rand_like(
        obs3['view_slot_02_image'][0])
    full3 = enc.forward_full(obs3)
    assert not torch.allclose(full['z_views'][0, 2], full3['z_views'][0, 2])
    assert torch.equal(full['z_views'][0, 0], full3['z_views'][0, 0])
    assert not torch.allclose(full['z_global'][0], full3['z_global'][0])

    # changing one active view's CAMERA changes that view's z_v -- once the
    # key path is live (the zero query and the zero ray output make it inert
    # at init, tested separately) -- and only through that view before fusion
    with torch.no_grad():
        enc.spatial_query.normal_(0.0, 0.02)
        enc.ray_mlp[-1].weight.normal_(0.0, 0.02)
    full = enc.forward_full(obs)
    obs4 = {key: val.clone() for key, val in obs.items()}
    g = torch.Generator().manual_seed(41)
    obs4['view_slot_00_cam'][0] = _cam(1, g)[0]
    full4 = enc.forward_full(obs4)
    assert not torch.allclose(full['z_views'][0, 0], full4['z_views'][0, 0])
    assert torch.equal(full['z_views'][0, 2], full4['z_views'][0, 2])

    bad = _fake_obs(k, 1, seed=5, mask=torch.zeros(1, k))
    try:
        enc(bad)
        raise AssertionError('an all-masked row should raise')
    except ValueError:
        pass
    bad2 = _fake_obs(k, 1, seed=6, mask=torch.full((1, k), 0.5))
    try:
        enc(bad2)
        raise AssertionError('a non-binary mask should raise')
    except ValueError:
        pass
    print('masking OK (inactive rows zero and ignored, including their invalid '
          'cameras; live camera changes only its own view; bad masks raise)')


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
    cam = base['view_slot_00_cam'].clone()

    def one(slot):
        o = {key: torch.zeros_like(val) for key, val in base.items()}
        for key in ('robot0_eef_pos', 'robot0_eef_quat',
                    'robot0_gripper_qpos'):
            o[key] = base[key].clone()
        o[f'view_slot_{slot:02d}_image'] = img.clone()
        o[f'view_slot_{slot:02d}_cam'] = cam.clone()
        o['view_mask'] = torch.zeros(2, k)
        o['view_mask'][:, slot] = 1.0
        return o

    o0, o5 = one(0), one(5)
    out0 = enc(o0)
    assert tuple(out0.shape) == (2, 521)
    assert bool(torch.isfinite(out0).all())
    f0 = enc.forward_full(o0)
    f5 = enc.forward_full(o5)
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


def test_cam_validation():
    k = 3
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, 2, seed=12)

    for drop in (f'view_slot_{k-1:02d}_cam',):
        missing = {key: val for key, val in obs.items() if key != drop}
        try:
            enc(missing)
            raise AssertionError('a missing camera key should raise')
        except KeyError:
            pass
    # camera size disagreeing with the image
    bad = {key: val.clone() for key, val in obs.items()}
    bad['view_slot_01_cam'][:, 8] = H - 1
    try:
        enc(bad)
        raise AssertionError('an h/w mismatch should raise')
    except RuntimeError:
        pass
    # degenerate quaternion on a LIVE slot -> NaN rays, so it must raise
    bad2 = {key: val.clone() for key, val in obs.items()}
    bad2['view_slot_00_cam'][0, 3:7] = 0.0
    try:
        enc(bad2)
        raise AssertionError('a zero quaternion on a live slot should raise')
    except ValueError:
        pass
    # ... but the same camera on an INACTIVE slot is fine (never read)
    ok = {key: val.clone() for key, val in obs.items()}
    ok['view_slot_00_cam'][0, 3:7] = 0.0
    ok['view_mask'] = obs['view_mask'].clone()
    ok['view_mask'][0, 0] = 0.0
    assert bool(torch.isfinite(enc(ok)).all())
    print('camera validation OK (missing key / size mismatch / dead quaternion '
          'on a live slot raise; the same camera on an inactive slot is '
          'ignored)')


def test_ray_branch_inert_then_live():
    k = 3
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, 2, seed=13)

    # At init the zero query makes the attention logits identically zero, so
    # the KEY path is inert; the zero ray output additionally makes e == 0 for
    # ANY input. R's added path is therefore bit-inert and the flag changes
    # nothing.
    _, rays = enc._prepare_slot(
        obs['view_slot_00_image'], obs['view_slot_00_cam'], train=False)
    assert bool((enc.ray_mlp(rays) == 0).all())
    with torch.no_grad():
        a = enc(obs)
        enc.use_ray_encoding = False
        b = enc(obs)
        enc.use_ray_encoding = True
    assert torch.equal(a, b), (a - b).abs().max()

    # A camera change moves the rays, but with a zero query it cannot reach
    # the output yet.
    g = torch.Generator().manual_seed(131)
    obs_cam = {key: val.clone() for key, val in obs.items()}
    obs_cam['view_slot_00_cam'][0] = _cam(1, g)[0]
    assert torch.equal(enc(obs), enc(obs_cam))

    # After the query moves (one optimizer step in a real run) but with the
    # ray output still zero, the key path is live and the camera still has no
    # effect: the rays are severed.
    with torch.no_grad():
        enc.spatial_query.normal_(0.0, 0.02)
    assert torch.equal(enc(obs), enc(obs_cam))

    # Once the output layer is non-zero too, the camera changes the output.
    with torch.no_grad():
        enc.ray_mlp[-1].weight.normal_(0.0, 0.02)
        c = enc(obs)
        d = enc(obs_cam)
    assert not torch.allclose(c, d), (c - d).abs().max()
    assert bool(torch.isfinite(c).all())
    print('ray branch OK (dead key path at init -> live with the query; '
          'rays severed until the output layer moves; camera then matters)')


def test_gradients():
    k = 4
    enc = _build_encoder(k)
    enc.train()
    obs = _fake_obs(k, 2, seed=11)

    def grads():
        enc.zero_grad(set_to_none=True)
        enc(obs).sum().backward()
        return {name: p.grad for name, p in (
            ('conv1', enc.backbone.conv1.weight),
            ('layer3', enc.backbone.layer3[0].conv1.weight),
            ('proj', enc.proj.weight),
            ('value_norm', enc.value_norm.weight),
            ('key_norm', enc.key_norm.weight),
            ('ray_mlp_out', enc.ray_mlp[-1].weight),
            ('ray_mlp_first', enc.ray_mlp[0].weight),
            ('intra_attn', enc.intra_attn.in_proj_weight),
            ('spatial_query', enc.spatial_query),
            ('fusion', enc.fusion.in_proj_weight),
            ('fusion_query', enc.fusion_query),
        )}

    # --- init: the key path is dead (zero query), everything else is live ---
    g = grads()
    for name in ('conv1', 'layer3', 'proj', 'value_norm', 'intra_attn',
                 'spatial_query', 'fusion', 'fusion_query'):
        p = g[name]
        assert p is not None and bool(torch.isfinite(p).all()), name
        assert float(p.abs().sum()) > 0, name
    for name in ('key_norm', 'ray_mlp_out', 'ray_mlp_first'):
        p = g[name]
        assert p is not None and bool((p == 0).all()), (
            f'{name} should be exactly zero at init (q == 0)')

    # --- after the query moves, the key path is live; the zero ray output ---
    # still kills only the FIRST ray-MLP layer
    with torch.no_grad():
        enc.spatial_query.normal_(0.0, 0.02)
    g = grads()
    for name in ('key_norm', 'ray_mlp_out'):
        p = g[name]
        assert p is not None and bool(torch.isfinite(p).all()), name
        assert float(p.abs().sum()) > 0, name
    assert g['ray_mlp_first'] is not None
    assert bool((g['ray_mlp_first'] == 0).all())

    # --- after the output layer moves, the first layer follows ---
    with torch.no_grad():
        enc.ray_mlp[-1].weight.normal_(0.0, 0.02)
    g = grads()
    p = g['ray_mlp_first']
    assert p is not None and bool(torch.isfinite(p).all())
    assert float(p.abs().sum()) > 0
    print('gradients OK (live paths at init; key path/ray MLP activate in the '
          'documented order: query -> output layer -> first layer)')


def test_parameter_count():
    counts = []
    for k in (1, 11):
        enc = _build_encoder(k)
        n = sum(p.numel() for p in enc.parameters())
        counts.append(n)
        assert n == R_PARAMS, (k, n)
    s = SpatialReadoutObsEncoder(
        shape_meta=_shape_meta(1),
        rgb_model=get_resnet('resnet18', weights=None),
        fused_dim=512, n_heads=8, crop_shape=[CROP, CROP], random_crop=True,
        use_group_norm=True, share_rgb_model=True, imagenet_norm=True)
    n_s = sum(p.numel() for p in s.parameters())
    assert n_s == S_PARAMS, n_s
    b = ViewConditionedObsEncoder(
        shape_meta=_shape_meta(1),
        rgb_model=get_resnet('resnet18', weights=None),
        fused_dim=512, n_heads=8, cond_dim=128,
        use_plucker=True, use_eef_hist=True, eef_hist_steps=HIST_STEPS,
        crop_shape=[CROP, CROP], random_crop=True, use_group_norm=True,
        share_rgb_model=True, imagenet_norm=True)
    n_b = sum(p.numel() for p in b.parameters())
    assert n_s < counts[0] < n_b, (n_s, counts[0], n_b)
    print(f'parameter count OK (R {counts[0]:,} = S {n_s:,} + '
          f'{counts[0] - n_s:,}; B {n_b:,})')


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
            config_name='train_diffusion_unet_image_workspace_enc_r_latent')
    assert cfg.task.name == 'enc_r_image_abs_multiview_az75'
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
        'ray_spatial_readout_obs_encoder.RaySpatialReadoutObsEncoder'), (
        obs_cfg._target_)
    assert obs_cfg.fused_dim == 512
    assert obs_cfg.n_heads == 8
    assert obs_cfg.ray_hidden_dim == 512
    assert float(obs_cfg.ray_scale) == 1.0
    assert bool(obs_cfg.use_ray_encoding) is True
    assert tuple(obs_cfg.crop_shape) == (76, 76)
    assert bool(obs_cfg.share_rgb_model) is True
    assert bool(obs_cfg.use_group_norm) is True
    assert bool(obs_cfg.imagenet_norm) is True
    for key in ('cond_dim', 'use_plucker', 'use_eef_hist', 'eef_hist_steps',
                'eef_hist_key', 'resize_shape'):
        assert key not in obs_cfg, key
    enc = hydra.utils.instantiate(obs_cfg)
    assert tuple(enc.output_shape()) == (521,)
    assert enc.n_slots == 11
    assert enc.token_grid == (GRID, GRID)
    assert enc.total_stride == STRIDE
    assert enc.cam_keys == [f'view_slot_{k:02d}_cam' for k in range(11)]
    OmegaConf.resolve(cfg)
    assert cfg.task.dataset.dataset_path == (
        'data/multiview/square_ph_ring13.zarr')
    print('R config OK (ray encoder, K=11 pool [1..11], probe 128/az0=6, '
          'derived cam_keys)')


def test():
    test_architecture()
    test_ray_convention()
    test_crop_ray_alignment()
    test_token_grid()
    test_output_shape()
    test_forward_contract()
    test_inactive_and_independence()
    test_permutation()
    test_n1()
    test_cam_validation()
    test_ray_branch_inert_then_live()
    test_gradients()
    test_parameter_count()
    test_width_interface()
    test_config()
    print('ALL RAY SPATIAL-READOUT ENCODER CHECKS PASSED')


if __name__ == '__main__':
    test()
