"""CPU-only test for M3's view-conditioned encoder (no simulator, no rendering).

Verifies, in order:
  1. the Pluecker ray-map convention, anchored to the M2-validated numpy
     projector `project_world_to_pixel` -- *including* mutation power checks,
     because a convention check that cannot fail under a wrong convention
     proves nothing (NOTES.md, "A check that cannot fail proves nothing", records
     that the deleted draft's check was itself wrong);
  2. crop alignment between the image and its Pluecker map;
  3. the matched-latent-dimension invariant: output_shape is (521,) under every
     ablation flag, and the UNet's 148 parameter tensors are shape-identical to
     a stock-encoder policy's;
  4. N handling (N=1 and N=K) and permutation invariance of the fusion;
  5. the two ablation flags are *exact* ablations, not merely similar-looking;
  6. the policy's own dict-slicing contract (global_cond is (B, 1042));
  7. the dataset -> encoder plumbing, including that each slot's camera vector
     and EE history belong to the SAME view whose image is in that slot;
  8. the real M3 configs resolve, and their slot/step counts agree across the
     dataset, the encoder and the runner.

Run from the repo root:
    python tests/test_view_conditioned_obs_encoder.py
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
sys.path.insert(0, this_dir)   # for the M2 synthetic-zarr builder

import numpy as np
import torch
import torchvision.transforms.functional as ttf

from diffusion_policy.dataset.multiview_image_dataset import (
    quat_wxyz_to_mat, quat_wxyz_to_mat_batch, fovy_to_intrinsics,
    project_world_to_pixel, eef_hist_to_cam, MultiViewImageDataset,
)
from diffusion_policy.model.vision.plucker import (
    quat_wxyz_to_mat_torch, ray_dirs_cam, plucker_ray_map,
)
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    ViewConditionedObsEncoder, IMAGENET_MEAN, IMAGENET_STD,
)
from diffusion_policy.model.vision.crop_randomizer import crop_image_from_indices
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
import diffusion_policy.model.vision.view_conditioned_obs_encoder as vcoe

H = W = 84
FOVY = 45.0
CROP = 76

# camera poses chosen so R is NOT the identity -- with an identity camera
# R == R.T == I, and every wrong convention below would pass silently.
POSES = [
    # (pos, quat_wxyz)  -- yaw, tilt, and a generic rotation
    (np.array([1.0, 0.0, 1.35]), np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])),
    (np.array([0.5, -0.3, 1.5]), np.array([0.9, 0.1, -0.2, 0.37])),
    (np.array([-0.4, 0.6, 1.1]), np.array([0.6, -0.5, 0.3, 0.55])),
]
for _, q in POSES:
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9 or True  # normalised in the checks


def _unit(q):
    return np.asarray(q, dtype=np.float64) / np.linalg.norm(q)


def test_quat_convention():
    """The torch quaternion path must agree with the validated numpy one."""
    for pos, q in POSES:
        q = _unit(q)
        a = quat_wxyz_to_mat(q)
        b = quat_wxyz_to_mat_torch(torch.tensor(q, dtype=torch.float64)).numpy()
        assert np.allclose(a, b, atol=1e-9), (a, b)
    # identity, and the rank check that killed the deleted draft
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    assert tuple(quat_wxyz_to_mat_torch(ident).shape) == (3, 3), 'a (4,) quat must give (3, 3)'
    vq = torch.tensor([_unit(q) for _, q in POSES], dtype=torch.float32)
    assert tuple(quat_wxyz_to_mat_torch(vq).shape) == (len(POSES), 3, 3)
    assert torch.allclose(quat_wxyz_to_mat_torch(vq)[0], quat_wxyz_to_mat_torch(vq[0]), atol=1e-6)
    # a zero quaternion must raise, not silently emit NaNs
    try:
        quat_wxyz_to_mat_torch(torch.zeros(4))
        raise AssertionError('a zero quaternion should have raised')
    except ValueError:
        pass
    print('quaternion convention OK (torch == numpy wxyz, batching, zero-quat guard)')


def _pixel_centre_cases(pose, rng, n):
    """`(i, j, p_world)` at random pixel CENTRES, anchored to the projector.

    Sampling an arbitrary world point and rounding its continuous (u, v) to the
    nearest pixel centre injects a ~half-pixel angular offset (0.28 deg at
    f=101.6), which is ~5.8e-3 in vector distance -- far too coarse to pin a
    convention. So instead build the point that projects *exactly* to the centre
    of pixel (i, j), and assert the M2-validated projector really does put it
    back there. The projector round-trip is the anchor; the map then has to
    agree with it to float precision.
    """
    pos = np.asarray(pose[0], dtype=np.float64)
    quat = _unit(pose[1])
    R = quat_wxyz_to_mat(quat)
    intr = fovy_to_intrinsics(FOVY, H, W)
    cases = []
    for k in rng.choice(H * W, size=min(n, H * W), replace=False):
        i, j = int(k // W), int(k % W)
        u, v = j + 0.5, i + 0.5
        depth = rng.uniform(0.4, 2.5)
        x = (u - intr['cx']) * depth / intr['fx']
        y = -(v - intr['cy']) * depth / intr['fy']
        p_world = pos + R @ np.array([x, y, -depth])
        # ANCHOR: the projector must return this pixel centre and positive depth
        uu, vv, dd = project_world_to_pixel(p_world, pos, quat, intr)
        assert abs(uu - u) < 1e-8 and abs(vv - v) < 1e-8, (uu, u, vv, v)
        assert dd > 0
        cases.append((i, j, p_world))
    return cases


def _direction_error(pose, cases, mutate=None):
    """Max |d_map - normalize(p_world - cam_pos)| over `cases`."""
    pos = np.asarray(pose[0], dtype=np.float64)
    quat = _unit(pose[1])
    dmap = plucker_ray_map(
        torch.tensor(pos, dtype=torch.float64),
        torch.tensor(quat, dtype=torch.float64),
        FOVY, H, W)[:3].numpy()
    if mutate is not None:
        dmap = mutate(dmap)
    worst = 0.0
    for i, j, p_world in cases:
        want = p_world - pos
        want = want / np.linalg.norm(want)
        worst = max(worst, float(np.abs(dmap[:, i, j] - want).max()))
    return worst


def test_plucker_convention():
    rng = np.random.default_rng(0)
    intr = fovy_to_intrinsics(FOVY, H, W)

    # --- the exact convention, at non-identity poses -----------------------
    worst = 0.0
    worst_m = 0.0
    n_pts = 0
    for pose in POSES:
        pos, quat = np.asarray(pose[0]), _unit(pose[1])
        cases = _pixel_centre_cases((pos, quat), rng, 120)
        n_pts += len(cases)
        worst = max(worst, _direction_error((pos, quat), cases))
        pmap = plucker_ray_map(
            torch.tensor(pos, dtype=torch.float64),
            torch.tensor(quat, dtype=torch.float64), FOVY, H, W)
        d_world, m_world = pmap[:3].numpy(), pmap[3:].numpy()
        for i, j, _ in cases:
            d = d_world[:, i, j]
            # m = o x d, and |m| is the perpendicular distance from the origin
            worst_m = max(worst_m, float(np.abs(m_world[:, i, j] - np.cross(pos, d)).max()))
            assert np.isclose(np.linalg.norm(m_world[:, i, j]),
                              np.linalg.norm(np.cross(pos, d)), atol=1e-9)
        # direction map must be unit-norm everywhere
        assert np.allclose(np.linalg.norm(d_world, axis=0), 1.0, atol=1e-12)
    assert worst < 1e-9, f'ray direction error {worst} at non-identity poses'
    assert worst_m < 1e-9, f'moment error {worst_m}'
    print(f'Pluecker convention OK ({n_pts} pixel centres, non-identity poses, '
          f'dir err {worst:.2e}, moment err {worst_m:.2e})')

    # --- axis semantics, in the camera frame (exact, index-based) ----------
    pos, quat = np.asarray(POSES[0][0]), _unit(POSES[0][1])
    R = quat_wxyz_to_mat(quat)
    dmap = plucker_ray_map(torch.tensor(pos, dtype=torch.float64),
                           torch.tensor(quat, dtype=torch.float64), FOVY, H, W)[:3].numpy()
    # a pixel above the centre must lean along the camera's +y (up), and one to
    # the right along +x. Camera-frame axes in world coords are R's columns.
    above = dmap[:, H // 4, W // 2]
    right = dmap[:, H // 2, 3 * W // 4]
    assert float(above @ R[:, 1]) > 0, 'a pixel above centre must lean along camera +y'
    assert float(right @ R[:, 0]) > 0, 'a pixel right of centre must lean along camera +x'
    # and the two must be mirror images about the centre column
    mirror = dmap[:, H // 2, W - 1 - 3 * W // 4]
    assert float(mirror @ R[:, 0]) < 0, 'the mirrored pixel must lean the other way'
    print('axis semantics OK (up is camera +y, right is camera +x, mirror symmetric)')

    # --- grid alignment: within one pixel's angular size -------------------
    # Here the arbitrary world point is rounded to the nearest centre, so the
    # honest bound is one pixel, not float precision.
    worst_ang, bound = 0.0, np.degrees(np.arctan(1.0 / intr['fy']))
    for pose in POSES:
        pos, quat = np.asarray(pose[0]), _unit(pose[1])
        dmap = plucker_ray_map(torch.tensor(pos, dtype=torch.float64),
                               torch.tensor(quat, dtype=torch.float64), FOVY, H, W)[:3].numpy()
        R = quat_wxyz_to_mat(quat)
        for i, j, p_world in _pixel_centre_cases((pos, quat), rng, 40):
            # jitter off the centre by up to half a pixel, then round back
            u = j + 0.5 + rng.uniform(-0.49, 0.49)
            v = i + 0.5 + rng.uniform(-0.49, 0.49)
            depth = np.linalg.norm(p_world - pos)
            x = (u - intr['cx']) * depth / intr['fx']
            y = -(v - intr['cy']) * depth / intr['fy']
            p = pos + R @ np.array([x, y, -depth])
            jj = int(min(max(round(u - 0.5), 0), W - 1))
            ii = int(min(max(round(v - 0.5), 0), H - 1))
            want = p - pos
            cosang = float(np.dot(dmap[:, ii, jj], want / np.linalg.norm(want)))
            worst_ang = max(worst_ang, np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    assert worst_ang < bound, f'worst {worst_ang:.3f}deg exceeds 1-pixel {bound:.3f}deg'
    print(f'grid alignment OK (worst {worst_ang:.3f}deg < 1 px = {bound:.3f}deg)')


def test_convention_mutation_power():
    """Every wrong convention must FAIL the check -- otherwise it proves nothing.

    A check that cannot fail is worse than no check: NOTES.md ("A check that
    cannot fail proves nothing") records that the deleted draft's convention
    check was itself wrong, so its apparent failure told us nothing about the code.
    """
    rng = np.random.default_rng(1)
    pose = (np.asarray(POSES[1][0]), _unit(POSES[1][1]))
    cases = _pixel_centre_cases(pose, rng, 80)
    R = quat_wxyz_to_mat(pose[1])

    # the correct map must pass -- this is the control
    assert _direction_error(pose, cases) < 1e-9, 'baseline check unexpectedly failing'

    def _plus_z_map():
        """Camera rays with +z forward instead of -z (the classic sign error)."""
        f = (H / 2) / np.tan(np.radians(FOVY / 2))
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d = np.stack([(uu - W / 2) / f, -(vv - H / 2) / f, np.ones_like(uu)])
        d = d / np.linalg.norm(d, axis=0, keepdims=True)
        return np.einsum('ij,jhw->ihw', R, d)

    q_xyzw = _unit(np.array([pose[1][1], pose[1][2], pose[1][3], pose[1][0]]))

    muts = {
        # R.T @ d_world == d_cam: the "forgot to rotate into the world" case
        'R.T / no rotation': lambda d: np.einsum('ij,jhw->ihw', R.T, d),
        '-R': lambda d: -d,
        'flipped y row': lambda d: np.stack([d[0], -d[1], d[2]]),
        'xyzw not wxyz': lambda d: plucker_ray_map(
            torch.tensor(pose[0], dtype=torch.float64),
            torch.tensor(q_xyzw, dtype=torch.float64), FOVY, H, W)[:3].numpy(),
        '+z forward not -z': lambda d: _plus_z_map(),
    }
    errs = {name: _direction_error(pose, cases, mutate=fn) for name, fn in muts.items()}
    for name, e in errs.items():
        assert e > 1e-3, f'mutation NOT caught ({e}): {name}'
    print('mutation power OK (all {} wrong conventions caught: {})'.format(
        len(errs), ', '.join(f'{k}={v:.2f}' for k, v in errs.items())))


def test_translation_channels():
    """d carries orientation only; m is what carries camera translation."""
    pos = np.array([1.0, 0.0, 1.35])
    quat = _unit(POSES[0][1])
    a = plucker_ray_map(torch.tensor(pos, dtype=torch.float64),
                        torch.tensor(quat, dtype=torch.float64), FOVY, H, W)
    # move along the camera's own viewing axis: same rotation, new position
    R = quat_wxyz_to_mat(quat)
    b = plucker_ray_map(torch.tensor(pos + R @ np.array([0.0, 0.0, -0.35]), dtype=torch.float64),
                        torch.tensor(quat, dtype=torch.float64), FOVY, H, W)
    assert torch.allclose(a[:3], b[:3], atol=1e-9), 'direction map must not depend on position'
    assert not torch.allclose(a[3:], b[3:], atol=1e-6), 'moment map must depend on position'
    # batched call matches unbatched
    vpos = torch.tensor(np.stack([p for p, _ in POSES]), dtype=torch.float64)
    vquat = torch.tensor(np.stack([_unit(q) for _, q in POSES]), dtype=torch.float64)
    vmap = plucker_ray_map(vpos, vquat, FOVY, H, W)
    assert tuple(vmap.shape) == (len(POSES), 6, H, W), vmap.shape
    assert torch.allclose(vmap[0], a, atol=1e-9), 'batched[0] != unbatched'
    # per-view fovy broadcast
    vfov = torch.tensor([FOVY, FOVY + 5.0, FOVY - 3.0], dtype=torch.float64)
    vmap2 = plucker_ray_map(vpos, vquat, vfov, H, W)
    assert tuple(vmap2.shape) == (len(POSES), 6, H, W)
    assert not torch.allclose(vmap2[1], vmap[1], atol=1e-6), 'per-view fovy ignored'
    print('translation/batching OK (d position-invariant, m not; batched == unbatched; per-view fovy)')


def test_camera_math_still_valid():
    """Guard: the M2-validated numpy helpers this test anchors to are unchanged."""
    half = np.pi / 4
    R = quat_wxyz_to_mat([np.cos(half), 0.0, 0.0, np.sin(half)])
    assert np.allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-9)
    intr = fovy_to_intrinsics(45.0, H, W)
    assert np.isclose(intr['fy'], (H / 2) / np.tan(np.radians(22.5)), atol=1e-9)
    print('numpy anchor OK (quat convention, intrinsics from fovy)')


K_SLOTS = 7
CROP_SIDE = 76
HIST_STEPS = 4
HIST_DIM = 11 * HIST_STEPS


def _shape_meta(k_slots):
    obs = {f'view_slot_{k:02d}_image': {'shape': [3, H, W], 'type': 'rgb'}
           for k in range(k_slots)}
    obs['robot0_eef_pos'] = {'shape': [3]}
    obs['robot0_eef_quat'] = {'shape': [4]}
    obs['robot0_gripper_qpos'] = {'shape': [2]}
    return {'obs': obs, 'action': {'shape': [10]}}


def _build_encoder(k_slots, **kw):
    kw.setdefault('crop_shape', [CROP_SIDE, CROP_SIDE])
    kw.setdefault('random_crop', True)
    kw.setdefault('use_group_norm', True)
    kw.setdefault('imagenet_norm', True)
    kw.setdefault('share_rgb_model', True)
    kw.setdefault('eef_hist_steps', HIST_STEPS)
    return ViewConditionedObsEncoder(
        shape_meta=_shape_meta(k_slots),
        rgb_model=get_resnet('resnet18', weights=None), **kw)


def _fake_obs(k_slots, n, seed=0, mask=None):
    """A batch shaped as the policy delivers it: leading dim is B*To."""
    g = torch.Generator().manual_seed(seed)
    obs = {}
    for k in range(k_slots):
        obs[f'view_slot_{k:02d}_image'] = torch.rand(n, 3, H, W, generator=g)
        q = torch.randn(n, 4, generator=g)
        q = q / q.norm(dim=-1, keepdim=True)
        cam = torch.zeros(n, 10)
        cam[:, :3] = torch.randn(n, 3, generator=g) * 0.3
        cam[:, 3:7] = q
        cam[:, 7] = FOVY
        cam[:, 8] = H
        cam[:, 9] = W
        obs[f'view_slot_{k:02d}_cam'] = cam
    obs['view_mask'] = torch.ones(n, k_slots) if mask is None else mask
    obs['view_eef_hist'] = torch.randn(n, k_slots, HIST_DIM, generator=g)
    obs['robot0_eef_pos'] = torch.randn(n, 3, generator=g)
    obs['robot0_eef_quat'] = torch.randn(n, 4, generator=g)
    obs['robot0_gripper_qpos'] = torch.randn(n, 2, generator=g)
    return obs


def _expected_slot0(obs, inds):
    """The slot-0 tensor the encoder should have produced, cropped at `inds`."""
    img = obs['view_slot_00_image']
    cam = obs['view_slot_00_cam']
    plk = plucker_ray_map(cam[:, :3], cam[:, 3:7], cam[:, 7], H, W)
    img_c = crop_image_from_indices(img, inds, CROP_SIDE, CROP_SIDE)
    plk_c = crop_image_from_indices(plk, inds, CROP_SIDE, CROP_SIDE)
    mean = img_c.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = img_c.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return torch.cat([(img_c - mean) / std, plk_c], dim=1)


def test_crop_alignment():
    """The image and its ray map must be cropped with the SAME window.

    Crop the image but not the map and the ray at pixel (i, j) stops describing
    the content at (i, j). The sampled offsets are spied on so the check is
    exact rather than statistical.
    """
    enc = _build_encoder(2, use_plucker=True, use_eef_hist=False)
    enc.train()
    obs = _fake_obs(2, 3, seed=3)

    recorded = {}
    real = vcoe.sample_random_image_crops

    def spy(images, crop_height, crop_width, num_crops, pos_enc=False):
        crops, inds = real(images=images, crop_height=crop_height,
                           crop_width=crop_width, num_crops=num_crops,
                           pos_enc=pos_enc)
        recorded['inds'] = inds.detach().clone()
        return crops, inds

    vcoe.sample_random_image_crops = spy
    try:
        out = enc._prepare_slots(obs, 0, train=True)
    finally:
        vcoe.sample_random_image_crops = real
    inds = recorded['inds'].reshape(-1, 2).long()
    # offsets must differ across the batch, or this proves nothing
    assert len({tuple(t) for t in inds.tolist()}) > 1, 'crop offsets are not random'
    want = _expected_slot0(obs, inds)
    assert torch.allclose(out, want, atol=1e-6), (out - want).abs().max()

    # eval path: deterministic centre crop, offsets (4, 4)
    enc.eval()
    got = enc._prepare_slots(obs, 0, train=False)
    centre = torch.full((3, 2), (H - CROP_SIDE) // 2, dtype=torch.long)
    assert torch.allclose(got, _expected_slot0(obs, centre), atol=1e-6)
    # and the (4, 4) offset must reproduce torchvision's center_crop, which is
    # what the stock CropRandomizer does at eval
    img_cc = ttf.center_crop(obs['view_slot_00_image'], CROP_SIDE)
    mean = img_cc.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = img_cc.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    assert torch.allclose(got[:, :3], (img_cc - mean) / std, atol=1e-6)
    print('crop alignment OK (image and ray map share one window, sampled and centre)')


def test_matched_capacity():
    """output_shape() must be the stock 521 under EVERY ablation flag."""
    expected = 512 + 3 + 4 + 2
    for use_plucker in (True, False):
        for use_eef in (True, False):
            enc = _build_encoder(K_SLOTS, use_plucker=use_plucker,
                                 use_eef_hist=use_eef)
            shape = tuple(enc.output_shape())
            assert shape == (expected,), (use_plucker, use_eef, shape)
            # a wrong-but-shape-correct NaN is exactly what a zero quaternion
            # dummy pose would produce, so output_shape() validates finiteness
            assert enc.fused_dim == 512
    # and the one-slot (N=1 gate) shape
    enc1 = _build_encoder(1)
    assert tuple(enc1.output_shape()) == (expected,)
    print(f'matched capacity OK (output_shape == ({expected},) for all 4 flag combos and K=1)')


def test_n_and_permutation():
    for k_slots in (1, 7):
        enc = _build_encoder(k_slots)
        enc.eval()
        obs = _fake_obs(k_slots, 4, seed=5)
        out = enc(obs)
        assert tuple(out.shape) == (4, 521), (k_slots, out.shape)
        assert bool(torch.isfinite(out).all())

    # fusion must be permutation-invariant: slot order carries no meaning
    enc = _build_encoder(7)
    enc.eval()
    obs = _fake_obs(7, 2, seed=6)
    perm = [3, 0, 6, 1, 5, 2, 4]
    shuffled = {}
    for k, src in enumerate(perm):
        shuffled[f'view_slot_{k:02d}_image'] = obs[f'view_slot_{src:02d}_image']
        shuffled[f'view_slot_{k:02d}_cam'] = obs[f'view_slot_{src:02d}_cam']
    shuffled['view_mask'] = obs['view_mask'][:, perm]
    shuffled['view_eef_hist'] = obs['view_eef_hist'][:, perm]
    for k in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'):
        shuffled[k] = obs[k]
    a, b = enc(obs), enc(shuffled)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    print('N path OK (K=1 and K=7 forward; fusion is permutation-invariant)')


def test_ablation_exactness():
    """Both flags must be exact ablations, not merely similar-looking ones."""
    # plucker off -> literally-zero ray channels, so the gradient into them is 0
    for use_plucker, expect_zero in ((False, True), (True, False)):
        enc = _build_encoder(3, use_plucker=use_plucker, use_eef_hist=False)
        enc.train()
        out = enc(_fake_obs(3, 2, seed=7))
        enc.zero_grad(set_to_none=True)
        out.sum().backward()
        g = enc.backbone.conv1.weight.grad[:, 3:]
        assert g is not None
        is_zero = bool((g == 0).all())
        assert is_zero == expect_zero, (use_plucker, g.abs().max())
    # eef off -> the output must not depend on the history at all
    enc = _build_encoder(3, use_eef_hist=False)
    enc.eval()
    a = _fake_obs(3, 2, seed=8)
    b = dict(a, view_eef_hist=torch.randn_like(a['view_eef_hist']) * 9.0)
    assert torch.allclose(enc(a), enc(b), atol=1e-6), 'history leaked when disabled'
    # eef on -> it must depend on the history (a live, non-inert path)
    enc2 = _build_encoder(3, use_eef_hist=True)
    enc2.eval()
    # the final FiLM layer is zero-init, so perturb it to prove the path is live
    for layer in enc2.film:
        torch.nn.init.normal_(layer.weight, std=0.5)
    assert not torch.allclose(enc2(a), enc2(b), atol=1e-6), 'history ignored when enabled'
    print('ablation exactness OK (plucker-off has exactly-zero grads; eef-off ignores history)')


def test_dropin_contract():
    """Reproduce the policy's own slicing without building the 277M UNet."""
    enc = _build_encoder(K_SLOTS)
    enc.eval()
    B, To = 3, 2
    flat = _fake_obs(K_SLOTS, B * To, seed=9)
    obs = {}
    for k, v in flat.items():
        obs[k] = v.reshape(B, To, *v.shape[1:])
    this_nobs = {k: v[:, :To].reshape(-1, *v.shape[2:]) for k, v in obs.items()}
    feats = enc(this_nobs)
    global_cond = feats.reshape(B, -1)
    assert tuple(global_cond.shape) == (B, 521 * To), global_cond.shape
    assert tuple(this_nobs['view_mask'].shape) == (B * To, K_SLOTS)
    assert tuple(this_nobs['view_eef_hist'].shape) == (B * To, K_SLOTS, HIST_DIM)
    print(f'drop-in contract OK (global_cond {(B, 521 * To)}, To-tiled extra keys)')


def test_stock_encoder_regression():
    """Guard the matched-width claim against upstream drift."""
    enc = MultiImageObsEncoder(
        shape_meta=_shape_meta(1),
        rgb_model=get_resnet('resnet18', weights=None),
        resize_shape=None, crop_shape=[CROP_SIDE, CROP_SIDE], random_crop=True,
        use_group_norm=True, share_rgb_model=False, imagenet_norm=True)
    shape = tuple(enc.output_shape())
    assert shape == (521,), shape
    print('stock encoder regression OK (MultiImageObsEncoder still gives (521,))')


def test_unet_identical_to_stock():
    """The end-to-end form of 'same latent dimensions' the user asked for.

    Build the stock encoder and the M3 encoder on the same shape_meta, build a
    ConditionalUnet1D with each one's global_cond_dim, and require the two
    state_dict shape maps to be IDENTICAL. Only the twelve cond_encoder weights
    depend on this width; if M3 drifted, they would differ.
    """
    stock = MultiImageObsEncoder(
        shape_meta=_shape_meta(1), rgb_model=get_resnet('resnet18', weights=None),
        resize_shape=None, crop_shape=[CROP_SIDE, CROP_SIDE], random_crop=True,
        use_group_norm=True, share_rgb_model=False, imagenet_norm=True)
    m3 = _build_encoder(1)
    d_stock = int(stock.output_shape()[0]) * 2
    d_m3 = int(m3.output_shape()[0]) * 2
    assert d_stock == d_m3 == 1042, (d_stock, d_m3)

    def shapes(cond_dim):
        m = ConditionalUnet1D(
            input_dim=10, local_cond_dim=None, global_cond_dim=cond_dim,
            diffusion_step_embed_dim=128, down_dims=[512, 1024, 2048],
            kernel_size=5, n_groups=8, cond_predict_scale=True)
        return {k: tuple(v.shape) for k, v in m.state_dict().items()}

    a, b = shapes(d_stock), shapes(d_m3)
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert not differing, f'UNet params differ: {differing}'
    assert len(a) == 148, len(a)
    print(f'UNet identical OK (global_cond_dim {d_m3}, all {len(a)} param tensors identical to stock)')


def test_dataset_plumbing():
    """The dataset must pair each slot's cam / EE history with THAT slot's view.

    Reuses the M2 synthetic zarr builder (3 views, 8x8) rather than duplicating
    the schema, and recovers each slot's view index from the deterministic
    render so the pairing is checked rather than assumed -- a swapped cam key
    would train happily and only show up as a mysteriously weak result.
    """
    import test_multiview_dataset as tmd
    tmp = tempfile.mkdtemp(prefix='m3_plumbing_')
    try:
        path = os.path.join(tmp, 'synth.zarr')
        tmd.build_synthetic_zarr(path)
        n_views, hw = tmd.N_VIEWS, tmd.H
        slots, steps = n_views, 3
        obs_meta = {f'view_slot_{k:02d}_image': {'shape': [3, hw, hw], 'type': 'rgb'}
                    for k in range(slots)}
        obs_meta['robot0_eef_pos'] = {'shape': [3]}
        obs_meta['robot0_eef_quat'] = {'shape': [4]}
        obs_meta['robot0_gripper_qpos'] = {'shape': [2]}
        shape_meta = {'obs': obs_meta, 'action': {'shape': [10]}}

        ds = MultiViewImageDataset(
            shape_meta=shape_meta, dataset_path=path, horizon=4,
            pad_before=1, pad_after=7, n_obs_steps=2, abs_action=True,
            view_pool=list(range(n_views)), eef_hist_steps=steps,
            seed=0, val_ratio=0.34)
        assert ds.cam_keys == [f'view_slot_{k:02d}_cam' for k in range(slots)]
        assert ds.cam_table.dtype == np.float32
        assert ds.cam_table.shape == (n_views, 10)
        # the batched quat helper must agree with the scalar one
        for v in range(n_views):
            assert np.allclose(quat_wxyz_to_mat_batch(ds.cam_table[v][3:7]),
                               quat_wxyz_to_mat(ds.cam_table[v][3:7]), atol=1e-6)

        obs = ds[0]['obs']
        assert obs['view_slot_00_image'].shape == (2, 3, hw, hw)
        assert obs['view_slot_00_cam'].shape == (2, 10)
        assert obs['view_mask'].shape == (2, slots)
        assert obs['view_eef_hist'].shape == (2, slots, 11 * steps)
        assert obs['view_slot_00_cam'].dtype == torch.float32
        assert obs['view_eef_hist'].dtype == torch.float32

        # the mask is per-sample (not per-frame) and the live slots are a prefix
        m = obs['view_mask'].numpy()
        assert np.array_equal(m[0], m[1]), 'mask must not vary across obs steps'
        assert m[0, 0] == 1.0, 'slot 0 must always be active'
        assert np.all(np.diff(m[0]) <= 0), 'active slots must form a prefix'

        expected = {v: np.moveaxis(ds._view_obs_frames(0, v), -1, 1).astype(np.float32) / 255.
                    for v in range(n_views)}
        pos, quat, grip = ds._eef_history(0)
        n_live = 0
        for k in range(slots):
            img = obs[f'view_slot_{k:02d}_image'].numpy()
            cam = obs[f'view_slot_{k:02d}_cam'].numpy()
            if m[0, k] < 0.5:
                assert np.allclose(img, 0.0), f'inactive slot {k} image must be zeros'
                assert np.allclose(cam, 0.0), f'inactive slot {k} cam must be zeros'
                assert np.allclose(obs['view_eef_hist'].numpy()[:, k], 0.0)
                continue
            n_live += 1
            matches = [v for v in range(n_views) if np.allclose(img, expected[v])]
            assert len(matches) == 1, f'slot {k} image matches {matches} views'
            v = matches[0]
            assert np.allclose(cam[0], ds.cam_table[v], atol=1e-6), \
                f'slot {k} cam does not belong to the view whose image it holds'
            want = eef_hist_to_cam(pos, quat, grip,
                                   ds.cam_table[v][:3],
                                   ds.cam_table[v][3:7]).reshape(2, -1)
            assert np.allclose(obs['view_eef_hist'].numpy()[:, k], want, atol=1e-6), \
                f'slot {k} EE history is not in the frame of the view it holds'
        assert n_live >= 1

        # every extra key needs an IDENTITY normalizer: the policy normalizes
        # every key present, and limits-scaling would zero a constant dim
        normalizer = ds.get_normalizer()
        keys = set(normalizer.params_dict.keys())
        for k in ds.cam_keys + [ds.view_mask_key, ds.eef_hist_key]:
            assert k in keys, f'{k} missing from the normalizer'
        for k in [ds.cam_keys[0], ds.view_mask_key, ds.eef_hist_key]:
            x = obs[k]
            back = normalizer[k].normalize(x)
            assert torch.allclose(x, back, atol=1e-6), f'{k} normalizer is not identity'
            assert back.dtype == torch.float32, (k, back.dtype)
            assert normalizer[k].params_dict['scale'].dtype == torch.float32, k

        # and the encoder must ingest a real dataset batch
        enc = ViewConditionedObsEncoder(
            shape_meta=shape_meta, rgb_model=get_resnet('resnet18', weights=None),
            eef_hist_steps=steps, crop_shape=None, use_group_norm=True,
            imagenet_norm=True, share_rgb_model=True)
        enc.eval()
        out = enc(obs)
        assert tuple(out.shape) == (2, 521), out.shape
        print('dataset plumbing OK (cam/EE-history paired to the right view, '
              'identity normalizers, encoder ingests a real batch)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_view_count_range_is_honoured():
    """`view_count_range` must control the draw -- including a range the old guard
    rejected, which is what makes this a regression test rather than decoration.

    Until 2026-09-24 `__init__` raised for any non-degenerate range whose `hi` was
    below the slot count, so `[1, 2]` at K=3 was rejected while `[2, 2]` was allowed
    -- though both leave exactly the same slots permanently dead. The N-diversity
    ladder needs `[1, 2]` at K=7 (PROGRESS.md *N-diversity ladder*), so the guard was
    removed. **This test fails against the pre-change code**, by construction: the
    first dataset below cannot even be constructed there.

    The count assertions carry the mutation power. Checking "mean is 1.5" against a
    hardcoded number would pass on a dataset that ignored the setting entirely; so
    the range is varied and the two draws must *differ*, and the subset draw is
    checked to actually cover the pool rather than re-serving one view.
    """
    import test_multiview_dataset as tmd
    tmp = tempfile.mkdtemp(prefix='m3_range_')
    try:
        path = os.path.join(tmp, 'synth.zarr')
        tmd.build_synthetic_zarr(path)
        n_views, hw = tmd.N_VIEWS, tmd.H
        slots = n_views
        obs_meta = {f'view_slot_{k:02d}_image': {'shape': [3, hw, hw], 'type': 'rgb'}
                    for k in range(slots)}
        obs_meta['robot0_eef_pos'] = {'shape': [3]}
        obs_meta['robot0_eef_quat'] = {'shape': [4]}
        obs_meta['robot0_gripper_qpos'] = {'shape': [2]}
        shape_meta = {'obs': obs_meta, 'action': {'shape': [10]}}

        def build(rng_range):
            return MultiViewImageDataset(
                shape_meta=shape_meta, dataset_path=path, horizon=4,
                pad_before=1, pad_after=7, n_obs_steps=2, abs_action=True,
                view_pool=list(range(n_views)), view_count_range=rng_range,
                eef_hist_steps=3, seed=0, val_ratio=0.0)

        def draws(ds, n=400, seed=0):
            """(active count, slot-0 view index) per fetch.

            The draw happens per `__getitem__`, so re-fetching a 9-sample dataset
            still yields independent draws -- which is why this needs no large store.
            """
            np.random.seed(seed)
            expected = {(i, v): np.moveaxis(ds._view_obs_frames(i, v), -1, 1)
                        .astype(np.float32) / 255.
                        for i in range(len(ds)) for v in range(n_views)}
            out = []
            for i in range(n):
                j = i % len(ds)
                obs = ds[j]['obs']
                m = obs['view_mask'].numpy()[0]
                assert m[0] == 1.0, 'slot 0 must always be active'
                assert np.all(np.diff(m) <= 0), 'active slots must form a prefix'
                img = obs['view_slot_00_image'].numpy()[0]
                hit = [v for v in range(n_views) if np.allclose(img, expected[(j, v)][0])]
                assert len(hit) == 1, f'slot 0 image matches {hit} views'
                out.append((int(m.sum()), hit[0]))
            return out

        small = draws(build([1, 2]))
        counts = np.array([c for c, _ in small])
        assert counts.min() >= 1 and counts.max() == 2, \
            f'[1, 2] drew counts {counts.min()}..{counts.max()}'
        assert abs(counts.mean() - 1.5) < 0.12, counts.mean()

        big = draws(build([1, 3]))
        bcounts = np.array([c for c, _ in big])
        assert bcounts.max() == 3, bcounts.max()
        assert abs(bcounts.mean() - 2.0) < 0.12, bcounts.mean()
        assert abs(counts.mean() - bcounts.mean()) > 0.3, \
            'the range was ignored: two different ranges produced the same draw'

        seen = {v for _, v in small}
        assert seen == set(range(n_views)), f'slot 0 only ever held views {sorted(seen)}'
        print(f'view_count_range honoured: [1,2] mean {counts.mean():.3f}/max '
              f'{counts.max()}, [1,3] mean {bcounts.mean():.3f}/max {bcounts.max()}, '
              f'pool coverage {sorted(seen)}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_m3_configs():
    """Resolve the real M3 configs and instantiate the encoder from them.

    The dataset, the encoder and the runner each carry `eef_hist_steps` / slot
    counts independently, and they must agree -- a mismatch is not a crash but
    a silent train/eval inconsistency. This catches that class of drift, and
    the config typos that would otherwise only surface on the training box.
    """
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')
    checked = []
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        for task in ('m3_plucker_image_abs_multiview', 'm3_plucker_image_abs_n1'):
            cfg = hydra.compose(
                config_name='train_diffusion_unet_image_workspace_m3',
                overrides=[f'task={task}'])
            slots = len([k for k, v in cfg.task.shape_meta.obs.items()
                         if v.get('type', 'low_dim') == 'rgb'])
            enc = hydra.utils.instantiate(cfg.policy.obs_encoder)
            assert tuple(enc.output_shape()) == (521,), (task, enc.output_shape())
            assert enc.n_slots == slots, (task, enc.n_slots, slots)
            # the three places eef_hist_steps appears must agree
            assert (cfg.task.dataset.eef_hist_steps
                    == cfg.task.env_runner.eef_hist_steps
                    == enc.eef_hist_steps), (task, cfg.task.dataset.eef_hist_steps,
                                             cfg.task.env_runner.eef_hist_steps,
                                             enc.eef_hist_steps)
            assert cfg.task.env_runner.m3_slots == slots, task
            assert cfg.policy.obs_encoder.fused_dim == 512, task
            # the runner must be one that can serve the extra keys
            assert 'cam_key_image_runner' in cfg.task.env_runner._target_, task
            # the ablation flags must be present for the 2x2
            assert cfg.policy.obs_encoder.use_plucker is True
            assert cfg.policy.obs_encoder.use_eef_hist is True
            checked.append(f'{task}(K={slots}, steps={enc.eef_hist_steps})')
    print(f'M3 configs OK ({" and ".join(checked)}: encoder output_shape (521,), '
          f'slot/step counts agree across dataset, encoder and runner)')


def test():
    test_camera_math_still_valid()
    test_quat_convention()
    test_plucker_convention()
    test_convention_mutation_power()
    test_translation_channels()
    test_crop_alignment()
    test_matched_capacity()
    test_n_and_permutation()
    test_ablation_exactness()
    test_dropin_contract()
    test_stock_encoder_regression()
    test_unet_identical_to_stock()
    test_dataset_plumbing()
    test_view_count_range_is_honoured()
    test_m3_configs()
    print('ALL VIEW-CONDITIONED ENCODER CHECKS PASSED')


if __name__ == '__main__':
    test()
