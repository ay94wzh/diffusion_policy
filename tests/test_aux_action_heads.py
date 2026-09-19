"""
CPU-only tests for M4's per-view auxiliary action heads (no simulator, no GPU).

Run from the repo root:
    python tests/test_aux_action_heads.py

Four things are checked, in rough order of how much they buy:

1. `action_to_cam`'s rotation convention, WITH mutation power. The repo has
   twice shipped a convention check that could not fail (PROGRESS.md sections
   9.2 and 10.2), so every mutation below must produce a large, quantified
   error -- including the one a wrong implementation is most likely to ship.
2. The dataset emits and pairs the target correctly, recoverable from the
   renders alone (a swapped cam table would train happily and read as "aux
   didn't help").
3. The policy's aux term: masking, gradient path, zero-init, and the two
   properties the ablation rests on -- that `forward` is unchanged, and that
   weight 0 reproduces the parent's loss and gradients BIT-FOR-BIT.
4. The M4 configs compose, and the model they instantiate is byte-identical to
   M3's (the user's constraint: hold the model fixed, add one part).
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
sys.path.insert(0, this_dir)          # for `import test_multiview_dataset`

import numpy as np
import torch

from diffusion_policy.dataset.multiview_image_dataset import (
    MultiViewImageDataset, action_to_cam, rot6d_to_mat, quat_wxyz_to_mat,
    eef_hist_to_cam, AUX_ACTION_KEY, ACTION_DIM)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.per_view_aux_head import PerViewAuxActionHead
from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    ViewConditionedObsEncoder)
from diffusion_policy.policy.diffusion_unet_image_policy import (
    DiffusionUnetImagePolicy)
from diffusion_policy.policy.diffusion_unet_image_policy_aux import (
    DiffusionUnetImagePolicyAux, masked_view_mean)
from diffusion_policy.common.pytorch_util import dict_apply

AUX_N_STEPS = 8


def _rand_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# 1. the transform
# ---------------------------------------------------------------------------

def test_rot6d_matches_pytorch3d():
    """`rot6d_to_mat` must be pytorch3d's `rotation_6d_to_matrix`, not merely
    'a' rotation. The stored actions were produced by that code path, so a
    hand-rolled Gram-Schmidt that disagrees on row-vs-column or the b3 sign
    would be wrong in a way nothing else in training would catch."""
    tf = RotationTransformer('rotation_6d', 'matrix')
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        R = quat_wxyz_to_mat(_rand_quat(rng))
        d6 = np.concatenate([R[0, :], R[1, :]])       # the rows-0-1 convention
        ours = rot6d_to_mat(d6)
        theirs = np.asarray(tf.forward(d6))
        worst = max(worst, np.abs(ours - theirs).max())
    assert worst < 1e-12, worst
    print(f'  rot6d_to_mat == RotationTransformer(rotation_6d->matrix) '
          f'(max|diff| {worst:.2e} over 200 random rotations)')


def test_action_to_cam_convention():
    """The camera-frame action transform, with mutation power.

    Every fixture camera is NON-IDENTITY and every fixture pose is a real
    rotation: at an identity camera all of these conventions agree, which is
    precisely how a broken check passes (PROGRESS.md section 9.2).
    """
    rng = np.random.default_rng(1)
    worst_anchor = 0.0
    worst_sibling = 0.0
    for _ in range(100):
        cam_pos = rng.normal(size=3) * 0.3 + np.array([0.5, 0.0, 1.3])
        cam_q = _rand_quat(rng)
        ee_q = _rand_quat(rng)
        R_c = quat_wxyz_to_mat(cam_q)
        R_e = quat_wxyz_to_mat(ee_q)
        p = rng.normal(size=3)
        gripper = 0.7
        d6 = np.concatenate([R_e[0, :], R_e[1, :]])
        action = np.concatenate([p, d6, [gripper]])

        out = action_to_cam(action, cam_pos, cam_q)

        # anchored to the numpy camera math M2's gate 2 validated against the
        # simulator -- NOT to the code under test
        worst_anchor = max(worst_anchor,
                           np.abs(out[:3] - R_c.T @ (p - cam_pos)).max(),
                           np.abs(rot6d_to_mat(out[3:9]) - R_c.T @ R_e).max())
        # and pinned to its M2-validated sibling, which rotates an EE pose with
        # the same convention
        hist = eef_hist_to_cam(p, ee_q, np.zeros(2), cam_pos, cam_q)
        worst_sibling = max(worst_sibling, np.abs(out[3:9] - hist[3:9]).max())
        # the gripper is frame-independent and must pass through untouched
        assert out[9] == np.float32(gripper)
        # NOTE the tolerance: `out` is float32 by contract (the dataset emits
        # float32 and the normalizer's scale is float32), so the comparison
        # against float64 references bottoms out at ~1e-7 relative, not 1e-16.
        # The mutation thresholds below are 1e-3, four orders above this floor.
        assert np.abs(out[:3] - R_c.T @ (p - cam_pos)).max() < 1e-6

    assert worst_anchor < 1e-6, worst_anchor
    assert worst_sibling < 1e-6, worst_sibling
    print(f'  action_to_cam anchored to quat_wxyz_to_mat (max|diff| '
          f'{worst_anchor:.2e}) and pinned to eef_hist_to_cam '
          f'(max|diff| {worst_sibling:.2e}) over 100 non-identity cameras')

    # ---- mutation power: each of these must be CAUGHT ----------------------
    cam_pos = np.array([0.62, -0.11, 1.28])
    cam_q = _rand_quat(np.random.default_rng(7))
    ee_q = _rand_quat(np.random.default_rng(8))
    R_c = quat_wxyz_to_mat(cam_q)
    R_e = quat_wxyz_to_mat(ee_q)
    p = np.array([0.15, -0.22, 1.05])
    d6 = np.concatenate([R_e[0, :], R_e[1, :]])
    action = np.concatenate([p, d6, [0.7]])
    good = action_to_cam(action, cam_pos, cam_q)

    def rot_blk(a):
        return a[3:9]

    mutations = {
        'R_c instead of R_c^T (pos)': (R_c @ (p - cam_pos)) - good[:3],
        'forgot -cam_pos': (R_c.T @ p) - good[:3],
        'rot6d left in the base frame': rot_blk(good) - d6,
        'pos left in the base frame': good[:3] - p,
        'first two COLUMNS instead of rows': (
            rot_blk(good)
            - np.concatenate([(R_c.T @ R_e)[:, 0], (R_c.T @ R_e)[:, 1]])),
        'R_e R_c^T (wrong order)': (
            rot_blk(good) - np.concatenate([(R_e @ R_c.T)[0, :], (R_e @ R_c.T)[1, :]])),
        '6x6 linear shortcut on the 6d vector': (
            rot_blk(good) - (R_c.T @ d6.reshape(2, 3).T).T.reshape(-1)),
    }
    for name, err in mutations.items():
        e = float(np.abs(err).max())
        assert e > 1e-3, f'mutation NOT caught ({name}): max|err| = {e:.2e}'
    print(f'  mutation power OK on {len(mutations)} mutations '
          f'(all > 1e-3, worst-case convention bugs are catchable)')

    # The executable form of "why the round trip exists": the 6x6 shortcut is
    # EXACTLY right at an identity camera and wrong everywhere else. If this
    # assertion ever fails, the test above has stopped being meaningful.
    ident_q = np.array([1.0, 0.0, 0.0, 0.0])
    ident_c = np.zeros(3)
    good_i = action_to_cam(action, ident_c, ident_q)
    shortcut_i = (np.eye(3) @ d6.reshape(2, 3).T).T.reshape(-1)
    assert np.abs(rot_blk(good_i) - shortcut_i).max() < 1e-6   # float32 floor
    shortcut_c = (R_c.T @ d6.reshape(2, 3).T).T.reshape(-1)
    assert np.abs(rot_blk(good) - shortcut_c).max() > 1e-3
    print('  6x6 shortcut passes at R_c=I and fails at a real pose '
          '(so an identity-camera test is vacuous)')

    # and the transform is invertible: cam -> base recovers the original action
    action_back = np.concatenate([
        R_c @ good[:3] + cam_pos,
        np.concatenate([(R_c @ rot6d_to_mat(good[3:9]))[0, :],
                        (R_c @ rot6d_to_mat(good[3:9]))[1, :]]),
        good[9:]])
    assert np.abs(action_back - action).max() < 1e-6
    print('  cam -> base round trip recovers the original action')


# ---------------------------------------------------------------------------
# 2. the dataset
# ---------------------------------------------------------------------------

def _slot_shape_meta(n_slots, hw):
    obs = {f'view_slot_{k:02d}_image': {'shape': [3, hw, hw], 'type': 'rgb'}
           for k in range(n_slots)}
    obs['robot0_eef_pos'] = {'shape': [3]}
    obs['robot0_eef_quat'] = {'shape': [4]}
    obs['robot0_gripper_qpos'] = {'shape': [2]}
    return {'obs': obs, 'action': {'shape': [10]}}


def _distinct_cam_poses(path, n_views):
    """Overwrite the synthetic builder's camera poses with DISTINCT,
    NON-IDENTITY ones.

    The shared builder writes identity quaternions and gives two of the three
    views the same position -- under which every rotation convention agrees and
    a swapped cam table can hide. This is the fixture that makes the plumbing
    test meaningful.
    """
    import zarr
    root = zarr.open(path, mode='r+')
    axes = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])[:n_views]
    ang = np.deg2rad(np.array([25.0, -35.0, 45.0]))[:n_views] / 2.0
    quat = np.concatenate([np.cos(ang)[:, None], axes * np.sin(ang)[:, None]], axis=1)
    pos = np.array([[0.60, 0.00, 1.30],
                    [1.00, 0.30, 1.10],
                    [0.40, -0.20, 1.40]])[:n_views]
    root['meta']['view_cam_pos'][:] = pos
    root['meta']['view_cam_quat_wxyz'][:] = quat


def _build_m4_dataset(tmp, emit_aux_action=True, **over):
    """M4-mode dataset over the M2 synthetic zarr with distinct cam poses."""
    import test_multiview_dataset as tmd
    path = os.path.join(tmp, 'synth.zarr')
    # The M2 fixture's episodes are 5/7/4 steps, shorter than horizon=16, which
    # SequenceSampler cannot window. Lengthen them for this test only (the
    # builder reads the global at call time).
    saved = tmd.EPISODE_LENGTHS
    tmd.EPISODE_LENGTHS = (40, 25, 30)
    try:
        # lossless: the plumbing test identifies each slot's view from its
        # pixels, which jpeg2k would perturb
        tmd.build_synthetic_zarr(path, codec=None)
    finally:
        tmd.EPISODE_LENGTHS = saved
    _distinct_cam_poses(path, tmd.N_VIEWS)
    shape_meta = _slot_shape_meta(tmd.N_VIEWS, tmd.H)
    kwargs = dict(
        shape_meta=shape_meta, dataset_path=path, horizon=16,
        pad_before=1, pad_after=7, n_obs_steps=2, abs_action=True,
        view_pool=list(range(tmd.N_VIEWS)), view_count_range=[1, tmd.N_VIEWS],
        eef_hist_steps=4, emit_aux_action=emit_aux_action,
        aux_n_steps=AUX_N_STEPS, seed=42, val_ratio=0.0)
    kwargs.update(over)
    return MultiViewImageDataset(**kwargs), shape_meta, tmd


def _identify_slot_views(ds, idx, batch, tmd):
    """``{slot: view}`` for the LIVE slots, recovered from the RENDERS only.

    Independent of the aux target and of `cam_table`: this is what makes the
    pairing checkable rather than assumed. Inactive slots are zero-filled, so
    they are skipped rather than matched.
    """
    n_slots = len(ds.rgb_keys)
    mask = batch['obs'][ds.view_mask_key].numpy()               # (To,K)
    found = dict()
    for k in range(n_slots):
        if mask[0, k] < 0.5:
            continue
        slot_img = batch['obs'][ds.rgb_keys[k]].numpy()          # (To,3,H,W)
        for v in range(tmd.N_VIEWS):
            cand = np.moveaxis(ds._view_obs_frames(idx, v), -1, 1
                               ).astype(np.float32)[:ds.n_obs_steps] / 255.
            if np.array_equal(cand, slot_img):
                found[k] = v
                break
        else:
            raise AssertionError(f'slot {k}: no view matches its image')
    return found


def test_aux_target_plumbing():
    tmp = tempfile.mkdtemp(prefix='m4_plumbing_')
    try:
        ds, shape_meta, tmd = _build_m4_dataset(tmp)
        C = AUX_N_STEPS
        n_slots = tmd.N_VIEWS
        max_live = 0

        for idx in (0, 3, 7, 11):
            batch = ds[idx]
            assert AUX_ACTION_KEY in batch, sorted(batch.keys())
            aux = batch[AUX_ACTION_KEY].numpy()
            assert aux.shape == (2, n_slots, C * ACTION_DIM), aux.shape
            assert aux.dtype == np.float32, aux.dtype

            mask = batch['obs'][ds.view_mask_key].numpy()             # (To,K)
            # inactive slots are EXACTLY zero, the invariant the loss mask and
            # the encoder's zeroed tokens both rest on
            assert np.all(aux[mask < 0.5] == 0.0)

            # every live slot holds a DISTINCT view, and that view is the one
            # whose image is in the slot
            views = _identify_slot_views(ds, idx, batch, tmd)
            assert len(views) == int(mask[0].sum()), (views, mask[0])
            live = list(views.values())
            assert len(set(live)) == len(live), live
            max_live = max(max_live, len(live))

            # recompute the expectation independently, from the raw stored
            # action and the identified view's pose: catches an off-by-one in
            # the cam table (the failure that trains happily and shows up only
            # as a mysteriously weak result)
            action_chunk = ds.sampler.sample_sequence(idx)['action']
            win = np.arange(2)[:, None] + np.arange(C)[None, :]
            for slot, v in views.items():
                cam = ds.cam_table[v]
                want = action_to_cam(action_chunk[win], cam[:3], cam[3:7])
                got = aux[:, slot].reshape(2, C, ACTION_DIM)
                assert np.abs(want - got).max() < 1e-6, (idx, slot, v)
        # a run of single-live-slot samples would make the pairing check weak
        assert max_live > 1, 'no sample drew more than one view'
        print(f'  aux target plumbing OK (shape (To,{n_slots},{C * ACTION_DIM}), '
              f'slots zeroed when inactive, view recovered from renders matches '
              f'the pose used)')

        # RNG neutrality: turning the flag on must not change the view draw, or
        # the ablation's two arms would stop being RNG-locked
        ds_off, _, _ = _build_m4_dataset(tmp, emit_aux_action=False)
        for idx in (0, 5, 9):
            np.random.seed(1234)
            b_on = ds[idx]
            np.random.seed(1234)
            b_off = ds_off[idx]
            assert torch.equal(b_on['obs']['view_mask'], b_off['obs']['view_mask'])
            for k in ds.cam_keys:
                assert torch.equal(b_on['obs'][k], b_off['obs'][k])
            assert AUX_ACTION_KEY not in b_off
        print('  emit_aux_action consumes no RNG (view draw identical at the '
              'same seed) and the flag-off sample has no aux key')

        # the ctor rejects a chunk that would reach into pad_after, and accepts
        # the largest legal one -- the boundary checked from BOTH sides
        # (obs step `to` reads action[to : to+C], so the last index touched is
        # C + n_obs_steps - 2 and it must stay <= horizon - 1).
        ds_max, _, _ = _build_m4_dataset(tmp, aux_n_steps=15)   # 15 + 2 - 2 = 15
        assert ds_max.aux_n_steps == 15
        try:
            _build_m4_dataset(tmp, aux_n_steps=16)              # index 16
        except ValueError as e:
            assert 'pad_after' in str(e), e
        else:
            raise AssertionError('ctor accepted aux_n_steps that reaches padding')
        print('  ctor accepts aux_n_steps <= horizon - n_obs_steps + 1 (15) and '
              'rejects one past it (16)')

        # the normalizer entry exists, is float32, and round-trips the target
        norm = ds.get_normalizer()
        entry = norm[AUX_ACTION_KEY]
        aux0 = ds[0][AUX_ACTION_KEY]
        back = entry.unnormalize(entry.normalize(aux0))
        assert back.shape == aux0.shape
        assert torch.allclose(back, aux0, atol=1e-5)
        assert entry.params_dict['scale'].dtype == torch.float32
        # the camera-frame pos block must NOT be identity -- it is fitted
        assert not torch.allclose(
            entry.params_dict['offset'][:3], torch.zeros(3))
        print(f'  normalizer[{AUX_ACTION_KEY}] fitted, float32, invertible '
              f'(scale[:3]={entry.params_dict["scale"][:3].tolist()})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. the policy
# ---------------------------------------------------------------------------

def _make_encoder(shape_meta):
    return ViewConditionedObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet('resnet18', weights=None),
        fused_dim=512, n_heads=8, cond_dim=128,
        use_plucker=True, use_eef_hist=True, eef_hist_steps=4,
        crop_shape=None, random_crop=False, use_group_norm=True,
        share_rgb_model=True, imagenet_norm=True)


def _make_policy(cls, shape_meta, **over):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    sched = DDPMScheduler(
        num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
        beta_schedule='squaredcos_cap_v2', variance_type='fixed_small',
        clip_sample=True, prediction_type='epsilon')
    kwargs = dict(
        shape_meta=shape_meta, noise_scheduler=sched,
        obs_encoder=_make_encoder(shape_meta), horizon=16,
        n_action_steps=8, n_obs_steps=2, num_inference_steps=100,
        obs_as_global_cond=True, diffusion_step_embed_dim=16,
        down_dims=(32, 64), kernel_size=5, n_groups=8,
        cond_predict_scale=True)
    kwargs.update(over)
    return cls(**kwargs)


def _batch_from(ds, idxs):
    samples = [ds[i] for i in idxs]
    return {
        'obs': {k: torch.stack([s['obs'][k] for s in samples])
                for k in samples[0]['obs']},
        'action': torch.stack([s['action'] for s in samples]),
        AUX_ACTION_KEY: torch.stack([s[AUX_ACTION_KEY] for s in samples]),
    }


def test_masked_view_mean():
    """Hand-computed, including the varying-active-count case that the per-row
    divide exists for.

    `pair_loss` carries one entry per ACTIVE pair, in the row-major (row, col)
    order `active.nonzero()` yields -- matching how the policy builds it, so
    this fixture would catch an implementation that assumed one entry per slot.
    """
    active = torch.tensor([[True, True, False],
                           [False, True, False]])
    # active pairs row-major: (0,0) (0,1) (1,1)
    pair = torch.tensor([1.0, 3.0, 10.0])
    # row 0: mean(1, 3) = 2 ; row 1: 10 ; overall (2 + 10) / 2 = 6
    got = float(masked_view_mean(pair, active))
    assert abs(got - 6.0) < 1e-6, got
    # the discriminator: dividing by the TOTAL active count instead of each
    # row's own count gives (1+3+10)/3 = 4.67, i.e. rows with fewer active
    # views would be silently down-weighted
    assert abs(sum(pair.tolist()) / 3 - got) > 1.0
    # an all-active row set degenerates to a plain mean
    all_on = torch.ones(2, 3, dtype=torch.bool)
    assert abs(float(masked_view_mean(torch.ones(6), all_on)) - 1.0) < 1e-6
    print('  masked_view_mean matches hand-computed values, divides per row '
          '(not by the global active count)')


def test_forward_full_and_loss():
    tmp = tempfile.mkdtemp(prefix='m4_policy_')
    try:
        ds, shape_meta, tmd = _build_m4_dataset(tmp)
        normalizer = ds.get_normalizer()
        batch = _batch_from(ds, [0, 1])
        B = 2

        off = _make_policy(DiffusionUnetImagePolicyAux, shape_meta,
                           aux_loss_weight=0.0, aux_n_steps=AUX_N_STEPS)
        on = _make_policy(DiffusionUnetImagePolicyAux, shape_meta,
                          aux_loss_weight=1.0, aux_n_steps=AUX_N_STEPS)
        ref = _make_policy(DiffusionUnetImagePolicy, shape_meta)
        for pol in (off, on):
            pol.set_normalizer(normalizer)
        ref.set_normalizer(normalizer)

        # give all three the SAME weights, so every difference below is the
        # aux term and nothing else
        sd = on.state_dict()
        for pol in (off, ref):
            missing, unexpected = pol.load_state_dict(sd, strict=False)
            assert not missing, missing
            assert all('aux_head' in k for k in unexpected), unexpected
        for pol in (off, on, ref):
            pol.eval()          # deterministic crop; no RNG in the encoder

        # --- forward == forward_full: a pure code move, so BIT-identical ----
        nobs = normalizer.normalize(batch['obs'])
        this = dict_apply(nobs, lambda x: x[:, :2].reshape(-1, *x.shape[2:]))
        enc = on.obs_encoder
        full = enc.forward_full(this)
        plain = enc.forward(this)
        assert torch.equal(plain[:, :512], full['z_global'])
        assert torch.equal(plain, torch.cat(
            [full['z_global']] + [this[k] for k in enc.low_dim_keys], dim=-1))
        assert tuple(enc.output_shape()) == (521,), enc.output_shape()
        print('  forward(obs) == forward_full(obs) bit-for-bit, '
              'output_shape still (521,)')

        # --- the zero-weight arm reproduces the parent EXACTLY --------------
        torch.manual_seed(0)
        l_ref = ref.compute_loss(batch)
        torch.manual_seed(0)
        l_off = off.compute_loss(batch)
        assert torch.equal(l_ref, l_off), (float(l_ref), float(l_off))
        for p1, p2 in zip(ref.parameters(), off.parameters()):
            assert torch.equal(p1.detach(), p2.detach())
        ref.zero_grad(set_to_none=True)
        off.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        ref.compute_loss(batch).backward()
        torch.manual_seed(0)
        off.compute_loss(batch).backward()
        n_grad = 0
        for (n1, p1), (n2, p2) in zip(ref.named_parameters(), off.named_parameters()):
            if p1.grad is None:
                assert p2.grad is None, n1
                continue
            assert p1.grad.shape == p2.grad.shape, n1
            assert torch.equal(p1.grad, p2.grad), n1
            n_grad += 1
        assert n_grad > 100, n_grad
        print(f'  aux_loss_weight=0 is bit-identical to the parent loss and all '
              f'{n_grad} gradients (the anti-drift pin on the copied body)')

        # --- the weight-1 arm is exactly diff + w * aux ---------------------
        torch.manual_seed(0)
        l_on = on.compute_loss(batch)
        assert torch.allclose(
            l_on, l_off + 1.0 * on.last_aux_loss, atol=1e-6), (
            float(l_on), float(l_off), on.last_aux_loss)
        assert on.last_aux_loss > 0, on.last_aux_loss
        print(f'  aux_loss_weight=1 loss == diff + aux '
              f'(diff {float(l_off):.4f}, aux {on.last_aux_loss:.4f}, '
              f'ratio {on.last_aux_loss / float(l_off):.3f})')

        # --- masking: inactive slots are exactly zero and contribute nothing
        nobs = normalizer.normalize(batch['obs'])
        this = dict_apply(nobs, lambda x: x[:, :2].reshape(-1, *x.shape[2:]))
        enc_out = on.obs_encoder.forward_full(this)
        active = enc_out['view_active']
        assert not bool(active.all()), 'fixture never leaves a slot inactive'
        assert torch.count_nonzero(enc_out['z_views'][~active]) == 0
        aux_a = on.aux_loss(enc_out, batch, B)
        poisoned = dict(batch)
        poisoned[AUX_ACTION_KEY] = batch[AUX_ACTION_KEY].clone()
        # `active` is the encoder's flat (B*To, K); the batch target is
        # (B, To, K, C*Da) -- same values, different layout
        poisoned[AUX_ACTION_KEY][~active.reshape(B, 2, -1)] += 1e3
        assert torch.allclose(aux_a, on.aux_loss(enc_out, poisoned, B)), (
            'an inactive slot changed the aux loss')
        print('  inactive slots are exactly zero in z_views and cannot affect '
              'the aux loss')

        # --- the head starts at exactly zero (this repo's added-branch rule)
        head = PerViewAuxActionHead(512, AUX_N_STEPS * ACTION_DIM)
        assert torch.count_nonzero(head(torch.randn(4, 512))) == 0
        # ... so the aux gradient reaches the trunk only once the head has
        # moved off zero. Fill it and the path must be live.
        with torch.no_grad():
            head.net[-1].weight.normal_(0, 0.01)
        assert torch.count_nonzero(head(torch.randn(4, 512))) > 0
        on.obs_encoder.backbone.conv1.weight.grad = None
        with torch.no_grad():
            on.aux_head.net[-1].weight.normal_(0, 0.01)
        on.aux_loss(on.obs_encoder.forward_full(this), batch, B).backward()
        g = on.obs_encoder.backbone.conv1.weight.grad
        assert g is not None and torch.count_nonzero(g) > 0
        assert torch.count_nonzero(g[:, :3]) > 0, 'image channels got no gradient'
        assert torch.count_nonzero(g[:, 3:]) > 0, 'ray channels got no gradient'
        print('  head is zero at init; once non-zero the aux gradient reaches '
              'conv1 (image AND ray channels)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. configs
# ---------------------------------------------------------------------------

def test_m4_configs():
    """The configs compose, the counts agree, and the MODEL IS UNCHANGED.

    The last one is the user's constraint made executable: M4 must add a branch
    and a loss, not alter the architecture the M3 result was measured on.
    """
    import inspect
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')

    # the landmine: an aux key that leaked into **kwargs would blow up inside
    # scheduler.step at ROLLOUT time, hours into a run
    sig = inspect.signature(DiffusionUnetImagePolicyAux.__init__)
    for key in ('aux_loss_weight', 'aux_n_steps', 'aux_hidden_dim'):
        assert key in sig.parameters, key

    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(config_name='train_diffusion_unet_image_workspace_m4')
        cfg_m3 = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_m3')

    assert 'diffusion_unet_image_policy_aux' in cfg.policy._target_, cfg.policy._target_
    assert cfg.task.dataset.emit_aux_action is True
    assert cfg.policy.aux_n_steps == cfg.task.dataset.aux_n_steps, (
        cfg.policy.aux_n_steps, cfg.task.dataset.aux_n_steps)
    assert cfg.policy.aux_loss_weight == 1.0
    assert 'kwargs' not in cfg.policy, 'aux params must be named, not **kwargs'

    # THE CONSTRAINT: the encoder is byte-identical to M3's
    assert OmegaConf.to_container(cfg.policy.obs_encoder) == \
        OmegaConf.to_container(cfg_m3.policy.obs_encoder), \
        'M4 changed the encoder; the model is supposed to be held fixed'
    for key in ('horizon', 'n_obs_steps', 'n_action_steps', 'obs_as_global_cond'):
        assert cfg[key] == cfg_m3[key], key

    enc = hydra.utils.instantiate(cfg.policy.obs_encoder)
    assert tuple(enc.output_shape()) == (521,), enc.output_shape()
    slots = len([k for k, v in cfg.task.shape_meta.obs.items()
                 if v.get('type', 'low_dim') == 'rgb'])
    assert enc.n_slots == slots == cfg.task.env_runner.m3_slots

    # instantiate the WHOLE policy through the real hydra path -- the encoder
    # wiring and the named aux kwargs are what is under test (a small UNet
    # keeps it CPU-cheap). `kwargs` must stay EMPTY: an aux key that landed
    # there would be forwarded into scheduler.step and raise TypeError at
    # ROLLOUT time, hours into a run.
    pol = hydra.utils.instantiate(
        cfg.policy, down_dims=[32, 64], diffusion_step_embed_dim=16)
    assert isinstance(pol, DiffusionUnetImagePolicyAux), type(pol)
    assert pol.kwargs == {}, pol.kwargs
    assert pol.aux_loss_weight == 1.0, pol.aux_loss_weight
    assert pol.aux_n_steps == cfg.task.dataset.aux_n_steps, (
        pol.aux_n_steps, cfg.task.dataset.aux_n_steps)
    assert pol.aux_head.out_dim == pol.aux_n_steps * pol.action_dim
    assert pol.aux_head.in_dim == enc.fused_dim
    # and the width the UNet sees is still M1/L1/M3's
    assert pol.obs_feature_dim == 521, pol.obs_feature_dim

    head = PerViewAuxActionHead(enc.fused_dim,
                                cfg.policy.aux_n_steps * 10,
                                cfg.policy.aux_hidden_dim)
    n_aux = sum(p.numel() for p in head.parameters())
    print(f'M4 configs OK (encoder identical to M3, output_shape (521,), '
          f'aux head {n_aux:,} params = {100 * n_aux / 290_000_000:.3f}% of the '
          f'policy)')


def test():
    test_rot6d_matches_pytorch3d()
    test_action_to_cam_convention()
    test_aux_target_plumbing()
    test_masked_view_mean()
    test_forward_full_and_loss()
    test_m4_configs()
    print('ALL AUX ACTION HEAD CHECKS PASSED')


if __name__ == '__main__':
    test()
