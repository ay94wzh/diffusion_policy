"""CPU-only tests for PLAN candidate 2: z_g pose-set supervision (no simulator,
no GPU).

Run from the repo root:
    python tests/test_zg_pose_supervision.py

What is checked, in rough order of how much it buys:

1. `mat_to_rot6d` is the EXACT inverse of the pinned `rot6d_to_mat` (rows 0
   and 1 of R), with mutation power -- the columns-instead-of-rows bug is
   invisible at an identity matrix, so every mutation is exerted at a general
   rotation (NOTES.md, "A check that cannot fail proves nothing").
2. The dataset emits the pooled pose set correctly: row i really is pool
   position i (the fixture's pool is a PERMUTATION, so a sorted-order or
   ring-index implementation fails here), the mask marks exactly the views
   whose renders are in the sample's slots, dead rows are exactly zero, and
   the flag consumes no RNG.
3. The policy's aux term reads the FUSED latent, scores ALL rows against the
   zero-filled target (dead rows are scored as zero; the anti-cheat pin shows
   no input-independent head can reach zero), starts at zero, and weight 0
   remains bit-identical to the parent's loss and all gradients (the
   anti-drift pin, as in tests/test_aux_action_heads.py).
4. The configs compose, the model is byte-identical to M3/m4base's, and the
   two train configs differ on EXACTLY the intended key set (the one-variable
   claim made executable).
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
sys.path.insert(0, this_dir)          # for `import test_multiview_dataset`

import numpy as np
import torch

import test_aux_action_heads as tah
from probe_relpose import rot6d_from_mat
from diffusion_policy.dataset.multiview_image_dataset import (
    MultiViewImageDataset, mat_to_rot6d, rot6d_to_mat, quat_wxyz_to_mat,
    AUX_VIEW_POSE_KEY, AUX_VIEW_POSE_MASK_KEY, VIEW_POSE_DIM)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.model.vision.view_pose_set_head import ViewPoseSetHead
from diffusion_policy.policy.diffusion_unet_image_policy import (
    DiffusionUnetImagePolicy)
from diffusion_policy.policy.diffusion_unet_image_policy_aux import (
    DiffusionUnetImagePolicyAux)
from diffusion_policy.policy.diffusion_unet_image_policy_zgpose import (
    DiffusionUnetImagePolicyZgPose)
from diffusion_policy.common.pytorch_util import dict_apply

# a PERMUTATION of the 3-view synthetic fixture's pool: ring index != pool
# position, which is the only fixture under which a pool-ordering bug cannot
# hide (a sorted pool would make the two coincide).
V_POOL = [2, 0, 1]


def _build_zg_dataset(tmp, **over):
    """Pose-set-mode dataset over the M2 synthetic zarr, permuted pool."""
    kwargs = dict(emit_aux_action=False, emit_aux_view_pose=True,
                  view_pool=list(V_POOL))
    kwargs.update(over)
    return tah._build_m4_dataset(tmp, **kwargs)


def _batch_from_zg(ds, idxs):
    samples = [ds[i] for i in idxs]
    return {
        'obs': {k: torch.stack([s['obs'][k] for s in samples])
                for k in samples[0]['obs']},
        'action': torch.stack([s['action'] for s in samples]),
        AUX_VIEW_POSE_KEY: torch.stack([s[AUX_VIEW_POSE_KEY] for s in samples]),
        AUX_VIEW_POSE_MASK_KEY: torch.stack(
            [s[AUX_VIEW_POSE_MASK_KEY] for s in samples]),
    }


# ---------------------------------------------------------------------------
# 1. the inverse convention
# ---------------------------------------------------------------------------

def test_mat_to_rot6d_convention():
    """`mat_to_rot6d` must be the exact inverse of the pinned `rot6d_to_mat`,
    and agree with the two independent copies of the convention that already
    exist (`RotationTransformer`, `probe_relpose.rot6d_from_mat`)."""
    tf = RotationTransformer('matrix', 'rotation_6d')
    rng = np.random.default_rng(0)
    worst_tf = 0.0
    worst_probe = 0.0
    worst_rt = 0.0
    for _ in range(200):
        R = quat_wxyz_to_mat(tah._rand_quat(rng))
        ours = mat_to_rot6d(R)
        worst_tf = max(worst_tf,
                       np.abs(ours - np.asarray(tf.forward(R))).max())
        worst_probe = max(worst_probe,
                          np.abs(ours - rot6d_from_mat(R)).max())
        worst_rt = max(worst_rt, np.abs(rot6d_to_mat(ours) - R).max())
    assert worst_tf < 1e-12, worst_tf
    assert worst_probe < 1e-12, worst_probe
    assert worst_rt < 1e-12, worst_rt
    print(f'  mat_to_rot6d == RotationTransformer(matrix->rotation_6d) '
          f'(max|diff| {worst_tf:.2e}), == probe_relpose.rot6d_from_mat '
          f'({worst_probe:.2e}), round-trips through rot6d_to_mat '
          f'({worst_rt:.2e}) over 200 random rotations')

    # batched form must agree with the row-wise one
    Rs = np.stack([quat_wxyz_to_mat(tah._rand_quat(rng)) for _ in range(7)])
    batch = mat_to_rot6d(Rs)
    assert batch.shape == (7, 6), batch.shape
    for i in range(7):
        assert np.array_equal(batch[i], mat_to_rot6d(Rs[i]))
    print('  batched (...,3,3) form matches row-wise')

    # ---- mutation power, exerted at a GENERAL rotation --------------------
    R = quat_wxyz_to_mat(tah._rand_quat(np.random.default_rng(7)))
    good = mat_to_rot6d(R)
    mutations = {
        'first two COLUMNS instead of rows': (
            good - np.concatenate([R[:, 0], R[:, 1]])),
        'rows 0 and 2 instead of 0 and 1': (
            good - np.concatenate([R[0, :], R[2, :]])),
        'sign-flipped first row': (
            good - np.concatenate([-R[0, :], R[1, :]])),
    }
    for name, err in mutations.items():
        e = float(np.abs(err).max())
        assert e > 1e-3, f'mutation NOT caught ({name}): max|err| = {e:.2e}'
    print(f'  mutation power OK on {len(mutations)} mutations '
          f'(all > 1e-3)')

    # ... and the executable form of why identity cameras prove nothing: the
    # columns-vs-rows mutation is EXACTLY zero at R = I
    assert np.abs(mat_to_rot6d(np.eye(3))
                  - np.concatenate([np.eye(3)[:, 0], np.eye(3)[:, 1]])).max() \
        < 1e-12
    print('  columns-vs-rows mutation is invisible at R=I '
          '(so an identity-camera check is vacuous)')


# ---------------------------------------------------------------------------
# 2. the dataset
# ---------------------------------------------------------------------------

def test_dataset_pose_set_emission():
    tmp = tempfile.mkdtemp(prefix='zgpose_ds_')
    try:
        ds, shape_meta, tmd = _build_zg_dataset(tmp)
        V = len(V_POOL)

        # the pool -> row mapping is pinned directly, and it is NOT sorted
        assert ds.pool_pose.shape == (V, VIEW_POSE_DIM), ds.pool_pose.shape
        assert ds._pool_pos == {int(v): i for i, v in enumerate(V_POOL)}
        assert ds.pool_pose.dtype == np.float32

        max_live = 0
        for idx in (0, 3, 7, 11):
            batch = ds[idx]
            assert AUX_VIEW_POSE_KEY in batch, sorted(batch.keys())
            assert AUX_VIEW_POSE_MASK_KEY in batch, sorted(batch.keys())
            tgt = batch[AUX_VIEW_POSE_KEY].numpy()
            msk = batch[AUX_VIEW_POSE_MASK_KEY].numpy()
            assert tgt.shape == (2, V, VIEW_POSE_DIM), tgt.shape
            assert msk.shape == (2, V), msk.shape
            assert tgt.dtype == np.float32 and msk.dtype == np.float32

            # which views are live, recovered from the RENDERS only -- never
            # from cam_table, so a cam-table index bug cannot hide
            views = tah._identify_slot_views(ds, idx, batch, tmd)
            live_pos = sorted(ds._pool_pos[int(v)] for v in views.values())
            got_pos = sorted(np.where(msk[0] > 0.5)[0].tolist())
            assert got_pos == live_pos, (idx, got_pos, live_pos)
            assert int(msk[0].sum()) == len(views)
            max_live = max(max_live, len(views))

            # live rows equal the INDEPENDENTLY computed pose (via
            # probe_relpose's own inverse), dead rows are exactly zero
            for i in live_pos:
                v = V_POOL[i]
                want = np.concatenate([
                    ds.cam_table[v][:3].astype(np.float64),
                    rot6d_from_mat(quat_wxyz_to_mat(ds.cam_table[v][3:7])),
                ])
                assert np.abs(tgt[:, i] - want).max() < 1e-6, (idx, i, v)
            # the EXACT zeros are load-bearing now: the all-rows loss scores
            # dead rows as zero, and that is what makes the optimum
            # sample-dependent (see the anti-cheat pin in the policy test)
            assert np.all(tgt[msk < 0.5] == 0.0)
            # the live rows are pairwise distinct (a constant row would make
            # the row-target check above pass vacuously)
            live_rows = tgt[0, live_pos]
            for a in range(len(live_pos)):
                for b in range(a + 1, len(live_pos)):
                    assert np.abs(live_rows[a] - live_rows[b]).max() > 1e-6

            # tiled over obs steps: the target is per-sample
            assert np.array_equal(tgt[0], tgt[1])
            assert np.array_equal(msk[0], msk[1])
        assert max_live > 1, 'no sample drew more than one view'
        print(f'  pose-set emission OK: (To,{V},{VIEW_POSE_DIM}) + mask, rows == '
              f'pool positions (permuted pool {V_POOL}), live from renders, '
              f'dead rows exactly zero')

        # flag off: keys absent, and the flag consumes NO RNG -- the view draw
        # (and, with M4's flag on, the aux action) is identical at one seed
        ds_off, _, _ = _build_zg_dataset(tmp, emit_aux_view_pose=False)
        ds_a_on, _, _ = _build_zg_dataset(
            tmp, emit_aux_action=True, emit_aux_view_pose=True)
        ds_a_off, _, _ = _build_zg_dataset(
            tmp, emit_aux_action=True, emit_aux_view_pose=False)
        for idx in (0, 5, 9):
            np.random.seed(1234)
            b_on = ds_a_on[idx]
            np.random.seed(1234)
            b_off = ds_a_off[idx]
            assert torch.equal(b_on['obs'][ds.view_mask_key],
                               b_off['obs'][ds.view_mask_key])
            for k in ds.cam_keys:
                assert torch.equal(b_on['obs'][k], b_off['obs'][k])
            assert torch.equal(b_on[tah.AUX_ACTION_KEY], b_off[tah.AUX_ACTION_KEY])
            assert AUX_VIEW_POSE_KEY not in b_off
            assert AUX_VIEW_POSE_MASK_KEY not in b_off
        np.random.seed(1234)
        b_off = ds_off[0]
        assert AUX_VIEW_POSE_KEY not in b_off
        assert AUX_VIEW_POSE_MASK_KEY not in b_off
        print('  emit_aux_view_pose consumes no RNG (draw and aux_action '
              'identical at the same seed) and the flag-off sample has no '
              'pose keys')

        # the ctor rejects the flag without a pool, at construction
        try:
            _build_zg_dataset(tmp, view_pool=None)
        except ValueError as e:
            assert 'view_pool' in str(e), e
        else:
            raise AssertionError('ctor accepted emit_aux_view_pose without pool')
        print('  ctor rejects emit_aux_view_pose without view_pool')

        # the normalizer entry is IDENTITY, float32, and the mask is not a
        # normalizer key (it never passes through the normalizer)
        norm = ds.get_normalizer()
        entry = norm[AUX_VIEW_POSE_KEY]
        assert AUX_VIEW_POSE_MASK_KEY not in norm.params_dict
        assert entry.params_dict['scale'].dtype == torch.float32
        assert torch.all(entry.params_dict['scale'] == 1)
        assert torch.all(entry.params_dict['offset'] == 0)
        t4 = ds[0][AUX_VIEW_POSE_KEY]
        assert torch.equal(entry.normalize(t4), t4)
        nobs = norm.normalize(ds[0]['obs'])          # must not raise
        assert AUX_VIEW_POSE_KEY not in nobs
        print('  normalizer[aux_view_pose] is identity float32; mask not '
              'registered; normalize(obs) works')

        # the validation copy carries the flag and the same table (with a
        # nonzero val_ratio -- the fixture's default 0.0 leaves the val
        # sampler empty by construction)
        ds_v, _, _ = _build_zg_dataset(tmp, val_ratio=0.5)
        val = ds_v.get_validation_dataset()
        assert val.emit_aux_view_pose is True
        assert np.array_equal(val.pool_pose, ds_v.pool_pose)
        assert len(val) > 0, 'fixture val split is empty'
        assert AUX_VIEW_POSE_KEY in val[0], sorted(val[0].keys())
        print('  validation dataset carries the flag, the table and the keys')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. the policy
# ---------------------------------------------------------------------------

def test_zgpose_policy_loss():
    tmp = tempfile.mkdtemp(prefix='zgpose_policy_')
    try:
        ds, shape_meta, tmd = _build_zg_dataset(tmp)
        V = len(V_POOL)
        normalizer = ds.get_normalizer()
        batch = _batch_from_zg(ds, [0, 1])
        B = 2

        off = tah._make_policy(DiffusionUnetImagePolicyZgPose, shape_meta,
                               aux_loss_weight=0.0, aux_n_views=V)
        on = tah._make_policy(DiffusionUnetImagePolicyZgPose, shape_meta,
                              aux_loss_weight=1.0, aux_n_views=V)
        ref = tah._make_policy(DiffusionUnetImagePolicy, shape_meta)
        for pol in (off, on):
            pol.set_normalizer(normalizer)
        ref.set_normalizer(normalizer)

        sd = on.state_dict()
        for pol in (off, ref):
            missing, unexpected = pol.load_state_dict(sd, strict=False)
            assert not missing, missing
            assert all('aux_head' in k for k in unexpected), unexpected
        for pol in (off, on, ref):
            pol.eval()

        # --- forward == forward_full: a pure code move, BIT-identical --------
        nobs = normalizer.normalize(batch['obs'])
        this = dict_apply(nobs, lambda x: x[:, :2].reshape(-1, *x.shape[2:]))
        enc = on.obs_encoder
        full = enc.forward_full(this)
        plain = enc.forward(this)
        assert torch.equal(plain[:, :512], full['z_global'])
        assert tuple(enc.output_shape()) == (521,), enc.output_shape()
        print('  forward(obs) == forward_full(obs) bit-for-bit, '
              'output_shape still (521,)')

        # --- the head starts at exactly zero, so the aux value at init IS the
        # hand-computed mean target-square over all rows (this is also what
        # makes the driver's pre-launch calibration dataset-only) ------------
        assert torch.count_nonzero(ViewPoseSetHead(512, V, VIEW_POSE_DIM)(
            torch.randn(4, 512))) == 0
        tgt = batch[AUX_VIEW_POSE_KEY].numpy()
        mskb = batch[AUX_VIEW_POSE_MASK_KEY].numpy() > 0.5
        # ALL rows are scored (dead rows against their zero targets), so at
        # the zero-init head the aux value is the plain mean target-square
        want = np.mean(tgt ** 2)
        enc_out = on.obs_encoder.forward_full(this)
        aux_init = float(on.aux_loss(enc_out, batch, B))
        assert abs(aux_init - want) < 1e-6, (aux_init, want)
        print(f'  zero-init head: aux at init == hand-computed mean '
              f'target-square over all rows ({aux_init:.4f})')

        # --- the zero-weight arm reproduces the parent EXACTLY ---------------
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
        for (n1, p1), (n2, p2) in zip(ref.named_parameters(),
                                      off.named_parameters()):
            if p1.grad is None:
                assert p2.grad is None, n1
                continue
            assert p1.grad.shape == p2.grad.shape, n1
            assert torch.equal(p1.grad, p2.grad), n1
            n_grad += 1
        assert n_grad > 100, n_grad
        for name, p in off.aux_head.named_parameters():
            assert p.grad is None, f'{name} got a gradient at weight 0'
        print(f'  aux_loss_weight=0 is bit-identical to the parent loss and all '
              f'{n_grad} shared gradients; the head gets none')

        # --- the weight-1 arm is exactly diff + w * aux ----------------------
        torch.manual_seed(0)
        l_on = on.compute_loss(batch)
        assert torch.allclose(
            l_on, l_off + 1.0 * on.last_aux_loss, atol=1e-6), (
            float(l_on), float(l_off), on.last_aux_loss)
        assert on.last_aux_loss > 0, on.last_aux_loss
        print(f'  aux_loss_weight=1 loss == diff + aux (diff {float(l_off):.4f}, '
              f'aux {on.last_aux_loss:.4f}, ratio '
              f'{on.last_aux_loss / float(l_off):.3f})')

        # --- ALL rows are scored, dead ones too (their targets are zero) ----
        aux_a = on.aux_loss(enc_out, batch, B)
        poisoned = dict(batch)
        poisoned[AUX_VIEW_POSE_KEY] = batch[AUX_VIEW_POSE_KEY].clone()
        poisoned[AUX_VIEW_POSE_KEY][batch[AUX_VIEW_POSE_MASK_KEY] < 0.5] += 1e3
        assert float(on.aux_loss(enc_out, poisoned, B)) - float(aux_a) > 1.0, (
            'poisoning a dead pose row did not change the aux loss -- the '
            'loss is not scoring it, and the constant-head cheat is back')
        poisoned2 = dict(batch)
        poisoned2[AUX_VIEW_POSE_KEY] = batch[AUX_VIEW_POSE_KEY].clone()
        poisoned2[AUX_VIEW_POSE_KEY][batch[AUX_VIEW_POSE_MASK_KEY] > 0.5] += 1e3
        assert float(on.aux_loss(enc_out, poisoned2, B)) - float(aux_a) > 1.0
        print('  every row is scored: poisoning DEAD rows raises the aux loss '
              'too (that is what forbids the constant-head solution)')

        # --- the anti-cheat pin: the target is sample-dependent (liveness
        # varies), so no input-independent head -- the per-row constant
        # included -- can reach zero. If this fails, the run's premise is gone.
        np.random.seed(0)     # immediately before the draws; nothing between
        ac = _batch_from_zg(ds, range(8))
        tc = ac[AUX_VIEW_POSE_KEY].numpy().reshape(-1, V, VIEW_POSE_DIM)
        mc = ac[AUX_VIEW_POSE_MASK_KEY].numpy().reshape(-1, V) > 0.5
        n = mc.sum(0)
        Mc = tc.shape[0]
        assert ((n > 0) & (n < Mc)).any(), \
            'no pool row is partially live -- the floor would be 0'
        const = float(np.mean((tc - tc.mean(0)) ** 2))
        assert const > 1e-4, const
        assert float(np.mean(tc ** 2)) > const, 'zero head beats the constant'
        print(f'  anti-cheat pin: the best per-row constant scores {const:.4f} '
              f'> 0 (liveness must come from z_g)')

        # --- the gradient path: head -> z_g -> fusion -> conv1 --------------
        with torch.no_grad():
            on.aux_head.net[-1].weight.normal_(0, 0.01)
        for n, p in on.obs_encoder.named_parameters():
            p.grad = None
        on.aux_loss(enc_out, batch, B).backward()
        g = on.obs_encoder.backbone.conv1.weight.grad
        assert g is not None and torch.count_nonzero(g) > 0
        assert torch.count_nonzero(g[:, :3]) > 0, 'image channels got no gradient'
        assert torch.count_nonzero(g[:, 3:]) > 0, 'ray channels got no gradient'
        gf = on.obs_encoder.fusion.in_proj_weight.grad
        assert gf is not None and torch.count_nonzero(gf) > 0, \
            'the FUSION got no gradient -- the supervision is not reaching z_g'
        print('  once non-zero, the aux gradient reaches the fusion and conv1 '
              '(image AND ray channels)')

        # --- loud checks, each with the key name in the message --------------
        def expect_valueerror(fn, needle):
            try:
                fn()
            except ValueError as e:
                assert needle in str(e), (needle, e)
            else:
                raise AssertionError(f'no ValueError mentioning {needle!r}')

        bad_v = dict(batch)
        bad_v[AUX_VIEW_POSE_KEY] = batch[AUX_VIEW_POSE_KEY][:, :, :V - 1]
        expect_valueerror(lambda: on.aux_loss(enc_out, bad_v, B),
                          AUX_VIEW_POSE_KEY)
        bad_p = dict(batch)
        bad_p[AUX_VIEW_POSE_KEY] = batch[AUX_VIEW_POSE_KEY][..., :8]
        expect_valueerror(lambda: on.aux_loss(enc_out, bad_p, B),
                          AUX_VIEW_POSE_KEY)
        pol4 = tah._make_policy(DiffusionUnetImagePolicyZgPose, shape_meta,
                                aux_loss_weight=1.0, aux_n_views=V + 1)
        pol4.set_normalizer(normalizer)
        expect_valueerror(lambda: pol4.aux_loss(enc_out, batch, B),
                          AUX_VIEW_POSE_KEY)
        bad_m = dict(batch)
        bad_m[AUX_VIEW_POSE_MASK_KEY] = batch[AUX_VIEW_POSE_MASK_KEY].clone()
        bad_m[AUX_VIEW_POSE_MASK_KEY][0, 0, :] = 0.0     # a frame with no live view
        expect_valueerror(lambda: on.aux_loss(enc_out, bad_m, B), 'no live view')
        bad_c = dict(batch)
        bad_c[AUX_VIEW_POSE_MASK_KEY] = batch[AUX_VIEW_POSE_MASK_KEY].clone()
        dead = int(np.where(mskb[0, 0] == 0)[0][0])
        bad_c[AUX_VIEW_POSE_MASK_KEY][0, 0, dead] = 1.0  # count desync vs encoder
        expect_valueerror(lambda: on.aux_loss(enc_out, bad_c, B),
                          'live-view count')
        print('  loud checks fire: wrong V, wrong pose dim, wrong aux_n_views, '
              'no-live-view frame, mask/encoder count desync')

        # --- N=1: one live view, finite, equal to the (zero-head) target, and
        # position-dependent -----------------------------------------------
        # a FRESH policy: `on`'s head was filled in the gradient block above,
        # so it no longer predicts zero
        pol_n1 = tah._make_policy(DiffusionUnetImagePolicyZgPose, shape_meta,
                                  aux_loss_weight=1.0, aux_n_views=V)
        pol_n1.set_normalizer(normalizer)
        pol_n1.eval()
        ds1, _, _ = _build_zg_dataset(tmp, view_count_range=[1, 1])
        b1 = _batch_from_zg(ds1, [0, 1])
        enc1 = pol_n1.obs_encoder.forward_full(
            dict_apply(normalizer.normalize(b1['obs']),
                       lambda x: x[:, :2].reshape(-1, *x.shape[2:])))
        a1 = float(pol_n1.aux_loss(enc1, b1, B))
        t1 = b1[AUX_VIEW_POSE_KEY].numpy()
        m1 = b1[AUX_VIEW_POSE_MASK_KEY].numpy() > 0.5
        assert (m1.sum(axis=-1) == 1).all(), 'N=1 fixture is not single-view'
        w1 = np.mean(t1 ** 2)          # all rows scored; dead ones are zero
        assert np.isfinite(a1) and abs(a1 - w1) < 1e-6, (a1, w1)
        # the live row's POSITION matters: make one frame carry a DIFFERENT
        # view's pose (the next pool position's, as the dataset would emit for
        # that draw) and the loss must move
        m1m = b1[AUX_VIEW_POSE_MASK_KEY].clone()
        t1m = b1[AUX_VIEW_POSE_KEY].clone()
        p_pos = int(np.where(m1m[0, 0].numpy() > 0.5)[0][0])
        q_pos = (p_pos + 1) % V
        m1m[0, 0] = 0.0
        m1m[0, 0, q_pos] = 1.0
        t1m[0, 0] = 0.0
        t1m[0, 0, q_pos] = torch.from_numpy(ds1.pool_pose[q_pos])
        b1m = dict(b1)
        b1m[AUX_VIEW_POSE_KEY] = t1m
        b1m[AUX_VIEW_POSE_MASK_KEY] = m1m
        a1m = float(pol_n1.aux_loss(enc1, b1m, B))
        # exact hand value first (pins the all-rows semantics), then the move
        expected = float(np.mean(b1m[AUX_VIEW_POSE_KEY].numpy() ** 2))
        assert abs(a1m - expected) < 1e-6, (a1m, expected)
        assert abs(a1m - a1) > 1e-5, (a1, a1m)
        print(f'  N=1 samples: aux finite, == the all-rows mean target-square '
              f'({a1:.4f}), and moves when the live view changes ({a1m:.4f})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. the configs
# ---------------------------------------------------------------------------

def test_zgpose_configs():
    """The configs compose, the counts agree, the MODEL IS UNCHANGED, and the
    zgpose train config differs from m4base's on EXACTLY the intended keys."""
    import inspect
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')

    sig = inspect.signature(DiffusionUnetImagePolicyZgPose.__init__)
    for key in ('aux_loss_weight', 'aux_n_views', 'aux_hidden_dim'):
        assert key in sig.parameters, key

    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_zgpose_latent')
        cfg_base = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_m4base_latent')
        cfg_m3 = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_m3')

    assert 'diffusion_unet_image_policy_zgpose' in cfg.policy._target_, \
        cfg.policy._target_
    assert 'latentprobe' in cfg._target_, cfg._target_
    assert cfg.task.dataset.emit_aux_view_pose is True
    assert cfg.task.dataset.emit_aux_action is False
    assert list(cfg.task.dataset.view_pool) == list(range(1, 12))
    assert tuple(cfg.task.dataset.view_count_range) == (1, 11)
    assert cfg.task.env_runner.m3_slots == 11 == cfg.policy.aux_n_views
    assert cfg.policy.aux_loss_weight == 0.3573   # the loud placeholder
    assert 'aux_n_steps' not in cfg.policy, 'the new policy has no aux_n_steps'
    assert 'kwargs' not in cfg.policy, 'aux params must be named, not **kwargs'
    assert cfg.latent_probe.n_states == 128
    assert cfg.latent_probe.seed == 12345
    assert cfg.latent_probe.az0_view == 6
    assert cfg.latent_probe.out_dir == 'latent_snapshots'
    assert OmegaConf.to_container(cfg.latent_probe) == \
        OmegaConf.to_container(cfg_base.latent_probe)

    # THE CONSTRAINT: the encoder is byte-identical to M3's and m4base's
    assert OmegaConf.to_container(cfg.policy.obs_encoder) == \
        OmegaConf.to_container(cfg_m3.policy.obs_encoder), \
        'zgpose changed the encoder; the model is supposed to be held fixed'
    assert OmegaConf.to_container(cfg.policy.obs_encoder) == \
        OmegaConf.to_container(cfg_base.policy.obs_encoder)
    for key in ('horizon', 'n_obs_steps', 'n_action_steps', 'obs_as_global_cond'):
        assert cfg[key] == cfg_m3[key], key

    enc = hydra.utils.instantiate(cfg.policy.obs_encoder)
    assert tuple(enc.output_shape()) == (521,), enc.output_shape()
    pol = hydra.utils.instantiate(
        cfg.policy, down_dims=[32, 64], diffusion_step_embed_dim=16)
    assert isinstance(pol, DiffusionUnetImagePolicyZgPose), type(pol)
    assert isinstance(pol, DiffusionUnetImagePolicyAux), type(pol)
    assert pol.kwargs == {}, pol.kwargs
    assert pol.aux_n_views == 11
    assert pol.aux_head.n_views == 11 and pol.aux_head.pose_dim == VIEW_POSE_DIM
    assert pol.aux_head.in_dim == enc.fused_dim
    assert pol.obs_feature_dim == 521, pol.obs_feature_dim
    print('  configs compose; encoder identical to M3/m4base; policy '
          'instantiates with the pose head and empty kwargs')

    # ---- the one-variable pin: EVERY other resolved key is identical ------
    def flatten(d, prefix=''):
        out = {}
        for k, v in d.items():
            p = f'{prefix}.{k}' if prefix else k
            if isinstance(v, dict):
                out.update(flatten(v, p))
            else:
                out[p] = json.dumps(v, sort_keys=True, default=str)
        return out

    a = flatten(OmegaConf.to_container(cfg, resolve=False))
    b = flatten(OmegaConf.to_container(cfg_base, resolve=False))
    diffs = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    expected = {
        'name',
        'task.name',
        'task.emit_aux_action',
        'task.emit_aux_view_pose',
        'task.aux_action_frame',
        'task.aux_n_steps',
        'task.dataset.emit_aux_action',
        'task.dataset.emit_aux_view_pose',
        'task.dataset.aux_action_frame',
        'task.dataset.aux_n_steps',
        'policy._target_',
        'policy.aux_loss_weight',
        'policy.aux_n_views',
        'policy.aux_n_steps',
    }
    assert diffs == expected, (
        f'unexpected config diffs: {sorted(diffs - expected)}; '
        f'missing expected diffs: {sorted(expected - diffs)}')
    print(f'  the train configs differ on exactly the intended {len(diffs)} '
          f'keys (supervision target/locus); everything else is byte-equal')


def test():
    test_mat_to_rot6d_convention()
    test_dataset_pose_set_emission()
    test_zgpose_policy_loss()
    test_zgpose_configs()
    print('ALL ZG POSE-SET SUPERVISION CHECKS PASSED')


if __name__ == '__main__':
    test()
