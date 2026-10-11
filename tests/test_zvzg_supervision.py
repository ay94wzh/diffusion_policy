"""CPU-only tests for the combined-supervision arm (`zvzg`): the z_v
base-frame action supervision AND the z_g pooled camera-pose-set supervision,
both on at once (no simulator, no GPU).

Run from the repo root:
    python tests/test_zvzg_supervision.py

What is checked, in rough order of how much it buys:

1. The dataset emits BOTH targets together, from the ONE view draw: the
   base-frame action chunk on live slots (bit-identical to the stored action
   slice, identical across live slots) and the pooled pose set with rows ==
   pool positions (the fixture's pool is a PERMUTATION, so a sorted-order or
   ring-index implementation fails here); dead rows are exactly zero on both;
   the two live masks agree per frame; and neither flag changes the draw
   stream (turn either off and the view_mask/cam keys are identical at the
   same seed).
2. The policy's three-way bit-identity pins, each load-bearing for one
   "one variable" claim:
     (w_v=0, w_g=0)  ==  DiffusionUnetImagePolicy        (base path)
     (w_v=W, w_g=0)  ==  DiffusionUnetImagePolicyAux     (m4base's term)
     (w_v=0, w_g=W)  ==  DiffusionUnetImagePolicyZgPose  (zgpose's term)
   loss AND every shared gradient, torch.equal -- with the state-dict key
   remap the zgpose reference needs (its `aux_head` IS the pose head, and a
   naive load raises on the 80-vs-99 output shape).
3. The guards: each head gets gradient only when its own weight > 0 (the
   inactive head's .grad is None, the parent tests' idiom); the pose term
   reaches the fusion and conv1; the pose floor is sample-dependent with both
   flags on (the zgpose REVISION 2 anti-cheat pin).
4. The configs compose, the model is byte-identical to M3/m4base/zgpose's,
   and the new train config differs from EACH parent on exactly the intended
   key set (the two one-variable claims made executable).
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
sys.path.insert(0, this_dir)          # for `import test_aux_action_heads`

import numpy as np
import torch

import test_aux_action_heads as tah
from probe_relpose import rot6d_from_mat
from diffusion_policy.dataset.multiview_image_dataset import (
    AUX_ACTION_KEY, AUX_VIEW_POSE_KEY, AUX_VIEW_POSE_MASK_KEY, VIEW_POSE_DIM,
    ACTION_DIM, quat_wxyz_to_mat)
from diffusion_policy.policy.diffusion_unet_image_policy import (
    DiffusionUnetImagePolicy)
from diffusion_policy.policy.diffusion_unet_image_policy_aux import (
    DiffusionUnetImagePolicyAux)
from diffusion_policy.policy.diffusion_unet_image_policy_zgpose import (
    DiffusionUnetImagePolicyZgPose)
from diffusion_policy.policy.diffusion_unet_image_policy_zvzg import (
    DiffusionUnetImagePolicyZvZg)
from diffusion_policy.common.pytorch_util import dict_apply

# a PERMUTATION of the 3-view synthetic fixture's pool: ring index != pool
# position, which is the only fixture under which a pool-ordering bug cannot
# hide (a sorted pool would make the two coincide).
V_POOL = [2, 0, 1]

# non-unit pose weight for the pins: at w=1.0 a weight-folding bug (ignoring
# the weight, or double-applying it) would be invisible in every comparison.
W_A = 1.0            # the action weight m4clear/m4base use
W_P = 0.8            # deliberately != 1.0


def _build_both_dataset(tmp, **over):
    """Both-target dataset over the M2 synthetic zarr, permuted pool."""
    kwargs = dict(emit_aux_action=True, aux_action_frame='base',
                  emit_aux_view_pose=True, view_pool=list(V_POOL))
    kwargs.update(over)
    return tah._build_m4_dataset(tmp, **kwargs)


def _batch_both(ds, idxs):
    samples = [ds[i] for i in idxs]
    return {
        'obs': {k: torch.stack([s['obs'][k] for s in samples])
                for k in samples[0]['obs']},
        'action': torch.stack([s['action'] for s in samples]),
        AUX_ACTION_KEY: torch.stack([s[AUX_ACTION_KEY] for s in samples]),
        AUX_VIEW_POSE_KEY: torch.stack(
            [s[AUX_VIEW_POSE_KEY] for s in samples]),
        AUX_VIEW_POSE_MASK_KEY: torch.stack(
            [s[AUX_VIEW_POSE_MASK_KEY] for s in samples]),
    }


# ---------------------------------------------------------------------------
# 1. the dataset
# ---------------------------------------------------------------------------

def test_dataset_joint_emission():
    tmp = tempfile.mkdtemp(prefix='zvzg_ds_')
    try:
        ds, shape_meta, tmd = _build_both_dataset(tmp)
        V = len(V_POOL)
        K = len(ds.rgb_keys)
        C = tah.AUX_N_STEPS
        max_live = 0
        for idx in (0, 3, 7, 11):
            batch = ds[idx]
            assert AUX_ACTION_KEY in batch, sorted(batch.keys())
            assert AUX_VIEW_POSE_KEY in batch and \
                AUX_VIEW_POSE_MASK_KEY in batch, sorted(batch.keys())
            aux = batch[AUX_ACTION_KEY].numpy()
            tgt = batch[AUX_VIEW_POSE_KEY].numpy()
            msk = batch[AUX_VIEW_POSE_MASK_KEY].numpy()
            assert aux.shape == (2, K, C * ACTION_DIM), aux.shape
            assert tgt.shape == (2, V, VIEW_POSE_DIM), tgt.shape
            assert msk.shape == (2, V), msk.shape
            assert all(a.dtype == np.float32 for a in (aux, tgt, msk))

            # which views are live, recovered from RENDERS only -- never from
            # cam_table, so a cam-table index bug cannot hide
            mask = batch['obs'][ds.view_mask_key].numpy()            # (To,K)
            views = tah._identify_slot_views(ds, idx, batch, tmd)
            assert len(views) == int(mask[0].sum()), (views, mask[0])
            max_live = max(max_live, len(views))

            # --- the base-frame action target ---------------------------
            # dead slots exactly zero; live slots == the stored action slice,
            # bit-for-bit; all live slots IDENTICAL (the view-invariant
            # property that is this arm's z_v term)
            assert np.all(aux[mask < 0.5] == 0.0)
            action_chunk = ds.sampler.sample_sequence(idx)['action']
            win = np.arange(2)[:, None] + np.arange(C)[None, :]
            want = action_chunk[win].astype(np.float32)              # (To,C,Da)
            live_slots = sorted(views)
            for slot in live_slots:
                got = aux[:, slot].reshape(2, C, ACTION_DIM)
                assert np.array_equal(got, want), (idx, slot)
            for s in live_slots[1:]:
                assert np.array_equal(aux[:, s], aux[:, live_slots[0]]), (idx, s)

            # --- the pooled pose set ------------------------------------
            # the mask marks exactly the pool positions of the identified
            # views; dead rows are exactly zero; live rows equal the
            # INDEPENDENTLY computed pose (via probe_relpose's own inverse)
            live_pos = sorted(ds._pool_pos[int(v)] for v in views.values())
            got_pos = sorted(np.where(msk[0] > 0.5)[0].tolist())
            assert got_pos == live_pos, (idx, got_pos, live_pos)
            assert int(msk[0].sum()) == len(views)
            for i in live_pos:
                v = V_POOL[i]
                want_pose = np.concatenate([
                    ds.cam_table[v][:3].astype(np.float64),
                    rot6d_from_mat(quat_wxyz_to_mat(ds.cam_table[v][3:7])),
                ])
                assert np.abs(tgt[:, i] - want_pose).max() < 1e-6, (idx, i, v)
            assert np.all(tgt[msk < 0.5] == 0.0)

            # --- the two masks agree: same draw, same liveness ----------
            assert int(msk[0].sum()) == int(mask[0].sum()), (idx, msk[0], mask[0])
        assert max_live > 1, 'no sample drew more than one view'
        print(f'  joint emission OK: base-frame chunks (To,{K},{C*ACTION_DIM}) == '
              f'the stored action slice on live slots (identical across them), '
              f'pose set (To,{V},{VIEW_POSE_DIM}) rows == pool positions '
              f'(permuted pool {V_POOL}), both zeroed when dead, masks agree')

        # --- RNG neutrality: neither flag (nor their combination) changes the
        # draw stream. The action flag alone, the pose flag alone and both off
        # must all leave view_mask/cam identical at the same seed, and each
        # flag-off build must lack its keys.
        ds_a, _, _ = _build_both_dataset(tmp, emit_aux_view_pose=False)
        ds_p, _, _ = _build_both_dataset(tmp, emit_aux_action=False)
        ds_off, _, _ = _build_both_dataset(
            tmp, emit_aux_action=False, emit_aux_view_pose=False)
        for idx in (0, 5, 9):
            np.random.seed(1234)
            b_both = ds[idx]
            for other in (ds_a, ds_p, ds_off):
                np.random.seed(1234)
                b = other[idx]
                assert torch.equal(b_both['obs'][ds.view_mask_key],
                                   b['obs'][ds.view_mask_key])
                for k in ds.cam_keys:
                    assert torch.equal(b_both['obs'][k], b['obs'][k])
            np.random.seed(1234)
            b_a = ds_a[idx]
            assert AUX_ACTION_KEY in b_a and AUX_VIEW_POSE_KEY not in b_a
            np.random.seed(1234)
            b_p = ds_p[idx]
            assert AUX_VIEW_POSE_KEY in b_p and AUX_ACTION_KEY not in b_p
            np.random.seed(1234)
            b_o = ds_off[idx]
            assert AUX_ACTION_KEY not in b_o and AUX_VIEW_POSE_KEY not in b_o
        print('  both flags together consume no RNG (draw + cam keys identical '
              'across both-on / action-only / pose-only / both-off at one seed)')

        # the ctor rejects the pose flag without a pool, at construction
        try:
            _build_both_dataset(tmp, view_pool=None)
        except ValueError as e:
            assert 'view_pool' in str(e), e
        else:
            raise AssertionError('ctor accepted emit_aux_view_pose without pool')
        print('  ctor rejects emit_aux_view_pose without view_pool')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. the policy: three-way pins
# ---------------------------------------------------------------------------

def _load_into(target_pol, source_sd, remap_pose=False):
    """Load `source_sd` into `target_pol`, remapping pose_head.* -> aux_head.*
    when the target's own `aux_head` IS the pose head (zgpose). Returns the
    unexpected keys (must be exactly the heads the target does not have)."""
    sd = source_sd
    if remap_pose:
        sd = {k: v for k, v in sd.items()
              if not k.startswith(('aux_head.', 'pose_head.'))}
        sd.update({'aux_head.' + k[len('pose_head.'):]: v
                   for k, v in source_sd.items() if k.startswith('pose_head.')})
    missing, unexpected = target_pol.load_state_dict(sd, strict=False)
    assert not missing, missing
    return unexpected


def _grads_equal(pol_ref, pol, target_name=None):
    """Compare every gradient of pol_ref against pol's, name-by-name (with an
    optional name remap for the pose-head-under-a-different-attribute case).
    Returns the number of non-None gradients compared."""
    named = dict(pol.named_parameters())
    n_grad = 0
    for name, p_ref in pol_ref.named_parameters():
        target = target_name(name) if target_name else name
        assert target in named, target
        p = named[target]
        if p_ref.grad is None:
            assert p.grad is None, (name, target)
            continue
        assert p.grad.shape == p_ref.grad.shape, (name, target)
        assert torch.equal(p_ref.grad, p.grad), (name, target)
        n_grad += 1
    return n_grad


def test_policy_three_way_pins():
    tmp = tempfile.mkdtemp(prefix='zvzg_policy_')
    try:
        ds, shape_meta, tmd = _build_both_dataset(tmp)
        V = len(V_POOL)
        normalizer = ds.get_normalizer()
        batch = _batch_both(ds, [0, 1])
        B = 2

        both = tah._make_policy(DiffusionUnetImagePolicyZvZg, shape_meta,
                                aux_loss_weight=W_A, aux_pose_weight=W_P,
                                aux_n_steps=tah.AUX_N_STEPS, aux_n_views=V)
        zero = tah._make_policy(DiffusionUnetImagePolicyZvZg, shape_meta,
                                aux_loss_weight=0.0, aux_pose_weight=0.0,
                                aux_n_steps=tah.AUX_N_STEPS, aux_n_views=V)
        act = tah._make_policy(DiffusionUnetImagePolicyZvZg, shape_meta,
                               aux_loss_weight=W_A, aux_pose_weight=0.0,
                               aux_n_steps=tah.AUX_N_STEPS, aux_n_views=V)
        pos = tah._make_policy(DiffusionUnetImagePolicyZvZg, shape_meta,
                               aux_loss_weight=0.0, aux_pose_weight=W_P,
                               aux_n_steps=tah.AUX_N_STEPS, aux_n_views=V)
        ref = tah._make_policy(DiffusionUnetImagePolicy, shape_meta)
        ref_a = tah._make_policy(DiffusionUnetImagePolicyAux, shape_meta,
                                 aux_loss_weight=W_A,
                                 aux_n_steps=tah.AUX_N_STEPS)
        ref_p = tah._make_policy(DiffusionUnetImagePolicyZgPose, shape_meta,
                                 aux_loss_weight=W_P, aux_n_views=V)

        # every instance gets `both`'s weights, so every difference below is a
        # weight setting or the compared class -- never initialization
        sd = both.state_dict()
        for pol in (zero, act, pos):
            missing, unexpected = pol.load_state_dict(sd, strict=True)
        unexpected = _load_into(ref, sd)
        assert all(k.startswith(('aux_head.', 'pose_head.')) for k in unexpected), \
            unexpected
        unexpected = _load_into(ref_a, sd)
        assert all(k.startswith('pose_head.') for k in unexpected), unexpected
        unexpected = _load_into(ref_p, sd, remap_pose=True)
        assert unexpected == [], unexpected
        for pol in (both, zero, act, pos, ref, ref_a, ref_p):
            pol.set_normalizer(normalizer)
            pol.eval()          # deterministic crop; no RNG in the encoder

        # --- (0, 0) == the base policy, loss and gradients BIT-identical ----
        torch.manual_seed(0)
        l_ref = ref.compute_loss(batch)
        torch.manual_seed(0)
        l_zero = zero.compute_loss(batch)
        assert torch.equal(l_ref, l_zero), (float(l_ref), float(l_zero))
        assert zero.last_aux_action_loss == 0.0 == zero.last_aux_pose_loss
        assert zero.last_aux_loss == 0.0
        ref.zero_grad(set_to_none=True)
        zero.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        ref.compute_loss(batch).backward()
        torch.manual_seed(0)
        zero.compute_loss(batch).backward()
        n = _grads_equal(ref, zero)
        assert n > 100, n
        for name, p in zero.named_parameters():
            if name.startswith(('aux_head.', 'pose_head.')):
                assert p.grad is None, f'{name} got a gradient at (0, 0)'
        print(f'  (0,0) == DiffusionUnetImagePolicy: loss and all {n} shared '
              f'gradients bit-identical; both heads untouched')

        # --- (W_A, 0) == m4base's term, BIT-identical -----------------------
        torch.manual_seed(0)
        l_a_ref = ref_a.compute_loss(batch)
        torch.manual_seed(0)
        l_act = act.compute_loss(batch)
        assert torch.equal(l_a_ref, l_act), (float(l_a_ref), float(l_act))
        assert act.last_aux_pose_loss == 0.0
        assert act.last_aux_loss == W_A * act.last_aux_action_loss
        ref_a.zero_grad(set_to_none=True)
        act.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        ref_a.compute_loss(batch).backward()
        torch.manual_seed(0)
        act.compute_loss(batch).backward()
        n = _grads_equal(ref_a, act)
        assert n > 100, n
        for name, p in act.named_parameters():
            if name.startswith('pose_head.'):
                assert p.grad is None, f'{name} got a gradient at w_g=0'
        print(f'  (w_v=W, 0) == DiffusionUnetImagePolicyAux: loss and all {n} '
              f'gradients bit-identical; the pose head gets none')

        # --- (0, W_P) == zgpose's term, BIT-identical -----------------------
        # (the remap: ref_p's `aux_head` is the pose head; a naive load would
        # raise on the 80-vs-27 output shape -- the fixture's V=3)
        torch.manual_seed(0)
        l_p_ref = ref_p.compute_loss(batch)
        torch.manual_seed(0)
        l_pos = pos.compute_loss(batch)
        assert torch.equal(l_p_ref, l_pos), (float(l_p_ref), float(l_pos))
        assert pos.last_aux_action_loss == 0.0
        assert pos.last_aux_loss == W_P * pos.last_aux_pose_loss
        ref_p.zero_grad(set_to_none=True)
        pos.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        ref_p.compute_loss(batch).backward()
        torch.manual_seed(0)
        pos.compute_loss(batch).backward()
        n = _grads_equal(
            ref_p, pos,
            target_name=lambda name: 'pose_head.' + name[len('aux_head.'):]
            if name.startswith('aux_head.') else name)
        assert n > 100, n
        for name, p in pos.named_parameters():
            if name.startswith('aux_head.'):
                assert p.grad is None, f'{name} got a gradient at w_v=0'
        print(f'  (0, w_g=W) == DiffusionUnetImagePolicyZgPose: loss and all '
              f'{n} gradients bit-identical (pose head via the remap); the '
              f'action head gets none')

        # --- per-term equality on ONE shared encoder pass --------------------
        nobs = normalizer.normalize(batch['obs'])
        this = dict_apply(nobs, lambda x: x[:, :2].reshape(-1, *x.shape[2:]))
        enc = both.obs_encoder.forward_full(this)
        assert tuple(both.obs_encoder.output_shape()) == (521,), \
            both.obs_encoder.output_shape()
        la_combo = both.aux_loss(enc, batch, B)
        lp_combo = both.aux_pose_loss(enc, batch, B)
        la_ref = ref_a.aux_loss(enc, batch, B)
        lp_ref = ref_p.aux_loss(enc, batch, B)
        assert torch.equal(la_combo, la_ref), 'action term != m4base term'
        assert torch.equal(lp_combo, lp_ref), 'pose term != zgpose term'
        print(f'  per-term equality on one encoder pass (action '
              f'{float(la_combo):.4f} == m4base; pose {float(lp_combo):.4f} == '
              f'zgpose)')

        # --- the composition: both == zero + w_v*L_a + w_g*L_p --------------
        torch.manual_seed(0)
        l_both = both.compute_loss(batch)
        # ... and compute_loss recorded exactly the standalone methods' values
        assert abs(both.last_aux_action_loss - float(la_combo)) < 1e-6
        assert abs(both.last_aux_pose_loss - float(lp_combo)) < 1e-6
        assert torch.allclose(
            l_both,
            l_zero.float() + W_A * both.last_aux_action_loss
            + W_P * both.last_aux_pose_loss, atol=1e-6), (
            float(l_both), float(l_zero), both.last_aux_action_loss,
            both.last_aux_pose_loss)
        assert both.last_aux_loss == (W_A * both.last_aux_action_loss
                                      + W_P * both.last_aux_pose_loss)
        print(f'  both-on loss == diff + w_v*L_a + w_g*L_p (diff '
              f'{float(l_zero):.4f}, L_a {both.last_aux_action_loss:.4f}, '
              f'L_p {both.last_aux_pose_loss:.4f}, weighted aux '
              f'{both.last_aux_loss:.4f})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_policy_pose_floor_and_gradient():
    """The two properties the pose term's premise rests on, with both flags on:
    a sample-dependent floor no constant head can reach, and a gradient path
    that actually reaches the fusion (the zgpose pins, re-exercised on the
    combined policy)."""
    tmp = tempfile.mkdtemp(prefix='zvzg_floor_')
    try:
        ds, shape_meta, tmd = _build_both_dataset(tmp)
        V = len(V_POOL)

        # --- anti-cheat: the best per-row constant cannot reach zero --------
        np.random.seed(0)     # immediately before the draws; nothing between
        ac = _batch_both(ds, range(8))
        tc = ac[AUX_VIEW_POSE_KEY].numpy().reshape(-1, V, VIEW_POSE_DIM)
        mc = ac[AUX_VIEW_POSE_MASK_KEY].numpy().reshape(-1, V) > 0.5
        n = mc.sum(0)
        Mc = tc.shape[0]
        assert ((n > 0) & (n < Mc)).any(), \
            'no pool row is partially live -- the floor would be 0'
        const = float(np.mean((tc - tc.mean(0)) ** 2))
        assert const > 1e-4, const
        assert float(np.mean(tc ** 2)) > const, 'zero head beats the constant'
        print(f'  anti-cheat pin (both flags on): best per-row constant scores '
              f'{const:.4f} > 0 (liveness must come from z_g)')

        # --- the gradient path: pose head -> z_g -> fusion -> conv1 ---------
        pol = tah._make_policy(DiffusionUnetImagePolicyZvZg, shape_meta,
                               aux_loss_weight=W_A, aux_pose_weight=W_P,
                               aux_n_steps=tah.AUX_N_STEPS, aux_n_views=V)
        pol.set_normalizer(ds.get_normalizer())
        pol.eval()
        batch = _batch_both(ds, [0, 1])
        nobs = pol.normalizer.normalize(batch['obs'])
        this = dict_apply(nobs, lambda x: x[:, :2].reshape(-1, *x.shape[2:]))

        with torch.no_grad():
            pol.pose_head.net[-1].weight.normal_(0, 0.01)
            pol.aux_head.net[-1].weight.normal_(0, 0.01)

        # the pose term alone must reach the fusion (its whole point).
        # A fresh forward per backward: the first backward frees the graph.
        for p in pol.obs_encoder.parameters():
            p.grad = None
        enc = pol.obs_encoder.forward_full(this)
        pol.aux_pose_loss(enc, batch, 2).backward()
        gf = pol.obs_encoder.fusion.in_proj_weight.grad
        assert gf is not None and torch.count_nonzero(gf) > 0, \
            'the pose term did not reach the fusion'
        g = pol.obs_encoder.backbone.conv1.weight.grad
        assert g is not None and torch.count_nonzero(g[:, :3]) > 0
        assert torch.count_nonzero(g[:, 3:]) > 0

        # the action term must reach z_views's pathway too (conv1, like M4's)
        for p in pol.obs_encoder.parameters():
            p.grad = None
        enc = pol.obs_encoder.forward_full(this)
        pol.aux_loss(enc, batch, 2).backward()
        g = pol.obs_encoder.backbone.conv1.weight.grad
        assert g is not None and torch.count_nonzero(g) > 0

        # and with BOTH weights on, each head receives its own gradient
        for p in pol.parameters():
            p.grad = None
        torch.manual_seed(0)
        pol.compute_loss(batch).backward()
        for name, p in pol.named_parameters():
            if name.startswith(('aux_head.', 'pose_head.')):
                assert p.grad is not None and torch.count_nonzero(p.grad) > 0, \
                    f'{name} got no gradient with both weights on'
        print('  pose term reaches the fusion + conv1 (image and ray channels); '
              'action term reaches conv1; both heads train with both on')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. the configs
# ---------------------------------------------------------------------------

def test_zvzg_configs():
    """The configs compose, the MODEL IS UNCHANGED, and the new train config
    differs from EACH parent on exactly the intended keys -- the two
    one-variable claims (vs m4base and vs zgpose) made executable."""
    import inspect
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')

    sig = inspect.signature(DiffusionUnetImagePolicyZvZg.__init__)
    for key in ('aux_loss_weight', 'aux_n_steps', 'aux_hidden_dim',
                'aux_pose_weight', 'aux_n_views'):
        assert key in sig.parameters, key

    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_zvzg_latent')
        cfg_m3 = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_m3')
        cfg_base = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_m4base_latent')
        cfg_zg = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_zgpose_latent')

    assert 'diffusion_unet_image_policy_zvzg' in cfg.policy._target_, \
        cfg.policy._target_
    assert 'latentprobe' in cfg._target_, cfg._target_
    # BOTH targets on, action target in the base frame
    assert cfg.task.dataset.emit_aux_action is True
    assert cfg.task.dataset.emit_aux_view_pose is True
    assert cfg.task.dataset.aux_action_frame == 'base'
    assert list(cfg.task.dataset.view_pool) == list(range(1, 12))
    assert tuple(cfg.task.dataset.view_count_range) == (1, 11)
    assert cfg.task.env_runner.m3_slots == 11 == cfg.policy.aux_n_views
    assert cfg.policy.aux_n_steps == cfg.task.dataset.aux_n_steps == 8
    # the registered action weight, and the LOUD PLACEHOLDER for the pose
    # weight (a non-1.33 value by design, so the driver's launch gate can
    # prove its override landed; see the config's comment)
    assert cfg.policy.aux_loss_weight == 1.0
    assert cfg.policy.aux_pose_weight == 1.0
    assert 'kwargs' not in cfg.policy, 'aux params must be named, not **kwargs'
    assert cfg.latent_probe.n_states == 128
    assert cfg.latent_probe.seed == 12345
    assert cfg.latent_probe.az0_view == 6
    assert cfg.latent_probe.out_dir == 'latent_snapshots'
    assert OmegaConf.to_container(cfg.latent_probe) == \
        OmegaConf.to_container(cfg_base.latent_probe)

    # THE CONSTRAINT: the encoder is byte-identical to M3's, m4base's and
    # zgpose's -- the arm changes the loss, not the model
    for ref_cfg, who in ((cfg_m3, 'm3'), (cfg_base, 'm4base'), (cfg_zg, 'zgpose')):
        assert OmegaConf.to_container(cfg.policy.obs_encoder) == \
            OmegaConf.to_container(ref_cfg.policy.obs_encoder), \
            f'zvzg changed the encoder vs {who}'
    for key in ('horizon', 'n_obs_steps', 'n_action_steps', 'obs_as_global_cond'):
        assert cfg[key] == cfg_m3[key], key

    enc = hydra.utils.instantiate(cfg.policy.obs_encoder)
    assert tuple(enc.output_shape()) == (521,), enc.output_shape()
    pol = hydra.utils.instantiate(
        cfg.policy, down_dims=[32, 64], diffusion_step_embed_dim=16)
    assert isinstance(pol, DiffusionUnetImagePolicyZvZg), type(pol)
    assert isinstance(pol, DiffusionUnetImagePolicyAux), type(pol)
    assert pol.kwargs == {}, pol.kwargs
    assert pol.aux_n_views == 11
    assert pol.aux_head.out_dim == pol.aux_n_steps * pol.action_dim
    assert pol.aux_head.in_dim == enc.fused_dim
    assert pol.pose_head.n_views == 11 and pol.pose_head.pose_dim == VIEW_POSE_DIM
    assert pol.pose_head.in_dim == enc.fused_dim
    assert pol.obs_feature_dim == 521, pol.obs_feature_dim
    print('  configs compose; encoder identical to M3/m4base/zgpose; both '
          'heads instantiate with empty kwargs')

    # ---- the one-variable pins: EVERY other resolved key is identical -----
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

    def diffs_vs(other):
        b = flatten(OmegaConf.to_container(other, resolve=False))
        return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}

    expected_vs_base = {
        'name',
        'task.name',
        'task.emit_aux_view_pose',
        'task.dataset.emit_aux_view_pose',
        'policy._target_',
        'policy.aux_n_views',
        'policy.aux_pose_weight',
    }
    got = diffs_vs(cfg_base)
    assert got == expected_vs_base, (
        f'vs m4base: unexpected {sorted(got - expected_vs_base)}; '
        f'missing {sorted(expected_vs_base - got)}')
    print(f'  vs m4base: exactly the intended {len(got)} keys (the pose term '
          f'and its emission)')

    expected_vs_zg = {
        'name',
        'task.name',
        'task.emit_aux_action',
        'task.aux_action_frame',
        'task.aux_n_steps',
        'task.dataset.emit_aux_action',
        'task.dataset.aux_action_frame',
        'task.dataset.aux_n_steps',
        'policy._target_',
        'policy.aux_loss_weight',
        'policy.aux_n_steps',
        'policy.aux_pose_weight',
    }
    got = diffs_vs(cfg_zg)
    assert got == expected_vs_zg, (
        f'vs zgpose: unexpected {sorted(got - expected_vs_zg)}; '
        f'missing {sorted(expected_vs_zg - got)}')
    print(f'  vs zgpose: exactly the intended {len(got)} keys (the action term '
          f'and its emission)')


def test():
    test_dataset_joint_emission()
    test_policy_three_way_pins()
    test_policy_pose_floor_and_gradient()
    test_zvzg_configs()
    print('ALL ZVZG COMBINED-SUPERVISION CHECKS PASSED')


if __name__ == '__main__':
    test()
