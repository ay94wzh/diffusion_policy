"""CPU-only tests for the z-supervision round (`PLAN_Z.md`): the z_v
shared-semantics alignment loss (cell A `zalign`), the z_g cross-view action
consistency loss (cell B `zcons`), and the joint cell C (`zboth`), all on the
committed m4base recipe (no simulator, no GPU).

Run from the repo root:
    python tests/test_z_supervision.py

What is checked, in rough order of how much it buys:

1. The dataset's SECOND draw: `obs2` exists iff `emit_second_draw` is on, with
   exactly the obs key set minus the view-independent low-dim keys; live
   overlap views carry the same render as draw 1; dead slots are exactly zero;
   and the draw rides a stateless per-sample private stream -- bit-identical
   under different global seeds (a pure function of `(seed, idx)`), and turning
   the flag on leaves draw 1's tensors AND the global numpy state untouched
   (the RNG-lockstep claim the driver header registers).
2. The shared semantic head: NOT zero at init (a zero-init head would make
   L_align identically 0 with exactly-zero gradients), and both construction
   and forward consume no global torch RNG.
3. The policy's bit-identity pins, each load-bearing for one claim:
     (align=0, cons=0)  ==  DiffusionUnetImagePolicyAux   (the m4base control)
   loss AND every shared gradient, torch.equal; the semantic head's own .grad
   is None at (0,0); the weighted decomposition l == diff + 1.0*aux + w*term
   holds at positive weights; cons_weight > 0 without obs2 raises loudly.
4. Whole-step RNG neutrality with a CROPPING encoder in train mode: after the
   reference and the zsup `compute_loss` (align-only AND cons), the global
   torch RNG state is equal -- the pin that fails if the second encoder pass
   is not under a save/restore.
5. Cons identities: `obs2 == obs` (deterministic crops) gives cons_loss exactly
   0.0; a genuinely different draw gives > 0; a hand-computed value pins the
   reduction, including a non-trivial loss_mask.
6. Align mechanics on a hand-computed fixture with an N=1 row (excluded from
   the mean, not 0/0); an all-N=1 batch returns exactly 0.0 with head grads
   zeros-not-None; and on the real model the align gradient provably reaches
   the trunk (conv1, the fusion) -- the anti-head-collapse pin.
7. The configs compose; the task clone differs from m4_base's by exactly
   {name, emit_second_draw}; the train config differs from m4base_latent's by
   exactly the seven registered keys; and the full policy instantiates with
   obs_feature_dim 521 and an empty `kwargs` (the rollout-TimeError guard).
"""
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

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import test_aux_action_heads as tah
from diffusion_policy.dataset.multiview_image_dataset import (
    AUX_ACTION_KEY, SECOND_OBS_KEY)
from diffusion_policy.model.vision.shared_semantic_head import (
    SharedSemanticHead)
from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    ViewConditionedObsEncoder)
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.policy.diffusion_unet_image_policy_aux import (
    DiffusionUnetImagePolicyAux)
from diffusion_policy.policy.diffusion_unet_image_policy_zsup import (
    DiffusionUnetImagePolicyZSup, masked_pair_mean)


def _np_state():
    st = np.random.get_state()
    return (st[0], tuple(int(x) for x in st[1]), int(st[2]))


def _build_z_dataset(tmp, emit_second_draw=False, **over):
    """m4-mode dataset fixture (3 views, distinct cam poses) + the new flag."""
    return tah._build_m4_dataset(
        tmp, emit_second_draw=emit_second_draw, **over)


def _batch_from_z(ds, idxs, with_obs2=True):
    """Batch dict incl. top-level SECOND_OBS_KEY and AUX_ACTION_KEY."""
    samples = [ds[i] for i in idxs]
    batch = {
        'obs': {k: torch.stack([s['obs'][k] for s in samples])
                for k in samples[0]['obs']},
        'action': torch.stack([s['action'] for s in samples]),
    }
    if AUX_ACTION_KEY in samples[0]:
        batch[AUX_ACTION_KEY] = torch.stack(
            [s[AUX_ACTION_KEY] for s in samples])
    if with_obs2:
        assert SECOND_OBS_KEY in samples[0]
        batch[SECOND_OBS_KEY] = {
            k: torch.stack([s[SECOND_OBS_KEY][k] for s in samples])
            for k in samples[0][SECOND_OBS_KEY]}
    return batch


def _slot_view_map(ds, sample_side, idx, tmd):
    """{slot: view} for the LIVE slots of one side ('obs' or obs2 dict),
    recovered from the renders only (the aux-test helper, reused)."""
    return tah._identify_slot_views(
        ds, idx, {'obs': sample_side}, tmd)


# ---------------------------------------------------------------------------
# 1. the dataset: the second draw
# ---------------------------------------------------------------------------

def test_dataset_second_draw():
    tmp = tempfile.mkdtemp(prefix='z_dataset_')
    try:
        d_off = os.path.join(tmp, 'off')
        d_on = os.path.join(tmp, 'on')
        os.makedirs(d_off)
        os.makedirs(d_on)
        ds_off, shape_meta, tmd = _build_z_dataset(d_off)
        ds_on, _, _ = _build_z_dataset(d_on, emit_second_draw=True)

        # -- flag off: nothing new anywhere
        s_off = ds_off[0]
        assert SECOND_OBS_KEY not in s_off
        try:
            ds_off._m3_second_draw(0)
        except RuntimeError as e:
            assert 'emit_second_draw' in str(e), e
        else:
            raise AssertionError('_m3_second_draw ran with the flag off')
        print('  flag off: no obs2 key; _m3_second_draw refuses')

        # -- schema: obs keys minus the view-independent low-dim keys
        idx = 2
        s = ds_on[idx]
        assert SECOND_OBS_KEY in s
        obs2 = s[SECOND_OBS_KEY]
        expected = (set(s['obs']) - set(ds_on.lowdim_keys))
        assert set(obs2) == expected, (
            set(obs2) ^ expected)
        To, K = ds_on.n_obs_steps, ds_on.n_slots
        H = tmd.H
        for k, v in obs2.items():
            if k.endswith('_image'):
                assert tuple(v.shape) == (To, 3, H, H), (k, v.shape)
            elif k.endswith('_cam'):
                assert tuple(v.shape) == (To, 10), (k, v.shape)
        assert tuple(obs2[ds_on.view_mask_key].shape) == (To, K)
        assert tuple(obs2[ds_on.eef_hist_key].shape) == (To, K, 44)
        print('  obs2 schema == obs minus low-dim keys, shapes as obs')

        # -- masks: every row has >= 1 live view; dead slots exactly zero
        m2 = obs2[ds_on.view_mask_key]
        assert float(m2.min()) in (0.0, 1.0)
        assert (m2.sum(dim=1) >= 1).all(), m2
        for k, rk in enumerate(ds_on.rgb_keys):
            dead_rows = m2[:, k] < 0.5
            if bool(dead_rows.any()):
                assert float(obs2[rk][dead_rows].abs().sum()) == 0.0, rk
        print('  obs2: live rows >=1, dead slots exactly zero')

        # -- overlap views carry the SAME render on both sides
        v1 = _slot_view_map(ds_on, s['obs'], idx, tmd)
        v2 = _slot_view_map(ds_on, obs2, idx, tmd)
        shared = {v for v in v1.values()} & {v for v in v2.values()}
        for v in shared:
            slot1 = [k for k, vv in v1.items() if vv == v][0]
            slot2 = [k for k, vv in v2.items() if vv == v][0]
            assert torch.equal(
                s['obs'][ds_on.rgb_keys[slot1]],
                obs2[ds_on.rgb_keys[slot2]]), v
        print(f'  overlap views reproduce the same render '
              f'({len(shared)} shared view(s) this draw)')

        # -- purity: obs2 is a pure function of (seed, idx)
        np.random.seed(11)
        a = ds_on[idx][SECOND_OBS_KEY]
        np.random.seed(99_999)
        b = ds_on[idx][SECOND_OBS_KEY]
        for k in a:
            assert torch.equal(a[k], b[k]), k
        print('  obs2 bit-identical under different global numpy seeds')

        # -- RNG neutrality: flag on leaves draw 1 AND the global state alone
        np.random.seed(7)
        s_on = ds_on[3]
        state_on = _np_state()
        np.random.seed(7)
        s_ref = ds_off[3]
        state_off = _np_state()
        assert state_on == state_off, 'the second draw consumed global RNG'
        for k in s_ref['obs']:
            assert torch.equal(s_on['obs'][k], s_ref['obs'][k]), k
        assert torch.equal(s_on[AUX_ACTION_KEY], s_ref[AUX_ACTION_KEY])
        # ... and the state also matches right after draw 1 alone
        np.random.seed(7)
        ds_off[3]      # re-fetch: same consumption as above
        assert _np_state() == state_off
        print('  flag on: draw 1 + aux + global numpy state identical to '
              'flag off')

        # -- val copy carries the flag
        val = ds_on.get_validation_dataset()
        assert val.emit_second_draw is True
        assert val.sampler_keys == ds_on.sampler_keys
        print('  get_validation_dataset carries emit_second_draw')

        # -- ctor guard: no pool -> refuse
        one_meta = tah._slot_shape_meta(1, H)
        path = os.path.join(d_on, 'synth.zarr')
        try:
            ds_cls = type(ds_on)
            ds_cls(shape_meta=one_meta, dataset_path=path, horizon=16,
                   n_obs_steps=2, abs_action=True, view_subset=[0],
                   emit_second_draw=True, seed=42)
        except ValueError as e:
            assert 'emit_second_draw' in str(e), e
        else:
            raise AssertionError('ctor accepted emit_second_draw w/o pool')
        print('  ctor rejects emit_second_draw without view_pool')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. the head
# ---------------------------------------------------------------------------

def test_semantic_head():
    torch.manual_seed(0)
    head = SharedSemanticHead(in_dim=16, hidden_dim=8)
    # NOTE: a bare construction consumes global RNG like any nn.Module -- the
    # neutrality claim is about the POLICY building it under the private
    # save/reseed/restore (pinned in `_make_pair`, not here).
    # NOT zero-init: the last layer must be live (zero-init would make L_align
    # identically 0 with exactly-zero gradients)
    wsum = sum(float(p.abs().sum()) for p in head.parameters())
    assert wsum > 0.0
    x = torch.randn(6, 16)
    state = torch.get_rng_state()
    y = head(x)
    assert torch.equal(torch.get_rng_state(), state), (
        'head forward consumed global RNG')
    assert tuple(y.shape) == (6, 16)
    yn = torch.nn.functional.normalize(y, dim=-1)
    assert float((yn.std(dim=0)).mean()) > 0.0
    try:
        head(torch.randn(6, 15))
    except ValueError as e:
        assert '16' in str(e), e
    else:
        raise AssertionError('head accepted a wrong last dim')
    print('  head: not zero-init; construction/forward RNG-neutral; '
          'wrong dim raises')


# ---------------------------------------------------------------------------
# 3. the policy pins
# ---------------------------------------------------------------------------

def _make_pair(shape_meta, normalizer):
    """Reference aux policy + zsup policy sharing the aux state dict."""
    torch.manual_seed(0)
    ref = tah._make_policy(
        DiffusionUnetImagePolicyAux, shape_meta, aux_loss_weight=1.0)
    ref.set_normalizer(normalizer)
    state_ref = torch.get_rng_state()
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    torch.manual_seed(0)
    pol = tah._make_policy(
        DiffusionUnetImagePolicyZSup, shape_meta, aux_loss_weight=1.0)
    # construction-lockstep pin: the semantic head is built under a private
    # stream, so constructing the zsup policy consumes exactly the global RNG
    # the aux policy consumed.
    assert torch.equal(torch.get_rng_state(), state_ref), (
        'zsup construction consumed global RNG')
    missing, unexpected = pol.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith('semantic_head.') for k in missing), missing
    pol.set_normalizer(normalizer)
    return ref, pol


def test_policy_pins():
    tmp = tempfile.mkdtemp(prefix='z_policy_')
    try:
        ds, shape_meta, tmd = _build_z_dataset(
            tmp, emit_second_draw=True)
        normalizer = ds.get_normalizer()
        batch = _batch_from_z(ds, [0, 1, 2])
        ref, pol = _make_pair(shape_meta, normalizer)

        # ---- (0, 0) == the aux control, loss and ALL shared gradients
        def run(p, seed):
            p.zero_grad(set_to_none=True)
            torch.manual_seed(seed)
            l = p.compute_loss(batch)
            l.backward()
            return l
        l_ref = run(ref, 0)
        l_zero = run(pol, 0)
        assert torch.equal(l_ref, l_zero), (float(l_ref), float(l_zero))
        n_grad = tah_zvzg_grads_equal(ref, pol)
        assert n_grad > 100, n_grad
        assert pol.semantic_head.net[0].weight.grad is None
        assert pol.last_align_loss == 0.0 and pol.last_cons_loss == 0.0
        print(f'  (0,0) == DiffusionUnetImagePolicyAux bit-for-bit '
              f'({n_grad} shared gradients equal)')

        # ---- (W, 0): decomposition l == diff + 1.0*aux + W*align
        pol_al = tah._make_policy(
            DiffusionUnetImagePolicyZSup, shape_meta, aux_loss_weight=1.0,
            align_weight=0.8)
        sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
        missing, unexpected = pol_al.load_state_dict(sd, strict=False)
        assert not unexpected
        pol_al.set_normalizer(normalizer)
        torch.manual_seed(0)
        l_al = pol_al.compute_loss(batch)
        decomp = (pol_al.last_diff_loss + 1.0 * pol_al.last_aux_loss
                  + 0.8 * pol_al.last_align_loss)
        assert abs(float(l_al.detach()) - decomp) < 1e-6, (
            float(l_al.detach()), decomp)
        assert pol_al.last_align_loss > 0.0
        pol_al.zero_grad(set_to_none=True)
        l_al.backward()
        g = pol_al.semantic_head.net[0].weight.grad
        assert g is not None and float(g.abs().sum()) > 0.0
        print(f'  (0.8, 0): l == diff + 1.0*aux + 0.8*align '
              f'(align {pol_al.last_align_loss:.4f}); head gets gradient')

        # ---- (0, W): decomposition with the cons term
        pol_co = tah._make_policy(
            DiffusionUnetImagePolicyZSup, shape_meta, aux_loss_weight=1.0,
            cons_weight=0.6)
        missing, unexpected = pol_co.load_state_dict(
            {k: v.detach().clone() for k, v in ref.state_dict().items()},
            strict=False)
        assert not unexpected
        pol_co.set_normalizer(normalizer)
        torch.manual_seed(0)
        l_co = pol_co.compute_loss(batch)
        decomp = (pol_co.last_diff_loss + 1.0 * pol_co.last_aux_loss
                  + 0.6 * pol_co.last_cons_loss)
        assert abs(float(l_co.detach()) - decomp) < 1e-6, (
            float(l_co.detach()), decomp)
        assert pol_co.last_cons_loss > 0.0, pol_co.last_cons_loss
        print(f'  (0, 0.6): l == diff + 1.0*aux + 0.6*cons '
              f'(cons {pol_co.last_cons_loss:.2e})')

        # ---- loud guard: cons on, obs2 absent
        batch_no2 = _batch_from_z(ds, [0, 1], with_obs2=False)
        try:
            pol_co.compute_loss(batch_no2)
        except ValueError as e:
            assert 'emit_second_draw' in str(e), e
        else:
            raise AssertionError('cons ran without obs2')
        print('  cons without obs2 raises with the emit_second_draw hint')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def tah_zvzg_grads_equal(ref, pol):
    """`_grads_equal` via the zvzg test's helper (name-identical params)."""
    import test_zvzg_supervision as tzv
    return tzv._grads_equal(ref, pol)


# ---------------------------------------------------------------------------
# 4. whole-step RNG neutrality, with a cropping encoder in train mode
# ---------------------------------------------------------------------------

def _make_crop_encoder(shape_meta):
    return ViewConditionedObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet('resnet18', weights=None),
        fused_dim=512, n_heads=8, cond_dim=128,
        use_plucker=True, use_eef_hist=True, eef_hist_steps=4,
        crop_shape=(6, 6), random_crop=True, use_group_norm=True,
        share_rgb_model=True, imagenet_norm=True)


def test_step_rng_neutrality():
    tmp = tempfile.mkdtemp(prefix='z_rng_')
    try:
        ds, shape_meta, tmd = _build_z_dataset(tmp, emit_second_draw=True)
        batch = _batch_from_z(ds, [0, 1])
        enc = _make_crop_encoder(shape_meta)
        sd_enc = {k: v.detach().clone() for k, v in enc.state_dict().items()}

        def policy(cls, **over):
            enc_i = _make_crop_encoder(shape_meta)
            enc_i.load_state_dict(
                {k: v.detach().clone() for k, v in sd_enc.items()})
            p = tah._make_policy(cls, shape_meta, obs_encoder=enc_i,
                                 aux_loss_weight=1.0, **over)
            p.set_normalizer(ds.get_normalizer())
            p.train()
            return p

        ref = policy(DiffusionUnetImagePolicyAux)
        torch.manual_seed(3)
        ref.compute_loss(batch)
        state_ref = torch.get_rng_state()

        for over, tag in [
                (dict(align_weight=0.8), 'align-only'),
                (dict(cons_weight=0.6), 'cons-only'),
                (dict(align_weight=0.8, cons_weight=0.6), 'both')]:
            p = policy(DiffusionUnetImagePolicyZSup, **over)
            torch.manual_seed(3)
            p.compute_loss(batch)
            assert torch.equal(torch.get_rng_state(), state_ref), (
                f'{tag}: global torch RNG state moved (crop save/restore '
                f'missing?)')
        print('  train-mode cropping step: RNG state after align/cons/both '
              '== the reference')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. cons identities and the reduction
# ---------------------------------------------------------------------------

def test_cons_identities():
    tmp = tempfile.mkdtemp(prefix='z_cons_')
    try:
        # deterministic crops: the fixture encoder has random_crop=False
        ds, shape_meta, tmd = _build_z_dataset(tmp, emit_second_draw=True)
        batch = _batch_from_z(ds, list(range(8)))
        pol = tah._make_policy(
            DiffusionUnetImagePolicyZSup, shape_meta, aux_loss_weight=1.0,
            cons_weight=0.6)
        pol.set_normalizer(ds.get_normalizer())

        # identical draws -> exactly zero
        batch_same = dict(batch)
        batch_same[SECOND_OBS_KEY] = batch['obs']
        torch.manual_seed(0)
        l_same = pol.compute_loss(batch_same)
        assert pol.last_cons_loss == 0.0, pol.last_cons_loss
        torch.manual_seed(0)
        l_ref = pol.compute_loss(batch)
        assert abs(l_same.item() - (l_ref.item()
               - 0.6 * pol.last_cons_loss)) < 1e-6  # both-weight algebra
        print('  obs2 == obs: cons_loss exactly 0.0')

        # genuinely different draws -> strictly positive
        torch.manual_seed(0)
        pol.compute_loss(batch)
        assert pol.last_cons_loss > 0.0, pol.last_cons_loss
        print(f'  independent draws: cons_loss {pol.last_cons_loss:.3e} > 0')

        # hand-computed reduction incl. a non-trivial loss_mask. NOTE the
        # convention mirrored from the parent's diff loss: masked elements are
        # ZEROED and the mean runs over ALL elements (not over the masked
        # ones) -- with the committed all-True mask the two agree, but the
        # fixture keeps them apart.
        pred = torch.randn(2, 3, 4)
        pred2 = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, 4, dtype=torch.bool)
        mask[0, 1, :] = False
        mask[1, 2, 2:] = False
        got = float(DiffusionUnetImagePolicyZSup.cons_loss(pol, pred, pred2,
                                                           mask))
        d = (pred2 - pred) ** 2
        per_row = [(d[b] * mask[b].float()).sum() / d[b].numel()
                   for b in range(2)]
        expected = float(torch.stack(per_row).mean())
        assert abs(got - expected) < 1e-6, (got, expected)
        # all-True mask reduces to a plain MSE
        full = torch.ones(2, 3, 4, dtype=torch.bool)
        got_full = float(DiffusionUnetImagePolicyZSup.cons_loss(
            pol, pred, pred2, full))
        assert abs(got_full
                   - float(torch.nn.functional.mse_loss(pred, pred2))) < 1e-7
        print('  reduction matches a hand-computed masked MSE')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 6. align mechanics
# ---------------------------------------------------------------------------

def test_align_mechanics():
    tmp = tempfile.mkdtemp(prefix='z_align_')
    try:
        ds, shape_meta, tmd = _build_z_dataset(tmp, emit_second_draw=True)
        pol = tah._make_policy(
            DiffusionUnetImagePolicyZSup, shape_meta, aux_loss_weight=1.0,
            align_weight=0.8)

        # -- hand-computed fixture: identity head so the outputs are z itself
        pol.semantic_head = torch.nn.Identity()
        z = torch.tensor([
            [[3., 4.], [1., 0.], [0., 0.]],      # row 0: two live pairs 1
            [[0., 0.], [0., 5.], [0., 0.]],      # row 1: N=1 -> no pair
            [[0., 1.], [0., -1.], [1., 0.]]])    # row 2: three live
        active = torch.tensor([
            [True, True, False],
            [False, True, False],
            [True, True, True]])
        got = float(pol.align_loss(z, active))
        # row 0: ||(0.6,0.8) - (1,0)||^2 = 0.8; row 2: (4 + 2 + 2)/3 = 8/3;
        # treated as the mean over the two rows WITH a pair
        expected = (0.8 + 8.0 / 3.0) / 2.0
        assert abs(got - expected) < 1e-6, (got, expected)
        # the per-row-then-rows denominator, not the pooled pair count
        pooled = (0.8 * 1 + (8.0 / 3.0) * 3) / 4.0
        assert abs(got - pooled) > 1e-3, 'rows are not weighted per row'
        print(f'  hand-computed align {got:.4f} == per-row mean (N=1 row '
              f'excluded, not NaN)')

        # -- all-N=1: exactly zero, backward OK, grads zeros not None
        pol2 = tah._make_policy(
            DiffusionUnetImagePolicyZSup, shape_meta, aux_loss_weight=1.0,
            align_weight=0.8)
        z1 = torch.randn(3, 3, 512)
        a1 = torch.tensor([[True, False, False],
                           [False, True, False],
                           [False, False, True]])
        l0 = pol2.align_loss(z1, a1)
        assert float(l0) == 0.0
        pol2.zero_grad(set_to_none=True)
        l0.backward()
        g0 = pol2.semantic_head.net[0].weight.grad
        assert g0 is not None and float(g0.abs().sum()) == 0.0
        print('  all-N=1: exactly 0.0; head grads zeros (not None)')

        # -- gradient reaches the trunk (the anti-head-collapse pin)
        pol3 = tah._make_policy(
            DiffusionUnetImagePolicyZSup, shape_meta, aux_loss_weight=1.0,
            align_weight=0.8)
        pol3.set_normalizer(ds.get_normalizer())
        batch = _batch_from_z(ds, list(range(6)), with_obs2=False)
        nobs = pol3.normalizer.normalize(batch['obs'])
        from diffusion_policy.common.pytorch_util import dict_apply
        this_nobs = dict_apply(nobs,
            lambda x: x[:, :pol3.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        out = pol3.obs_encoder.forward_full(this_nobs)
        pol3.zero_grad(set_to_none=True)
        pol3.align_loss(out['z_views'], out['view_active']).backward()
        named = dict(pol3.named_parameters())
        conv1 = named['obs_encoder.backbone.conv1.weight'].grad
        assert conv1 is not None and float(conv1.abs().sum()) > 0.0
        # NOTE: the fusion's parameters get NO gradient from this term, by
        # construction -- z_views is UPSTREAM of the MHA (z_g is downstream);
        # the align pressure reaches the per-view encoder trunk, which is
        # exactly where idea 1 is supposed to bite. The zvzg/pose arms, whose
        # term sits on z_g, are the ones that must show fusion gradients.
        fusion = named['obs_encoder.fusion.in_proj_weight'].grad
        assert fusion is None or float(fusion.abs().sum()) == 0.0
        print('  align gradient reaches conv1 (the trunk), not the fusion '
              '(z_views is upstream of the MHA)')

        # -- masked_pair_mean: rows with zero pairs are excluded, not 0/0
        pl = torch.tensor([2.0, 4.0, 1.0])
        rows = torch.tensor([0, 0, 2])
        val = float(masked_pair_mean(pl, rows, 4))
        assert abs(val - ((2.0 + 4.0) / 2 + 1.0) / 2) < 1e-7, val
        print('  masked_pair_mean: per-row mean, then over rows with pairs')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 7. the configs
# ---------------------------------------------------------------------------

def _flatten(obj, prefix=''):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f'{prefix}{k}.'))
    elif isinstance(obj, (list, tuple)):
        out[prefix[:-1]] = str(obj)
    else:
        out[prefix[:-1]] = obj
    return out


def _diff_keys(a, b):
    """Dotted keys whose unresolved values differ (resolve=False on purpose:
    interpolations must compare as their expressions)."""
    fa = _flatten(OmegaConf.to_container(a, resolve=False))
    fb = _flatten(OmegaConf.to_container(b, resolve=False))
    return {k for k in (set(fa) | set(fb)) if fa.get(k) != fb.get(k)}


def test_z_configs():
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_z_latent')
        base = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_m4base_latent')
        cfg_small = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_z_latent',
            overrides=['policy.down_dims=[32,64]',
                       'policy.diffusion_step_embed_dim=16'])

    assert 'DiffusionUnetImagePolicyZSup' in cfg.policy._target_
    assert float(cfg.policy.align_weight) == 0.0
    assert float(cfg.policy.cons_weight) == 0.0
    assert int(cfg.policy.semantic_hidden_dim) == 256
    assert cfg.policy.aux_loss_weight == 1.0
    assert cfg.policy.aux_n_steps == 8 == cfg.task.dataset.aux_n_steps
    assert cfg.task.dataset.emit_second_draw is False
    assert 'LatentProbe' in cfg._target_
    print('  z config: defaults are the aux control (both weights 0.0)')

    # the task clone is m4_base's except name + the new dataset flag
    t = OmegaConf.to_container(cfg.task, resolve=True)
    tb = OmegaConf.to_container(base.task, resolve=True)
    assert t.pop('name') == 'z_base_image_abs_multiview_az75'
    assert tb.pop('name') == 'm4_base_image_abs_multiview_az75'
    assert t['dataset'].pop('emit_second_draw') is False
    tb['dataset'].pop('emit_second_draw', None)
    assert t == tb, 'the z task clone differs from the m4base task'
    print('  task clone == m4base task except {name, emit_second_draw}')

    # the workspace diff is exactly the allowlist
    got = _diff_keys(cfg, base)
    expected = {
        'name', 'task.name', 'task.dataset.emit_second_draw',
        'policy._target_', 'policy.align_weight', 'policy.cons_weight',
        'policy.semantic_hidden_dim'}
    assert got == expected, (sorted(got - expected), sorted(expected - got))
    print(f'  workspace diff vs m4base_latent is exactly the {len(expected)} '
          f'registered keys')

    # the encoder block is byte-identical (a one-variable claim)
    assert (OmegaConf.to_container(cfg.policy.obs_encoder)
            == OmegaConf.to_container(base.policy.obs_encoder))
    print('  encoder block byte-identical to m4base_latent')

    # the full policy instantiates; cond width unchanged; kwargs empty
    pol = hydra.utils.instantiate(cfg_small.policy)
    assert pol.obs_feature_dim == 521
    assert pol.kwargs == {}
    assert pol.semantic_head.in_dim == 512
    assert tuple(pol.obs_encoder.output_shape()) == (521,)
    print('  policy instantiates (obs_feature_dim 521, semantic_head.in_dim '
          '512, kwargs empty)')


def test():
    print('--- dataset: the second draw ---')
    test_dataset_second_draw()
    print('--- the shared semantic head ---')
    test_semantic_head()
    print('--- the policy pins ---')
    test_policy_pins()
    print('--- step-level RNG neutrality ---')
    test_step_rng_neutrality()
    print('--- cons identities ---')
    test_cons_identities()
    print('--- align mechanics ---')
    test_align_mechanics()
    print('--- the configs ---')
    test_z_configs()
    print('ALL Z-SUPERVISION CHECKS PASSED')


if __name__ == '__main__':
    test()
