"""CPU-only checks for the per-view Gaussian heads + PoE fusion (PLAN_G.md stage 1).

Six groups, ordered by value:

1. the fusion math against a float64 numpy reference, with the masking failure
   that is SILENT at init (dead slots carry ``logvar = 0``, i.e. precision 1 --
   an unmasked formula adds one certain phantom expert per dead slot);
2. the N=1 / at-init identities that make the cells interpretable: PoE returns
   the single live mean bitwise, and cell C at init is bitwise cell D;
3. ``PerViewGaussianHead``: ``mu == z`` bitwise at init, exact clamp
   saturation, no RNG at forward;
4. the encoder subclass: the returned-dict contract (``z_views`` unchanged and
   raw, dead rows zero, ``z_global_logvar`` None outside ``poe``), the RNG
   lockstep (``ProbViewConditionedObsEncoder`` consumes exactly the parent's
   stream, so every shared parameter AND every later draw is bit-identical to
   the deterministic baseline's), B/C/D state-dict bit-identity, and the
   branch-discrimination pins (perturbing the logvar branch moves ``poe`` only);
5. the policy: ``aux_input`` selects the tensor it says it selects (poison
   test, both directions), and ``aux_input='z_v'`` + the parent encoder is
   bit-identical to ``DiffusionUnetImagePolicyAux`` in loss and all gradients;
6. the probe extension: the prob-probe workspace's snapshots are a strict
   superset of the parent's (``zv/zg/vg_n1/state_idx`` bit-identical), and the
   new scalars move when the Gaussian head moves.

Run from the repo root:
    python tests/test_prob_gaussian_fusion.py
"""
import inspect
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

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

import test_aux_action_heads as tah
from diffusion_policy.dataset.multiview_image_dataset import AUX_ACTION_KEY
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.per_view_gaussian import (
    PerViewGaussianHead, masked_weighted_mean, poe_fuse, uniform_fuse)
from diffusion_policy.model.vision.prob_view_conditioned_obs_encoder import (
    ProbViewConditionedObsEncoder)
from diffusion_policy.policy.diffusion_unet_image_policy_aux import (
    DiffusionUnetImagePolicyAux)
from diffusion_policy.policy.diffusion_unet_image_policy_prob import (
    DiffusionUnetImagePolicyProb)
from diffusion_policy.workspace.train_diffusion_unet_image_workspace_latentprobe import (
    TrainDiffusionUnetImageWorkspaceLatentProbe)
from diffusion_policy.workspace.train_diffusion_unet_image_workspace_probprobe import (
    TrainDiffusionUnetImageWorkspaceProbProbe)

HIST_DIM = 44            # 11 per step x 4 steps (the config default)
AZ0_VIEW = 1             # the fixture's ring: az [-30, 0, 30] -> ring 1 is az_0


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _make_prob_encoder(shape_meta, mode='mha', **over):
    kwargs = dict(
        shape_meta=shape_meta,
        rgb_model=get_resnet('resnet18', weights=None),
        fused_dim=512, n_heads=8, cond_dim=128,
        use_plucker=True, use_eef_hist=True, eef_hist_steps=4,
        crop_shape=None, random_crop=False, use_group_norm=True,
        share_rgb_model=True, imagenet_norm=True,
        fusion_mode=mode, logvar_min=-6.0, logvar_max=2.0,
        gauss_hidden=256, head_init_seed=42)
    kwargs.update(over)
    return ProbViewConditionedObsEncoder(**kwargs)


def _fake_obs(shape_meta, n, live_by_row, hw=8, seed=0):
    """A valid obs dict for the encoder, built from shape_meta alone."""
    g = torch.Generator().manual_seed(seed)
    obs = {}
    rgb_keys = sorted(k for k, v in shape_meta['obs'].items()
                      if v.get('type', 'low_dim') == 'rgb')
    K = len(rgb_keys)
    for k, key in enumerate(rgb_keys):
        obs[key] = torch.randn(n, 3, hw, hw, generator=g)
        cam = torch.zeros(n, 10)
        cam[:, 0] = 0.5 + 0.1 * k            # distinct positions
        cam[:, 3] = 1.0                      # identity quat, w first
        cam[:, 7] = 45.0
        cam[:, 8] = hw
        cam[:, 9] = hw
        obs[key.replace('_image', '_cam')] = cam
    mask = torch.zeros(n, K)
    for i, slots in enumerate(live_by_row):
        mask[i, slots] = 1.0
    obs['view_mask'] = mask
    obs['view_eef_hist'] = torch.randn(n, K, HIST_DIM, generator=g)
    for key, dim in (('robot0_eef_pos', 3), ('robot0_eef_quat', 4),
                     ('robot0_gripper_qpos', 2)):
        obs[key] = torch.zeros(n, dim)
    return obs


def _grads_equal(a, b):
    """Name-by-name gradient lockstep (None-agreement included). -> count."""
    n = 0
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        if p1.grad is None:
            assert p2.grad is None, n1
            continue
        assert p1.grad.shape == p2.grad.shape, n1
        assert torch.equal(p1.grad, p2.grad), n1
        n += 1
    return n


# ---------------------------------------------------------------------------
# 1. the fusion math
# ---------------------------------------------------------------------------

def _fusion_fixture(N=4, K=5, D=6, seed=0):
    """Random fixture with the model's real init layout on dead slots:
    ``mu = 0`` and ``logvar = 0``. The N=1 row's live slot is at logvar=2.0
    (precision e^-2 << 1), which is what makes the phantom-expert mutation
    below fail loudly rather than by a rounding."""
    torch.manual_seed(seed)
    active = torch.zeros(N, K, dtype=torch.bool)
    active[0, 2] = True                       # N=1
    active[1, :] = True                       # full
    active[2, :3] = True
    active[3, 1] = active[3, 4] = True
    mu = torch.zeros(N, K, D)
    lv = torch.zeros(N, K, D)
    for i in range(N):
        for j in range(K):
            if active[i, j]:
                mu[i, j] = torch.randn(D)
                lv[i, j] = -6.0 + 8.0 * torch.rand(1).item()
    lv[0, 2] = 2.0
    return mu, lv, active


def test_fusion_vs_numpy_reference():
    mu, lv, active = _fusion_fixture()
    mu_g, lv_g = poe_fuse(mu, lv, active)

    mu64, lv64 = mu.double().numpy(), lv.double().numpy()
    a = active.numpy()
    p = np.exp(-lv64) * a[..., None]
    denom = p.sum(1)
    ref_mu = (p * mu64).sum(1) / denom
    ref_lv = -np.log(denom)
    rel_mu = np.abs(mu_g.double().numpy() - ref_mu).max() / np.abs(ref_mu).max()
    err_lv = np.abs(lv_g.double().numpy() - ref_lv).max()
    assert rel_mu < 1e-5, rel_mu
    assert err_lv < 1e-4, err_lv

    # ---- mutation power: the UNMASKED formula must fail loudly --------------
    # dead slots carry logvar=0 -> p=1, so the naive denominator gains four
    # phantom experts on the N=1 row (4.135 vs e^-2 = 0.135)
    p_naive = torch.exp(-lv)
    naive_mu = (p_naive * mu).sum(1) / p_naive.sum(1)
    naive_lv = -torch.log(p_naive.sum(1))
    gap_mu = float((naive_mu[0] - mu_g[0]).norm() / mu_g[0].norm())
    gap_lv = float((naive_lv[0] - lv_g[0]).abs().max())
    assert gap_mu > 0.5, gap_mu
    assert gap_lv > 1.0, gap_lv
    print(f'  poe_fuse == numpy float64 (rel {rel_mu:.2e}); the unmasked formula '
          f'fails by {gap_mu:.2f} rel / {gap_lv:.2f} log-var (phantom experts)')

    # ---- dead-slot values cannot leak ---------------------------------------
    mu2, lv2 = mu.clone(), lv.clone()
    dead = ~active
    mu2[dead] = torch.randn(int(dead.sum()), mu.shape[-1]) * 100.0
    lv2[dead] = -50.0
    mu_g2, lv_g2 = poe_fuse(mu2, lv2, active)
    assert torch.equal(mu_g, mu_g2) and torch.equal(lv_g, lv_g2)
    print('  dead-slot values (incl. huge weights) do not move either output')


def test_fusion_n1_and_init_identity():
    mu, lv, active = _fusion_fixture()
    mu_g, lv_g = poe_fuse(mu, lv, active)
    # N=1: weights p/p == 1.0 exactly, so the mean is BITWISE the live slot's
    assert torch.equal(mu_g[0], mu[0, 2])
    # the logvar round-trips through -log(exp(-.)): ~1 ulp, not exact
    assert torch.allclose(lv_g[0], lv[0, 2], atol=1e-5)
    print('  N=1: mu_g bitwise == the single live mean; logvar_g within 1 ulp')

    # C-at-init == D, bitwise: logvar == 0 -> p == 1.0, the same float 1.0 the
    # uniform weight uses
    mu0 = torch.randn(*mu.shape)
    mu0[~active] = 0.0
    zero_lv = torch.zeros_like(lv)
    m_poe, l_poe = poe_fuse(mu0, zero_lv, active)
    m_uni = uniform_fuse(mu0, active)
    assert torch.equal(m_poe, m_uni)
    # at init every live slot claims precision 1, so the fused logvar is the
    # per-row constant -log(N_live) -- the UNet never sees it, and it is NOT
    # part of the C==D pin (which is about the mean)
    n_live = active.sum(1).float()
    assert torch.allclose(
        l_poe, (-torch.log(n_live)).unsqueeze(-1).expand_as(l_poe), atol=1e-6)
    # and `uniform_fuse` is the masked mean (to rounding: it normalizes each
    # weight BEFORE summing, which is exactly what buys the N=1 bitwise pin)
    ref = (mu0 * active.unsqueeze(-1)).sum(1) / active.sum(1, keepdim=True)
    assert torch.allclose(m_uni, ref, atol=1e-6)
    print('  C-at-init == D bitwise (PoE at logvar=0 == the uniform masked mean)')


def test_fusion_bounds_and_permutation():
    # the clamp is the overflow guarantee: logvar in [-6, 2] over K=11
    K, D = 11, 6
    active = torch.ones(2, K, dtype=torch.bool)
    mu = torch.randn(2, K, D) * 1e4
    lv = torch.empty(2, K, D)
    lv[:, ::2] = -6.0
    lv[:, 1::2] = 2.0
    m, l = poe_fuse(mu, lv, active)
    assert bool(torch.isfinite(m).all()) and bool(torch.isfinite(l).all())
    # mu_g is a convex combination, per dimension
    assert bool((m.abs() <= mu.abs().amax(dim=1) + 1e-3).all())
    # sum p in [11 e^-6, 11 e^6] -> logvar_g in [-6-log11, 2-log11]
    assert float(l.max()) <= 2.0 - np.log(11) + 1e-3
    assert float(l.min()) >= -6.0 - np.log(11) - 1e-3
    print('  bounds: finite at both clamp rails, |mu_g| bounded by the max '
          'live |mu|, logvar_g analytic')

    # permutation invariance (the fusion must not depend on slot order)
    mu, lv, active = _fusion_fixture()
    m0, l0 = poe_fuse(mu, lv, active)
    perm = torch.randperm(mu.shape[1])
    m1, l1 = poe_fuse(mu[:, perm], lv[:, perm], active[:, perm])
    assert torch.allclose(m0, m1, atol=1e-6) and torch.allclose(l0, l1, atol=1e-6)

    # guards, loud
    for bad in (
            lambda: masked_weighted_mean(mu, torch.zeros_like(mu), active),
            lambda: masked_weighted_mean(mu, torch.ones_like(mu), active.double()),
            lambda: masked_weighted_mean(mu[:, :2], torch.ones_like(mu), active),
    ):
        try:
            bad()
            assert False, 'guard did not fire'
        except ValueError:
            pass
    print('  permutation-invariant; zero-weight / dtype / shape guards fire')


# ---------------------------------------------------------------------------
# 2. the head
# ---------------------------------------------------------------------------

def test_gaussian_head():
    torch.manual_seed(0)
    head = PerViewGaussianHead(12, hidden_dim=8)
    z = torch.randn(7, 12)
    mu, lv = head(z)
    assert torch.equal(mu, z)                      # residual + zero-init
    assert torch.equal(lv, torch.zeros_like(lv))   # zero-init, 0 is unclamped

    with torch.no_grad():
        head.logvar_net[-1].bias.fill_(50.0)
    _, lv_hi = head(z)
    assert bool((lv_hi == 2.0).all())
    with torch.no_grad():
        head.logvar_net[-1].bias.fill_(-50.0)
    _, lv_lo = head(z)
    assert bool((lv_lo == -6.0).all())

    with torch.no_grad():
        gen = torch.Generator().manual_seed(1)
        head.mu_net[-1].weight.normal_(generator=gen)
    mu2, _ = head(z)
    assert torch.allclose(mu2 - z, head.mu_net(z), atol=1e-6)

    st = torch.get_rng_state()
    head(z)
    assert torch.equal(st, torch.get_rng_state()), 'head consumed RNG'
    print('  head: mu == z bitwise at init; clamp saturates exactly; no RNG')

    for bad in (dict(logvar_min=1.0), dict(logvar_max=-3.0),
                dict(logvar_min=float('-inf'))):
        try:
            PerViewGaussianHead(12, **bad)
            assert False, bad
        except ValueError:
            pass
    try:
        head(torch.randn(3, 13))
        assert False, 'shape guard did not fire'
    except ValueError:
        pass
    print('  head ctor/dim guards fire')


# ---------------------------------------------------------------------------
# 3. the encoder
# ---------------------------------------------------------------------------

def test_encoder_contract():
    shape_meta = tah._slot_shape_meta(4, 8)
    obs = _fake_obs(shape_meta, 3, [[0, 1, 2, 3], [0], [1, 2]])
    live_row, live_slot = 1, 0
    for mode in ('mha', 'poe', 'mean'):
        torch.manual_seed(0)
        enc = _make_prob_encoder(shape_meta, mode)
        enc.eval()
        assert tuple(enc.output_shape()) == (521,), enc.output_shape()
        out = enc.forward_full(obs)
        assert set(out) == {'z_global', 'z_views', 'view_active',
                            'mu_views', 'logvar_views', 'z_global_logvar'}
        assert tuple(out['z_views'].shape[:2]) == tuple(out['view_active'].shape)
        plain = enc.forward(obs)
        assert torch.equal(plain, torch.cat(
            [out['z_global']] + [obs[k] for k in enc.low_dim_keys], dim=-1))
        dead = ~out['view_active']
        assert int(torch.count_nonzero(out['z_views'][dead])) == 0
        assert int(torch.count_nonzero(out['mu_views'][dead])) == 0
        assert int(torch.count_nonzero(out['logvar_views'][dead])) == 0
        if mode == 'poe':
            assert out['z_global_logvar'].shape == out['z_global'].shape
            assert bool(torch.isfinite(out['z_global_logvar']).all())
        else:
            assert out['z_global_logvar'] is None
        # the N=1 row: poe/mean degenerate to the live mean bitwise; mha does
        # NOT (it applies the learned attention output map) -- the pin must
        # discriminate, and it does only because out_proj is not identity
        if mode in ('poe', 'mean'):
            assert torch.equal(out['z_global'][live_row],
                               out['mu_views'][live_row, live_slot])
        else:
            assert not torch.equal(out['z_global'][live_row],
                                   out['mu_views'][live_row, live_slot])
    print('  contract (521, dead rows zero, z_global_logvar only in poe, N=1 '
          'degeneracy in poe/mean not mha) holds in all three modes')


def test_encoder_init_lockstep():
    shape_meta = tah._slot_shape_meta(4, 8)
    torch.manual_seed(0)
    parent = tah._make_encoder(shape_meta)
    st_parent = torch.get_rng_state()
    torch.manual_seed(0)
    prob = _make_prob_encoder(shape_meta, 'poe')
    st_prob = torch.get_rng_state()
    assert torch.equal(st_parent, st_prob), (
        'the prob encoder consumed global RNG: the Gaussian head must be '
        'built from its private stream')
    sd_p, sd_c = parent.state_dict(), prob.state_dict()
    assert not (set(sd_p) - set(sd_c))
    assert set(sd_c) - set(sd_p) == {k for k in sd_c
                                     if k.startswith('gauss_head.')}
    for k in sd_p:
        assert torch.equal(sd_p[k], sd_c[k]), k
    print(f'  private-stream init: RNG state + all {len(sd_p)} shared tensors '
          f'bit-identical to the parent; extras are gauss_head.* only')

    # B/C/D: identical construction, identical parameters
    encs = []
    for mode in ('mha', 'poe', 'mean'):
        torch.manual_seed(0)
        encs.append(_make_prob_encoder(shape_meta, mode))
    sds = [e.state_dict() for e in encs]
    assert set(sds[0]) == set(sds[1]) == set(sds[2])
    for s in sds[1:]:
        for k in sds[0]:
            assert torch.equal(sds[0][k], s[k]), k
    print('  B/C/D state dicts bit-identical under one seed')

    # B-at-init == A: same weights, and mu == z means the attention runs over
    # the very same tokens -- bitwise
    obs = _fake_obs(shape_meta, 2, [[0, 1, 2, 3], [2]])
    parent.eval()
    encs[0].eval()
    oa = parent.forward_full(obs)
    ob = encs[0].forward_full(obs)
    assert torch.equal(oa['z_views'], ob['z_views'])
    assert torch.equal(oa['z_global'], ob['z_global'])
    live = ob['view_active']
    assert torch.equal(ob['mu_views'][live], ob['z_views'][live])
    print('  B-at-init == A bitwise (z_views and z_global); mu == z on live slots')


def test_encoder_branch_discrimination():
    shape_meta = tah._slot_shape_meta(3, 8)
    obs = _fake_obs(shape_meta, 2, [[0, 1, 2], [1]])
    encs = {}
    for mode in ('mha', 'poe', 'mean'):
        torch.manual_seed(0)
        encs[mode] = _make_prob_encoder(shape_meta, mode)
        encs[mode].eval()
    base = {m: e.forward_full(obs) for m, e in encs.items()}

    for e in encs.values():
        with torch.no_grad():
            # a HETEROGENEOUS perturbation of the logvar branch: only precision
            # RATIOS enter the PoE mean, so a uniform bias shift would (rightly)
            # leave poe's z_global alone
            e.gauss_head.logvar_net[-1].weight.normal_(
                mean=0.0, std=0.05,
                generator=torch.Generator().manual_seed(3))
    moved = {m: e.forward_full(obs) for m, e in encs.items()}
    assert not torch.equal(moved['poe']['z_global'], base['poe']['z_global'])
    for m in ('mha', 'mean'):
        assert torch.equal(moved[m]['z_global'], base[m]['z_global']), m
        assert torch.equal(moved[m]['z_views'], base[m]['z_views']), m
    print('  logvar branch moves poe only (mha/mean bit-identical)')

    for e in encs.values():
        with torch.no_grad():
            e.gauss_head.mu_net[-1].bias.add_(0.25)
    moved2 = {m: e.forward_full(obs) for m, e in encs.items()}
    for m in ('mha', 'poe', 'mean'):
        assert not torch.equal(moved2[m]['z_global'], moved[m]['z_global']), m
    print('  mu branch moves all three fusions')

    # a NaN logvar must be caught, not propagated
    with torch.no_grad():
        encs['poe'].gauss_head.logvar_net[-1].bias.fill_(float('nan'))
    try:
        encs['poe'].forward_full(obs)
        assert False, 'NaN logvar not caught'
    except ValueError:
        pass
    print('  NaN logvar raises (the loud finite/clamp check fires)')


def test_encoder_gradients():
    shape_meta = tah._slot_shape_meta(3, 8)
    obs = _fake_obs(shape_meta, 2, [[0, 1, 2], [2]])
    for mode, want_logvar in (('poe', True), ('mha', False), ('mean', False)):
        torch.manual_seed(0)
        enc = _make_prob_encoder(shape_meta, mode)
        enc.eval()
        out = enc.forward_full(obs)
        out['z_global'].sum().backward()
        g_mu = enc.gauss_head.mu_net[-1].weight.grad
        g_lv = enc.gauss_head.logvar_net[-1].weight.grad
        assert g_mu is not None and float(g_mu.abs().sum()) > 0, mode
        if want_logvar:
            assert g_lv is not None and float(g_lv.abs().sum()) > 0, mode
        else:
            assert g_lv is None, mode
    print('  gradients: mu branch live in all modes; logvar branch live in poe, '
          'grad-None in mha/mean')


# ---------------------------------------------------------------------------
# 4. the policy
# ---------------------------------------------------------------------------

def test_policy_aux_input_switch():
    tmp = tempfile.mkdtemp(prefix='prob_policy_switch_')
    try:
        ds, shape_meta, _tmd = tah._build_m4_dataset(tmp)
        normalizer = ds.get_normalizer()
        batch = _batch_from(ds, [0, 1])
        torch.manual_seed(0)
        enc = _make_prob_encoder(shape_meta, 'poe')
        pol_mu = tah._make_policy(
            DiffusionUnetImagePolicyProb, shape_meta, obs_encoder=enc,
            aux_loss_weight=1.0, aux_n_steps=tah.AUX_N_STEPS,
            aux_input='mu_v')
        pol_zv = tah._make_policy(
            DiffusionUnetImagePolicyProb, shape_meta,
            obs_encoder=_make_prob_encoder(shape_meta, 'poe'),
            aux_loss_weight=1.0, aux_n_steps=tah.AUX_N_STEPS,
            aux_input='z_v')
        for pol in (pol_mu, pol_zv):
            pol.set_normalizer(normalizer)
            pol.eval()
        torch.manual_seed(0)
        l = pol_mu.compute_loss(batch)
        assert np.isfinite(float(l))
        assert pol_mu.last_aux_action_loss > 0
        assert pol_mu.last_aux_loss == pol_mu.last_aux_action_loss, (
            'for this cell the unweighted per-term and the parent aux key must '
            'agree (same convention as m4base)')

        nobs = normalizer.normalize(batch['obs'])
        this = {k: v[:, :2].reshape(-1, *v.shape[2:])
                for k, v in nobs.items()}
        enc_out = enc.forward_full(this)

        # the aux head is zero-initialized: make it input-sensitive, or the
        # poison test below would be vacuous (the output is 0 for any input)
        for pol in (pol_mu, pol_zv):
            with torch.no_grad():
                last = pol.aux_head.net[-1]
                last.weight.normal_(
                    mean=0.0, std=0.05,
                    generator=torch.Generator().manual_seed(7))
                last.bias.fill_(0.1)

        def poisoned(key, delta):
            e2 = dict(enc_out)
            e2[key] = enc_out[key].clone()
            e2[key][enc_out['view_active']] += delta
            return e2

        clean = pol_mu.aux_loss(enc_out, batch, 2)
        assert torch.equal(pol_mu.aux_loss(poisoned('z_views', 1e3), batch, 2),
                           clean), "aux_input='mu_v' read z_views"
        assert not torch.equal(pol_mu.aux_loss(poisoned('mu_views', 1e3), batch, 2),
                               clean), "aux_input='mu_v' did not read mu_views"
        enc_zv = pol_zv.obs_encoder.forward_full(this)
        clean_zv = pol_zv.aux_loss(enc_zv, batch, 2)
        e2 = dict(enc_zv)
        e2['mu_views'] = enc_zv['mu_views'].clone()
        e2['mu_views'][enc_zv['view_active']] += 1e3
        assert torch.equal(pol_zv.aux_loss(e2, batch, 2), clean_zv)
        e3 = dict(enc_zv)
        e3['z_views'] = enc_zv['z_views'].clone()
        e3['z_views'][enc_zv['view_active']] += 1e3
        assert not torch.equal(pol_zv.aux_loss(e3, batch, 2), clean_zv)
        print('  aux_input selects exactly its tensor (poison test, both '
              'directions); last_aux_action_loss == last_aux_loss')

        # the missing-key guard
        try:
            pol_mu.aux_loss({'z_views': enc_out['z_views'],
                             'view_active': enc_out['view_active']}, batch, 2)
            assert False, 'missing mu_views not caught'
        except ValueError:
            pass
        # ctor gates
        try:
            tah._make_policy(DiffusionUnetImagePolicyProb, shape_meta,
                             aux_input='mu_v')     # plain parent encoder
            assert False, "aux_input='mu_v' accepted a non-Gaussian encoder"
        except ValueError:
            pass
        try:
            tah._make_policy(DiffusionUnetImagePolicyProb, shape_meta,
                             aux_input='bogus')
            assert False
        except ValueError:
            pass
        assert 'aux_input' in inspect.signature(
            DiffusionUnetImagePolicyProb.__init__).parameters
        assert pol_mu.kwargs == {}
        print('  ctor gates fire; aux_input is a named param; kwargs empty')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _batch_from(ds, idxs):
    samples = [ds[i] for i in idxs]
    return {
        'obs': {k: torch.stack([s['obs'][k] for s in samples])
                for k in samples[0]['obs']},
        'action': torch.stack([s['action'] for s in samples]),
        AUX_ACTION_KEY: torch.stack([s[AUX_ACTION_KEY] for s in samples]),
    }


def test_policy_zv_bit_identity_vs_parent():
    tmp = tempfile.mkdtemp(prefix='prob_policy_pin_')
    try:
        ds, shape_meta, _tmd = tah._build_m4_dataset(tmp)
        normalizer = ds.get_normalizer()
        batch = _batch_from(ds, [0, 1])
        torch.manual_seed(0)
        ref = tah._make_policy(DiffusionUnetImagePolicyAux, shape_meta,
                               aux_loss_weight=1.0, aux_n_steps=tah.AUX_N_STEPS)
        torch.manual_seed(0)
        prob = tah._make_policy(DiffusionUnetImagePolicyProb, shape_meta,
                                aux_loss_weight=1.0,
                                aux_n_steps=tah.AUX_N_STEPS, aux_input='z_v')
        for pol in (ref, prob):
            pol.set_normalizer(normalizer)
            pol.eval()
        for (n1, p1), (n2, p2) in zip(ref.named_parameters(),
                                      prob.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1.detach(), p2.detach()), n1
        ref.zero_grad(set_to_none=True)
        prob.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        l_ref = ref.compute_loss(batch)
        torch.manual_seed(0)
        l_prob = prob.compute_loss(batch)
        assert torch.equal(l_ref, l_prob), (float(l_ref), float(l_prob))
        torch.manual_seed(0)
        ref.compute_loss(batch).backward()
        torch.manual_seed(0)
        prob.compute_loss(batch).backward()
        n = _grads_equal(ref, prob)
        assert n > 100, n
        print(f"  aux_input='z_v' + parent encoder == DiffusionUnetImagePolicyAux "
              f"bit-for-bit (loss + all {n} gradients)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. the configs
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
    """Dotted keys whose unresolved values differ (or that exist on one side
    only). resolve=False on purpose: interpolations must compare as their
    expressions, so `${name}` does not become a timestamp."""
    fa = _flatten(OmegaConf.to_container(a, resolve=False))
    fb = _flatten(OmegaConf.to_container(b, resolve=False))
    return {k for k in (set(fa) | set(fb)) if fa.get(k) != fb.get(k)}


def test_prob_configs():
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_prob_latent')
        base = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_m4base_latent')
        poe = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_prob_latent',
            overrides=['policy.obs_encoder.fusion_mode=poe'])
        cfg_small = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_prob_latent',
            overrides=['policy.down_dims=[32,64]',
                       'policy.diffusion_step_embed_dim=16'])
        m3 = hydra.compose(config_name='train_diffusion_unet_image_workspace_m3')
        m4 = hydra.compose(config_name='train_diffusion_unet_image_workspace_m4')

    enc = cfg.policy.obs_encoder
    assert enc.fusion_mode == 'mha'                  # the committed cell is B
    assert poe.policy.obs_encoder.fusion_mode == 'poe'
    assert cfg.policy.aux_input == 'mu_v'
    assert float(enc.logvar_min) == -6.0 and float(enc.logvar_max) == 2.0
    assert int(enc.gauss_hidden) == 256
    assert int(enc.head_init_seed) == 42
    assert cfg.latent_probe.save_gaussian is True
    assert 'ProbViewConditionedObsEncoder' in enc._target_
    assert 'DiffusionUnetImagePolicyProb' in cfg.policy._target_
    assert 'ProbProbe' in cfg._target_
    assert cfg.policy.aux_loss_weight == 1.0
    assert cfg.policy.aux_n_steps == 8 == cfg.task.dataset.aux_n_steps

    # the task clone changed nothing but its name
    t = OmegaConf.to_container(cfg.task, resolve=True)
    tb = OmegaConf.to_container(base.task, resolve=True)
    assert t.pop('name') == 'prob_base_image_abs_multiview_az75'
    assert tb.pop('name') == 'm4_base_image_abs_multiview_az75'
    assert t == tb, 'the prob task clone differs from the m4base task'
    print('  task clone == m4base task except name')

    # the workspace diff is exactly the allowlist
    got = _diff_keys(cfg, base)
    expected = {
        'name', 'task.name', '_target_', 'latent_probe.save_gaussian',
        'policy._target_', 'policy.aux_input', 'policy.obs_encoder._target_',
        'policy.obs_encoder.fusion_mode', 'policy.obs_encoder.logvar_min',
        'policy.obs_encoder.logvar_max', 'policy.obs_encoder.gauss_hidden',
        'policy.obs_encoder.head_init_seed'}
    assert got == expected, (sorted(got - expected), sorted(expected - got))
    print(f'  workspace diff vs m4base_latent is exactly the {len(expected)} '
          f'registered keys')

    # the full policy instantiates, and the UNet cond width is unchanged
    pol = hydra.utils.instantiate(cfg_small.policy)
    assert pol.obs_feature_dim == 521
    assert pol.kwargs == {}
    assert pol.aux_input == 'mu_v'
    assert tuple(pol.obs_encoder.output_shape()) == (521,)
    print('  policy instantiates from the composed config (obs_feature_dim 521, '
          'kwargs empty)')

    # the existing m3/m4 encoder pin still holds: the new files touched nothing
    assert (OmegaConf.to_container(m4.policy.obs_encoder)
            == OmegaConf.to_container(m3.policy.obs_encoder))
    print('  m4 vs m3 encoder configs still identical (no upstream drift)')


# ---------------------------------------------------------------------------
# 6. the probe extension
# ---------------------------------------------------------------------------

class _FakePolicy:
    def __init__(self, encoder, normalizer):
        self.obs_encoder = encoder
        self.normalizer = normalizer

    def parameters(self):
        return self.obs_encoder.parameters()


def _probe_dataset_cfg(zarr_path):
    shape_meta = tah._slot_shape_meta(3, 8)
    return OmegaConf.create(dict(
        _target_='diffusion_policy.dataset.multiview_image_dataset.'
                 'MultiViewImageDataset',
        shape_meta=shape_meta, dataset_path=str(zarr_path),
        horizon=2, pad_before=0, pad_after=1, n_obs_steps=1, abs_action=True,
        view_pool=[0, 1, 2], view_count_range=[1, 3], eef_hist_steps=2,
        emit_aux_action=True, aux_n_steps=2, seed=0, val_ratio=0.0))


def _probe_shim(cls, ds_cfg, out_dir, **extra):
    ws = cls.__new__(cls)                 # bypass workspace init (no hydra/wandb)
    ws.cfg = OmegaConf.create(
        {'task': {'dataset': OmegaConf.to_container(ds_cfg)}})
    ws._latent_probe_cfg = OmegaConf.create(dict(
        n_states=4, seed=777, az0_view=AZ0_VIEW, save_raw=True,
        out_dir='latent_snapshots', save_gaussian=extra.pop('save_gaussian', False)))
    ws._output_dir = out_dir
    ws.epoch = 0
    ws._probe_cache = None
    ws._latent_probe_hook = ws._run_latent_probe
    for k, v in extra.items():
        setattr(ws, k, v)
    return ws


def test_probe_superset(tmp):
    from tests.test_multiview_dataset import (build_synthetic_zarr,
                                              N_VIEWS)
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
    register_codecs()
    zarr_path = os.path.join(tmp, 'synthetic.zarr')
    build_synthetic_zarr(zarr_path)
    # distinct camera poses per view (the fixture duplicates two), so the
    # slot->view map the cache asserts is well-defined
    root = zarr.open(zarr_path, mode='a')
    root['meta/view_cam_pos'][:] = np.array(
        [[1.0, 0.0, 1.0], [0.5, 0.0, 1.35], [1.2, 0.1, 0.9]])
    ds_cfg = _probe_dataset_cfg(zarr_path)

    dataset = hydra.utils.instantiate(ds_cfg)
    normalizer = dataset.get_normalizer()
    encoder = _make_prob_encoder(
        OmegaConf.to_container(ds_cfg.shape_meta, resolve=True), 'poe',
        eef_hist_steps=2)
    encoder.eval()
    pol = _FakePolicy(encoder, normalizer)

    out_p = os.path.join(tmp, 'run_parent')
    out_c = os.path.join(tmp, 'run_child')
    os.makedirs(out_p, exist_ok=True)
    os.makedirs(out_c, exist_ok=True)
    ws_p = _probe_shim(TrainDiffusionUnetImageWorkspaceLatentProbe, ds_cfg, out_p)
    ws_c = _probe_shim(TrainDiffusionUnetImageWorkspaceProbProbe, ds_cfg, out_c,
                       save_gaussian=True, _save_gaussian=True,
                       _logvar_min=-6.0, _logvar_max=2.0)

    ws_p.epoch = 0
    stats_p = ws_p._latent_probe_hook(pol)
    ws_c.epoch = 0
    stats_c = ws_c._latent_probe_hook(pol)

    sd_p = os.path.join(out_p, 'latent_snapshots')
    sd_c = os.path.join(out_c, 'latent_snapshots')
    a = np.load(os.path.join(sd_p, 'epoch_0000.npz'))
    b = np.load(os.path.join(sd_c, 'epoch_0000.npz'))
    for key in ('zv', 'zg', 'zg_n1', 'state_idx'):
        assert np.array_equal(a[key], b[key]), key
    for key in ('mu', 'lv', 'lv_n1', 'lvg'):
        assert key in b.files, (key, b.files)
    assert b['mu'].shape == (4, N_VIEWS, 512) and b['mu'].dtype == np.float16
    assert b['lvg'].shape == (4, 512)
    print('  prob snapshot is a superset: zv/zg/zg_n1/state_idx bit-identical, '
          'mu/lv/lv_n1/lvg present')

    assert set(stats_c) >= set(stats_p)
    assert stats_c['latent_probe/mu_resid_rel'] == 0.0        # mu == z at init
    assert stats_c['latent_probe/lv_mean'] == 0.0
    assert stats_c['latent_probe/lv_std'] == 0.0
    assert stats_c['latent_probe/lv_view_spread'] == 0.0
    assert stats_c['latent_probe/lv_at_min_frac'] == 0.0
    assert stats_c['latent_probe/lv_at_max_frac'] == 0.0
    # the probe's full draw is all-live: uniform precision -> lvg = -log(K)
    assert abs(stats_c['latent_probe/lvg_mean']
               - (-np.log(N_VIEWS))) <= 1e-5
    assert stats_c['latent_probe/lvg_n1_vs_lv'] <= 1e-5       # N=1 degeneracy
    print('  init scalars exact: mu_resid_rel/lv_* all 0, N=1 PoE degeneracy held')

    meta_c = json.load(open(os.path.join(sd_c, 'meta.json')))
    for k in ('mu', 'lv', 'lv_n1', 'lvg'):
        assert k in meta_c['arrays'], meta_c['arrays']
    meta_p = json.load(open(os.path.join(sd_p, 'meta.json')))
    assert set(meta_p['arrays']) == {'zv', 'zg', 'zg_n1', 'state_idx'}

    # a check that can fail: perturb the mu branch; the extra must move
    with torch.no_grad():
        encoder.gauss_head.mu_net[-1].bias.add_(0.5)
    ws_c.epoch = 1
    stats_c1 = ws_c._latent_probe_hook(pol)
    assert stats_c1['latent_probe/mu_resid_rel'] > 0.0
    b1 = np.load(os.path.join(sd_c, 'epoch_0001.npz'))
    assert not np.array_equal(b['mu'], b1['mu'])
    print('  mutation check: perturbing mu_net moves mu_resid_rel and the array')


# ---------------------------------------------------------------------------

def test():
    tmp = tempfile.mkdtemp(prefix='prob_fusion_test_')
    try:
        test_fusion_vs_numpy_reference()
        test_fusion_n1_and_init_identity()
        test_fusion_bounds_and_permutation()
        test_gaussian_head()
        test_encoder_contract()
        test_encoder_init_lockstep()
        test_encoder_branch_discrimination()
        test_encoder_gradients()
        test_policy_aux_input_switch()
        test_policy_zv_bit_identity_vs_parent()
        test_prob_configs()
        test_probe_superset(tmp)
        print('ALL PROB GAUSSIAN-FUSION CHECKS PASSED')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test()
