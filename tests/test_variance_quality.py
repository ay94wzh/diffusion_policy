"""CPU-only test for analyze_variance_quality.py (no checkpoint, no zarr, no GPU).

Run from the repo root:
    python tests/test_variance_quality.py

What this pins, and why each check has a mutation next to it
------------------------------------------------------------
`analyze_variance_quality.py` decides whether the round's predicted variance
means anything, and it does so through statistics that fail *quietly* when they
are wrong: a rank correlation with a sign error, a shuffle control that can
never fail, a subset re-fusion that silently counts a dead slot as a certain
expert. Each convention below is therefore paired with an input on which a
broken version would give a different answer, and the degenerate cases that make
a broken check wrongly *pass* are constructed explicitly -- the same discipline
`tests/test_relpose_probe.py` documents for the probe.

What is NOT tested here: whether the real checkpoint's variance carries this
signal. That is the measurement the script exists to make, and it needs the box.
"""
import os
import sys
import json
import shutil
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from types import SimpleNamespace

import numpy as np
import torch

from analyze_variance_quality import (
    scalar_lv, lv_view_spread, _rankdata, spearman, perm_pvalue, shuffle_control,
    verdict_corr, subset_fuse, n_sensitivity, leave_one_out, overconfidence,
    ridge_residuals, per_view_table, register_gates, log_scalars_at_epoch,
    build_readings, summary_markdown)
from diffusion_policy.model.vision.per_view_gaussian import poe_fuse


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# R1 -- the spread statistic
# ---------------------------------------------------------------------------


def test_scalar_lv_is_mean_over_dims():
    lv = torch.zeros(2, 3, 4)
    lv[..., 0] = 2.0
    _check(torch.allclose(scalar_lv(lv), torch.full((2, 3), 0.5)),
           'scalar_lv must be the mean over latent dims')
    print('  scalar_lv is the dim-mean .................... ok')


def test_lv_view_spread_zero_and_alive():
    # identical views -> the exact-zero reading the registration calls "C is a
    # re-parameterized D"; the mutation is one view disagreeing.
    same = torch.full((2, 3, 4), -1.0)
    act = torch.ones(2, 3, dtype=torch.bool)
    v = lv_view_spread(same, act)
    _check(v == 0.0, f'identical views must read exactly 0, got {v}')
    diff = same.clone()
    diff[:, 1, :] = 1.0
    v2 = lv_view_spread(diff, act)
    _check(v2 > 0.0, 'one disagreeing view must make the spread positive')
    # dead slots must not count: a dead slot's lv is 0 by construction and would
    # inflate the std if the mask were ignored
    act2 = act.clone()
    act2[:, 2] = False
    dead_noise = same.clone()
    dead_noise[:, 2, :] = 5.0
    _check(lv_view_spread(dead_noise, act2) == 0.0,
           'dead slots must be excluded from the spread')
    print('  lv_view_spread: zero / alive / mask ....... OK')


# ---------------------------------------------------------------------------
# ranks and correlations
# ---------------------------------------------------------------------------


def test_rankdata_averages_ties():
    r = _rankdata([10.0, 20.0, 20.0, 30.0])
    _check(np.allclose(r, [1.0, 2.5, 2.5, 4.0]), f'tie ranks wrong: {r}')
    r2 = _rankdata([5.0, 5.0, 5.0])
    _check(np.allclose(r2, [2.0, 2.0, 2.0]), f'all-tied ranks wrong: {r2}')
    print('  _rankdata tie handling ....................... ok')


def test_spearman_signs_and_degeneracy():
    x = np.arange(50.0)
    _check(abs(spearman(x, 3.0 * x + 1.0) - 1.0) < 1e-12, 'monotone must read +1')
    _check(abs(spearman(x, -x) + 1.0) < 1e-12, 'anti-monotone must read -1')
    _check(np.isnan(spearman(x, np.zeros_like(x))), 'constant input must be NaN')
    # rank-based, so a monotone nonlinearity must NOT change the value
    _check(abs(spearman(x, np.exp(x / 20.0)) - 1.0) < 1e-12,
           'spearman must be invariant to monotone transforms')
    print('  spearman sign / nonlinearity / NaN ........... ok')


def test_perm_pvalue_is_one_sided():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    pos = x + 0.6 * rng.normal(size=200)          # planted positive
    rho_p, p_pos = perm_pvalue(x, pos, n_perm=400, seed=1)
    _check(rho_p > 0.3 and p_pos < 0.01, f'planted positive: rho={rho_p} p={p_pos}')
    rho_n, p_neg = perm_pvalue(x, -pos, n_perm=400, seed=1)
    _check(rho_n < 0 and p_neg > 0.5,
           f'a NEGATIVE rho must not read as significant: rho={rho_n} p={p_neg}')
    print('  perm_pvalue one-sidedness (the sign trap) .... ok')


def test_shuffle_control_can_fail():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    y = 0.8 * x + 0.5 * rng.normal(size=300)
    floor = shuffle_control(x, y, n_shuffles=200, seed=0)
    rho = spearman(x, y)
    _check(rho > floor, f'planted signal {rho} must beat the floor {floor}')
    # mutation: break the pairing and the same data must fall under the floor
    broke = rng.permutation(y)
    _check(spearman(x, broke) <= floor,
           'after breaking the pairing the statistic must fall under its floor')
    print('  shuffle control: beats / is beaten ........... ok')


def test_verdict_corr_branches():
    _check(verdict_corr(0.9, 0.001, rho_min=0.5).startswith('supported'),
           'strong + significant must read supported')
    _check(verdict_corr(0.7, 0.30, rho_min=0.5).startswith('weak'),
           'a big rho with a bad p must read weak')
    _check(verdict_corr(0.9, 0.001, rho_min=0.2, floor=0.95).startswith('weak'),
           'a rho under the shuffle floor must read weak')
    _check('finding-against' in verdict_corr(-0.6, 0.999, rho_min=0.5),
           'a negative rho is a finding, not a null')
    _check('undefined' in verdict_corr(float('nan'), float('nan')),
           'a constant input must be undefined')
    print('  verdict_corr branches (all five) ............. ok')


# ---------------------------------------------------------------------------
# R5 -- the offline fusion
# ---------------------------------------------------------------------------


def test_subset_fuse_is_the_production_fusion():
    torch.manual_seed(0)
    mu = torch.randn(5, 11, 8)
    lv = torch.rand(5, 11, 8) - 3.0
    mask = torch.zeros(5, 11, dtype=torch.bool)
    mask[:, [0, 3, 7]] = True
    got = subset_fuse(mu, lv, mask)
    ref = poe_fuse(mu, lv, mask)
    _check(torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]),
           'subset_fuse must BE poe_fuse, not a re-derivation')
    # the phantom-expert trap: dead slots carry lv=0 (precision 1) -- garbage on
    # a dead slot must not move the fusion
    dirty = mu.clone()
    dirty[:, 1, :] = 1e3
    _check(torch.equal(subset_fuse(dirty, lv, mask)[0], got[0]),
           'a dead slot must not enter the fusion (phantom expert)')
    # N=1: bitwise the live mean
    one = torch.zeros(5, 11, dtype=torch.bool)
    one[:, 3] = True
    _check(torch.equal(subset_fuse(mu, lv, one)[0], mu[:, 3]),
           'at N=1 the fused mean must be bitwise the live mean')
    print('  subset_fuse == production / mask / N=1 ....... ok')


def test_subset_fuse_rejects_bad_masks():
    mu = torch.randn(2, 3, 4)
    lv = torch.zeros(2, 3, 4)
    good = torch.ones(2, 3, dtype=torch.bool)
    for bad, why in ((good[:, :0], 'no live views'),
                     (torch.zeros(2, 3, dtype=torch.bool), 'all dead'),
                     (good.float(), 'not bool')):
        try:
            subset_fuse(mu, lv, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{why} must raise')
    print('  subset_fuse rejects empty/non-bool masks ...... ok')


def test_n_sensitivity_known_limits():
    # identical means and identical logvars across views: lvg(N) = lv - log N
    # (strictly decreasing) and the drift is exactly 0 -- the closed form the
    # implementation must reproduce.
    mu = torch.full((4, 11, 6), 2.0)
    lv = torch.full((4, 11, 6), -1.0)
    curve = n_sensitivity(mu, lv, n_draws=3, seed=0)
    for n in (1, 5, 11):
        want = -1.0 - float(np.log(n))
        _check(abs(curve[str(n)]['lvg_mean'] - want) < 1e-5,
               f'lvg({n}) should be {want}, got {curve[str(n)]["lvg_mean"]}')
        _check(curve[str(n)]['drift_mean'] < 1e-6, f'drift({n}) should be 0')
    _check(curve['1']['lvg_mean'] > curve['11']['lvg_mean'], 'lvg must fall with N')
    # mutation: with views that disagree, drift(1) must exceed drift(11)
    torch.manual_seed(1)
    mu2 = torch.randn(64, 11, 6)
    lv2 = torch.zeros(64, 11, 6)
    c2 = n_sensitivity(mu2, lv2, n_draws=20, seed=0)
    _check(c2['1']['drift_mean'] > c2['11']['drift_mean'],
           'drift must decay as more views join (was 0 in the degenerate case)')
    print('  n_sensitivity closes to lv-logN / decays ..... ok')


def test_leave_one_out_finds_the_outlier():
    # 10 consensus views + one outlier: dropping the outlier must move the fused
    # mean most, and the reported row must carry its slot
    mu = torch.zeros(16, 11, 5)
    mu[:, 6, :] = 50.0
    lv = torch.zeros(16, 11, 5)
    rows = leave_one_out(mu, lv)
    top = max(rows, key=lambda r: r['drop_drift'])
    _check(top['slot'] == 6, f'expected slot 6 on top, got {top}')
    others = [r['drop_drift'] for r in rows if r['slot'] != 6]
    _check(top['drop_drift'] > max(others) + 1e-6,
           'the outlier removal must dominate every other removal')
    print('  leave_one_out ranks the outlier .............. ok')


def test_overconfidence_extremes():
    # (a) means spread wide, logvar very negative -> implied sigma far below the
    # between-view spread -> the frac reads 1. The spread must be in EVERY dim:
    # a dim with identical means has zero disagreement and would (correctly)
    # read as not-under-confident, which is what the first version of this test
    # did by accident.
    mu = torch.zeros(8, 11, 4)
    mu[:, :, :] = torch.linspace(-1, 1, 11).unsqueeze(-1)
    lv = torch.full((8, 11, 4), -6.0)
    act = torch.ones(8, 11, dtype=torch.bool)
    o = overconfidence(mu, lv, act)
    _check(o['frac_sigma_below_between'] == 1.0, f'(a) {o}')
    # (b) identical means -> zero between-view spread; sigma/between blows up
    mu_same = torch.zeros(8, 11, 4)
    o2 = overconfidence(mu_same, torch.zeros(8, 11, 4), act)
    _check(o2['frac_sigma_below_between'] == 0.0, f'(b) {o2}')
    print('  overconfidence extremes (0 and 1) ............ ok')


# ---------------------------------------------------------------------------
# the ridges and the table
# ---------------------------------------------------------------------------


def test_ridge_residuals_split_and_recovery():
    torch.manual_seed(3)
    n, d = 400, 12
    X = torch.randn(n, d)
    w = torch.randn(d, 3)
    Y = X @ w
    res, n_tr = ridge_residuals(X, Y)
    _check(n_tr == 320 and res.shape[0] == n - n_tr,
           f'split wrong: n_tr={n_tr} res={tuple(res.shape)}')
    _check(float(res.max()) < 1e-2, f'exact linear target must decode: {res.max()}')
    # mutation: an unrelated target must NOT decode -- residuals at its own scale
    Y2 = torch.randn(n, 3)
    res2, _ = ridge_residuals(X, Y2)
    _check(float(res2.mean()) > 0.5, 'an unrelated target must not decode')
    print('  ridge_residuals: split / recovery / null ..... ok')


def test_per_view_table_groups_by_ring():
    m, k = 2, 3
    lv_s = torch.tensor([[-1.0, -2.0, 0.0], [-1.0, -2.0, 0.0]])
    act = torch.ones(m, k, dtype=torch.bool)
    act[1, 1] = False                      # slot 1 dead on frame 1
    err_aux = torch.full((m, k), 0.5)
    err_state = torch.full((m, k), 2.0)
    ring_of_slot = {0: 5, 1: 9, 2: 11}
    az = {5: -15.0, 9: 45.0, 11: 75.0}
    table = per_view_table(ring_of_slot, act, lv_s, err_aux, err_state, az)
    by_ring = {r['ring']: r for r in table}
    _check([r['ring'] for r in table] == [5, 9, 11], 'rings must be sorted')
    _check(by_ring[5]['n_pairs'] == 2 and by_ring[9]['n_pairs'] == 1,
           f"pair counts wrong: {[r['n_pairs'] for r in table]}")
    _check(abs(by_ring[9]['mean_lv'] + 2.0) < 1e-9,
           'the mean must exclude the dead frame')
    _check(by_ring[11]['azimuth_deg'] == 75.0, 'azimuth lookup wrong')
    print('  per_view_table groups by ring, skips dead ... ok')


# ---------------------------------------------------------------------------
# gates and the log cross-check
# ---------------------------------------------------------------------------


def test_register_gates_refuses_non_poe_encoders():
    def stub(emits_mu, mode):
        enc = SimpleNamespace(emits_mu=emits_mu, fusion_mode=mode)
        return SimpleNamespace(obs_encoder=enc)
    try:
        register_gates(stub(False, 'poe'))
    except SystemExit as e:
        _check('emits no per-view Gaussians' in str(e), str(e))
    else:
        raise AssertionError('a non-Gaussian encoder must be refused')
    try:
        register_gates(stub(True, 'mha'))
    except SystemExit as e:
        _check('--assert-dead-logvar' in str(e), str(e))
    else:
        raise AssertionError('a non-PoE cell must be refused for the full read')
    register_gates(stub(True, 'poe'))       # must not raise
    print('  register_gates refuses / accepts ............. ok')


def test_log_scalars_at_epoch():
    tmp = tempfile.mkdtemp(prefix='vq_test_')
    try:
        path = os.path.join(tmp, 'logs.json.txt')
        rows = [
            dict(epoch=0, **{'latent_probe/zv_pair_ratio': 0.5,
                             'latent_probe/lv_view_spread': 0.0}),
            dict(epoch=200, **{'latent_probe/zv_pair_ratio': 0.256,
                               'latent_probe/lv_view_spread': 0.447}),
            dict(epoch=200, train_loss=0.01),          # a step row, no probe keys
        ]
        with open(path, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        got = log_scalars_at_epoch(path, epoch=200)
        _check(got['lv_view_spread'] == 0.447 and 'train_loss' not in got,
               f'extraction wrong: {got}')
        _check(log_scalars_at_epoch(os.path.join(tmp, 'nope.txt')) == {},
               'a missing log must read as absent, not crash')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('  log_scalars_at_epoch extracts e200 ........... ok')


def test_build_readings_glue_smoke():
    """The glue, not the math: every index that can be transposed is exercised.

    `build_readings` juggles five tensors through three different indexings
    (pairs, states, frames x dims), and on the box it only ever runs behind a
    GPU + checkpoint + zarr. This synthetic object runs the SAME function, so a
    transposed pair index or an off-by-one in the fused-logvar reshape fails
    here instead of after the server run.
    """
    torch.manual_seed(0)
    S, K, D, To = 24, 11, 8, 2                  # 12 states x To frames
    lv = torch.randn(S, K, D) * 0.5 - 1.0
    mu = torch.randn(S, K, D)
    active = torch.ones(S, K, dtype=torch.bool)
    err_aux = (lv.mean(-1).exp() * 0.01).clamp_min(1e-6)   # error tied to lv
    lvg = -torch.logsumexp(-lv, dim=1)                      # the PoE closed form
    c = dict(mu=mu, lv=lv, active=active, err_aux=err_aux,
             eef=torch.randn(S, 3), zg=mu.mean(1), lvg=lvg,
             diff_loss=torch.rand(S // To) * 0.01,
             azimuths={i: (i - 5) * 15.0 for i in range(K)},
             stats=dict(ring_of_slot={i: i for i in range(K)},
                        n_states=S // To, K=K, dim=D),
             config=dict(n_perm=200, n_shuffles=50, n_draws_per_level=5, seed=0))
    readings = build_readings(c, {'lv_view_spread': 0.447})
    for key in ('r1_lv_view_spread', 'r2_view_quality', 'r3_pair_error',
                'r4_fused_vs_error', 'r5_n_sensitivity', 'r6_overconfidence'):
        _check(key in readings, f'{key} missing from the readings block')
    _check(len(readings['r2_view_quality']['per_view']) == K,
           'the per-view table must carry every ring view')
    _check(readings['r2_view_quality']['per_view'][5]['azimuth_deg'] == 0.0,
           'ring 5 must be the azimuth-0 view')
    _check(readings['r3_pair_error']['n_pairs'] == S * K, 'pair count wrong')
    order = sorted(readings['r5_n_sensitivity']['lvg_by_n'], key=int)
    _check(order[0] == '1' and order[-1] == str(K), f'R5 levels wrong: {order}')
    loo = readings['r5_n_sensitivity']['leave_one_out']
    _check(len(loo) == K and loo[0]['ring'] == 0,
           'the leave-one-out rows must map slots back to rings')
    # err_aux was built FROM lv, so the aux reading must read positive -- the
    # state reading decodes an independent random target and must NOT be
    # asserted positive (that would be a test that cannot fail honestly)
    _check(readings['r3_pair_error']['rho_lv_vs_err_aux']['rho'] > 0,
           'the aux reading must be positive on a planted pairing')
    _check(np.isfinite(readings['r3_pair_error']['rho_lv_vs_err_state']['rho']),
           'the state reading must be a number, not NaN')
    _check(readings['r4_fused_vs_error']['rho_lvg_vs_state_residual']['rho']
           is not None, 'R4a must be present when lvg is')
    _check(readings['r3_pair_error']['n_test_pairs'] < readings['r3_pair_error']['n_pairs'],
           'the state reading must run on the held-out split only')
    md = summary_markdown(dict(
        checkpoint='/tmp/fake.ckpt', settings={'fusion_mode': 'poe', 'aux_input': 'mu_v'},
        config={'n_states': S // To, 'seed': 0}, readings=readings))
    _check('## R6 overconfidence' in md and 'az_0 removal drift' in md,
           'the markdown digest must render every section')
    print('  build_readings + summary_markdown glue ...... ok')


def test():
    print('=' * 70)
    print('variance quality -- CPU checks')
    print('=' * 70)
    test_scalar_lv_is_mean_over_dims()
    test_lv_view_spread_zero_and_alive()
    test_rankdata_averages_ties()
    test_spearman_signs_and_degeneracy()
    test_perm_pvalue_is_one_sided()
    test_shuffle_control_can_fail()
    test_verdict_corr_branches()
    test_subset_fuse_is_the_production_fusion()
    test_subset_fuse_rejects_bad_masks()
    test_n_sensitivity_known_limits()
    test_leave_one_out_finds_the_outlier()
    test_overconfidence_extremes()
    test_ridge_residuals_split_and_recovery()
    test_per_view_table_groups_by_ring()
    test_register_gates_refuses_non_poe_encoders()
    test_log_scalars_at_epoch()
    test_build_readings_glue_smoke()
    print('=' * 70)
    print('ALL VARIANCE QUALITY CHECKS PASSED')
    print('=' * 70)


if __name__ == '__main__':
    test()
