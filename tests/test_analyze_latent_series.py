"""CPU-only tests for analyze_latent_series.py (numpy only: no torch, no zarr,
no simulator). Runs on the laptop AND on the training box.

Builds a synthetic per-epoch snapshot series with KNOWN plants and checks that
the analyzer recovers them -- and, per this project's house rule, that every
gate can FAIL when it should. The mutation checks are the point:

  * an orthogonal rotation of a whole epoch must leave CKA ~= 1 while moving
    the strict measures (the two readings are different, on purpose);
  * rolling the view axis must move the per-view argmin/argmax;
  * a meta-consistent but wrong az0_ring must drop cka(zg_n1, zv[:,az0]);
  * a shuffled state_idx / a wrong K / a NaN / too few epochs must each stop
    the analysis loudly, and --allow-nonfinite must downgrade a NaN to a skip;
  * a 10%-perturbed logged scalar must FAIL the recheck, and an absent log
    must report not_run -- never a vacuous PASS.

The real npz of the clear run live on the training box; this fixture mirrors
their layout (fp16 arrays, meta.json, snapshots.jsonl) exactly at a 16x4x8
scale, and the driver runs THIS file as its CPU gate before the real analysis.

Run from the repo root:
    python tests/test_analyze_latent_series.py
    python tests/test_analyze_latent_series.py --write-fixture /tmp/ls_fix
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(this_dir)
os.chdir(repo_root)
sys.path.insert(0, repo_root)

import analyze_latent_series as als  # noqa: E402

S_DEF, K_DEF, D_DEF = 16, 4, 8
FREEZE_FROM = 3
AZ0_POS = 2          # pool_ring [1, 2, 3, 4] with az0_ring 3
AZ0_RING = 3
BEHAVIOUR = [(1, 0.30), (3, 0.70), (6, 0.70), (9, 0.71)]


# ---------------------------------------------------------------------------
# an independent re-implementation of the hook's scalars (the recheck's other
# side): written from the hook's formulas with einsum, so the recheck gate
# compares two implementations, not one implementation with itself.
# ---------------------------------------------------------------------------
def _ref_rel_dist(a, b):
    diff = np.sqrt(np.einsum('ij,ij->i', a - b, a - b))
    na = np.sqrt(np.einsum('ij,ij->i', a, a))
    nb = np.sqrt(np.einsum('ij,ij->i', b, b))
    return float(np.mean(diff / np.maximum(0.5 * (na + nb), 1e-6)))


def _ref_spread(x):
    sd = x.std(axis=0, ddof=1).max()
    return float(sd / max(np.linalg.norm(x, axis=1).mean(), 1e-12))


def _ref_pr(x):
    lam = np.linalg.svd(x - x.mean(axis=0, keepdims=True), compute_uv=False) ** 2
    return float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-30))


def _ref_scalars(zv, zg, zn):
    S, K = zv.shape[0], zv.shape[1]
    iu = np.triu_indices(S, 1)
    per_view = [_ref_spread(zv[:, j, :]) for j in range(K)]
    pair = [_ref_rel_dist(zv[:, i, :], zv[:, j, :])
            for i in range(K) for j in range(i + 1, K)]
    cross = [_ref_rel_dist(zv[:, v, :][iu[0]], zv[:, v, :][iu[1]])
             for v in range(K)]
    return {
        'zv_norm_mean': float(np.linalg.norm(zv, axis=-1).mean()),
        'zv_rel_spread_mean': float(np.mean(per_view)),
        'zv_rel_spread_min': float(np.min(per_view)),
        'zg_norm_mean': float(np.linalg.norm(zg, axis=-1).mean()),
        'zg_rel_spread': _ref_spread(zg),
        'zg_n1_norm_mean': float(np.linalg.norm(zn, axis=-1).mean()),
        'zg_n1_rel_spread': _ref_spread(zn),
        'zv_pair_ratio': float(np.mean(pair) / max(np.mean(cross), 1e-12)),
        'zv_pr': _ref_pr(zv.reshape(-1, zv.shape[-1])),
        'zg_pr': _ref_pr(zg),
    }


# ---------------------------------------------------------------------------
# the fixture
# ---------------------------------------------------------------------------
def build_arrays(n_epochs=10, S=S_DEF, K=K_DEF, D=D_DEF, freeze_from=FREEZE_FROM,
                 seed=0):
    """The planted series.

    View constructions are chosen so the per-view checks have known answers:
      view 1 = full-rank random  -> argmax participation ratio;
      view 2 = near-constant row direction -> argmin relative spread (and the
               az_0 view: zg_n1 = zv[:, 2] @ Q with Q orthogonal, making
               cka(zg_n1, zv_az0) exactly 1 by construction).
    Callers must use different seeds before drawing anything else.
    """
    rng = np.random.RandomState(seed)

    def view(j):
        if j == 1:
            return rng.normal(size=(S, D))
        if j == 2:
            # near-constant row direction: the argmin-spread plant. Not TOO
            # extreme -- a spread near the fp16 storage floor would make the
            # recheck fail on storage noise, which is the fixture's artifact,
            # not the analyzer's (the real min spread is ~0.2, not ~0.007).
            u = 1.0 + 0.2 * rng.normal(size=(S, 1))
            v = rng.normal(size=(1, D))
            return u * v
        r = {0: 2, 3: 3}.get(j, 2)
        return rng.normal(size=(S, r)) @ rng.normal(size=(r, D))

    B = np.stack([view(j) for j in range(K)], axis=1)          # (S, K, D)
    delta = rng.normal(size=(S, K, D)) * 0.7
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))               # orthogonal
    Rg = rng.normal(size=(D, D))
    state_idx = np.sort(rng.choice(100000, S, replace=False))
    Zs, Gs, Ns = [], [], []
    for e in range(n_epochs):
        w = max(0.0, float(freeze_from - e)) / float(freeze_from)
        Z = B + w * delta
        Zs.append(Z)
        Gs.append(Z.mean(axis=1) @ Rg)
        Ns.append(Z[:, AZ0_POS, :] @ Q)
    return Zs, Gs, Ns, state_idx


def write_fixture(root, n_epochs=10, S=S_DEF, K=K_DEF, D=D_DEF,
                  freeze_from=FREEZE_FROM, seed=0, write_log=True):
    """Writes meta.json, epoch_XXXX.npz (fp16), snapshots.jsonl and a
    logs.json.txt with the fp32 scalars, exactly as the hook's layout."""
    os.makedirs(root, exist_ok=True)
    Zs, Gs, Ns, state_idx = build_arrays(n_epochs, S, K, D, freeze_from, seed)
    meta = {'tool': 'test_fixture', 'n_states': S, 'K': K, 'fused_dim': D,
            'seed': 12345, 'pool_ring': [1, 2, 3, 4], 'az0_ring': AZ0_RING,
            'az0_pos': AZ0_POS, 'conditions': ['full', 'n1_az0']}
    with open(os.path.join(root, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    scalars = {}
    for e in range(n_epochs):
        zv, zg, zn = Zs[e], Gs[e], Ns[e]
        np.savez(os.path.join(root, 'epoch_%04d.npz' % e),
                 zv=zv.astype(np.float16), zg=zg.astype(np.float16),
                 zg_n1=zn.astype(np.float16), state_idx=state_idx)
        scalars[e] = _ref_scalars(zv, zg, zn)

    with open(os.path.join(root, 'snapshots.jsonl'), 'w') as f:
        for e in range(n_epochs):
            f.write(json.dumps(dict(epoch=e, **scalars[e])) + '\n')

    if write_log:
        with open(os.path.join(root, 'logs.json.txt'), 'w') as f:
            f.write(json.dumps({'train_loss': 1.0, 'global_step': 0, 'epoch': 0}) + '\n')
            f.write('{not json at all\n')                       # malformed, counted
            for e in range(n_epochs):
                d = {'train_loss': 0.1, 'global_step': e, 'epoch': e}
                for k, v in scalars[e].items():
                    d['latent_probe/' + k] = v
                for be, bs in BEHAVIOUR:
                    if e == be:
                        d['test/mean_score'] = bs
                        for i in range(3):
                            d['test/sim_max_reward_%d' % (100000 + i)] = float(i < 2)
                f.write(json.dumps(d) + '\n')
    return scalars


def _run(root, extra=(), log='logs.json.txt', out='out'):
    argv = ['-s', root, '-o', os.path.join(root, out)]
    if log:
        argv += ['-l', os.path.join(root, log)]
    return als.run(argv + list(extra))


def _tmpdir():
    return tempfile.mkdtemp(prefix='ls_test_')


# ---------------------------------------------------------------------------
# 1-2. rel_dist: formula, and the cross-pin to the real probe when it imports
# ---------------------------------------------------------------------------
def test_rel_dist():
    rng = np.random.RandomState(1)
    a = rng.normal(size=(9, 5))
    b = rng.normal(size=(9, 5))
    mine = als.rel_dist_np(a, b)
    assert abs(mine - _ref_rel_dist(a, b)) < 1e-12
    assert abs(als.rel_dist_np(a, a)) < 1e-12
    # scale-free: the denominator is what makes it comparable across epochs
    # whose norms change 3x (drop it and this check fails)
    assert abs(als.rel_dist_np(100 * a, 100 * b) - mine) < 1e-9
    print('rel_dist formula OK (einsum pin, identity, scale-free)')

    try:
        import torch
        from probe_relpose import _rel_dist as real
    except Exception as e:                                    # pragma: no cover
        print(f'  SKIPPED (probe_relpose import unavailable: {e}) '
              f'-- cross-pin NOT run; it runs on the training box')
        return
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    assert abs(float(real(ta, tb)) - mine) < 1e-12
    print('rel_dist cross-pin to probe_relpose._rel_dist OK')


# ---------------------------------------------------------------------------
# 3-5. CKA / Procrustes: invariances, and mutations that must break them
# ---------------------------------------------------------------------------
def test_cka():
    rng = np.random.RandomState(2)
    x = rng.normal(size=(32, 10))
    assert abs(als.linear_cka(x, x) - 1.0) < 1e-9
    Q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    assert abs(als.linear_cka(x, x @ Q) - 1.0) < 1e-9        # rotation gauge
    assert abs(als.linear_cka(x, 3.0 * x + 5.0) - 1.0) < 1e-9  # scale+shift
    y = x + rng.normal(size=x.shape) * 0.5
    assert als.linear_cka(x, y) < 0.98

    def cka_uncentered(a, b):
        num = float(np.sum((a.T @ b) ** 2))
        den = float(np.sqrt(np.sum((a.T @ a) ** 2) * np.sum((b.T @ b) ** 2)))
        return num / den if den > 0 else 0.0
    # centering is load-bearing: without it a shift reads as a change
    assert abs(cka_uncentered(x, x + 7.0) - 1.0) > 1e-3
    print('CKA invariances OK (rotation, scale+shift, centering matters)')

    pf = als.permutation_floor(x, x, repeats=8, seed=0)
    # the floor is the MEASURED chance level (~d/(n+d) at this shape), not 0;
    # a broken permutation (or none) reads 1, which this catches
    assert pf['max'] < 0.6, pf
    assert als.linear_cka(x, x) > pf['max'] + 0.4
    hs = als.half_split(x, x)
    assert hs['gap'] >= 0.0
    # ... and the gap MEASURES something: perturb one half's structure and it
    # opens (a bug comparing the same rows twice would keep it at exactly 0)
    x_het = x.copy()
    x_het[1::2] += rng.normal(size=x_het[1::2].shape) * 2.0
    assert als.half_split(x_het, x)['gap'] > 0.05
    assert hs['gap'] == 0.0
    print(f"permutation floor OK (measured max {pf['max']:.3f}, not assumed 0), "
          f"half-split gap {hs['gap']:.3f} (opens to "
          f"{als.half_split(x_het, x)['gap']:.3f} on heterogeneous halves)")

    # the matched null is what rel/cos/proc are read against -- and it is not 0
    nd = als.matched_null(x, seed=1)
    assert 1.2 < nd['rel'] < 1.7, nd          # ~sqrt(2): unrelated directions
    assert nd['cos'] > 0.9, nd
    assert nd['proc'] > 0.4, nd
    print(f"matched null OK (rel {nd['rel']:.2f}, cos {nd['cos']:.2f}, "
          f"proc {nd['proc']:.2f} -- read every value against these)")


def test_procrustes():
    rng = np.random.RandomState(3)
    x = rng.normal(size=(20, 4))
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    # rigid rotation -> 0, down to the formula's ~1e-8 cancellation floor
    assert als.procrustes_resid(x, x @ Q) < 1e-6
    one = rng.normal(size=(30, 1))
    for c in (2.0, 0.5):
        want = 2.0 * abs(1.0 - c) / (1.0 + c)
        assert abs(als.procrustes_resid(one, c * one) - want) < 1e-6
    assert als.procrustes_resid(x, x + rng.normal(size=x.shape) * 0.3) > 0.05
    print('procrustes OK (rotation -> 0, scale analytic, noise > 0)')


# ---------------------------------------------------------------------------
# 6. settle / frozen rules
# ---------------------------------------------------------------------------
def test_settle_rules():
    eps = list(range(10))
    vals = [1.0, 0.7, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    s = als.settle_epoch(vals, eps, tol=0.05)
    assert s['settled'] and s['settle_epoch'] == 4, s
    assert als.frozen_epoch(vals, eps, 0.1) == 4
    # scale invariance of the settle rule (catches an absolute-tolerance bug)
    s2 = als.settle_epoch([v * 1e-3 for v in vals], eps, tol=0.05)
    assert s2['settled'] and s2['settle_epoch'] == 4, s2
    # a strictly monotone ramp never settles -- reported, never clamped
    ramp = [0.1 * e for e in eps]
    assert not als.settle_epoch(ramp, eps, tol=0.05)['settled']
    # tol=0 can fail: a series that keeps jittering at 1e-6 is still moving
    jit = [1.0, 0.5] + [1e-6, 2e-6] * 4
    assert not als.settle_epoch(jit, eps, tol=0.0)['settled']
    # a curve whose limit is 0 is read correctly (the drift-to-final case)
    dec = [1.0 / (1.0 + e) for e in eps]
    s3 = als.settle_epoch(dec, eps, tol=0.05)
    assert s3['settled'] and s3['settle_epoch'] == 6, s3
    print('settle/frozen rules OK (planted, scale-invariant, not_settled is valid)')


# ---------------------------------------------------------------------------
# 7. per-view plants + the view-axis roll mutation
# ---------------------------------------------------------------------------
def test_per_view_and_roll():
    Zs, _, _, _ = build_arrays()
    pv = als.per_view_stats(Zs[-1])
    assert int(np.argmin(pv['rel_spread'])) == 2, pv['rel_spread']
    assert int(np.argmax(pv['pr'])) == 1, pv['pr']
    rolled = als.per_view_stats(np.roll(Zs[-1], 1, axis=1))
    # the roll moves the plants by exactly one slot -> the checks have power
    assert int(np.argmin(rolled['rel_spread'])) == 3, rolled['rel_spread']
    assert int(np.argmax(rolled['pr'])) == 2, rolled['pr']
    print('per-view plants OK (spread argmin 2, PR argmax 1; roll -> 3, 2)')


# ---------------------------------------------------------------------------
# 8. end to end on the fixture
# ---------------------------------------------------------------------------
def test_end_to_end():
    root = _tmpdir()
    try:
        write_fixture(root)
        out, code = _run(root, extra=['--strict', '--refs', '0,5,9'])
        assert code == 0, code
        g = out['gates']
        assert g['state_idx_identical'] and not g['layout']
        assert g['recheck']['status'] == 'pass', g['recheck']
        assert g['chance_levels']['valid']
        # the two settles differ by construction: evidence is constant from
        # epoch 3, its epoch-to-epoch DRIFT is zero from epoch 4
        assert out['settle']['scalars']['zv_norm_mean']['settle_epoch'] <= 4
        d = out['settle']['drift']['zv_flat']
        assert d['rel_prev']['settle_epoch'] == 4, d['rel_prev']
        assert d['rel_prev_frozen'] == 4
        # fused plant: zg_n1 is an orthogonal image of zv[:, az0_pos]
        ckas = out['fused']['n1_vs_zv_az0']['cka']
        assert min(ckas) > 0.99, min(ckas)
        # behaviour plant
        assert out['settle']['behaviour']['settle_epoch'] == 3
        assert out['behaviour']['rollout_epochs'] == [1, 3, 6, 9]
        assert abs(out['gates']['recheck']['per_scalar']['zv_pair_ratio']) < 5e-3
        print(f"end-to-end OK (gates pass; scalar settle 3, drift settle 4; "
              f"recheck max rel {g['recheck']['max_rel_diff']:.1e})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cka_gate_can_fail():
    root = _tmpdir()
    try:
        write_fixture(root)
        out, code = _run(root, extra=['--strict', '--cka-perm-gate', '0.05'])
        assert code == 0                      # voiding CKA is a flag, not a stop
        assert out['gates']['chance_levels']['valid'] is False
        assert any('CKA readings void' in w for w in out['warnings'])
        print('cka gate OK (a floor above the gate voids CKA, loudly)')
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_az0_mapping_mutation():
    root = _tmpdir()
    try:
        write_fixture(root)
        with open(os.path.join(root, 'meta.json')) as f:
            meta = json.load(f)
        meta['az0_ring'] = 1          # meta-CONSISTENT but wrong view
        meta['az0_pos'] = 0
        with open(os.path.join(root, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        out, code = _run(root, extra=['--strict'])
        assert code == 0
        assert min(out['fused']['n1_vs_zv_az0']['cka']) < 0.9
        print('az0 mapping mutation OK (wrong slot drops cka(n1, zv_az0) < 0.9)')
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# 9-11. gates that must stop the run
# ---------------------------------------------------------------------------
def test_state_idx_gate():
    root = _tmpdir()
    try:
        write_fixture(root)
        p = os.path.join(root, 'epoch_0005.npz')
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        rng = np.random.RandomState(9)
        d['state_idx'] = d['state_idx'][rng.permutation(len(d['state_idx']))]
        np.savez(p, **d)
        out, code = _run(root, extra=['--strict'])
        assert code == 3, code
        assert out['gates']['state_idx_identical'] is False
        assert out['gates']['state_idx_first_mismatch'] == 5
        assert out['series'] is None
        assert any('ANALYSIS NOT RUN' in w for w in out['warnings'])
        print('state_idx gate OK (shuffled rows abort, no series reported)')
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_shape_and_finite_gates():
    root = _tmpdir()
    try:
        write_fixture(root)
        p = os.path.join(root, 'epoch_0005.npz')
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        d['zv'] = np.zeros((d['zv'].shape[0], d['zv'].shape[1] + 1,
                            d['zv'].shape[2]), dtype=np.float16)
        np.savez(p, **d)
        out, code = _run(root, extra=['--strict'])
        assert code == 3 and out['series'] is None
        assert out['gates']['layout'], out['gates']['layout']
        print('shape gate OK (wrong K aborts)')
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root = _tmpdir()
    try:
        write_fixture(root)
        p = os.path.join(root, 'epoch_0005.npz')
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        d['zg'] = d['zg'].copy()
        d['zg'][0, 0] = np.nan
        np.savez(p, **d)
        out, code = _run(root, extra=['--strict'])
        assert code == 3 and out['series'] is None
        # ... and --allow-nonfinite downgrades it to a skip, loudly
        out2, code2 = _run(root, extra=['--strict', '--allow-nonfinite'], out='out2')
        assert code2 == 0
        assert out2['gates']['n_skipped'] == 1
        assert 5 not in out2['series']['epochs']
        assert any('skipped' in w for w in out2['warnings'])
        print('finite gate OK (NaN aborts; --allow-nonfinite warns + skips)')
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_min_epochs_and_hole():
    root = _tmpdir()
    try:
        write_fixture(root, n_epochs=2)
        try:
            _run(root)
            raise AssertionError('expected SystemExit for < --min-epochs')
        except SystemExit:
            pass
        print('min-epochs gate OK (2 epochs refused)')
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root = _tmpdir()
    try:
        write_fixture(root)
        os.remove(os.path.join(root, 'epoch_0004.npz'))
        out, code = _run(root, extra=['--strict'])
        assert code == 0
        assert 4 not in out['series']['epochs']
        assert out['gates']['epoch_holes'] == [4]
        assert any('holes' in w for w in out['warnings'])
        print('hole handling OK (warns, continues over the gap)')
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# 12. the recheck gate: pass / fail / not_run
# ---------------------------------------------------------------------------
def test_recheck_gate():
    root = _tmpdir()
    try:
        write_fixture(root)
        log = os.path.join(root, 'logs.json.txt')
        lines = open(log).read().split('\n')
        # perturb one logged scalar at epoch 5 by 10%
        for i, line in enumerate(lines):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('epoch') == 5 and 'latent_probe/zv_norm_mean' in d:
                d['latent_probe/zv_norm_mean'] *= 1.10
                lines[i] = json.dumps(d)
                break
        open(log, 'w').write('\n'.join(lines))
        out, code = _run(root, extra=['--strict'])
        assert code == 3, code
        assert out['gates']['recheck']['status'] == 'fail'
        assert any('recheck FAILED' in w for w in out['warnings'])
        # the failing scalar is named, and the others still pass
        assert out['gates']['recheck']['per_scalar']['zv_norm_mean'] > 0.05
        print('recheck gate OK (10% perturbation fails, scalar named)')
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root = _tmpdir()
    try:
        write_fixture(root)
        out, code = _run(root, log=None, extra=['--strict'])
        assert code == 0
        assert out['gates']['recheck'] is None
        md = open(os.path.join(root, 'out', 'summary.md')).read()
        assert 'not_run' in md
        print('recheck not_run OK (no log -> not_run, never a vacuous PASS)')
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# 13. scalars-only mode (the laptop dry run) + outputs
# ---------------------------------------------------------------------------
def test_scalars_only_and_outputs():
    root = _tmpdir()
    try:
        write_fixture(root)
        out, code = als.run(['-l', os.path.join(root, 'logs.json.txt'),
                             '--scalars-only', '-o', os.path.join(root, 'sout')])
        assert code == 0
        assert out['mode'] == 'scalars'
        assert len(out['series']['epochs']) == 10
        assert out['settle']['scalars']['zv_norm_mean']['settle_epoch'] == 3
        assert out['gates']['malformed'] == 1
        assert out['series']['per_view'] == {}
        print('scalars-only OK (10 epochs, settle 3, malformed line counted)')
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root = _tmpdir()
    try:
        write_fixture(root)
        out, code = _run(root, extra=['--strict'])
        p = os.path.join(root, 'out', 'analyze_latent_series.json')
        size = os.path.getsize(p)
        assert size < 1024 * 1024, size
        with open(p) as f:
            j = json.load(f)
        for k in ('schema', 'gates', 'settle', 'series', 'drift', 'fused',
                  'behaviour', 'meta', 'warnings'):
            assert k in j, k
        assert j['schema'] == 1
        md = open(os.path.join(root, 'out', 'summary.md')).read()
        for needle in ('## gates', '## scalars', '## representation drift',
                       '## per-view', '## fused latents', '## behaviour',
                       '## caveats'):
            assert needle in md, needle
        print(f'outputs OK (json {size/1024:.0f} KB, all summary sections present)')
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test():
    test_rel_dist()
    test_cka()
    test_procrustes()
    test_settle_rules()
    test_per_view_and_roll()
    test_end_to_end()
    test_cka_gate_can_fail()
    test_az0_mapping_mutation()
    test_state_idx_gate()
    test_shape_and_finite_gates()
    test_min_epochs_and_hole()
    test_recheck_gate()
    test_scalars_only_and_outputs()
    print('\nALL analyze_latent_series TESTS PASSED')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--write-fixture', default=None,
                    help='write the synthetic series here and exit (for '
                         'smoking the real CLI)')
    args = ap.parse_args()
    if args.write_fixture:
        write_fixture(args.write_fixture)
        print(f'fixture written to {args.write_fixture}')
    else:
        test()
