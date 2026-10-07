"""CPU-only tests for visualize_latent_distribution.py (numpy; matplotlib only
for the plot smoke -- skipped LOUDLY if absent, per this repo's rules).

Builds synthetic mini-snapshots with a KNOWN planted structure and checks both
that `compute` recovers it and that the gates can fail when they should:

  * the planted per-view offsets come back as separated view centroids;
  * shifting ONE view's latents moves that view's centroid by the predicted
    amount while a pair of untouched views stays put (mutation power);
  * a state_idx mismatch across epochs must stop the compute loudly;
  * PCA signs are deterministic -- two runs are byte-identical;
  * `plot` renders fig1/fig2 from the coords JSON, and fig3/fig4 from a
    synthetic analyze_latent_series-style digest.

Run from the repo root:
    python tests/test_visualize_latent_distribution.py
"""
import argparse
import itertools
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

import visualize_latent_distribution as vld  # noqa: E402

S_DEF, K_DEF, D_DEF = 24, 4, 8
POOL = [1, 2, 3, 4]          # ring indices; az0_ring 3 is at canonical slot 2
AZ0_RING, AZ0_POS = 3, 2
R_VIEW = 1.5                 # planted per-view offset radius (dims 0,1)
AMP_LO, AMP_HI = 0.2, 0.6    # planted per-state amplitude (dim K)
SHIFT = np.array([2.0, 0.0])  # mutation: view 0's in-plane shift


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _view_offsets():
    """K points on a circle of radius R_VIEW in dims (0, 1)."""
    out = np.zeros((K_DEF, D_DEF))
    for j in range(K_DEF):
        th = 2.0 * math.pi * j / K_DEF
        out[j, 0] = R_VIEW * math.cos(th)
        out[j, 1] = R_VIEW * math.sin(th)
    return out


def make_snapshots(root, epochs=(0, 1, 2), shift_view0=False,
                   state_idx_mutate_epoch=None):
    """Synthetic snapshot store whose z_v = view offsets + per-state amplitude
    along dim K + small noise; z_g = mean over views. Planted structure only --
    if `compute` cannot recover it, the analysis is wrong."""
    snap = os.path.join(root, 'latent_snapshots')
    os.makedirs(snap, exist_ok=True)
    meta = {'tool': 'test', 'n_states': S_DEF, 'K': K_DEF, 'fused_dim': D_DEF,
            'seed': 1, 'pool_ring': POOL, 'az0_ring': AZ0_RING,
            'az0_pos': AZ0_POS, 'conditions': ['full', 'n1_az0'], 'arrays': {}}
    with open(os.path.join(snap, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    rng = np.random.RandomState(0)
    offs = _view_offsets()
    amp = rng.uniform(AMP_LO, AMP_HI, size=S_DEF)
    base_si = np.arange(S_DEF, dtype=np.int64)
    for e in epochs:
        scale = 0.6 + 0.2 * e                # structure grows over epochs
        zv = (offs[None, :, :] * scale
              + rng.normal(scale=0.01, size=(S_DEF, K_DEF, D_DEF)))
        zv[:, :, K_DEF] += (amp * scale)[:, None]  # per-state variation, 1 dim
        if shift_view0:
            zv[:, 0, :2] += SHIFT            # in-plane mutation, view 0 only
        zg = zv.mean(axis=1) + rng.normal(scale=0.01, size=(S_DEF, D_DEF))
        zn = zg + rng.normal(scale=0.002, size=(S_DEF, D_DEF))
        si = base_si.copy()
        if state_idx_mutate_epoch == e:
            si[0] += 1
        np.savez(os.path.join(snap, f'epoch_{e:04d}.npz'),
                 zv=zv.astype(np.float16), zg=zg.astype(np.float16),
                 zg_n1=zn.astype(np.float16), state_idx=si)
    return snap


def make_digest(path):
    """A tiny analyze_latent_series-style digest for the fig3/fig4 smoke."""
    rng = np.random.RandomState(1)
    n_ep = 3
    per_view = {k: rng.uniform(0.1, 2.0, size=(n_ep, K_DEF)).tolist()
                for k in ('norm_mean', 'rel_spread', 'ratio')}
    order = [list(p) for p in itertools.combinations(range(K_DEF), 2)]
    vals = rng.uniform(0.2, 1.5, size=(2, len(order))).tolist()
    doc = {'series': {'epochs': [0, 1, 2],
                      'per_view': per_view,
                      'pair_matrix': {'epochs': [0, 2], 'order': order,
                                      'values': vals}},
           'meta': {'pool_ring': POOL}}
    with open(path, 'w') as f:
        json.dump(doc, f)
    return path


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_compute_recovers_planted_structure(tmp):
    snap = make_snapshots(os.path.join(tmp, 'structure'))
    out = os.path.join(tmp, 'structure', 'coords.json')
    rc = vld.run(['compute', '-s', snap, '-o', out, '--epochs', '0,1,2',
                  '--basis-epoch', '2'])
    assert rc == 0
    doc = json.load(open(out))
    assert doc['S'] == S_DEF and doc['K'] == K_DEF and doc['D'] == D_DEF
    assert [ep['epoch'] for ep in doc['epochs']] == [0, 1, 2]
    for ep in doc['epochs']:
        assert len(ep['zv_xy']) == S_DEF * K_DEF
        assert len(ep['zg_xy']) == S_DEF and len(ep['zg_n1_xy']) == S_DEF
    evr = doc['basis']['evr']
    assert evr[0] >= evr[1] > 0.0 and evr[0] <= 1.0
    pts = np.asarray(doc['epochs'][-1]['zv_xy']).reshape(S_DEF, K_DEF, 2)
    cen = pts.mean(axis=0)
    dmin = min(np.linalg.norm(cen[i] - cen[j])
               for i in range(K_DEF) for j in range(i + 1, K_DEF))
    assert dmin > 1.0, f'view centroids not separated: dmin={dmin:.3f}'
    print(f'  planted view structure recovered: min centroid distance '
          f'{dmin:.3f} (planted min {2 * R_VIEW * math.sin(math.pi / K_DEF):.3f})')


def test_mutation_shift_moves_only_that_view(tmp):
    base = make_snapshots(os.path.join(tmp, 'mut_base'))
    mut = make_snapshots(os.path.join(tmp, 'mut_shift'), shift_view0=True)
    ob = os.path.join(tmp, 'mut_base', 'coords.json')
    om = os.path.join(tmp, 'mut_shift', 'coords.json')
    for s, o in ((base, ob), (mut, om)):
        assert vld.run(['compute', '-s', s, '-o', o, '--epochs', '0,1,2',
                        '--basis-epoch', '2']) == 0

    def centroids(path):
        d = json.load(open(path))
        return np.asarray(d['epochs'][-1]['zv_xy']).reshape(
            S_DEF, K_DEF, 2).mean(axis=0)

    cb, cm = centroids(ob), centroids(om)
    # view0 vs view1: centroid difference moves by exactly SHIFT (in-plane),
    # so the projected distance follows |c0 - c1 + SHIFT|.
    expect = float(np.linalg.norm(np.array([R_VIEW, -R_VIEW]) + SHIFT))
    got = float(np.linalg.norm(cm[0] - cm[1]))
    assert abs(got - expect) / expect < 0.05, (got, expect)
    db = float(np.linalg.norm(cb[2] - cb[3]))
    dm = float(np.linalg.norm(cm[2] - cm[3]))
    assert abs(dm - db) / db < 0.02, (db, dm)
    print(f'  mutation power: d(0,1) {np.linalg.norm(cb[0] - cb[1]):.3f} -> '
          f'{got:.3f} (predicted {expect:.3f}); d(2,3) moved {abs(dm - db):.4f}')


def test_state_idx_mismatch_stops_compute(tmp):
    snap = make_snapshots(os.path.join(tmp, 'mut_si'), state_idx_mutate_epoch=1)
    out = os.path.join(tmp, 'mut_si', 'coords.json')
    try:
        vld.run(['compute', '-s', snap, '-o', out, '--epochs', '0,1,2',
                 '--basis-epoch', '2'])
    except SystemExit as e:
        assert 'state_idx' in str(e), str(e)
        assert not os.path.exists(out), 'coords written despite failed gate'
        print(f'  state_idx gate fired: {e}')
        return
    raise AssertionError('compute did not stop on a state_idx mismatch')


def test_determinism(tmp):
    snap = make_snapshots(os.path.join(tmp, 'det'))
    o1 = os.path.join(tmp, 'det', 'a.json')
    o2 = os.path.join(tmp, 'det', 'b.json')
    assert vld.run(['compute', '-s', snap, '-o', o1, '--epochs', '0,1,2',
                    '--basis-epoch', '2']) == 0
    assert vld.run(['compute', '-s', snap, '-o', o2, '--epochs', '0,1,2',
                    '--basis-epoch', '2']) == 0
    assert open(o1, 'rb').read() == open(o2, 'rb').read(), \
        'PCA signs not deterministic across runs'
    print('  determinism: two runs byte-identical')


def test_plot_smoke(tmp):
    try:
        import matplotlib  # noqa: F401
    except Exception as e:                                    # pragma: no cover
        print(f'  SKIPPED (matplotlib unavailable: {e}) -- plot smoke NOT run')
        return
    snap = make_snapshots(os.path.join(tmp, 'plot'))
    coords = os.path.join(tmp, 'plot', 'coords.json')
    assert vld.run(['compute', '-s', snap, '-o', coords, '--epochs', '0,1,2',
                    '--basis-epoch', '2']) == 0
    digest = make_digest(os.path.join(tmp, 'plot', 'digest.json'))
    figdir = os.path.join(tmp, 'plot', 'figures')
    paths = vld.plot(coords, figdir, digest)
    names = sorted(p.name for p in paths)
    assert names == ['fig1_zv_view_structure.png',
                     'fig2_zg_n1_invariance.png',
                     'fig3_per_view_curves.png',
                     'fig4_pair_matrix.png'], names
    for p in paths:
        assert p.stat().st_size > 10_000, (p, p.stat().st_size)
    print(f'  plot smoke: {len(paths)} figures, '
          f'min size {min(p.stat().st_size for p in paths) // 1024} KB')


def main():
    tests = [test_compute_recovers_planted_structure,
             test_mutation_shift_moves_only_that_view,
             test_state_idx_mismatch_stops_compute,
             test_determinism,
             test_plot_smoke]
    failures = []
    with tempfile.TemporaryDirectory(prefix='vld_test_') as tmp:
        for fn in tests:
            print(f'== {fn.__name__}')
            try:
                fn(tmp)
            except Exception as e:
                failures.append(fn.__name__)
                print(f'  FAIL: {type(e).__name__}: {e}')
    if failures:
        print(f'\nFAILED: {failures}')
        return 1
    print('\nALL PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
