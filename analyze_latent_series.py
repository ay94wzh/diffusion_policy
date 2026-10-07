#!/usr/bin/env python3
"""Analyze a per-epoch latent series: what the "clear" run's hook records.

Input: a run's ``latent_snapshots/`` (``meta.json`` + ``snapshots.jsonl`` +
``epoch_XXXX.npz``), written by
``train_diffusion_unet_image_workspace_latentprobe.py``. Every npz holds the
SAME fixed probe set encoded at one epoch -- ``zv`` (S, K, D) with canonical
slots (slot j == ring ``pool_ring[j]``), ``zg`` (S, D) at a full draw,
``zg_n1`` (S, D) at the single az_0 slot -- so rows are state-aligned across
epochs and every cross-epoch comparison is exact.

What it measures
----------------
* per-epoch scalars: the hook's own 10 summaries, recomputed from the stored
  (fp16) arrays -- the recheck gate compares them against the fp32 values the
  hook logged, which is what keeps this re-implementation honest;
* per-view detail the logged mean/min hides (K curves): norm, relative
  spread, participation ratio, pair/cross means, and the exact per-view
  decomposition of the hook's ``zv_pair_ratio``;
* representation drift across epochs under four complementary measures, which
  must never be read alone: row-aligned relative distance (strict, and
  gauge-sensitive: a rotation of the latent space inflates it), cosine
  distance (scale-free), linear CKA (rotation-invariant -- "same geometry up
  to rotation"), and Procrustes residual (movement beyond a rigid rotation);
* fused-latent relations relevant to M5's distillation premise: zg_n1 vs zg,
  both vs the az_0 view's z_v, and zg vs the uniform mean-pool of z_v;
* the behaviour line from the run's ``logs.json.txt`` (rollout epochs only).

Gates (a failed required gate is recorded loudly; with ``--strict`` the run
exits 3, so a driver stops instead of quoting numbers):
  layout/meta/shape/finite; state_idx identity across every npz (an abort --
  silently comparing mismatched rows would fabricate drift); scalar recheck;
  fp16 storage floor (a stated convention). Chance levels are *measured and
  reported*, never assumed: the plain linear CKA's is ~d/(n+d) (0.27 for
  zv_flat at the clear run's shape, 0.80 for zg) and Procrustes' is ~0.35 at
  n < d, so every similarity value is read against its own null. A gate that
  cannot run reports ``not_run``, never PASS.

Storage floor: the npz are fp16, so differences below ~4.9e-4 relative are
not measurable from this artifact; readings at or below the floor are
reported as bounds, never as values. A *deep* collapse (the future `[1,2]`
run measured 1.8e-7 relative spread) is far below this floor -- its npz can
only bound it; the fp32 scalars in logs.json.txt remain the deep instrument
(NOTES.md *Traps*).

Dependencies: numpy + stdlib only, so it runs on `miroc-server` (where the
npz live) and on the laptop (scalars-only mode, from the committed log). The
``_rel_dist`` formula is the same one in ``probe_relpose.py`` and the hook;
importing either would drag in torch/hydra/zarr, so it is re-implemented here
and cross-pinned by ``tests/test_analyze_latent_series.py`` whenever the real
one imports.

Usage::

    python analyze_latent_series.py -s <run>/latent_snapshots \
        -l <run>/logs.json.txt -o data/analysis_clear --strict
    python analyze_latent_series.py -l <run>/logs.json.txt --scalars-only -o /tmp/ls
"""
import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

SCHEMA = 1
FP16_REL_FLOOR = 2.0 ** -11          # per-element fp16 relative resolution
SCALAR_KEYS = (
    'zv_norm_mean', 'zv_rel_spread_mean', 'zv_rel_spread_min',
    'zg_norm_mean', 'zg_rel_spread', 'zg_n1_norm_mean', 'zg_n1_rel_spread',
    'zv_pair_ratio', 'zv_pr', 'zg_pr',
)
TARGETS = ('zv_flat', 'zg', 'zg_n1')
DRIFT_MEASURES = ('rel', 'cos', 'cka', 'proc')
FUSED_RELATIONS = ('n1_vs_full', 'n1_vs_zv_az0', 'full_vs_zv_az0',
                   'full_vs_mean_pool')


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _r6(x):
    """Round to 6 significant digits (JSON hygiene); None/non-finite pass."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0.0:
        return 0.0
    return float('%.6g' % x)


def _finite(x):
    return x is not None and math.isfinite(float(x))


def _utc_now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# ---------------------------------------------------------------------------
# statistics (pure numpy; unit-tested in tests/test_analyze_latent_series.py)
# ---------------------------------------------------------------------------
def rel_dist_np(a, b, eps=1e-6):
    """Mean over rows of ||a-b|| / (0.5(||a||+||b||)).

    The same definition as ``probe_relpose._rel_dist`` (line 341) and the
    latentprobe hook's module-level ``_rel_dist`` -- re-implemented here so
    this tool stays numpy-only.
    """
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    return float(np.mean(np.linalg.norm(a - b, axis=1) /
                         np.maximum(0.5 * (an + bn), eps)))


def rel_dist_rows(a, b, eps=1e-6):
    """Per-row version of rel_dist_np (for median/p90 and floor flags)."""
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    return np.linalg.norm(a - b, axis=1) / np.maximum(0.5 * (an + bn), eps)


def cos_dist_np(a, b, eps=1e-12):
    """Mean 1 - cos(a_r, b_r); also returns the count of rows with a ~zero
    norm (their direction is undefined and they are counted, not zeroed)."""
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    ok = (an > eps) & (bn > eps)
    n_zero = int((~ok).sum())
    if not ok.any():
        return float('nan'), n_zero
    c = np.sum(a[ok] * b[ok], axis=1) / (an[ok] * bn[ok])
    return float(np.mean(1.0 - c)), n_zero


def norm_ratio(a, b, eps=1e-12):
    """Mean over rows of ||b|| / ||a|| (a = earlier epoch, b = later)."""
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    return float(np.mean(bn / np.maximum(an, eps)))


def rel_spread(x):
    """The hook's ``_rel_spread``: max-over-dims std across rows over the mean
    row norm. ~1e-02 healthy, ~1e-04 and below degenerate (read it against the
    same checkpoint's random-init screen, never in absolute)."""
    sd = x.std(axis=0, ddof=1).max() if x.shape[0] > 1 else 0.0
    denom = max(float(np.linalg.norm(x, axis=1).mean()), 1e-12)
    return float(sd) / denom


def participation_ratio(x):
    """The hook's ``_participation_ratio``: (sum lam)^2 / sum lam^2 over the
    covariance spectrum (singular values of the centered rows)."""
    xc = x - x.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(xc, compute_uv=False)
    lam = sv.astype(np.float64) ** 2
    return float(lam.sum() ** 2 / max(float((lam ** 2).sum()), 1e-30))


def center_rows(x):
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x, y):
    """Linear CKA: ||Xc^T Yc||_F^2 / sqrt(||Xc^T Xc||_F^2 * ||Yc^T Yc||_F^2).

    Invariant to orthogonal transforms and isotropic scaling of either input
    (that is the point of reading it next to the strict measures).
    """
    xc = center_rows(x)
    yc = center_rows(y)
    xty = xc.T @ yc
    num = float(np.sum(xty * xty))
    xx = xc.T @ xc
    yy = yc.T @ yc
    den = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    return num / den if den > 0.0 else 0.0


def permutation_floor(x, y, repeats=8, seed=0):
    """CKA between x and row-permuted copies of y -- the chance floor the real
    CKA must sit far above for the reading to mean anything."""
    rng = np.random.RandomState(seed)
    n = x.shape[0]
    vals = [linear_cka(x, y[rng.permutation(n)]) for _ in range(int(repeats))]
    return {'mean': float(np.mean(vals)), 'max': float(np.max(vals)),
            'repeats': int(repeats)}


def split_half_reliability(prev, cur):
    """Two disjoint-state-half estimates of the SAME CKA reading, and their gap.

    This is the finite-sample floor for reading a CKA *difference*: both halves
    estimate one and the same population quantity, so their disagreement is
    sampling noise.

    Two forms that do NOT work, both tried and rejected here: CKA(x, x) on
    even/odd rows separately is exactly 1 by construction (a check that cannot
    fail), and CKA between two independently sampled halves reads ~1/n_half
    whatever the structure is (it measures chance, not shared geometry).
    """
    a = linear_cka(prev[::2], cur[::2])
    b = linear_cka(prev[1::2], cur[1::2])
    return {'even': float(a), 'odd': float(b), 'gap': float(abs(a - b))}


def procrustes_resid(x, y, eps=1e-12):
    """Fraction of the averaged energy NOT explained by a rigid rotation of one
    row-set onto the other (after centering).

    ``resid = sqrt(||Xc||^2 + ||Yc||^2 - 2 sum(svdvals(Yc^T Xc))) /
    (0.5 (||Xc|| + ||Yc||))``. A pure rotation gives 0; a pure *scale* change
    still reads |c-1| / ((1+c)/2) -- that is intended (it is a change of the
    tensors), and it is why this measure is read next to CKA. The sqrt of a
    cancelling difference of large numbers caps the numerical floor at
    ~1e-8 relative (far below any signal this instrument reads).
    """
    xc = center_rows(x)
    yc = center_rows(y)
    s = np.linalg.svd(yc.T @ xc, compute_uv=False)
    diff = float(np.sum(xc * xc) + np.sum(yc * yc) - 2.0 * float(np.sum(s)))
    denom = 0.5 * (float(np.linalg.norm(xc)) + float(np.linalg.norm(yc)))
    return math.sqrt(max(diff, 0.0)) / max(denom, eps)


def settle_epoch(values, epochs, tol=0.05, tail_frac=0.1, min_tail=3):
    """First epoch after which a series stays within tol x C of its final
    value, where C = max_t |v(t) - v_final| -- the curve's total excursion
    from its end point.

    The band is relative to the excursion, not to the final value: that makes
    the rule scale-free and lets it read a curve whose limit is 0 (every
    drift-to-final series), which a band relative to the final value cannot.
    ``settled`` additionally requires the last max(min_tail, tail_frac*n)
    epochs to sit inside the band -- a strictly monotone series therefore
    never settles, and that is reported, never clamped. The convention (tol,
    tail window) is part of the reading, so it is recorded in the output.
    """
    pairs = [(int(e), float(v)) for e, v in zip(epochs, values)
             if _finite(v)]
    n = len(pairs)
    if n < 2:
        return {'settled': False, 'settle_epoch': None, 'reason': 'too_few',
                'n': n}
    final = pairs[-1][1]
    dev = [abs(v - final) for _, v in pairs]
    C = max(dev)
    band = abs(float(tol)) * C + 1e-12
    k = min(n, max(int(min_tail), int(math.ceil(tail_frac * n))))
    tail = [v for _, v in pairs[-k:]]
    tail_dev = max(dev[-k:])
    settle = None
    for i in range(n):
        if all(d <= band for d in dev[i:]):
            settle = pairs[i][0]
            break
    settled = bool(tail_dev <= band and settle is not None)
    return {'settled': settled,
            'settle_epoch': settle if settled else None,
            'C': _r6(C), 'band': _r6(band), 'tail_dev': _r6(tail_dev),
            'tail_mean': _r6(float(np.mean(tail))),
            'tail_std': _r6(float(np.std(tail, ddof=1)) if k > 1 else 0.0),
            'final': _r6(final), 'n': n}


def frozen_epoch(values, epochs, floor=FP16_REL_FLOOR):
    """First epoch after which every later value is below the given floor
    (None if that never happens, e.g. a series still moving at the end)."""
    pairs = [(int(e), float(v)) for e, v in zip(epochs, values)
             if _finite(v)]
    for i in range(len(pairs)):
        if all(abs(v) < floor for _, v in pairs[i:]):
            return pairs[i][0]
    return None


def frozen_tail_count(values, floor=FP16_REL_FLOOR):
    """How many trailing values are below the floor (a single last-epoch
    point below it is not a freeze -- reported next to frozen_epoch)."""
    n = 0
    for v in reversed(values):
        if v is not None and _finite(v) and abs(v) < floor:
            n += 1
        else:
            break
    return n


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def resolve_snapshots(path):
    """Accept either the snapshots dir itself or a run dir containing one."""
    p = pathlib.Path(path)
    if (p / 'meta.json').is_file():
        return p
    alt = p / 'latent_snapshots'
    if (alt / 'meta.json').is_file():
        return alt
    raise FileNotFoundError(
        f'no meta.json under {p} or {alt} -- point -s at the snapshots dir')


def load_meta(snap_dir):
    with open(pathlib.Path(snap_dir) / 'meta.json') as f:
        return json.load(f)


def list_epoch_files(snap_dir):
    """{epoch: path} for every epoch_XXXX.npz; also the missing epochs inside
    the covered span."""
    snap_dir = pathlib.Path(snap_dir)
    files = {}
    for p in sorted(snap_dir.glob('epoch_*.npz')):
        stem = p.name[len('epoch_'):-len('.npz')]
        files[int(stem)] = p
    missing = []
    if files:
        have = set(files)
        missing = [e for e in range(min(have), max(have) + 1) if e not in have]
    return files, missing


def parse_epoch_spec(spec, available):
    """'all' | 'a,b,c' | 'a:b:c' -> (selected, dropped)."""
    avail = set(int(e) for e in available)
    if spec in (None, 'all'):
        want = sorted(avail)
    elif ':' in spec:
        parts = spec.split(':')
        start = int(parts[0]) if parts[0] else min(avail)
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else max(avail)
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        want = list(range(start, stop + 1, step))
    else:
        want = [int(x) for x in spec.split(',') if x.strip() != '']
    selected = sorted(e for e in want if e in avail)
    dropped = sorted(e for e in want if e not in avail)
    return selected, dropped


def load_epoch(snap_dir, e):
    p = pathlib.Path(snap_dir) / ('epoch_%04d.npz' % int(e))
    with np.load(str(p), allow_pickle=False) as z:
        return {'zv': z['zv'].astype(np.float64),
                'zg': z['zg'].astype(np.float64),
                'zg_n1': z['zg_n1'].astype(np.float64),
                'state_idx': z['state_idx'].astype(np.int64),
                'dtypes': {k: str(z[k].dtype) for k in ('zv', 'zg', 'zg_n1')}}


def check_arrays(entry, meta, first_state_idx):
    """Gate violations for one epoch's arrays (empty list = clean)."""
    bad = []
    S, K, D = int(meta['n_states']), int(meta['K']), int(meta['fused_dim'])
    zv, zg, zn = entry['zv'], entry['zg'], entry['zg_n1']
    if zv.shape != (S, K, D):
        bad.append(f'zv shape {zv.shape} != {(S, K, D)}')
    if zg.shape != (S, D):
        bad.append(f'zg shape {zg.shape} != {(S, D)}')
    if zn.shape != (S, D):
        bad.append(f'zg_n1 shape {zn.shape} != {(S, D)}')
    for name in ('zv', 'zg', 'zg_n1'):
        if not np.all(np.isfinite(entry[name])):
            bad.append(f'{name} has non-finite values')
    if first_state_idx is not None and not np.array_equal(
            entry['state_idx'], first_state_idx):
        n_diff = int((entry['state_idx'] != first_state_idx).sum())
        bad.append(f'state_idx differs from the first epoch in {n_diff} entries')
    return bad


def parse_log(path):
    """Parse the run's ``logs.json.txt`` OR a ``snapshots.jsonl``.

    Returns ``{'format', 'scalars': {epoch: {key: val}}, 'behaviour': {...},
    'malformed': n}``. Scalars are looked up under both the prefixed
    (logs.json.txt) and bare (snapshots.jsonl) names.
    """
    path = pathlib.Path(path)
    scalars = {}
    b_epochs, b_scores, b_n = [], [], []
    train_scores = []
    malformed = 0
    fmt = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if 'epoch' not in d:
                continue
            if fmt is None and any(
                    k in d for k in ('latent_probe/zv_norm_mean', 'zv_norm_mean')):
                fmt = ('logs.json.txt' if 'latent_probe/zv_norm_mean' in d
                       else 'snapshots.jsonl')
            e = int(d['epoch'])
            vals = {}
            for k in SCALAR_KEYS:
                v = d.get('latent_probe/' + k, d.get(k))
                if v is not None:
                    vals[k] = float(v)
            if vals:
                scalars[e] = vals
            if 'test/mean_score' in d:
                b_epochs.append(e)
                b_scores.append(float(d['test/mean_score']))
                b_n.append(int(sum(1 for k in d
                                   if k.startswith('test/sim_max_reward'))))
            if 'train/mean_score' in d:
                train_scores.append([e, float(d['train/mean_score'])])
    behaviour = None
    if b_epochs:
        behaviour = {'source': path.name,
                     'rollout_epochs': b_epochs,
                     'test_mean_score': b_scores,
                     'n_episodes': b_n,
                     'train_mean_score': train_scores or None}
    return {'format': fmt or 'unknown', 'scalars': scalars,
            'behaviour': behaviour, 'malformed': malformed}


# ---------------------------------------------------------------------------
# per-epoch statistics
# ---------------------------------------------------------------------------
def per_view_stats(zv):
    """Per-view (K) statistics: the ones the hook logged only as mean/min,
    plus the exact decomposition of its ``zv_pair_ratio``.

    * pair_mean_j = mean_{i != j} rel_dist(zv[:, i], zv[:, j]) -- same state,
      two views;   mean_j pair_mean_j is the hook's ``mean(pair)``.
    * cross_mean_j = mean over state pairs within view j -- view j against
      itself;  mean_j cross_mean_j is the hook's ``mean(cross)``.
    """
    S, K = zv.shape[0], zv.shape[1]
    iu = np.triu_indices(S, 1)
    out = {k: [] for k in ('norm_mean', 'rel_spread', 'pr', 'pair_mean',
                           'cross_mean', 'ratio')}
    for j in range(K):
        xj = zv[:, j, :]
        out['norm_mean'].append(float(np.linalg.norm(xj, axis=1).mean()))
        out['rel_spread'].append(rel_spread(xj))
        out['pr'].append(participation_ratio(xj))
        out['cross_mean'].append(rel_dist_np(xj[iu[0]], xj[iu[1]]))
    for j in range(K):
        vals = [rel_dist_np(zv[:, i, :], zv[:, j, :])
                for i in range(K) if i != j]
        out['pair_mean'].append(float(np.mean(vals)))
    out['ratio'] = [p / max(c, 1e-12)
                    for p, c in zip(out['pair_mean'], out['cross_mean'])]
    return out


def hook_scalars(zv, zg, zg_n1, pv):
    """The hook's own 10 summaries, recomputed. ``pv`` is per_view_stats(zv)."""
    return {
        'zv_norm_mean': float(np.linalg.norm(zv, axis=-1).mean()),
        'zv_rel_spread_mean': float(np.mean(pv['rel_spread'])),
        'zv_rel_spread_min': float(np.min(pv['rel_spread'])),
        'zg_norm_mean': float(np.linalg.norm(zg, axis=-1).mean()),
        'zg_rel_spread': rel_spread(zg),
        'zg_n1_norm_mean': float(np.linalg.norm(zg_n1, axis=-1).mean()),
        'zg_n1_rel_spread': rel_spread(zg_n1),
        'zv_pair_ratio': (float(np.mean(pv['pair_mean'])) /
                          max(float(np.mean(pv['cross_mean'])), 1e-12)),
        'zv_pr': participation_ratio(zv.reshape(-1, zv.shape[-1])),
        'zg_pr': participation_ratio(zg),
    }


def pair_matrix(zv):
    """The upper-triangle matrix of per-view-pair rel_dist, in a fixed order
    ([(i, j) for i in range(K) for j in range(i+1, K)])."""
    K = zv.shape[1]
    order, vals = [], []
    for i in range(K):
        for j in range(i + 1, K):
            order.append([i, j])
            vals.append(rel_dist_np(zv[:, i, :], zv[:, j, :]))
    return order, vals


def fused_stats(zv, zg, zg_n1, az0_pos):
    """The four fused-latent relations (each a dict of scalar summaries)."""
    za = zv[:, int(az0_pos), :]
    pool_mean = zv.mean(axis=1)
    rels = {'n1_vs_full': (zg_n1, zg),
            'n1_vs_zv_az0': (zg_n1, za),
            'full_vs_zv_az0': (zg, za),
            'full_vs_mean_pool': (zg, pool_mean)}
    out = {}
    for name, (a, b) in rels.items():
        rows = rel_dist_rows(a, b)
        c, n_zero = cos_dist_np(a, b)
        out[name] = {
            'rel': float(rows.mean()),
            'rel_median': float(np.median(rows)),
            'rel_p90': float(np.quantile(rows, 0.9)),
            'cos': c,
            'cos_n_zero_rows': n_zero,
            'cka': linear_cka(a, b),
            'norm_ratio': norm_ratio(a, b),
        }
    return out


def drift_vs(a, b):
    """The four drift measures of a against b (both (n, D))."""
    c, n_zero = cos_dist_np(a, b)
    return {'rel': rel_dist_np(a, b), 'cos': c, 'cos_n_zero_rows': n_zero,
            'cka': linear_cka(a, b), 'proc': procrustes_resid(a, b)}


def _targets(zv, zg, zn, S, K, D):
    return {'zv_flat': zv.reshape(S * K, D), 'zg': zg, 'zg_n1': zn}


# ---------------------------------------------------------------------------
# the streaming pass
# ---------------------------------------------------------------------------
def analyze_npz(snap_dir, meta, epochs, refs, cfg):
    """One streaming pass over the selected epochs.

    Ref-epoch arrays are pre-loaded into a small cache (a few tens of MB) so
    drift-vs-ref is computable at every epoch in the same pass; ``prev`` drift
    uses the previously selected epoch. Returns
    ``(series, drift, fused, gates, warnings, analyzed)``.
    """
    snap_dir = pathlib.Path(snap_dir)
    S, K, D = int(meta['n_states']), int(meta['K']), int(meta['fused_dim'])
    a0 = int(meta['az0_pos'])
    warnings = []
    gates = {'layout': [], 'state_idx_identical': True,
             'state_idx_first_mismatch': None, 'dtypes': None,
             'n_skipped': 0}

    files, missing = list_epoch_files(snap_dir)
    if missing:
        warnings.append(f'epoch holes: {missing} (analysis continues over them)')
    gates['epoch_holes'] = missing
    gates['n_npz'] = len(files)

    refs = sorted(set(int(r) for r in refs if int(r) in files)
                  | {int(epochs[-1])})
    ref_arrays = {}
    for r in refs:
        entry = load_epoch(snap_dir, r)
        ref_arrays[r] = _targets(entry['zv'], entry['zg'], entry['zg_n1'],
                                 S, K, D)

    series = {'epochs': [], 'scalars': {k: [] for k in SCALAR_KEYS},
              'per_view': {k: [] for k in ('norm_mean', 'rel_spread', 'pr',
                                           'pair_mean', 'cross_mean', 'ratio')},
              'pair_matrix': {'order': None, 'epochs': [], 'values': []}}
    drift = {'refs': list(refs), 'targets': {}, 'per_view': {}}
    for name in TARGETS:
        drift['targets'][name] = {
            'rel_prev': [], 'cos_prev': [], 'cka_prev': [], 'proc_prev': [],
            'norm_ratio_prev': [],
            'rel_vs_ref': {str(r): [] for r in refs},
            'cka_vs_ref': {str(r): [] for r in refs},
            'proc_vs_ref': {str(r): [] for r in refs}}
    for k in ('rel_prev', 'cos_prev', 'cka_prev'):
        drift['per_view'][k] = []
    fused = {name: {k: [] for k in ('rel', 'rel_median', 'rel_p90', 'cos',
                                    'cka', 'norm_ratio')}
             for name in FUSED_RELATIONS}

    prev = None
    prev_zv = None
    first_state_idx = None

    for e in epochs:
        entry = load_epoch(snap_dir, e)
        if first_state_idx is None:
            first_state_idx = entry['state_idx']
            gates['dtypes'] = entry['dtypes']
        bad = check_arrays(entry, meta, first_state_idx)
        if bad:
            fatal = any(('state_idx' in b or 'shape' in b) for b in bad)
            if not fatal and cfg['allow_nonfinite']:
                warnings.append(f'epoch {e}: {" | ".join(bad)} (skipped)')
                gates['n_skipped'] += 1
                continue
            gates['layout'].append({'epoch': e, 'violations': bad})
            if any('state_idx' in b for b in bad):
                gates['state_idx_identical'] = False
                if gates['state_idx_first_mismatch'] is None:
                    gates['state_idx_first_mismatch'] = e
            break

        zv, zg, zn = entry['zv'], entry['zg'], entry['zg_n1']
        pv = per_view_stats(zv)
        sc = hook_scalars(zv, zg, zn, pv)
        for k in SCALAR_KEYS:
            series['scalars'][k].append(sc[k])
        for k in series['per_view']:
            series['per_view'][k].append(pv[k])
        fs = fused_stats(zv, zg, zn, a0)
        for name, d in fs.items():
            for k in fused[name]:
                fused[name][k].append(d[k])
        if e in cfg['matrix_epochs']:
            order, vals = pair_matrix(zv)
            if series['pair_matrix']['order'] is None:
                series['pair_matrix']['order'] = order
            series['pair_matrix']['epochs'].append(e)
            series['pair_matrix']['values'].append(vals)

        cur = _targets(zv, zg, zn, S, K, D)
        for name in TARGETS:
            t = drift['targets'][name]
            if prev is None:
                for k in ('rel_prev', 'cos_prev', 'cka_prev', 'proc_prev',
                          'norm_ratio_prev'):
                    t[k].append(None)
            else:
                dv = drift_vs(cur[name], prev[name])
                t['rel_prev'].append(dv['rel'])
                t['cos_prev'].append(dv['cos'])
                t['cka_prev'].append(dv['cka'])
                t['proc_prev'].append(dv['proc'])
                # norm_ratio(prev, cur) = ||cur|| / ||prev||: >1 means the norm
                # GREW from the previous epoch (the function's own convention,
                # which the first version of this call had inverted)
                t['norm_ratio_prev'].append(norm_ratio(prev[name], cur[name]))
            for r in refs:
                if r == e:
                    t['rel_vs_ref'][str(r)].append(0.0)
                    t['cka_vs_ref'][str(r)].append(1.0)
                    t['proc_vs_ref'][str(r)].append(0.0)
                else:
                    dv = drift_vs(cur[name], ref_arrays[r][name])
                    t['rel_vs_ref'][str(r)].append(dv['rel'])
                    t['cka_vs_ref'][str(r)].append(dv['cka'])
                    t['proc_vs_ref'][str(r)].append(dv['proc'])
        for k in ('rel_prev', 'cos_prev', 'cka_prev'):
            if prev_zv is None:
                drift['per_view'][k].append([None] * K)
            else:
                row = []
                for j in range(K):
                    if k == 'rel_prev':
                        row.append(rel_dist_np(zv[:, j, :], prev_zv[:, j, :]))
                    elif k == 'cos_prev':
                        c, _ = cos_dist_np(zv[:, j, :], prev_zv[:, j, :])
                        row.append(c)
                    else:
                        row.append(linear_cka(zv[:, j, :], prev_zv[:, j, :]))
                drift['per_view'][k].append(row)

        series['epochs'].append(e)
        prev = cur
        prev_zv = zv.copy()

    gates['n_analyzed'] = len(series['epochs'])
    return series, drift, fused, gates, warnings


def epoch_arrays(snap_dir, meta, e):
    entry = load_epoch(snap_dir, e)
    S, K, D = int(meta['n_states']), int(meta['K']), int(meta['fused_dim'])
    return _targets(entry['zv'], entry['zg'], entry['zg_n1'], S, K, D)


def matched_null(x, seed=0):
    """The 'unrelated' level of every measure at this shape: x against an
    independent Gaussian with the same per-row norms.

    Needed because the floors are shape-dependent, not zero: the plain linear
    CKA's chance level is ~d/(n+d) (0.27 for the clear run's zv_flat, 0.80 for
    zg -- measured, not assumed), and Procrustes on n < d data aligns
    unrelated sets to ~0.35. Read every similarity value against its own null.
    """
    rng = np.random.RandomState(int(seed))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    r = rng.normal(size=x.shape)
    r *= norms / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
    c, _ = cos_dist_np(x, r)
    return {'rel': rel_dist_np(x, r), 'cos': c, 'cka': linear_cka(x, r),
            'proc': procrustes_resid(x, r)}


def chance_levels(arrays, prev_arrays, cfg):
    """Permutation / split-half / matched-null floors. ``arrays`` is the final
    analyzed epoch, ``prev_arrays`` the one before it (for the split-half
    reliability of the final consecutive CKA reading). CKA is declared void
    only when its permutation floor is so high that the measure cannot
    discriminate at all (>= --cka-perm-gate, default 0.9 -- the floors are
    otherwise reported for the reader, not used to gate)."""
    out = {'permutation_cka': {}, 'split_half': {}, 'matched_null': {},
           'valid': True, 'gate': cfg['cka_perm_gate']}
    for name, arr in arrays.items():
        out['permutation_cka'][name] = permutation_floor(
            arr, arr, repeats=cfg['perms'], seed=cfg['seed'])
        out['split_half'][name] = split_half_reliability(prev_arrays[name], arr)
        out['matched_null'][name] = matched_null(arr, seed=cfg['seed'] + 1)
    out['max_permutation'] = max(v['max']
                                 for v in out['permutation_cka'].values())
    if out['max_permutation'] >= cfg['cka_perm_gate']:
        out['valid'] = False
    return out


# ---------------------------------------------------------------------------
# recheck + settle tables
# ---------------------------------------------------------------------------
def recheck_scalars(series, log_scalars, tol):
    """Compare the npz-derived scalars against the fp32 ones the hook logged.

    Returns None (=> ``not_run``) when there is nothing to compare against --
    never a vacuous PASS.
    """
    if not log_scalars:
        return None
    per_scalar = {}
    n_cmp = 0
    for k in SCALAR_KEYS:
        worst = 0.0
        n = 0
        for i, e in enumerate(series['epochs']):
            ref = log_scalars.get(int(e), {}).get(k)
            got = series['scalars'][k][i]
            if ref is None or not _finite(got):
                continue
            worst = max(worst, abs(got - ref) / max(abs(ref), 1e-12))
            n += 1
        if n:
            per_scalar[k] = worst
            n_cmp += n
    if not per_scalar:
        return None
    max_rel = max(per_scalar.values())
    return {'status': 'pass' if max_rel <= tol else 'fail',
            'tol': tol, 'max_rel_diff': _r6(max_rel),
            'per_scalar': {k: _r6(v) for k, v in per_scalar.items()},
            'n_epoch_scalar_pairs': n_cmp}


def settle_table(series, drift, fused, behaviour, cfg):
    tol = cfg['tol']
    tf, mt = cfg['tail_frac'], cfg['min_tail']
    epochs = series['epochs']
    out = {'tol': tol, 'tail_frac': tf, 'min_tail': mt,
           'frozen_floor': cfg['storage_floor'], 'scalars': {},
           'drift': {}, 'per_view': {}, 'fused': {}}
    for k in SCALAR_KEYS:
        out['scalars'][k] = settle_epoch(series['scalars'][k], epochs, tol, tf, mt)
    if drift.get('targets'):
        for name in TARGETS:
            if name not in drift['targets']:
                continue
            d = drift['targets'][name]
            out['drift'][name] = {
                'rel_prev': settle_epoch(d['rel_prev'], epochs, tol, tf, mt),
                'cka_prev': settle_epoch(d['cka_prev'], epochs, tol, tf, mt),
                'proc_prev': settle_epoch(d['proc_prev'], epochs, tol, tf, mt),
                'rel_prev_frozen': frozen_epoch(d['rel_prev'], epochs,
                                                cfg['storage_floor']),
                'rel_prev_frozen_tail_count': frozen_tail_count(
                    d['rel_prev'], cfg['storage_floor']),
                'rel_vs_ref_settle': {
                    r: settle_epoch(v, epochs, tol, tf, mt)
                    for r, v in d['rel_vs_ref'].items()},
            }
    if drift.get('per_view') and series.get('per_view'):
        K = len(series['per_view']['rel_spread'][0])
        out['per_view'] = {
            'rel_prev': [settle_epoch(
                [drift['per_view']['rel_prev'][i][j] for i in range(len(epochs))],
                epochs, tol, tf, mt) for j in range(K)]}
    for name in fused:
        out['fused'][name] = settle_epoch(fused[name]['rel'], epochs, tol, tf, mt)
    if behaviour and len(behaviour['rollout_epochs']) >= 2:
        out['behaviour'] = settle_epoch(behaviour['test_mean_score'],
                                        behaviour['rollout_epochs'], tol, tf, mt)
    return out


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------
def _fmt(v, nd=4):
    if v is None:
        return '-'
    if isinstance(v, float):
        return ('%.*g' % (nd, v)) if math.isfinite(v) else '-'
    return str(v)


def _settle_str(s):
    if not s:
        return 'n/a'
    return str(s['settle_epoch']) if s.get('settled') else 'not_settled'


def summarize(out, cfg):
    meta = out['meta']
    L = []
    src = out['snapshot_dir'] or out['log']
    L.append(f"# latent series — {src}")
    L.append('')
    if out['series'] is None:
        L.append('ANALYSIS NOT RUN — a fatal gate failed, so no cross-epoch')
        L.append('statistic is reported (see gates below and the warnings).')
        L.append('')
        L.append('## gates')
        L.append('')
        L.append('```')
        L.append(json.dumps(out['gates'], indent=2, sort_keys=True,
                            default=_json_default))
        L.append('```')
        L.append('')
        for w in out['warnings']:
            L.append(f"- {w}")
        return '\n'.join(L) + '\n'
    epochs = out['series']['epochs']
    if out['mode'] == 'npz':
        L.append(f"{len(epochs)} epochs analyzed "
                 f"({epochs[0]}..{epochs[-1]}), S={meta['n_states']}, "
                 f"K={meta['K']}, D={meta['fused_dim']}, "
                 f"pool_ring={meta['pool_ring']}, az0_pos={meta['az0_pos']}")
    else:
        L.append(f"scalars-only: {len(epochs)} epochs ({epochs[0]}..{epochs[-1]})")
    L.append(f"settle convention: within tol={cfg['tol']} x C of the final value "
             f"(C = the curve's total excursion from it; settled also needs the "
             f"last max({cfg['min_tail']}, {cfg['tail_frac']:.0%}) epochs inside "
             f"the band); frozen floor={cfg['storage_floor']:.2e} relative")
    L.append('')

    g = out['gates']
    L.append('## gates')
    L.append('')
    L.append('| gate | value | verdict |')
    L.append('|---|---|---|')
    if out['mode'] == 'npz':
        L.append(f"| layout/meta/shape/finite | {g.get('n_analyzed', 0)} analyzed, "
                 f"{g.get('n_skipped', 0)} skipped, holes={g.get('epoch_holes')} | "
                 f"{'PASS' if not g.get('layout') else 'FAIL'} |")
        L.append(f"| state_idx identity | first mismatch: "
                 f"{g.get('state_idx_first_mismatch')} | "
                 f"{'PASS' if g.get('state_idx_identical') else 'FAIL'} |")
        rc = g.get('recheck')
        if rc:
            L.append(f"| scalar recheck | max rel {rc['max_rel_diff']:.2e} "
                     f"(tol {rc['tol']:.0e}, {rc['n_epoch_scalar_pairs']} pairs) | "
                     f"{rc['status'].upper()} |")
        else:
            L.append('| scalar recheck | no log given | not_run |')
        cf = g.get('chance_levels')
        if cf:
            per = ', '.join(f"{k} {v['max']:.3g}"
                            for k, v in cf['permutation_cka'].items())
            L.append(f"| cka permutation floor (max) | {per} "
                     f"(void at {cf['gate']}) | "
                     f"{'PASS' if cf['valid'] else 'FAIL (cka void)'} |")
            sh = ', '.join(f"{k} {v['gap']:.2g}"
                           for k, v in cf['split_half'].items())
            L.append(f"| cka split-half gap (final pair) | {sh} | "
                     f"finite-sample floor |")
            for k, v in cf['matched_null'].items():
                L.append(f"| chance level {k} (unrelated) | rel {v['rel']:.2f}, "
                         f"cos {v['cos']:.2f}, cka {v['cka']:.2f}, "
                         f"proc {v['proc']:.2f} | read against |")
        L.append(f"| fp16 storage floor | {FP16_REL_FLOOR:.1e} relative | convention |")
    else:
        L.append(f"| log parsed | malformed lines: {g.get('malformed', 0)} | "
                 f"{'PASS' if not g.get('malformed') else 'WARN'} |")
    L.append('')

    st = out['settle']
    L.append(f"## scalars — values at key epochs; settle at tol {st['tol']}")
    L.append('')
    L.append('| scalar | e0 | e50 | e100 | e150 | e200 | settle | tail mean ± std |')
    L.append('|---|---|---|---|---|---|---|---|')
    for k in SCALAR_KEYS:
        vals = out['series']['scalars'][k]
        cells = []
        for e in (0, 50, 100, 150, 200):
            cells.append(_fmt(vals[epochs.index(e)]) if e in epochs else '-')
        s = st['scalars'][k]
        L.append(f"| {k} | {' | '.join(cells)} | {_settle_str(s)} | "
                 f"{_fmt(s.get('tail_mean'))} ± {_fmt(s.get('tail_std'))} |")
    L.append('')

    if out['mode'] == 'npz':
        L.append('## representation drift (epoch vs previous selected epoch)')
        L.append('')
        L.append('| target | rel settle | rel frozen (trailing pts) | cka settle | cka tail | proc settle |')
        L.append('|---|---|---|---|---|---|')
        for name in TARGETS:
            d = st['drift'][name]
            froz = d['rel_prev_frozen']
            froz_s = ('-' if froz is None else
                      f"{froz} ({d.get('rel_prev_frozen_tail_count', '?')}pt)")
            L.append(f"| {name} | {_settle_str(d['rel_prev'])} | "
                     f"{froz_s} | {_settle_str(d['cka_prev'])} | "
                     f"{_fmt(d['cka_prev'].get('tail_mean'), 8)} | "
                     f"{_settle_str(d['proc_prev'])} |")
        L.append('')
        L.append('drift vs the fixed refs (rel):')
        L.append('')
        L.append('| target | ' + ' | '.join(f'vs e{r}' for r in out['drift']['refs']) + ' |')
        L.append('|---|' + '---|' * len(out['drift']['refs']))
        for name in TARGETS:
            d = st['drift'][name]['rel_vs_ref_settle']
            L.append(f"| {name} settle | " + ' | '.join(
                _settle_str(d[str(r)]) for r in out['drift']['refs']) + ' |')
        L.append('')

        L.append('## per-view (final analyzed epoch) + settle of per-view rel_prev')
        L.append('')
        L.append('| ring | norm | rel_spread | PR | pair_mean | cross_mean | ratio | rel_prev settle |')
        L.append('|---|---|---|---|---|---|---|---|')
        pv = out['series']['per_view']
        pool = meta['pool_ring']
        for j in range(len(pool)):
            L.append(f"| {pool[j]} | {_fmt(pv['norm_mean'][-1][j])} | "
                     f"{_fmt(pv['rel_spread'][-1][j])} | {_fmt(pv['pr'][-1][j])} | "
                     f"{_fmt(pv['pair_mean'][-1][j])} | "
                     f"{_fmt(pv['cross_mean'][-1][j])} | {_fmt(pv['ratio'][-1][j])} | "
                     f"{_settle_str(st['per_view']['rel_prev'][j])} |")
        L.append('')

        L.append('## fused latents (M5 premise)')
        L.append('')
        L.append('| epoch | rel(n1,full) | cka(n1,full) | rel(n1,zv_az0) | '
                 'cka(n1,zv_az0) | cka(full,zv_az0) | cka(full,mean_pool) |')
        L.append('|---|---|---|---|---|---|---|')
        fu = out['fused']
        for e in out['drift']['refs']:
            if e not in epochs:
                continue
            i = epochs.index(e)
            L.append(f"| {e} | {_fmt(fu['n1_vs_full']['rel'][i])} | "
                     f"{_fmt(fu['n1_vs_full']['cka'][i])} | "
                     f"{_fmt(fu['n1_vs_zv_az0']['rel'][i])} | "
                     f"{_fmt(fu['n1_vs_zv_az0']['cka'][i])} | "
                     f"{_fmt(fu['full_vs_zv_az0']['cka'][i])} | "
                     f"{_fmt(fu['full_vs_mean_pool']['cka'][i])} |")
        L.append('')
        L.append('| fused relation | rel settle | rel e0 | rel e-final | cka e-final |')
        L.append('|---|---|---|---|---|')
        for name in FUSED_RELATIONS:
            L.append(f"| {name} | {_settle_str(st['fused'][name])} | "
                     f"{_fmt(fu[name]['rel'][0])} | {_fmt(fu[name]['rel'][-1])} | "
                     f"{_fmt(fu[name]['cka'][-1])} |")
        L.append('')

    b = out['behaviour']
    if b:
        L.append('## behaviour (from the log)')
        L.append('')
        L.append('| epoch | test/mean_score | episodes |')
        L.append('|---|---|---|')
        for e, s, n in zip(b['rollout_epochs'], b['test_mean_score'],
                           b['n_episodes']):
            L.append(f"| {e} | {s:.3f} | {n} |")
        if 'behaviour' in st:
            L.append('')
            L.append(f"behaviour settles at epoch {_settle_str(st['behaviour'])} "
                     f"(tol {st['tol']})")
        L.append('')

    if out['warnings']:
        L.append('## warnings')
        L.append('')
        for w in out['warnings']:
            L.append(f"- {w}")
        L.append('')

    L.append('## caveats')
    L.append('')
    L.append(f"- npz are fp16: differences below ~{FP16_REL_FLOOR:.1e} relative")
    L.append('  are storage, not signal; deeper collapses can only be bounded by the npz')
    L.append('- 128 fixed states, one draw per condition, single seed, no held-out views:')
    L.append('  this is an instrument, not a ladder cell')
    L.append('- zg-vs-zv distances cross two latent spaces: read trends, not absolute values')
    L.append('- the four drift measures are read together; none alone is the answer')
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-s', '--snapshots', default=None,
                    help='latent_snapshots dir (or the run dir containing it)')
    ap.add_argument('-l', '--log', default=None,
                    help='logs.json.txt or snapshots.jsonl (optional)')
    ap.add_argument('-o', '--output_dir', required=True,
                    help='writes analyze_latent_series.json + summary.md here')
    ap.add_argument('--epochs', default='all', help="'all' | 'a,b,c' | 'a:b:c'")
    ap.add_argument('--refs', default='0,50,100,150,200',
                    help='drift reference epochs (the final analyzed epoch is always added)')
    ap.add_argument('--matrix-epochs', default='0,50,100,150,200',
                    help='epochs for the per-view-pair matrix')
    ap.add_argument('--scalars-only', action='store_true',
                    help='no npz: settle/behaviour from the log alone')
    ap.add_argument('--tol', type=float, default=0.05,
                    help="settle convention: fraction of the curve's total "
                         "excursion C from its final value")
    ap.add_argument('--tail-frac', type=float, default=0.1)
    ap.add_argument('--min-tail', type=int, default=3)
    ap.add_argument('--storage-floor', type=float, default=FP16_REL_FLOOR)
    ap.add_argument('--perms', type=int, default=8)
    ap.add_argument('--cka-perm-gate', type=float, default=0.9,
                    help='CKA is declared void only if its permutation floor '
                         '(the measured chance level) is >= this')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--recheck-tol', type=float, default=5e-3)
    ap.add_argument('--min-epochs', type=int, default=3)
    ap.add_argument('--allow-nonfinite', action='store_true',
                    help='warn + skip a non-finite epoch instead of stopping')
    ap.add_argument('--strict', action='store_true',
                    help='exit 3 when a required gate fails')
    ap.add_argument('-q', '--quiet', action='store_true')
    return ap.parse_args(argv)


def run(argv=None):
    """Programmatic entry point: returns (out_dict, exit_code)."""
    args = _parse_args(argv)
    t0 = time.time()
    cfg = {'tol': args.tol, 'tail_frac': args.tail_frac,
           'min_tail': args.min_tail, 'storage_floor': args.storage_floor,
           'perms': args.perms, 'cka_perm_gate': args.cka_perm_gate,
           'seed': args.seed, 'allow_nonfinite': args.allow_nonfinite,
           'min_epochs': args.min_epochs, 'matrix_epochs': set()}

    warnings = []
    out = {'schema': SCHEMA, 'tool': 'analyze_latent_series',
           'snapshot_dir': None, 'log': args.log, 'output_dir': args.output_dir,
           'created_utc': _utc_now(),
           'mode': 'scalars' if args.scalars_only else 'npz',
           'meta': None, 'epochs': None, 'gates': {}, 'settle': None,
           'series': None, 'drift': None, 'fused': None, 'behaviour': None,
           'warnings': warnings, 'required_gate_failures': []}
    series, drift, fused = None, {'refs': [], 'targets': {}, 'per_view': {}}, {}

    log = parse_log(args.log) if args.log else None
    if args.log and log['format'] == 'unknown':
        warnings.append(f'log {args.log}: not recognised as logs.json.txt or '
                        f'snapshots.jsonl')
    if log and log['malformed']:
        warnings.append(f'{log["malformed"]} malformed line(s) in {args.log}')

    if args.scalars_only:
        if not log:
            raise SystemExit('--scalars-only needs -l/--log')
        sc = log['scalars']
        epochs = sorted(sc)
        if not epochs:
            raise SystemExit(f'no latent_probe scalars found in {args.log}')
        if len(epochs) < args.min_epochs:
            raise SystemExit(f'only {len(epochs)} epochs with scalars '
                             f'(< --min-epochs {args.min_epochs})')
        series = {'epochs': epochs,
                  'scalars': {k: [sc[e].get(k) for e in epochs]
                              for k in SCALAR_KEYS},
                  'per_view': {}, 'pair_matrix': None}
        out['gates'] = {'malformed': log['malformed'], 'n_analyzed': len(epochs),
                        'recheck': None}
    else:
        if not args.snapshots:
            raise SystemExit('need -s/--snapshots (or --scalars-only with -l)')
        snap_dir = resolve_snapshots(args.snapshots)
        out['snapshot_dir'] = str(snap_dir)
        meta = load_meta(snap_dir)
        files, holes = list_epoch_files(snap_dir)
        if len(files) < args.min_epochs:
            raise SystemExit(f'{len(files)} npz in {snap_dir} '
                             f'(< --min-epochs {args.min_epochs})')
        if meta.get('az0_pos') != meta['pool_ring'].index(meta['az0_ring']):
            raise SystemExit(f'meta az0_pos={meta.get("az0_pos")} != '
                             f'pool_ring.index(az0_ring={meta["az0_ring"]})')
        first = load_epoch(snap_dir, min(files))
        if (first['zv'].shape[1] != int(meta['K']) or
                first['zv'].shape[2] != int(meta['fused_dim'])):
            raise SystemExit(f'meta K/D {meta["K"]}/{meta["fused_dim"]} != npz '
                             f'{first["zv"].shape[1:]}')
        epochs, dropped = parse_epoch_spec(args.epochs, sorted(files))
        if dropped:
            warnings.append(f'requested epochs not on disk: {dropped}')
        if not epochs:
            raise SystemExit(f'no epochs selected (--epochs {args.epochs})')
        refs, _ = parse_epoch_spec(args.refs, sorted(files))
        cfg['matrix_epochs'] = set(parse_epoch_spec(args.matrix_epochs,
                                                    sorted(files))[0])
        series, drift, fused, gates, warns = analyze_npz(
            snap_dir, meta, epochs, refs, cfg)
        warnings.extend(warns)
        if not gates['state_idx_identical'] or gates['layout']:
            gates['halted'] = True
            warnings.append(
                'ANALYSIS NOT RUN: a fatal gate failed '
                f'(state_idx_identical={gates["state_idx_identical"]}, '
                f'layout violations={len(gates["layout"])}) -- cross-epoch '
                'comparisons would be invalid, so no series is reported')
            series, drift, fused = None, {'refs': [], 'targets': {}, 'per_view': {}}, {}
        else:
            gates['recheck'] = recheck_scalars(
                series, log['scalars'] if log else None, args.recheck_tol)
            if gates['recheck'] and gates['recheck']['status'] == 'fail':
                warnings.append(
                    f'scalar recheck FAILED: max rel diff '
                    f'{gates["recheck"]["max_rel_diff"]:.3e} > tol '
                    f'{args.recheck_tol:.0e} -- the npz and the log may not be '
                    f'the same series')
            gates['fp16_rel_floor'] = FP16_REL_FLOOR
            if len(series['epochs']) >= 2:
                gates['chance_levels'] = chance_levels(
                    epoch_arrays(snap_dir, meta, series['epochs'][-1]),
                    epoch_arrays(snap_dir, meta, series['epochs'][-2]), cfg)
            if not gates['chance_levels']['valid']:
                warnings.append(
                    f'CKA permutation floor '
                    f'{gates["chance_levels"]["max_permutation"]:.3f} >= gate '
                    f'{args.cka_perm_gate} -- CKA readings void')
        out['meta'] = meta
        out['epochs'] = {'available': len(files), 'analyzed': epochs,
                         'missing': holes}
        out['gates'] = gates

    if series is not None:
        out['series'] = series
        out['drift'] = drift
        out['fused'] = fused
        behaviour = log['behaviour'] if log else None
        out['behaviour'] = behaviour
        out['settle'] = settle_table(series, drift, fused, behaviour, cfg)

    req = []
    if args.scalars_only:
        if log and log['malformed']:
            warnings.append('malformed log lines do not fail the run; counted only')
    else:
        if not out['gates'].get('state_idx_identical', True):
            req.append('state_idx')
        if out['gates'].get('layout'):
            req.append('layout')
        if out['gates'].get('halted'):
            req.append('halted')
        rk = out['gates'].get('recheck')
        if rk and rk['status'] == 'fail':
            req.append('recheck')
        if out['gates'].get('n_analyzed', 0) < args.min_epochs:
            req.append('min_epochs')
    out['required_gate_failures'] = req

    out['cost'] = {'wall_s': _r6(time.time() - t0)}
    if series is not None:
        out['cost']['epochs_analyzed'] = len(series['epochs'])
        if out['snapshot_dir']:
            out['cost']['npz_bytes'] = int(sum(
                p.stat().st_size for p in
                pathlib.Path(out['snapshot_dir']).glob('epoch_*.npz')))

    if args.output_dir:
        od = pathlib.Path(args.output_dir)
        od.mkdir(parents=True, exist_ok=True)
        with open(od / 'analyze_latent_series.json', 'w') as f:
            json.dump(_round_obj(json.loads(json.dumps(out, default=_json_default))),
                      f, separators=(',', ':'), sort_keys=True)
        with open(od / 'summary.md', 'w') as f:
            f.write(summarize(out, cfg))
        if not args.quiet:
            print(f"wrote {od / 'analyze_latent_series.json'}")
            print(f"wrote {od / 'summary.md'}")
            if req:
                print(f"required gate failures: {req}")
    code = 3 if (args.strict and req) else 0
    return out, code


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not JSON serialisable: {type(o)}')


def _round_obj(o):
    if isinstance(o, float):
        return _r6(o)
    if isinstance(o, list):
        return [_round_obj(x) for x in o]
    if isinstance(o, dict):
        return {k: _round_obj(v) for k, v in o.items()}
    return o


def main():
    _, code = run()
    return code


if __name__ == '__main__':
    sys.exit(main())
