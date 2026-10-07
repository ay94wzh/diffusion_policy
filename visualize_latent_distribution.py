"""
Latent-distribution visualization for the per-epoch latent probe (first built
for the clear run, `m4_aux_image_abs_multiview_az75`: K=11, pool ring 1..11).

Two modes, because the raw npz never leave miroc-server and the laptop is where
figures get made:

    compute   numpy only -- runs where the npz live (miroc-server).
              Reads a few epoch_XXXX.npz, fits ONE deterministic 2-D PCA basis
              on the basis epoch's z_v (flattened (S*K, D), row = state*K +
              canonical slot), projects every selected epoch's z_v, z_g and
              z_g_n1 onto it, and writes a small coordinates digest
              (data/analysis_clear/latent_viz_coords.json -- committed).

    plot      matplotlib -- runs on the laptop from the committed JSONs.
              Renders, into --out-dir:
                fig1  z_v PCA small multiples, colored by ring azimuth, with
                      per-view centroids (view structure; when it settles);
                fig2  z_g vs z_g_n1 overlay, first and last selected epoch
                      (the ~0.5% N-invariance story);
                fig3  per-view curves over the whole 201-epoch series
                      (needs the analyze_latent_series digest via -d);
                fig4  pairwise view-distance heatmaps (same digest).

Determinism: the SVD sign convention makes the largest-magnitude loading of
each component positive, so re-running compute reproduces identical
coordinates. The basis travels in the JSON, so any later latent can be
projected into the same frame:

    x_2d = (x - basis.mean) @ basis.v.T

Palette: the project's validated light-mode chart palette (blue/orange
categorical pair validated 2026-10-07: CVD dE 24.7, normal dE 33.6, both PASS;
diverging = blue <-> gray <-> red; sequential = the blue ramp).
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

import analyze_latent_series as als  # numpy-only; the npz schema owner

SCHEMA = 1
TOOL = 'visualize_latent_distribution.py'

DEFAULT_EPOCHS = '0,25,50,100,150,200'

# palette (light mode; see the module docstring)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
BLUE = '#2a78d6'      # categorical slot 1
ORANGE = '#eb6834'    # categorical slot 2
DIV_ANCHORS = [BLUE, '#f0efec', '#e34948']        # -az ... 0 ... +az
SEQ_ANCHORS = ['#cde2fb', '#0d366b']              # the blue ramp, light -> dark


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def pca_basis(x, n_components=2):
    """Deterministic PCA: returns (mean, v (n,D), evr (n,))."""
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    v = vt[:n_components].copy()
    for i in range(v.shape[0]):
        j = int(np.argmax(np.abs(v[i])))
        if v[i, j] < 0:                      # fixed sign convention
            v[i] *= -1.0
    var = s ** 2
    evr = var[:n_components] / max(float(var.sum()), 1e-30)
    return mean, v, evr


def _project(x, mean, v):
    return (x - mean) @ v.T


def compute(snapshots, out_path, epochs=DEFAULT_EPOCHS, basis_epoch=200):
    """Read the npz, write the coordinates digest. Raises SystemExit on a
    failed gate (the guard must be able to fail, loudly)."""
    snap_dir = als.resolve_snapshots(snapshots)
    meta = als.load_meta(snap_dir)
    files, missing = als.list_epoch_files(snap_dir)
    if not files:
        raise SystemExit(f'no epoch_*.npz under {snap_dir}')
    selected, dropped = als.parse_epoch_spec(epochs, files)
    if dropped:
        raise SystemExit(f'requested epochs not on disk: {dropped}')
    if not selected:
        raise SystemExit('no epochs selected')
    if int(basis_epoch) not in files:
        raise SystemExit(f'basis epoch {basis_epoch} not on disk')
    if missing:
        print(f'note: gap epochs inside the span (not read): {missing}')

    S, K, D = int(meta['n_states']), int(meta['K']), int(meta['fused_dim'])
    pool = [int(v) for v in meta['pool_ring']]
    if len(pool) != K:
        raise SystemExit(f'pool_ring has {len(pool)} entries, K={K}')

    # gates: shapes/finiteness/state_idx identity across every epoch read.
    first_state_idx = None
    loaded = {}
    for e in sorted(set(list(selected) + [int(basis_epoch)])):
        entry = als.load_epoch(snap_dir, e)
        bad = als.check_arrays(entry, meta, first_state_idx)
        if bad:
            raise SystemExit(f'epoch {e}: array gate FAILED: {bad}')
        if first_state_idx is None:
            first_state_idx = entry['state_idx']
        loaded[e] = entry

    # basis from the basis epoch's z_v; row order = state-major (s*K + slot),
    # the same convention the plot side reshapes with.
    flat = loaded[int(basis_epoch)]['zv'].reshape(S * K, D)
    mean, v, evr = pca_basis(flat)

    epochs_out = []
    for e in selected:
        ent = loaded[e]
        epochs_out.append({
            'epoch': int(e),
            'zv_xy': np.round(_project(ent['zv'].reshape(S * K, D),
                                       mean, v), 6).tolist(),
            'zg_xy': np.round(_project(ent['zg'], mean, v), 6).tolist(),
            'zg_n1_xy': np.round(_project(ent['zg_n1'], mean, v), 6).tolist(),
        })

    doc = {
        'schema': SCHEMA,
        'tool': TOOL,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'snapshots': str(snap_dir),
        'S': S, 'K': K, 'D': D,
        'basis_epoch': int(basis_epoch),
        'state_idx': [int(x) for x in first_state_idx],
        'pool_ring': pool,
        'azimuth_deg': [-90 + 15 * i for i in pool],
        'az0_pos': int(meta.get('az0_pos', -1)),
        'basis': {'mean': np.round(mean, 6).tolist(),
                  'v': np.round(v, 6).tolist(),
                  'evr': np.round(np.asarray(evr), 6).tolist()},
        'epochs': epochs_out,
    }
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(doc, f, separators=(',', ':'))
    return doc, out


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
def plot(coords_path, out_dir, digest_path=None, dpi=170):
    """Render the figures from the committed JSONs; returns the written paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
    })
    div = LinearSegmentedColormap.from_list('vld_div', DIV_ANCHORS)
    seq = LinearSegmentedColormap.from_list('vld_seq', SEQ_ANCHORS)

    def style_axes(ax):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=8)

    doc = json.load(open(coords_path))
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    S, K = int(doc['S']), int(doc['K'])
    az = np.asarray(doc['azimuth_deg'], float)
    norm = Normalize(az.min(), az.max())
    cols = [div(norm(a)) for a in az]
    epochs = doc['epochs']

    # ---- fig1: z_v PCA small multiples -----------------------------------
    ncol = min(3, len(epochs))
    nrow = int(np.ceil(len(epochs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow),
                             dpi=dpi, constrained_layout=True, squeeze=False)
    for ax, ep in zip(axes.flat, epochs):
        pts = np.asarray(ep['zv_xy']).reshape(S, K, 2)
        for j in range(K):
            ax.scatter(pts[:, j, 0], pts[:, j, 1], s=12, color=cols[j],
                       edgecolors=SURFACE, linewidths=0.4, alpha=0.85)
        cen = pts.mean(axis=0)
        ax.scatter(cen[:, 0], cen[:, 1], s=60, facecolors=cols,
                   edgecolors=INK, linewidths=0.8, zorder=5)
        style_axes(ax)
        ax.set_title(f"epoch {ep['epoch']}", color=INK)
    for ax in axes.flat[len(epochs):]:
        ax.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=div)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75,
                        ticks=list(az[::2]) + [az[-1]])
    cbar.set_label('ring azimuth (deg)', color=INK2)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BASELINE)
    fig.suptitle(f'z_v PCA (basis fit on epoch {doc["basis_epoch"]}) — dots: '
                 f'{S} probe states x {K} views; ring: per-view centroid',
                 color=INK)
    p = out_dir / 'fig1_zv_view_structure.png'
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)

    # ---- fig2: z_g vs z_g_n1 ---------------------------------------------
    sel = [epochs[0]] if len(epochs) == 1 else [epochs[0], epochs[-1]]
    fig, axes = plt.subplots(1, len(sel), figsize=(5.4 * len(sel), 5.0),
                             dpi=dpi, constrained_layout=True, squeeze=False)
    for ax, ep in zip(axes.flat, sel):
        zg = np.asarray(ep['zg_xy'])
        zn = np.asarray(ep['zg_n1_xy'])
        ax.scatter(zg[:, 0], zg[:, 1], s=30, color=BLUE, alpha=0.9,
                   edgecolors=SURFACE, linewidths=0.6, zorder=3,
                   label='z_g (all 11 views)')
        ax.scatter(zn[:, 0], zn[:, 1], s=30, facecolors='none',
                   edgecolors=ORANGE, linewidths=1.1, zorder=4,
                   label='z_g (N=1, az_0)')
        rel = np.linalg.norm(zn - zg, axis=1) / np.maximum(
            0.5 * (np.linalg.norm(zg, axis=1) + np.linalg.norm(zn, axis=1)),
            1e-12)
        cos = 1.0 - np.einsum('ij,ij->i', zg, zn) / np.maximum(
            np.linalg.norm(zg, axis=1) * np.linalg.norm(zn, axis=1), 1e-12)
        ax.text(0.03, 0.97, f'mean cos-dist {100 * cos.mean():.2f}%  ·  '
                            f'rel |dz|/|z| {100 * rel.mean():.1f}%',
                transform=ax.transAxes, va='top', color=INK2)
        style_axes(ax)
        ax.set_title(f"epoch {ep['epoch']}", color=INK)
    axes.flat[0].legend(loc='lower right', frameon=False, fontsize=8,
                        labelcolor=INK2)
    fig.suptitle('fused latent z_g: 11 views vs N=1 inference — the two '
                 'coincide to ~0.5%', color=INK)
    p = out_dir / 'fig2_zg_n1_invariance.png'
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)

    # ---- fig3 + fig4: the per-view series over all epochs -----------------
    if digest_path is not None:
        dg = json.load(open(digest_path))
        series = dg['series']
        pool = [int(v) for v in dg['meta']['pool_ring']]
        az_all = np.asarray([-90 + 15 * i for i in pool], float)
        cols_all = [div(norm(a)) for a in az_all]

        pv = series['per_view']
        epochs_all = series['epochs']
        panels = [('norm_mean', 'mean ‖z_v‖'),
                  ('rel_spread', 'relative spread'),
                  ('ratio', 'pair / cross ratio')]
        fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 8.0),
                                 sharex=True, dpi=dpi, squeeze=False)
        for ax, (key, ylab) in zip(axes.flat, panels):
            arr = np.asarray(pv[key])           # (n_epochs, K)
            for j in range(arr.shape[1]):
                ax.plot(epochs_all, arr[:, j], color=cols_all[j], lw=1.6,
                        alpha=0.9)
            style_axes(ax)
            ax.set_ylabel(ylab, color=INK2)
            if key == 'ratio':
                ax.axhline(1.0, color=BASELINE, lw=0.8, ls=(0, (4, 3)))
        # direct labels on the two extremes only -- a third mid label collided
        # with them in the smoke render (and the diverging colors identify the
        # ring order everywhere else). Label text comes from the slot's own
        # azimuth, not a hardcoded ring.
        for j in (0, K - 1):
            arr0 = np.asarray(pv[panels[0][0]])
            axes.flat[0].annotate(f'{az_all[j]:+g}°',
                                  xy=(epochs_all[-1], arr0[-1, j]),
                                  xytext=(4, 0), textcoords='offset points',
                                  color=INK2, fontsize=8, va='center')
        axes.flat[-1].set_xlabel('epoch', color=INK2)
        fig.suptitle('per-view z_v structure over training (11 ring views)',
                     color=INK)
        p = out_dir / 'fig3_per_view_curves.png'
        fig.savefig(p, facecolor=SURFACE)
        plt.close(fig)
        paths.append(p)

        pm = series['pair_matrix']
        order = pm['order']
        vals = np.asarray(pm['values'])         # (n_pm_epochs, n_pairs)
        Kk = len(pool)
        mats = []
        for row in vals:
            m = np.full((Kk, Kk), np.nan)
            for (i, j), val in zip(order, row):
                m[i, j] = m[j, i] = val
            mats.append(m)
        vmax = float(np.nanmax(vals))
        ncol4 = min(5, len(mats))
        nrow4 = int(np.ceil(len(mats) / ncol4))
        fig, axes = plt.subplots(nrow4, ncol4, figsize=(2.9 * ncol4, 3.3 * nrow4),
                                 dpi=dpi, constrained_layout=True,
                                 squeeze=False)
        ticks = list(range(Kk))
        labels = [f'{a:g}' for a in az_all]
        for ax, m, e in zip(axes.flat, mats, pm['epochs']):
            im = ax.imshow(m, cmap=seq, vmin=0.0, vmax=vmax)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, fontsize=7)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_title(f"epoch {e}", color=INK)
            ax.tick_params(colors=MUTED)
            for side in ('top', 'right', 'left', 'bottom'):
                ax.spines[side].set_visible(False)
        for ax in axes.flat[len(mats):]:
            ax.set_visible(False)
        cbar = fig.colorbar(im, ax=axes, shrink=0.8)
        cbar.set_label('mean rel. distance between view pairs', color=INK2)
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
        cbar.outline.set_edgecolor(BASELINE)
        fig.suptitle('pairwise z_v distance across the ring (axis labels: '
                     'azimuth deg)', color=INK)
        p = out_dir / 'fig4_pair_matrix.png'
        fig.savefig(p, facecolor=SURFACE)
        plt.close(fig)
        paths.append(p)
    else:
        print('note: no --digest given; fig3/fig4 skipped (loudly, not silently)')

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(argv=None):
    ap = argparse.ArgumentParser(
        description='latent-distribution digest (compute) and figures (plot)')
    sub = ap.add_subparsers(dest='mode', required=True)

    c = sub.add_parser('compute', help='npz -> coordinates JSON (numpy only)')
    c.add_argument('-s', '--snapshots', required=True,
                   help='latent_snapshots dir, or the run dir containing it')
    c.add_argument('-o', '--out', required=True)
    c.add_argument('--epochs', default=DEFAULT_EPOCHS)
    c.add_argument('--basis-epoch', type=int, default=200)

    p = sub.add_parser('plot', help='coordinates JSON -> figures (matplotlib)')
    p.add_argument('-i', '--coords', required=True)
    p.add_argument('-o', '--out-dir', required=True)
    p.add_argument('-d', '--digest', default=None,
                   help='analyze_latent_series.json (enables fig3/fig4)')
    p.add_argument('--dpi', type=int, default=170)

    args = ap.parse_args(argv)
    if args.mode == 'compute':
        doc, out = compute(args.snapshots, args.out, args.epochs,
                           args.basis_epoch)
        evr = doc['basis']['evr']
        print(f"compute: epochs={[e['epoch'] for e in doc['epochs']]} "
              f"S={doc['S']} K={doc['K']} D={doc['D']} "
              f"evr=({evr[0]:.3f}, {evr[1]:.3f})")
        print(f'-> {out} ({out.stat().st_size / 1024:.0f} KB)')
        return 0
    paths = plot(args.coords, args.out_dir, args.digest, args.dpi)
    for q in paths:
        print(f'-> {q} ({q.stat().st_size / 1024:.0f} KB)')
    return 0


if __name__ == '__main__':
    sys.exit(run())
