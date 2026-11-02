"""Registered means for the B1 residual-token A/B/C/D screen.

Reads the three `eval_log.json` files produced by `eval_novel_view.py` for each
cell and reports the plan doc's registered means using **strict success_rate**:

    seen   -- all 11 `azimuth_interp` viewpoints (trained 15-degree ring)
    interp -- the 10 off-grid viewpoints at |az| in
              {7.5, 22.5, 37.5, 52.5, 67.5}
    past   -- the 4 out-of-range viewpoints at |az| in {82.5, 90}
    el     -- el_0, el_m15, el_p15; also the m15/p15 mean and el_0 alone

It does not merge out-of-range into interpolation (the working-tree plan doc
requires them separate), and it does not use `mean_score`.

Usage:
    python summarize_enc_rt.py \
        --cell a=data/eval_interp_square_enc_rt_a,data/eval_heldout_square_enc_rt_a,data/eval_el_square_enc_rt_a \
        --cell b=... --cell c=... --cell d=...

Each cell value is three comma-separated paths (or directories containing
`eval_log.json`) in interp,heldout,el order. `--selftest` runs the split logic
on synthetic logs and fails if the definition drifts.
"""
import json
import os
import pathlib
import re
import statistics
import tempfile

import click

AZ_RE = re.compile(r'^az_(?P<sign>[pm])?(?P<deg>\d+(?:\.\d+)?)$')
EL_RE = re.compile(r'^el_(?P<sign>[pm])?(?P<deg>\d+)$')
INTERP_DEG = {7.5, 22.5, 37.5, 52.5, 67.5}
PAST_DEG = {82.5, 90.0}


def _log_path(value):
    path = pathlib.Path(value)
    if path.is_dir():
        path = path / 'eval_log.json'
    if not path.is_file():
        raise click.BadParameter(f'no eval_log.json at {value!r}')
    return path


def _load(value):
    with open(_log_path(value), 'r') as f:
        return json.load(f)


def _viewpoints(log):
    return {key.split('/')[1] for key in log
            if key.startswith('test/') and key.endswith('/success_rate')}


def _success(log, vp):
    key = f'test/{vp}/success_rate'
    if key not in log:
        raise KeyError(f'missing {key}')
    return float(log[key])


def _check_episodes(log, vp, expect):
    """Loud sanity check: one per-seed success entry per evaluated episode."""
    prefix = f'test/{vp}/success_'
    rate_key = f'test/{vp}/success_rate'
    n = sum(1 for key in log
            if key.startswith(prefix) and key != rate_key)
    if n != expect:
        raise RuntimeError(
            f'{vp}: {n} per-seed success entries, expected {expect}; the '
            'episode count or the log schema changed')


def _az_deg(vp):
    match = AZ_RE.match(vp)
    if match is None:
        return None
    deg = float(match.group('deg'))
    return -deg if match.group('sign') == 'm' else deg


def _el_deg(vp):
    match = EL_RE.match(vp)
    if match is None:
        return None
    deg = float(match.group('deg'))
    return -deg if match.group('sign') == 'm' else deg


def _mean(values, name):
    if len(values) == 0:
        raise RuntimeError(f'no viewpoints for {name}')
    return statistics.mean(values)


def _cell_means(interp, heldout, el, expect_episodes):
    seen_vps = sorted(_viewpoints(interp))
    for vp in seen_vps:
        _check_episodes(interp, vp, expect_episodes)
    held_vps = sorted(_viewpoints(heldout))
    for vp in held_vps:
        _check_episodes(heldout, vp, expect_episodes)
    el_vps = sorted(_viewpoints(el))
    for vp in el_vps:
        _check_episodes(el, vp, expect_episodes)

    seen = _mean([_success(interp, vp) for vp in seen_vps], 'seen')
    interp_vals = []
    past_vals = []
    for vp in held_vps:
        deg = _az_deg(vp)
        if deg is None:
            continue
        if abs(deg) in INTERP_DEG:
            interp_vals.append(_success(heldout, vp))
        elif abs(deg) in PAST_DEG:
            past_vals.append(_success(heldout, vp))
    if len(interp_vals) != 10:
        raise RuntimeError(
            f'interp split has {len(interp_vals)} viewpoints, expected 10: '
            f'{[vp for vp in held_vps if _az_deg(vp) is not None]}')
    if len(past_vals) != 4:
        raise RuntimeError(
            f'past split has {len(past_vals)} viewpoints, expected 4')

    el_map = {}
    for vp in el_vps:
        deg = _el_deg(vp)
        if deg is not None:
            el_map[deg] = _success(el, vp)
    if set(el_map) != {0.0, -15.0, 15.0}:
        raise RuntimeError(f'elevation viewpoints are {sorted(el_map)}, '
                           'expected el_0/el_m15/el_p15')
    el_pm = statistics.mean([el_map[-15.0], el_map[15.0]])
    el_all = statistics.mean([el_map[0.0], el_map[-15.0], el_map[15.0]])
    return dict(
        seen=seen,
        interp=_mean(interp_vals, 'interp'),
        past=_mean(past_vals, 'past'),
        el_0=el_map[0.0],
        el_m15=el_map[-15.0],
        el_p15=el_map[15.0],
        el_pm=el_pm,
        el_all=el_all,
        seen_vps=seen_vps,
        interp_vps=[vp for vp in held_vps
                    if _az_deg(vp) is not None and abs(_az_deg(vp)) in INTERP_DEG],
        past_vps=[vp for vp in held_vps
                  if _az_deg(vp) is not None and abs(_az_deg(vp)) in PAST_DEG],
    )


def _fmt(value):
    return f'{value:.3f}'


def _delta(value, ref):
    return f'{value - ref:+.3f}'


def _print_report(cells):
    labels = list(cells)
    ref_label = 'a' if 'a' in cells else labels[0]
    ref = cells[ref_label]
    for label in labels[1:]:
        for group in ('seen_vps', 'interp_vps', 'past_vps'):
            assert cells[label][group] == ref[group], (
                f'{label} {group} differs from {ref_label}: '
                f'{cells[label][group]} vs {ref[group]}')
    keys = [('seen', 'seen_11'), ('interp', 'interp_10'),
            ('past', 'past_4'), ('el_0', 'el_0'),
            ('el_m15', 'el_m15'), ('el_p15', 'el_p15'),
            ('el_pm', 'el_pm_mean'), ('el_all', 'el_all_mean')]
    header = 'cell  ' + '  '.join(name for _, name in keys)
    print(header)
    for label in labels:
        row = [label] + [_fmt(cells[label][key]) for key, _ in keys]
        print('  '.join(row))
    print(f'\ndelta vs {ref_label}:')
    print(header)
    for label in labels:
        row = [label] + [_delta(cells[label][key], ref[key]) for key, _ in keys]
        print('  '.join(row))

    # Per-viewpoint table, all cells side by side.
    all_vps = []
    seen = cells[labels[0]]['seen_vps']
    interp = cells[labels[0]]['interp_vps']
    past = cells[labels[0]]['past_vps']
    for group, vps in (('seen', seen), ('interp', interp), ('past', past)):
        for vp in vps:
            all_vps.append((group, vp))
    print('\nper-viewpoint success_rate:')
    print('group   viewpoint  ' + '  '.join(labels))
    for group, vp in all_vps:
        # interp/past live in the heldout log; seen lives in the interp log
        row = [group, vp]
        for label in labels:
            cell = cells[label]
            log_key = 'seen' if group == 'seen' else 'heldout'
            log = cell.get('_logs', {}).get(log_key)
            if log is None:
                row.append('n/a')
            else:
                row.append(_fmt(_success(log, vp)))
        print('  '.join(row))
    print('\nelevation success_rate:')
    print('viewpoint  ' + '  '.join(labels))
    for vp in ('el_0', 'el_m15', 'el_p15'):
        row = [vp]
        for label in labels:
            row.append(_fmt(_success(cells[label]['_logs']['el'], vp)))
        print('  '.join(row))


def _selftest():
    """Synthetic logs with known values; fail if the split/mean logic drifts."""
    with tempfile.TemporaryDirectory() as tmp:
        def make(path, vps, value, episodes=2):
            log = {}
            for i, vp in enumerate(vps):
                log[f'test/{vp}/success_rate'] = value + 0.01 * i
                for seed in range(episodes):
                    log[f'test/{vp}/success_{seed}'] = 1.0
                log[f'test/{vp}/sim_max_reward_{seed}'] = 1.0
            with open(path, 'w') as f:
                json.dump(log, f)

        seen = ['az_0', 'az_p15']
        offgrid = ['az_m7.5', 'az_p7.5', 'az_m22.5', 'az_p22.5',
                   'az_m37.5', 'az_p37.5', 'az_m52.5', 'az_p52.5',
                   'az_m67.5', 'az_p67.5',
                   'az_m82.5', 'az_p82.5', 'az_m90', 'az_p90']
        # Need exactly 11 seen and 10 interp; add the rest of the ring.
        seen += ['az_m15', 'az_m30', 'az_p30', 'az_m45', 'az_p45',
                 'az_m60', 'az_p60', 'az_m75', 'az_p75']
        make(os.path.join(tmp, 'interp.json'), seen, 0.5, episodes=2)
        make(os.path.join(tmp, 'heldout.json'), offgrid, 0.25, episodes=2)
        make(os.path.join(tmp, 'el.json'), ['el_0', 'el_m15', 'el_p15'],
             0.75, episodes=2)
        means = _cell_means(
            _load(os.path.join(tmp, 'interp.json')),
            _load(os.path.join(tmp, 'heldout.json')),
            _load(os.path.join(tmp, 'el.json')),
            expect_episodes=2)
        # interp names are the first 10 off-grid entries (indices 0..9);
        # past is entries 10..13.
        assert abs(means['interp'] - (0.25 + 0.01 * 4.5)) < 1e-12, means
        assert abs(means['past'] - (0.25 + 0.01 * 11.5)) < 1e-12, means
        assert abs(means['el_0'] - 0.75) < 1e-12
        assert abs(means['el_pm'] - (0.75 + 0.01 * 1.5)) < 1e-12
        print('SELFTEST PASS (seen/interp/past/elevation splits and means)')


@click.command()
@click.option('--cell', multiple=True,
              help='label=interp,heldout,el paths (each file or directory)')
@click.option('--expect-episodes', type=int, default=50,
              show_default=True, help='per-seed success entries per viewpoint')
@click.option('--selftest', is_flag=True,
              help='run a synthetic split check and exit')
def main(cell, expect_episodes, selftest):
    if selftest:
        _selftest()
        return
    if not cell:
        raise click.UsageError('pass at least one --cell label=interp,heldout,el')
    cells = {}
    for spec in cell:
        if '=' not in spec:
            raise click.BadParameter(f'--cell must be label=paths, got {spec!r}')
        label, paths = spec.split('=', 1)
        parts = [p for p in paths.split(',') if p]
        if len(parts) != 3:
            raise click.BadParameter(
                f'--cell {label}: expected interp,heldout,el (3 paths), '
                f'got {len(parts)}')
        interp, heldout, el = (_load(p) for p in parts)
        means = _cell_means(interp, heldout, el, expect_episodes)
        means['_logs'] = {'seen': interp, 'heldout': heldout, 'el': el}
        cells[label] = means
    _print_report(cells)


if __name__ == '__main__':
    main()
