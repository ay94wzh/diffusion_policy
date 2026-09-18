"""
Aggregate `eval_log.json` files from `eval_novel_view.py` into the Milestone 1
deliverable: a per-viewpoint success-rate table (the degradation curve).

Usage:
python summarize_novel_view.py data/eval_az5_s42/eval_log.json data/eval_az5_s43/eval_log.json
python summarize_novel_view.py --labels s42,s43 --out table.md <dir1> <dir2>

Each argument is either a path to an eval_log.json or a directory containing
one. Viewpoints are ordered by |azimuth| (az_0 first), so the table reads as a
degradation curve rather than in logging order.
"""

import sys
import os
import re
import json
import pathlib
import statistics
import click

# test/<viewpoint>/<metric>
LOG_KEY_PATTERN = re.compile(r'^test/(?P<vp>[^/]+)/(?P<metric>mean_score|success_rate)$')
# azimuth_sweep5 names: az_0, az_p15, az_m15, az_p30, az_m30
# azimuth_sweep5 names: az_0, az_p15, az_m15, az_p30, az_m30
AZ_PATTERN = re.compile(r'^az_(?P<sign>[pm])?(?P<deg>\d+)$')
# elevation_az0 names: el_0, el_p15, el_m15
EL_PATTERN = re.compile(r'^el_(?P<sign>[pm])?(?P<deg>\d+)$')


def viewpoint_sort_key(name):
    """az_* by |angle|, then el_* by |elevation|, then anything else.

    Within a group the zero offset comes first, then increasing magnitude, and
    negatives before positives at equal magnitude. Without the el_ branch the
    elevation rows fall into the catch-all and sort alphabetically
    (el_0, el_m15, el_p15), which reads as noise rather than a sweep.
    """
    for group, pattern in enumerate((AZ_PATTERN, EL_PATTERN)):
        match = pattern.match(name)
        if match is None:
            continue
        deg = float(match.group('deg'))
        sign = match.group('sign')
        if sign is None:
            return (group, 0.0, 0, name)
        return (group, abs(deg), 0 if sign == 'm' else 1, name)
    return (2, 0.0, 0, name)


def load_log(path):
    with open(path, 'r') as f:
        return json.load(f)


def collect(logs, labels):
    """-> {(viewpoint, metric): [value per run]}"""
    table = {}
    for log, label in zip(logs, labels):
        for key, value in log.items():
            match = LOG_KEY_PATTERN.match(key)
            if match is None:
                continue
            entry = table.setdefault((match.group('vp'), match.group('metric')), {})
            entry[label] = value
    return table


def fmt(value):
    return 'n/a' if value is None else f'{value:.3f}'


def mean_spread(values):
    """mean +/- population std over the runs that reported a value."""
    present = [v for v in values if v is not None]
    if len(present) == 0:
        return 'n/a'
    if len(present) == 1:
        return f'{present[0]:.3f}'
    return f'{statistics.mean(present):.3f} ± {statistics.pstdev(present):.3f}'


def render(table, labels, metric):
    viewpoints = sorted({vp for vp, m in table if m == metric}, key=viewpoint_sort_key)
    if len(viewpoints) == 0:
        return f'_(no `{metric}` entries found)_\n'
    header = ['viewpoint'] + labels + ['mean ± std']
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '---|' * len(header)]
    for vp in viewpoints:
        entry = table.get((vp, metric), {})
        cells = [vp] + [fmt(entry.get(label)) for label in labels]
        cells.append(mean_spread([entry.get(label) for label in labels]))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


@click.command()
@click.argument('paths', nargs=-1, required=True)
@click.option('--labels', default=None,
              help='Comma-separated labels, one per path (default: parent dir names).')
@click.option('--out', default=None, help='Also write the markdown table to this file.')
def main(paths, labels, out):
    resolved = []
    for path in paths:
        p = pathlib.Path(path)
        candidate = p / 'eval_log.json' if p.is_dir() else p
        if not candidate.exists():
            raise click.ClickException(f'no eval_log.json at {candidate}')
        resolved.append(candidate)

    if labels is not None:
        label_list = [s.strip() for s in labels.split(',')]
        if len(label_list) != len(resolved):
            raise click.ClickException(
                f'--labels has {len(label_list)} entries but {len(resolved)} paths were given')
    else:
        label_list = [p.parent.name for p in resolved]

    logs = [load_log(p) for p in resolved]
    table = collect(logs, label_list)

    blocks = ['# Novel-view evaluation summary\n',
              f'Runs: {", ".join(label_list)}\n']
    for metric in ('success_rate', 'mean_score'):
        blocks.append(f'\n## {metric}\n')
        blocks.append(render(table, label_list, metric))

    report = '\n'.join(blocks)
    click.echo(report)
    if out is not None:
        pathlib.Path(out).write_text(report)
        click.echo(f'wrote {out}')


if __name__ == '__main__':
    main()
