#!/bin/bash
# Sweep the clear run's checkpoint at HELD-OUT viewpoints: the 7.5-degree-offset
# ring (azimuth_offgrid) plus elevation for cross-cell comparability.
#
# Why: the clear run trained on every 15-degree pose -75..+75 (pool ring 1..11),
# so every azimuth_offgrid viewpoint is one it never saw -- the pure held-out
# test. Its in-distribution ring is already committed (data/eval_interp_square_m4clear,
# mean 0.747 at the same epoch-200 latest.ckpt).
#
# Registered expectation before the run (PROGRESS.md style): the midpoints should
# land near the mean of their neighbouring trained poses (ring neighbours sit at
# 0.70-0.84, so an offgrid mean of roughly 0.6-0.8 over +-7.5..+-67.5 means the
# encoder interpolates); +-82.5/+-90 may degrade (M2's low-value edge); an
# offgrid mean below ~0.4 would be real view-tiedness at unvisited poses.
#
# Runtime on miroc-server: stage 2 ~60 min (14 viewpoints x 50 paired episodes),
# stage 3 ~12 min. Outputs land in this repo's data/ (auto-tracked by the
# !data/eval_*/** gitignore rule), so the record travels by git after the run.
#
# Run to completion with:  setsid bash -c 'bash data/clear_heldout_eval.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
RUN=/data/zihan/runs/run_square_m4clear_s42_200ep
CKPT=$RUN/checkpoints/latest.ckpt
OUT=data/eval_heldout_square_m4clear
OUT_EL=data/eval_el_square_m4clear
OUT_SMOKE=data/eval_offgrid_smoke
GPU=${GPU:-cuda:0}
FORCE=${FORCE:-0}
LOG=data/clear_heldout_eval.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gate -- the preset exists, expands to exactly the 14 wanted
# viewpoints, and the summarizer's sort key orders fractional-degree names.
# A guard must be able to fail: every check below prints its evidence.
say "CPU gate: preset expansion + sort key"
if ! "$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import eval_novel_view as env
import summarize_novel_view as snv

vp = env.PRESETS['azimuth_offgrid']['viewpoints']
names = [v['name'] for v in vp]
azs = sorted(v.get('azimuth_deg', 0.0) for v in vp)
assert len(names) == len(set(names)) == 14, names
want = [-90, -82.5, -67.5, -52.5, -37.5, -22.5, -7.5,
        7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 90]
assert azs == want, azs
order = sorted(names, key=snv.viewpoint_sort_key)
mags = [snv.viewpoint_sort_key(n)[1] for n in order]
assert all(b >= a for a, b in zip(mags, mags[1:])), order
assert order[0] == 'az_m7.5' and order[1] == 'az_p7.5', order
assert snv.viewpoint_sort_key('az_p7.5')[:2] == (0, 7.5)
print(f'GATE_PASS: {len(names)} viewpoints, |az| order {order[:3]} ... {order[-2:]}')
PYEOF
then
    say "CPU_GATE_FAILED -- see the log tail above"; exit 1
fi
say "CPU_GATE_PASSED"

# ---- stage 1: smoke -- 4 episodes x 14 viewpoints before the 60-min sweep.
if [ -f "$OUT_SMOKE/eval_log.json" ]; then
    say "SMOKE_SKIPPED (already present)"
else
    say "SMOKE_START ($OUT_SMOKE)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_SMOKE" -d "$GPU" \
        --preset azimuth_offgrid --m3-slots 11 --eef-hist-steps 4 \
        --n-test 4 --n-envs 4 --n-test-vis 0 < /dev/null >> data/sweep_offgrid_smoke.log 2>&1
    rc=$?
    say "SMOKE_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "SMOKE_FAILED -- see data/sweep_offgrid_smoke.log"; exit 1; fi
    if ! "$PY" - "$OUT_SMOKE/eval_log.json" >> "$LOG" 2>&1 <<'PYEOF'
import collections, json, sys

log = json.load(open(sys.argv[1]))
vp_of = {k.split('/')[1] for k in log if k.startswith('test/')}
n_succ = sum(1 for k in log if k.startswith('test/') and k.endswith('/success_rate'))
cnt = collections.Counter(
    k.split('/')[1] for k in log
    if k.startswith('test/') and k.rsplit('/', 1)[1].startswith('success_')
    and k.rsplit('/', 1)[1][len('success_'):].isdigit())
assert len(vp_of) == 14 and n_succ == 14, (len(vp_of), n_succ)
assert set(cnt.values()) == {4}, dict(cnt)
print(f'SMOKE_GATE_PASS: {len(vp_of)} viewpoints x 4 episodes')
PYEOF
    then
        say "SMOKE_LAYOUT_FAILED"; exit 1
    fi
    say "SMOKE_GATE_PASSED"
fi

# ---- stage 2: the held-out sweep (14 viewpoints x 50 paired episodes).
if [ -f "$OUT/eval_log.json" ] && [ "$FORCE" != "1" ]; then
    say "SWEEP_SKIPPED (already present; FORCE=1 to redo)"
else
    [ -e "$CKPT" ] || { say "MISSING_CKPT $CKPT"; exit 1; }
    say "SWEEP_START ($GPU, azimuth_offgrid, 14 viewpoints x 50 paired episodes)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT" -d "$GPU" \
        --preset azimuth_offgrid --m3-slots 11 --eef-hist-steps 4 \
        --n-envs 14 --n-test-vis 0 < /dev/null >> data/sweep_square_m4clear_offgrid.log 2>&1
    rc=$?
    say "SWEEP_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "SWEEP_FAILED -- see data/sweep_square_m4clear_offgrid.log"; exit 1; fi
fi
"$PY" summarize_novel_view.py "$OUT" --out data/eval_heldout_square_m4clear_summary.txt \
    >> data/sweep_square_m4clear_offgrid.log 2>&1
say "HELDOUT_SUMMARY -> data/eval_heldout_square_m4clear_summary.txt"

# ---- stage 3: elevation (el_0 / +-15) -- the columns the M3/M4 cells carry.
if [ -f "$OUT_EL/eval_log.json" ] && [ "$FORCE" != "1" ]; then
    say "ELEV_SKIPPED (already present; FORCE=1 to redo)"
else
    say "ELEV_START ($GPU, elevation_az0, 3 viewpoints x 50 paired episodes)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_EL" -d "$GPU" \
        --preset elevation_az0 --m3-slots 11 --eef-hist-steps 4 \
        --n-envs 14 --n-test-vis 0 < /dev/null >> data/sweep_square_m4clear_el.log 2>&1
    rc=$?
    say "ELEV_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "ELEV_FAILED -- see data/sweep_square_m4clear_el.log"; exit 1; fi
fi
"$PY" summarize_novel_view.py "$OUT_EL" --out data/eval_el_square_m4clear_summary.txt \
    >> data/sweep_square_m4clear_el.log 2>&1
say "ELEV_SUMMARY -> data/eval_el_square_m4clear_summary.txt"
say "DONE"
