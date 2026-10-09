#!/bin/bash
# Sweep the zgpose checkpoint (PLAN candidate 2: z_g pose-set supervision on
# the clear-run recipe):
#   1. azimuth_interp  -- the in-distribution ring (all 11 poses trained here;
#      the trained-reference curve; m4base reads 0.796, m4clear 0.747);
#   2. azimuth_offgrid -- the pure held-out ring (7.5-degree offsets + +-90,
#      none trained; m4base 0.774 / m4clear 0.716 over +-7.5..+-67.5);
#      smoke first;
#   3. elevation_az0   -- the conditioning-route columns (m4base reads
#      0.840 / 0 / 0; m4clear 0.700 / 0 / 0).
# Runtime reference (clear run, GPU shared, 2026-10-07): smoke ~15 min,
# offgrid 77 min, interp ~46 min, elevation ~12-17 min.
#
# The comparison tables come from summarize_novel_view.py against the
# committed data/eval_{interp,heldout,el}_square_m4{base,clear}/ dirs; see
# data/zgpose_run.sh for the pre-registration this is read against.
#
# Run to completion with:  setsid bash -c 'bash data/zgpose_eval.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
RUN=/data/zihan/runs/run_square_zgpose_s42_200ep
CKPT=$RUN/checkpoints/latest.ckpt
OUT_INTERP=data/eval_interp_square_zgpose
OUT_HELDOUT=data/eval_heldout_square_zgpose
OUT_EL=data/eval_el_square_zgpose
OUT_SMOKE=data/eval_offgrid_smoke_zgpose
GPU=${GPU:-cuda:0}
FORCE=${FORCE:-0}
LOG=data/zgpose_eval.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gate -- the three presets expand to the wanted viewpoint
# counts with unique names. A guard must be able to fail: it prints evidence.
say "CPU gate: preset expansion"
if ! "$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import eval_novel_view as env

want = {'azimuth_interp': 11, 'azimuth_offgrid': 14, 'elevation_az0': 3}
for name, n in want.items():
    vp = env.PRESETS[name]['viewpoints']
    names = [v['name'] for v in vp]
    assert len(names) == len(set(names)) == n, (name, names)
    print(f'GATE_PASS: {name} -> {n} viewpoints ({names[0]} .. {names[-1]})')
print('GATE_PASS: all preset counts as wanted')
PYEOF
then
    say "CPU_GATE_FAILED -- see the log tail above"; exit 1
fi
say "CPU_GATE_PASSED"

[ -e "$CKPT" ] || { say "MISSING_CKPT $CKPT"; exit 1; }
say "CKPT $CKPT"

# ---- stage 1: in-distribution ring (azimuth_interp, 11 vp x 50 paired eps)
if [ -f "$OUT_INTERP/eval_log.json" ] && [ "$FORCE" != "1" ]; then
    say "INTERP_SKIPPED (already present; FORCE=1 to redo)"
else
    say "INTERP_START ($GPU, azimuth_interp, 11 viewpoints x 50 paired episodes)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_INTERP" -d "$GPU" \
        --preset azimuth_interp --m3-slots 11 --eef-hist-steps 4 \
        --n-envs 14 --n-test-vis 0 < /dev/null >> data/sweep_square_zgpose_interp.log 2>&1
    rc=$?
    say "INTERP_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "INTERP_FAILED -- see data/sweep_square_zgpose_interp.log"; exit 1; fi
fi
"$PY" summarize_novel_view.py "$OUT_INTERP" --out data/eval_square_zgpose_summary.txt \
    >> data/sweep_square_zgpose_interp.log 2>&1
say "INTERP_SUMMARY -> data/eval_square_zgpose_summary.txt"

# ---- stage 2: smoke, then the held-out ring (azimuth_offgrid)
if [ -f "$OUT_SMOKE/eval_log.json" ]; then
    say "SMOKE_SKIPPED (already present)"
else
    say "SMOKE_START ($OUT_SMOKE)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_SMOKE" -d "$GPU" \
        --preset azimuth_offgrid --m3-slots 11 --eef-hist-steps 4 \
        --n-test 4 --n-envs 4 --n-test-vis 0 < /dev/null >> data/sweep_zgpose_offgrid_smoke.log 2>&1
    rc=$?
    say "SMOKE_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "SMOKE_FAILED -- see data/sweep_zgpose_offgrid_smoke.log"; exit 1; fi
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
if [ -f "$OUT_HELDOUT/eval_log.json" ] && [ "$FORCE" != "1" ]; then
    say "HELDOUT_SKIPPED (already present; FORCE=1 to redo)"
else
    say "HELDOUT_START ($GPU, azimuth_offgrid, 14 viewpoints x 50 paired episodes)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_HELDOUT" -d "$GPU" \
        --preset azimuth_offgrid --m3-slots 11 --eef-hist-steps 4 \
        --n-envs 14 --n-test-vis 0 < /dev/null >> data/sweep_square_zgpose_offgrid.log 2>&1
    rc=$?
    say "HELDOUT_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "HELDOUT_FAILED -- see data/sweep_square_zgpose_offgrid.log"; exit 1; fi
fi
"$PY" summarize_novel_view.py "$OUT_HELDOUT" --out data/eval_heldout_square_zgpose_summary.txt \
    >> data/sweep_square_zgpose_offgrid.log 2>&1
say "HELDOUT_SUMMARY -> data/eval_heldout_square_zgpose_summary.txt"

# ---- stage 3: elevation (el_0 / +-15)
if [ -f "$OUT_EL/eval_log.json" ] && [ "$FORCE" != "1" ]; then
    say "ELEV_SKIPPED (already present; FORCE=1 to redo)"
else
    say "ELEV_START ($GPU, elevation_az0, 3 viewpoints x 50 paired episodes)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_EL" -d "$GPU" \
        --preset elevation_az0 --m3-slots 11 --eef-hist-steps 4 \
        --n-envs 14 --n-test-vis 0 < /dev/null >> data/sweep_square_zgpose_el.log 2>&1
    rc=$?
    say "ELEV_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "ELEV_FAILED -- see data/sweep_square_zgpose_el.log"; exit 1; fi
fi
"$PY" summarize_novel_view.py "$OUT_EL" --out data/eval_el_square_zgpose_summary.txt \
    >> data/sweep_square_zgpose_el.log 2>&1
say "ELEV_SUMMARY -> data/eval_el_square_zgpose_summary.txt"

say "SIZES $(du -sh "$OUT_INTERP" "$OUT_HELDOUT" "$OUT_EL" 2>/dev/null | tr '\n' ' ')"
say "DONE -- outputs are UNCOMMITTED. Read them first, then commit:"
say "  git add $OUT_INTERP $OUT_HELDOUT $OUT_EL $OUT_SMOKE"
say "  # and generate the three-way tables (m4base, m4clear, zgpose) with:"
say "  #   python summarize_novel_view.py data/eval_interp_square_m4base data/eval_interp_square_m4clear $OUT_INTERP"
say "  #   python summarize_novel_view.py data/eval_heldout_square_m4base data/eval_heldout_square_m4clear $OUT_HELDOUT"
say "  #   python summarize_novel_view.py data/eval_el_square_m4base data/eval_el_square_m4clear $OUT_EL"
