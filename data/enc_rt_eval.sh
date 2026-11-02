#!/bin/bash
# Evaluate one A/B/C/D residual-token cell at the fixed epoch-30 endpoint.
#
#   CELL=a|b|c|d
#   presets: azimuth_interp (11 seen), azimuth_offgrid (14; 10 interp +
#   4 out-of-range), elevation_az0 (3). 50 paired episodes per viewpoint,
#   N=1 inference via --m3-slots 11. This is the same protocol as enc_b_eval.sh.
#
# Run one cell at a time (two full sweeps on one GPU are slow and can contend):
#   CELL=a setsid bash -c 'bash data/enc_rt_eval.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
CELL=${CELL:-a}
case "$CELL" in
    a|b|c|d) ;;
    *) echo "CELL must be a|b|c|d (got $CELL)" >&2; exit 2 ;;
esac
RUN=/data/zihan/runs/run_square_enc_rt_${CELL}_s42_warm30
CKPT=$RUN/checkpoints/latest.ckpt
OUT_INTERP=data/eval_interp_square_enc_rt_${CELL}
OUT_HELDOUT=data/eval_heldout_square_enc_rt_${CELL}
OUT_EL=data/eval_el_square_enc_rt_${CELL}
OUT_SMOKE=data/eval_offgrid_smoke_enc_rt_${CELL}
GPU=${GPU:-cuda:0}
FORCE=${FORCE:-0}
LOG=data/enc_rt_eval_${CELL}.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: the run must be the fixed epoch-30 endpoint, not mid-loop latest.
if ! grep -q 'FINAL_CHECKPOINT .*epoch=30' "data/train_enc_rt_${CELL}.log" 2>/dev/null; then
    say "NOT_THE_FIXED_ENDPOINT: data/train_enc_rt_${CELL}.log has no FINAL_CHECKPOINT epoch=30"
    exit 1
fi
[ -e "$CKPT" ] || { say "MISSING_CKPT $CKPT"; exit 1; }
say "CKPT $CKPT (epoch-30 endpoint)"

# ---- stage 1: preset-expansion gate (must be able to fail and print evidence).
say "CPU gate: preset expansion"
if ! "$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import eval_novel_view as env
want = {'azimuth_interp': 11, 'azimuth_offgrid': 14, 'elevation_az0': 3}
for name, n in want.items():
    names = [v['name'] for v in env.PRESETS[name]['viewpoints']]
    assert len(names) == len(set(names)) == n, (name, names)
    print(f'GATE_PASS: {name} -> {n} viewpoints ({names[0]} .. {names[-1]})')
print('GATE_PASS: all preset counts as wanted')
PYEOF
then
    say "CPU_GATE_FAILED -- see the log tail above"; exit 1
fi
say "CPU_GATE_PASSED"

# ---- stage 2: in-distribution ring (11 trained poses).
if [ -f "$OUT_INTERP/eval_log.json" ] && [ "$FORCE" != "1" ]; then
    say "INTERP_SKIPPED (already present; FORCE=1 to redo)"
else
    say "INTERP_START ($GPU, azimuth_interp, 11 vp x 50 paired eps)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_INTERP" -d "$GPU" \
        --preset azimuth_interp --m3-slots 11 --eef-hist-steps 4 \
        --n-envs 14 --n-test-vis 0 < /dev/null \
        >> "data/sweep_square_enc_rt_${CELL}_interp.log" 2>&1
    rc=$?
    say "INTERP_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "INTERP_FAILED -- see the cell sweep log"; exit 1; fi
fi
"$PY" summarize_novel_view.py "$OUT_INTERP" \
    --out "data/eval_square_enc_rt_${CELL}_interp_summary.txt" \
    >> "data/sweep_square_enc_rt_${CELL}_interp.log" 2>&1
say "INTERP_SUMMARY -> data/eval_square_enc_rt_${CELL}_interp_summary.txt"

# ---- stage 3: off-grid smoke, then the full held-out ring.
if [ -f "$OUT_SMOKE/eval_log.json" ]; then
    say "SMOKE_SKIPPED (already present)"
else
    say "SMOKE_START ($OUT_SMOKE)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_SMOKE" -d "$GPU" \
        --preset azimuth_offgrid --m3-slots 11 --eef-hist-steps 4 \
        --n-test 4 --n-envs 4 --n-test-vis 0 < /dev/null \
        >> "data/sweep_square_enc_rt_${CELL}_offgrid_smoke.log" 2>&1
    rc=$?
    say "SMOKE_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "SMOKE_FAILED -- see the smoke log"; exit 1; fi
    if ! "$PY" - "$OUT_SMOKE/eval_log.json" >> "$LOG" 2>&1 <<'PYEOF'
import collections, json, sys
log = json.load(open(sys.argv[1]))
vp_of = {k.split('/')[1] for k in log if k.startswith('test/')}
n_succ = sum(1 for k in log
             if k.startswith('test/') and k.endswith('/success_rate'))
cnt = collections.Counter(k.split('/')[1] for k in log
                          if k.startswith('test/') and '/success_' in k
                          and not k.endswith('success_rate'))
missing = sorted(v for v in vp_of if cnt.get(v, 0) == 0)
print(f'SMOKE_GATE: viewpoints={len(vp_of)} success_rate_keys={n_succ} '
      f'per_viewpoint_episodes={sorted(set(cnt.values()))} missing={missing}')
assert len(vp_of) == 14 and n_succ == 14 and not missing
print('SMOKE_GATE_PASS')
PYEOF
    then
        say "SMOKE_GATE_FAILED"; exit 1
    fi
fi

if [ -f "$OUT_HELDOUT/eval_log.json" ] && [ "$FORCE" != "1" ]; then
    say "HELDOUT_SKIPPED (already present; FORCE=1 to redo)"
else
    say "HELDOUT_START ($GPU, azimuth_offgrid, 14 vp x 50 paired eps)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_HELDOUT" -d "$GPU" \
        --preset azimuth_offgrid --m3-slots 11 --eef-hist-steps 4 \
        --n-envs 14 --n-test-vis 0 < /dev/null \
        >> "data/sweep_square_enc_rt_${CELL}_offgrid.log" 2>&1
    rc=$?
    say "HELDOUT_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "HELDOUT_FAILED -- see the cell sweep log"; exit 1; fi
fi
"$PY" summarize_novel_view.py "$OUT_HELDOUT" \
    --out "data/eval_square_enc_rt_${CELL}_heldout_summary.txt" \
    >> "data/sweep_square_enc_rt_${CELL}_offgrid.log" 2>&1
say "HELDOUT_SUMMARY -> data/eval_square_enc_rt_${CELL}_heldout_summary.txt"

# ---- stage 4: elevation el_0/+-15 (descriptive).
if [ -f "$OUT_EL/eval_log.json" ] && [ "$FORCE" != "1" ]; then
    say "EL_SKIPPED (already present; FORCE=1 to redo)"
else
    say "EL_START ($GPU, elevation_az0, 3 vp x 50 paired eps)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_EL" -d "$GPU" \
        --preset elevation_az0 --m3-slots 11 --eef-hist-steps 4 \
        --n-envs 14 --n-test-vis 0 < /dev/null \
        >> "data/sweep_square_enc_rt_${CELL}_el.log" 2>&1
    rc=$?
    say "EL_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "EL_FAILED -- see the cell sweep log"; exit 1; fi
fi
"$PY" summarize_novel_view.py "$OUT_EL" \
    --out "data/eval_square_enc_rt_${CELL}_el_summary.txt" \
    >> "data/sweep_square_enc_rt_${CELL}_el.log" 2>&1
say "EL_SUMMARY -> data/eval_square_enc_rt_${CELL}_el_summary.txt"

# ---- stage 5: registered means for this cell (cross-cell deltas are run
# separately once all four logs exist).
say "REGISTERED_MEANS"
"$PY" summarize_enc_rt.py \
    --cell "${CELL}=${OUT_INTERP},${OUT_HELDOUT},${OUT_EL}" >> "$LOG" 2>&1
say "ALL_EVAL_DONE($CELL)"
