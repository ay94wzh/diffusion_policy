#!/bin/bash
# Sweep the prob checkpoints (cells B/C/D: MHA-over-mu / PoE / mean-of-mu), each
# through the same three presets as every arm since m4base, plus the smoke gate:
#   1. azimuth_interp  -- the in-distribution ring (all 11 poses trained here);
#      cell A (m4base) reads 0.796 there, m4clear 0.747, zgpose 0.735,
#      zvzg 0.751;
#   2. azimuth_offgrid -- the pure held-out ring (7.5-degree offsets +-90, none
#      trained; A 0.774 / m4clear 0.716 / zgpose 0.714 / zvzg 0.606 over
#      +-7.5..+-67.5); smoke first;
#   3. elevation_az0   -- the conditioning-route columns (A reads 0.840/0/0).
# The past-the-ring cells (+-82.5/+-90; A/m4clear 0.310, zgpose 0.140, zvzg
# 0.100) are the run header's separately-registered read-out.
# Runtime reference (clear run, shared GPU): smoke ~15 min, offgrid ~77 min,
# interp ~46 min, elevation ~12-17 min -> ~2.5 h per cell, ~7.5 h for three.
#
# The comparison tables come from summarize_novel_view.py against the committed
# data/eval_{interp,heldout,el}_square_{m4base,m4clear,zgpose,zvzg}/ dirs; see
# data/prob_run.sh for the pre-registration this is read against. Outputs land
# under data/eval_*/ which .gitignore whitelists wholesale -- commit them after
# a human read.
#
# Run to completion with:  setsid bash -c 'bash data/prob_eval.sh' < /dev/null &
# One cell:                CELLS=c GPU=cuda:1 bash data/prob_eval.sh
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
GPU=${GPU:-cuda:0}
CELLS=${CELLS:-b c d}
FORCE=${FORCE:-0}
LOG=data/prob_eval.log
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

for CELL in $CELLS; do
    RUN=/data/zihan/runs/run_square_prob_${CELL}_s42_200ep
    CKPT=$RUN/checkpoints/latest.ckpt
    OUT_INTERP=data/eval_interp_square_prob_${CELL}
    OUT_HELDOUT=data/eval_heldout_square_prob_${CELL}
    OUT_EL=data/eval_el_square_prob_${CELL}
    OUT_SMOKE=data/eval_offgrid_smoke_prob_${CELL}

    [ -e "$CKPT" ] || { say "MISSING_CKPT $CKPT (cell $CELL)"; exit 1; }
    say "CKPT cell=$CELL $CKPT"

    # ---- stage 1: in-distribution ring (azimuth_interp, 11 vp x 50 paired eps)
    if [ -f "$OUT_INTERP/eval_log.json" ] && [ "$FORCE" != "1" ]; then
        say "INTERP_SKIPPED cell=$CELL (already present; FORCE=1 to redo)"
    else
        say "INTERP_START cell=$CELL ($GPU, azimuth_interp, 11 vp x 50 paired eps)"
        "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_INTERP" -d "$GPU" \
            --preset azimuth_interp --m3-slots 11 --eef-hist-steps 4 \
            --n-envs 14 --n-test-vis 0 < /dev/null \
            >> data/sweep_square_prob_${CELL}_interp.log 2>&1
        rc=$?
        say "INTERP_EXIT=$rc cell=$CELL"
        if [ $rc -ne 0 ]; then say "INTERP_FAILED -- see data/sweep_square_prob_${CELL}_interp.log"; exit 1; fi
    fi
    "$PY" summarize_novel_view.py "$OUT_INTERP" \
        --out data/eval_square_prob_${CELL}_interp_summary.txt \
        >> data/sweep_square_prob_${CELL}_interp.log 2>&1
    say "INTERP_SUMMARY cell=$CELL -> data/eval_square_prob_${CELL}_interp_summary.txt"

    # ---- stage 2: smoke, then the held-out ring (azimuth_offgrid)
    if [ -f "$OUT_SMOKE/eval_log.json" ]; then
        say "SMOKE_SKIPPED cell=$CELL (already present)"
    else
        say "SMOKE_START cell=$CELL ($OUT_SMOKE)"
        "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_SMOKE" -d "$GPU" \
            --preset azimuth_offgrid --m3-slots 11 --eef-hist-steps 4 \
            --n-test 4 --n-envs 4 --n-test-vis 0 < /dev/null \
            >> data/sweep_prob_${CELL}_offgrid_smoke.log 2>&1
        rc=$?
        say "SMOKE_EXIT=$rc cell=$CELL"
        if [ $rc -ne 0 ]; then say "SMOKE_FAILED -- see data/sweep_prob_${CELL}_offgrid_smoke.log"; exit 1; fi
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
            say "SMOKE_LAYOUT_FAILED cell=$CELL"; exit 1
        fi
        say "SMOKE_GATE_PASSED cell=$CELL"
    fi
    if [ -f "$OUT_HELDOUT/eval_log.json" ] && [ "$FORCE" != "1" ]; then
        say "HELDOUT_SKIPPED cell=$CELL (already present; FORCE=1 to redo)"
    else
        say "HELDOUT_START cell=$CELL ($GPU, azimuth_offgrid, 14 vp x 50 paired eps)"
        "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_HELDOUT" -d "$GPU" \
            --preset azimuth_offgrid --m3-slots 11 --eef-hist-steps 4 \
            --n-envs 14 --n-test-vis 0 < /dev/null \
            >> data/sweep_square_prob_${CELL}_offgrid.log 2>&1
        rc=$?
        say "HELDOUT_EXIT=$rc cell=$CELL"
        if [ $rc -ne 0 ]; then say "HELDOUT_FAILED -- see data/sweep_square_prob_${CELL}_offgrid.log"; exit 1; fi
    fi
    "$PY" summarize_novel_view.py "$OUT_HELDOUT" \
        --out data/eval_heldout_square_prob_${CELL}_summary.txt \
        >> data/sweep_square_prob_${CELL}_offgrid.log 2>&1
    say "HELDOUT_SUMMARY cell=$CELL -> data/eval_heldout_square_prob_${CELL}_summary.txt"

    # ---- stage 3: elevation (el_0 / +-15)
    if [ -f "$OUT_EL/eval_log.json" ] && [ "$FORCE" != "1" ]; then
        say "ELEV_SKIPPED cell=$CELL (already present; FORCE=1 to redo)"
    else
        say "ELEV_START cell=$CELL ($GPU, elevation_az0, 3 vp x 50 paired eps)"
        "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_EL" -d "$GPU" \
            --preset elevation_az0 --m3-slots 11 --eef-hist-steps 4 \
            --n-envs 14 --n-test-vis 0 < /dev/null \
            >> data/sweep_square_prob_${CELL}_el.log 2>&1
        rc=$?
        say "ELEV_EXIT=$rc cell=$CELL"
        if [ $rc -ne 0 ]; then say "ELEV_FAILED -- see data/sweep_square_prob_${CELL}_el.log"; exit 1; fi
    fi
    "$PY" summarize_novel_view.py "$OUT_EL" \
        --out data/eval_el_square_prob_${CELL}_summary.txt \
        >> data/sweep_square_prob_${CELL}_el.log 2>&1
    say "ELEV_SUMMARY cell=$CELL -> data/eval_el_square_prob_${CELL}_summary.txt"

    say "SIZES cell=$CELL $(du -sh "$OUT_INTERP" "$OUT_HELDOUT" "$OUT_EL" 2>/dev/null | tr '\n' ' ')"
done

say "DONE -- outputs are UNCOMMITTED. Read them, then commit:"
say "  git add data/eval_interp_square_prob_* data/eval_heldout_square_prob_* data/eval_el_square_prob_* data/eval_offgrid_smoke_prob_*"
say "  # and build the FIVE-way tables (m4base, m4clear, zgpose, zvzg, prob_b/c/d) with:"
say "  #   python summarize_novel_view.py data/eval_interp_square_m4base data/eval_interp_square_m4clear data/eval_interp_square_zgpose data/eval_interp_square_zvzg data/eval_interp_square_prob_b data/eval_interp_square_prob_c data/eval_interp_square_prob_d"
say "  #   (same for heldout_ and el_ dirs)"
