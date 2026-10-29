#!/bin/bash
# Sweep the encoder-ladder RH cell:
#   1. azimuth_interp  -- the in-distribution ring (11 trained poses);
#   2. azimuth_offgrid -- the held-out ring (7.5-degree offsets + +-90),
#      with a 4-episode smoke gate first;
#   3. elevation_az0   -- el_0 / +-15 (descriptive cells).
# 50 paired episodes per viewpoint. Mirrors data/enc_s_eval.sh, single cell.
#
# RH reads the camera vectors (its rays) AND the per-sample base-frame
# trajectory window, so the wrapper must serve `eef_hist_base` too:
# --serve-eef-hist-base on every eval_novel_view call. The camera-frame
# `view_eef_hist` is still served (the M3 schema, the normalizer's identity
# entries); RH ignores it. Keep --m3-slots / --eef-hist-steps; do not reduce
# this to an M1-style sweep.
#
# Run with:
#   setsid bash -c 'bash data/enc_rh_eval.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
RUN=/data/zihan/runs/run_square_enc_rh_s42_200ep
CKPT=$RUN/checkpoints/latest.ckpt
OUT_INTERP=data/eval_interp_square_enc_rh
OUT_HELDOUT=data/eval_heldout_square_enc_rh
OUT_EL=data/eval_el_square_enc_rh
OUT_SMOKE=data/eval_offgrid_smoke_enc_rh
GPU=${GPU:-cuda:0}
FORCE=${FORCE:-0}
LOG=data/enc_rh_eval.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: preset-expansion gate (must be able to fail, and print evidence)
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

[ -e "$CKPT" ] || { say "MISSING_CKPT $CKPT"; exit 1; }
say "CKPT $CKPT"

# ---- stage 1: in-distribution ring
if [ -f "$OUT_INTERP/eval_log.json" ] && [ "$FORCE" != "1" ]; then
    say "INTERP_SKIPPED (already present; FORCE=1 to redo)"
else
    say "INTERP_START ($GPU, azimuth_interp, 11 vp x 50 paired eps)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_INTERP" -d "$GPU" \
        --preset azimuth_interp --m3-slots 11 --eef-hist-steps 4 --serve-eef-hist-base \
        --n-envs 14 --n-test-vis 0 < /dev/null \
        >> "data/sweep_square_enc_rh_interp.log" 2>&1
    rc=$?
    say "INTERP_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "INTERP_FAILED -- see the sweep log"; exit 1; fi
fi
"$PY" summarize_novel_view.py "$OUT_INTERP" \
    --out "data/eval_square_enc_rh_interp_summary.txt" \
    >> "data/sweep_square_enc_rh_interp.log" 2>&1
say "INTERP_SUMMARY -> data/eval_square_enc_rh_interp_summary.txt"

# ---- stage 2: smoke, then the held-out ring
if [ -f "$OUT_SMOKE/eval_log.json" ]; then
    say "SMOKE_SKIPPED (already present)"
else
    say "SMOKE_START ($OUT_SMOKE)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_SMOKE" -d "$GPU" \
        --preset azimuth_offgrid --m3-slots 11 --eef-hist-steps 4 --serve-eef-hist-base \
        --n-test 4 --n-envs 4 --n-test-vis 0 < /dev/null \
        >> "data/sweep_square_enc_rh_offgrid_smoke.log" 2>&1
    rc=$?
    say "SMOKE_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "SMOKE_FAILED -- see the smoke log"; exit 1; fi
    if ! "$PY" - "$OUT_SMOKE/eval_log.json" >> "$LOG" 2>&1 <<'PYEOF'
import collections, json, sys

log = json.load(open(sys.argv[1]))
vp_of = {k.split('/')[1] for k in log if k.startswith('test/')}
n_succ = sum(1 for k in log
             if k.startswith('test/') and k.endswith('/success_rate'))
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
    say "HELDOUT_START ($GPU, azimuth_offgrid, 14 vp x 50 paired eps)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_HELDOUT" -d "$GPU" \
        --preset azimuth_offgrid --m3-slots 11 --eef-hist-steps 4 --serve-eef-hist-base \
        --n-envs 14 --n-test-vis 0 < /dev/null \
        >> "data/sweep_square_enc_rh_offgrid.log" 2>&1
    rc=$?
    say "HELDOUT_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "HELDOUT_FAILED -- see the sweep log"; exit 1; fi
fi
"$PY" summarize_novel_view.py "$OUT_HELDOUT" \
    --out "data/eval_square_enc_rh_heldout_summary.txt" \
    >> "data/sweep_square_enc_rh_offgrid.log" 2>&1
say "HELDOUT_SUMMARY -> data/eval_square_enc_rh_heldout_summary.txt"

# ---- stage 3: elevation
if [ -f "$OUT_EL/eval_log.json" ] && [ "$FORCE" != "1" ]; then
    say "ELEV_SKIPPED (already present; FORCE=1 to redo)"
else
    say "ELEV_START ($GPU, elevation_az0, 3 vp x 50 paired eps)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$OUT_EL" -d "$GPU" \
        --preset elevation_az0 --m3-slots 11 --eef-hist-steps 4 --serve-eef-hist-base \
        --n-envs 14 --n-test-vis 0 < /dev/null \
        >> "data/sweep_square_enc_rh_el.log" 2>&1
    rc=$?
    say "ELEV_EXIT=$rc"
    if [ $rc -ne 0 ]; then say "ELEV_FAILED -- see the sweep log"; exit 1; fi
fi
"$PY" summarize_novel_view.py "$OUT_EL" \
    --out "data/eval_square_enc_rh_el_summary.txt" \
    >> "data/sweep_square_enc_rh_el.log" 2>&1
say "ELEV_SUMMARY -> data/eval_square_enc_rh_el_summary.txt"

say "SIZES $(du -sh "$OUT_INTERP" "$OUT_HELDOUT" "$OUT_EL" 2>/dev/null | tr '\n' ' ')"
say "DONE -- outputs are UNCOMMITTED. Read them first, then commit."
