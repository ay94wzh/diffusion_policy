#!/bin/bash
# The m4base instrument battery, on MIROC-SERVER (npz + checkpoints live under
# /data; this script must never run from the laptop). Everything here is
# symmetric across the A/B pair: m4base (base-frame aux target) AND m4clear
# (camera-frame, the control), so every comparison is within-pair.
#
#   1. analyze_latent_series.py --strict over m4base's 201 npz
#      -> data/analysis_m4base/{analyze_latent_series.json,summary.md}
#      (read against data/analysis_clear/ for the pre-registered series questions)
#   2. visualize_latent_distribution.py compute (6 epochs)
#      -> data/analysis_m4base/latent_viz_coords.json
#      (laptop renders: python3 visualize_latent_distribution.py plot -i ... \
#         -d data/analysis_m4base/analyze_latent_series.json -o data/analysis_m4base/figures)
#   3. collapse screens, matched range 11,11 + random-init, on BOTH cells
#      (m4clear has none yet -- created here for the pair)
#   4. probe_relpose grid on BOTH cells with IDENTICAL flags (the NOTES
#      *Latent probe* command scaled to this cell's K=11/range [1,11]), plus a
#      final gate that the two JSONs' draw fingerprints match -- the
#      documented comparability condition.
#
# THE GIT POLICY: this script runs no git command. Outputs are committed only
# after a human has read them; the 328 MB of npz stay under /data.
#
# Run:  setsid bash -c 'bash data/m4base_analysis.sh' < /dev/null &
#   FORCE_ANALYSIS=1 / FORCE_VIZ=1 / FORCE_SCREEN=1 / FORCE_PROBE=1 re-run one
#   stage whose output exists (FORCE=1 means all).
set -u
FORCE_ALL=${FORCE:-0}
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
RUN=/data/zihan/runs/run_square_m4base_s42_200ep
RUN_CLEAR=/data/zihan/runs/run_square_m4clear_s42_200ep
SNAP=$RUN/latent_snapshots
CKPT=$RUN/checkpoints/latest.ckpt
CKPT_CLEAR=$RUN_CLEAR/checkpoints/latest.ckpt
OUT=data/analysis_m4base
VIZ=$OUT/latent_viz_coords.json
EXPECT_NPZ=201
GPU=${GPU:-cuda:0}
LOG=data/m4base_analysis.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates -- the analyzer's and the viz script's own tests.
# A guard must be able to fail AND print why: status unpiped, log tail shown.
say "CPU gate: tests/test_analyze_latent_series.py"
if ! "$PY" tests/test_analyze_latent_series.py > /tmp/m4base_analysis_cpu_gate.log 2>&1; then
    say "CPU_GATE_FAILED (analyzer) -- last lines of the gate log follow"
    tail -n 20 /tmp/m4base_analysis_cpu_gate.log | tee -a "$LOG"
    exit 1
fi
say "CPU gate: tests/test_visualize_latent_distribution.py"
if ! "$PY" tests/test_visualize_latent_distribution.py > /tmp/m4base_viz_cpu_gate.log 2>&1; then
    say "CPU_GATE_FAILED (viz) -- last lines of the gate log follow"
    tail -n 20 /tmp/m4base_viz_cpu_gate.log | tee -a "$LOG"
    exit 1
fi
say "CPU_GATES_PASSED"

# ---- stage 1: snapshot gate, read from the files rather than assumed.
snapshot_gate() {
    "$PY" - "$SNAP" "$EXPECT_NPZ" <<'PYEOF'
import glob, json, os, sys
snap, expect = sys.argv[1], int(sys.argv[2])
if not os.path.isdir(snap):
    print(f'no snapshots dir at {snap}')
    sys.exit(1)
eps = sorted(int(os.path.basename(p).split('_')[1].split('.')[0])
             for p in glob.glob(os.path.join(snap, 'epoch_*.npz')))
meta = json.load(open(os.path.join(snap, 'meta.json')))
have = set(eps)
gaps = [e for e in range(eps[0], eps[-1] + 1) if e not in have] if eps else []
print(f'npz={len(eps)} first={eps[0] if eps else None} '
      f'last={eps[-1] if eps else None} gaps={gaps} '
      f'K={meta["K"]} pool={meta["pool_ring"]} az0_ring={meta["az0_ring"]}')
if len(eps) != expect or gaps or meta['K'] != 11:
    sys.exit(1)
PYEOF
}
if ! out=$(snapshot_gate 2>&1); then
    say "SNAPSHOT_GATE_FAILED (expected $EXPECT_NPZ contiguous npz, K=11, under $SNAP)"
    say "  raw: $out"
    exit 1
fi
say "SNAPSHOT_GATE $out"

# ---- stage 2: the analysis
if [ -f "$OUT/analyze_latent_series.json" ] && [ "${FORCE_ANALYSIS:-$FORCE_ALL}" != "1" ]; then
    say "ANALYSIS_SKIPPED (already present; FORCE_ANALYSIS=1 or FORCE=1 to redo)"
else
    say "ANALYSIS_START -> $OUT"
    "$PY" -u analyze_latent_series.py -s "$SNAP" -l "$RUN/logs.json.txt" \
        -o "$OUT" --strict >> data/analyze_m4base.log 2>&1
    rc=$?
    say "ANALYSIS_EXIT=$rc"
    if [ $rc -ne 0 ]; then
        say "ANALYSIS_FAILED -- see data/analyze_m4base.log"
        tail -n 20 data/analyze_m4base.log | tee -a "$LOG"
        exit $rc
    fi
fi

# ---- stage 3: latent-distribution coordinates (numpy-only compute)
if [ -f "$VIZ" ] && [ "${FORCE_VIZ:-$FORCE_ALL}" != "1" ]; then
    say "VIZ_SKIPPED (already present; FORCE_VIZ=1 or FORCE=1 to redo)"
else
    say "VIZ_START -> $VIZ"
    "$PY" -u visualize_latent_distribution.py compute -s "$SNAP" -o "$VIZ" \
        --epochs 0,25,50,100,150,200 --basis-epoch 200 >> data/compute_m4base_viz.log 2>&1
    rc=$?
    say "VIZ_EXIT=$rc"
    if [ $rc -ne 0 ]; then
        say "VIZ_FAILED -- see data/compute_m4base_viz.log"; exit $rc
    fi
fi

# ---- stage 4: collapse screens on BOTH cells, matched range + random-init.
# The baseline is per-architecture and must be measured, never borrowed.
for cell_ckpt in "m4base $CKPT" "m4clear $CKPT_CLEAR"; do
    set -- $cell_ckpt
    cell=$1; c=$2
    [ -e "$c" ] || { say "SCREEN_MISSING_CKPT $c"; exit 1; }
    if [ -f "data/screen_collapse/square_${cell}_latest.json" ] \
       && [ "${FORCE_SCREEN:-$FORCE_ALL}" != "1" ]; then
        say "SCREEN_SKIPPED_${cell} (already present)"
        continue
    fi
    say "SCREEN_START_${cell} (range 11,11)"
    "$PY" screen_collapse.py -c "$c" -d "$GPU" --view-count-range 11,11 \
        -o "data/screen_collapse/square_${cell}_latest.json" \
        >> "data/screen_square_${cell}.log" 2>&1
    rc=$?
    "$PY" screen_collapse.py -c "$c" -d "$GPU" --view-count-range 11,11 --random-init \
        -o "data/screen_collapse/square_${cell}_RANDOM_INIT.json" \
        >> "data/screen_square_${cell}.log" 2>&1
    rc2=$?
    say "SCREEN_DONE_${cell} exit=$rc/$rc2"
    [ $rc -eq 0 ] && [ $rc2 -eq 0 ] || { say "SCREEN_FAILED_${cell} -- see data/screen_square_${cell}.log"; exit 1; }
done

# ---- stage 5: the relpose grid probes, IDENTICAL flags on both cells (the
# NOTES *Latent probe* command scaled to K=11 / range [1,11]; --random-init-
# control on both so the draw fingerprints stay observable). Cost: the NOTES
# *Traps* entry budgets ~35 min per cell.
probe_one() {
    cell=$1; c=$2
    out="data/probe_relpose_grid_square_${cell}"
    if [ -f "$out/probe_relpose.json" ] && [ "${FORCE_PROBE:-$FORCE_ALL}" != "1" ]; then
        say "PROBE_SKIPPED_${cell} (already present)"
        return 0
    fi
    "$PY" -u probe_relpose.py -c "$c" -o "$out" -d "$GPU" \
        --view-count-range 1,11 --stability-ranges "1,2;1,6;1,11;11,11" \
        --stability-states 256 --stability-repeats 6 --n-samples 2000 \
        --mlp-steps 2000 --num-threads 4 --random-init-control \
        >> "data/probe_square_${cell}.log" 2>&1
    rc=$?
    say "PROBE_DONE_${cell} exit=$rc"
    [ $rc -eq 0 ] || { say "PROBE_FAILED_${cell} -- see data/probe_square_${cell}.log"; return 1; }
}
probe_one m4base "$CKPT" && probe_one m4clear "$CKPT_CLEAR" || exit 1

# the documented comparability gate: the two cells' draws must MATCH range by
# range, and mean_active must track the range (NOTES *Latent probe*: 1.5 means
# the override was silently ignored)
"$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import json
a = json.load(open('data/probe_relpose_grid_square_m4base/probe_relpose.json'))
b = json.load(open('data/probe_relpose_grid_square_m4clear/probe_relpose.json'))
fails = []
for r in sorted(set(a['latent_grid']) & set(b['latent_grid'])):
    fa = a['latent_grid'][r].get('draw_fingerprint')
    fb = b['latent_grid'][r].get('draw_fingerprint')
    print(f"PROBE_GATE range {r}: fingerprint base={fa} clear={fb} "
          f"{'MATCH' if fa == fb else 'MISMATCH'}")
    if fa != fb:
        fails.append(r)
for name, d in (('base', a), ('clear', b)):
    ma = d['stats']['mean_active']
    print(f"PROBE_GATE mean_active {name}={ma:.3f} (want ~6.0 for [1,11])")
    if not (5.5 < ma < 6.5):
        fails.append(f'mean_active_{name}')
if fails:
    raise SystemExit(f'PROBE_GATE_FAIL: {fails}')
print('PROBE_GATE PASS: draw fingerprints match, mean_active tracks [1,11]')
PYEOF
probe_rc=$?
say "PROBE_GATE_EXIT=$probe_rc"
[ $probe_rc -eq 0 ] || say "WARNING: probe comparability gate failed -- do not compare probe numbers across cells"

# ---- stage 6: sizes + the digest, read from the files rather than assumed
say "SIZES $(du -h "$OUT/analyze_latent_series.json" 2>/dev/null | cut -f1) analysis json, \
$(du -h "$VIZ" 2>/dev/null | cut -f1) viz coords, \
$(du -sh "$SNAP" 2>/dev/null | cut -f1) of npz left under /data"
say "----- data/analysis_m4base/summary.md -----"
cat "$OUT/summary.md" 2>/dev/null | tee -a "$LOG"
say "DONE -- outputs are UNCOMMITTED. Read them first, then commit:"
say "  git add $OUT data/screen_collapse/square_m4base_latest.json \
data/screen_collapse/square_m4base_RANDOM_INIT.json \
data/screen_collapse/square_m4clear_latest.json \
data/screen_collapse/square_m4clear_RANDOM_INIT.json \
data/probe_relpose_grid_square_m4base data/probe_relpose_grid_square_m4clear"
say "  # laptop, after committing: python3 visualize_latent_distribution.py plot \
-i $VIZ -d $OUT/analyze_latent_series.json -o $OUT/figures"
