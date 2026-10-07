#!/bin/bash
# Analyze the clear run's per-epoch latent series, and sweep its checkpoint.
#
# Runs ON `MIROC-SERVER` (the npz and the checkpoint live under /data; this
# script must never be run from the laptop -- the interpreter below does not
# exist there). Two stages, both idempotent:
#
#   1. analyze_latent_series.py over
#      /data/zihan/runs/run_square_m4clear_s42_200ep/latent_snapshots
#      (201 epochs x 1.71 MB = 328 MB; CPU-only, ~1.5-4 min), writing the small
#      committed digest data/analysis_clear/{analyze_latent_series.json,summary.md};
#   2. the clear run's ONLY saved checkpoint swept at azimuth_interp
#      (11 viewpoints x 50 paired episodes, ~46 min on one GPU) into
#      data/eval_interp_square_m4clear/ -- the in-distribution reference curve
#      PROGRESS.md records as "open, and cheap". --m3-slots 11 and
#      --eef-hist-steps 4 match the checkpoint's own task config.
#
# THE GIT POLICY: this script runs no git command. The outputs are small
# enough for git but they are committed only after a human has read them.
# The 328 MB of npz stay under /data; only the derived digest travels.
#
# Run:  setsid bash -c 'bash data/clear_analysis.sh' < /dev/null &
#   FORCE_ANALYSIS=1 / FORCE_SWEEP=1 re-run one stage whose output exists
#   (FORCE=1 still means both) -- this is what a code fix to the analyzer
#   needs, so the 46-min cached sweep is not repeated; GPU=cuda:1 picks the
#   other device for the sweep.
set -u
FORCE_ALL=${FORCE:-0}
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
RUN=/data/zihan/runs/run_square_m4clear_s42_200ep
SNAP=$RUN/latent_snapshots
OUT=data/analysis_clear
SWEEP=data/eval_interp_square_m4clear
CKPT=$RUN/checkpoints/latest.ckpt
EXPECT_NPZ=201
GPU=${GPU:-cuda:0}
LOG=data/clear_analysis.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gate -- the analyzer's own tests, mutation checks included.
# A guard must be able to fail AND print why: status unpiped, log tail shown.
say "CPU gate: tests/test_analyze_latent_series.py"
if ! "$PY" tests/test_analyze_latent_series.py > /tmp/clear_analysis_cpu_gate.log 2>&1; then
    say "CPU_GATE_FAILED -- not analyzing; last lines of the gate log follow"
    tail -n 20 /tmp/clear_analysis_cpu_gate.log | tee -a "$LOG"
    exit 1
fi
say "CPU_GATE_PASSED"

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
if len(eps) != expect or gaps:
    sys.exit(1)
PYEOF
}
if ! out=$(snapshot_gate 2>&1); then
    say "SNAPSHOT_GATE_FAILED (expected $EXPECT_NPZ contiguous npz under $SNAP)"
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
        -o "$OUT" --strict >> data/analyze_clear.log 2>&1
    rc=$?
    say "ANALYSIS_EXIT=$rc"
    if [ $rc -ne 0 ]; then
        say "ANALYSIS_FAILED -- stopping before the sweep"
        exit $rc
    fi
fi

# ---- stage 3: the sweep (idempotent; eval_novel_view blocks on a prompt when
# -o exists, which is why the guard is a file check and stdin is /dev/null)
if [ -f "$SWEEP/eval_log.json" ] && [ "${FORCE_SWEEP:-$FORCE_ALL}" != "1" ]; then
    say "SWEEP_SKIPPED (already present; FORCE_SWEEP=1 or FORCE=1 to redo)"
else
    if [ ! -e "$CKPT" ]; then
        say "SWEEP_MISSING_CKPT $CKPT"
        exit 1
    fi
    say "SWEEP_START ($GPU, azimuth_interp, 11 viewpoints x 50 paired episodes)"
    "$PY" eval_novel_view.py -c "$CKPT" -o "$SWEEP" -d "$GPU" \
        --preset azimuth_interp --m3-slots 11 --eef-hist-steps 4 \
        --n-envs 14 --n-test-vis 0 < /dev/null >> data/sweep_square_m4clear.log 2>&1
    rc=$?
    say "SWEEP_EXIT=$rc"
    if [ $rc -ne 0 ]; then
        say "SWEEP_FAILED -- see data/sweep_square_m4clear.log"
        exit $rc
    fi
fi
"$PY" summarize_novel_view.py "$SWEEP" --out data/eval_square_m4clear_summary.txt \
    >> data/sweep_square_m4clear.log 2>&1
say "SWEEP_SUMMARY -> data/eval_square_m4clear_summary.txt"

# ---- stage 4: sizes + the digest, read from the files rather than assumed
say "SIZES $(du -h "$OUT/analyze_latent_series.json" 2>/dev/null | cut -f1) analysis json, \
$(du -h "$OUT/summary.md" 2>/dev/null | cut -f1) summary md, \
$(du -sh "$SNAP" 2>/dev/null | cut -f1) of npz left under /data"
if [ "$(stat -c %s "$OUT/analyze_latent_series.json" 2>/dev/null || echo 0)" -gt 1000000 ]; then
    say "SIZE_WARNING analysis json exceeds 1 MB -- check before committing"
fi
say "----- data/analysis_clear/summary.md -----"
cat "$OUT/summary.md" 2>/dev/null | tee -a "$LOG"
say "----- data/eval_square_m4clear_summary.txt -----"
cat data/eval_square_m4clear_summary.txt 2>/dev/null | tee -a "$LOG"
say "DONE -- outputs are UNCOMMITTED. Read them first, then commit:"
say "  git add data/analysis_clear data/eval_interp_square_m4clear"
