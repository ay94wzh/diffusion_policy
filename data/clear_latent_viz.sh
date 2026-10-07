#!/bin/bash
# Compute the latent-distribution coordinates for the clear run, on miroc-server
# (the npz live there; the compute path is numpy-only). The laptop then renders
# the figures from the small committed JSON:
#
#   python3 visualize_latent_distribution.py plot \
#     -i data/analysis_clear/latent_viz_coords.json \
#     -d data/analysis_clear/analyze_latent_series.json \
#     -o data/analysis_clear/figures
#
# Runtime: seconds-to-minutes (6 epochs of 1.71 MB npz, one SVD each).
#
# Run to completion with:  setsid bash -c 'bash data/clear_latent_viz.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
RUN=/data/zihan/runs/run_square_m4clear_s42_200ep
SNAP=$RUN/latent_snapshots
OUT=data/analysis_clear/latent_viz_coords.json
LOG=data/clear_latent_viz.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# CPU gate first -- the script's own tests (compute needs only numpy; the plot
# smoke self-skips loudly when matplotlib is absent). A guard must be able to
# fail, and to print why: status unpiped, log tail shown.
say "CPU gate: tests/test_visualize_latent_distribution.py"
if ! "$PY" tests/test_visualize_latent_distribution.py > /tmp/clear_viz_cpu_gate.log 2>&1; then
    say "CPU_GATE_FAILED -- not computing (see /tmp/clear_viz_cpu_gate.log)"
    tail -n 20 /tmp/clear_viz_cpu_gate.log | tee -a "$LOG"
    exit 1
fi
say "CPU_GATE_PASSED"

say "COMPUTE_START -> $OUT"
"$PY" -u visualize_latent_distribution.py compute -s "$SNAP" -o "$OUT" \
    --epochs 0,25,50,100,150,200 --basis-epoch 200 >> data/compute_clear_viz.log 2>&1
rc=$?
say "COMPUTE_EXIT=$rc"
if [ $rc -ne 0 ]; then
    say "COMPUTE_FAILED -- see data/compute_clear_viz.log"
    tail -n 20 data/compute_clear_viz.log | tee -a "$LOG"
    exit $rc
fi
say "SIZES $(du -h "$OUT" | cut -f1) coords json"
say "DONE -- commit data/analysis_clear/latent_viz_coords.json, then render on the laptop"
