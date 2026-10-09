#!/bin/bash
# The zgpose instrument battery, on MIROC-SERVER (npz + checkpoints live under
# /data; this script must never run from the laptop). The same battery the
# m4base cell got, so every comparison is against committed artifacts:
#
#   1. analyze_latent_series.py --strict over zgpose's 201 npz
#      -> data/analysis_zgpose/{analyze_latent_series.json,summary.md}
#      (read against data/analysis_m4base/ and data/analysis_clear/ for the
#      pre-registered two-outcome diagnostic in data/zgpose_run.sh)
#   2. visualize_latent_distribution.py compute (6 epochs)
#      -> data/analysis_zgpose/latent_viz_coords.json
#      (laptop renders: python3 visualize_latent_distribution.py plot -i ... \
#         -d data/analysis_zgpose/analyze_latent_series.json -o data/analysis_zgpose/figures)
#   3. collapse screen, matched range 11,11 + random-init, on the zgpose cell
#      (m4base/m4clear are already committed; only this cell is new)
#   4. probe_relpose grid with IDENTICAL flags to the m4 pair, plus a final
#      gate that the draw fingerprints match the COMMITTED m4base probe JSON
#      -- the documented comparability condition, one side already in git.
#   5. the registered readouts, printed from the files (the criteria live in
#      data/zgpose_run.sh's header).
#
# THE GIT POLICY: this script runs no git command. Outputs are committed only
# after a human has read them; the ~330 MB of npz stay under /data.
#
# Run:  setsid bash -c 'bash data/zgpose_analysis.sh' < /dev/null &
#   FORCE_ANALYSIS=1 / FORCE_VIZ=1 / FORCE_SCREEN=1 / FORCE_PROBE=1 re-run one
#   stage whose output exists (FORCE=1 means all).
set -u
FORCE_ALL=${FORCE:-0}
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
RUN=/data/zihan/runs/run_square_zgpose_s42_200ep
SNAP=$RUN/latent_snapshots
CKPT=$RUN/checkpoints/latest.ckpt
OUT=data/analysis_zgpose
VIZ=$OUT/latent_viz_coords.json
EXPECT_NPZ=201
GPU=${GPU:-cuda:0}
LOG=data/zgpose_analysis.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates -- the analyzer's and the viz script's own tests.
# A guard must be able to fail AND print why: status unpiped, log tail shown.
say "CPU gate: tests/test_analyze_latent_series.py"
if ! "$PY" tests/test_analyze_latent_series.py > /tmp/zgpose_analysis_cpu_gate.log 2>&1; then
    say "CPU_GATE_FAILED (analyzer) -- last lines of the gate log follow"
    tail -n 20 /tmp/zgpose_analysis_cpu_gate.log | tee -a "$LOG"
    exit 1
fi
say "CPU gate: tests/test_visualize_latent_distribution.py"
if ! "$PY" tests/test_visualize_latent_distribution.py > /tmp/zgpose_viz_cpu_gate.log 2>&1; then
    say "CPU_GATE_FAILED (viz) -- last lines of the gate log follow"
    tail -n 20 /tmp/zgpose_viz_cpu_gate.log | tee -a "$LOG"
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
        -o "$OUT" --strict >> data/analyze_zgpose.log 2>&1
    rc=$?
    say "ANALYSIS_EXIT=$rc"
    if [ $rc -ne 0 ]; then
        say "ANALYSIS_FAILED -- see data/analyze_zgpose.log"
        tail -n 20 data/analyze_zgpose.log | tee -a "$LOG"
        exit $rc
    fi
fi

# ---- stage 3: latent-distribution coordinates (numpy-only compute)
if [ -f "$VIZ" ] && [ "${FORCE_VIZ:-$FORCE_ALL}" != "1" ]; then
    say "VIZ_SKIPPED (already present; FORCE_VIZ=1 or FORCE=1 to redo)"
else
    say "VIZ_START -> $VIZ"
    "$PY" -u visualize_latent_distribution.py compute -s "$SNAP" -o "$VIZ" \
        --epochs 0,25,50,100,150,200 --basis-epoch 200 >> data/compute_zgpose_viz.log 2>&1
    rc=$?
    say "VIZ_EXIT=$rc"
    if [ $rc -ne 0 ]; then
        say "VIZ_FAILED -- see data/compute_zgpose_viz.log"; exit $rc
    fi
fi

# ---- stage 4: collapse screen on the new cell, matched range + random-init.
# The baseline is per-architecture and must be measured, never borrowed
# (m4base/m4clear screens are already committed; only this cell is new).
if [ -f "data/screen_collapse/square_zgpose_latest.json" ] \
   && [ "${FORCE_SCREEN:-$FORCE_ALL}" != "1" ]; then
    say "SCREEN_SKIPPED (already present)"
else
    say "SCREEN_START (range 11,11)"
    "$PY" screen_collapse.py -c "$CKPT" -d "$GPU" --view-count-range 11,11 \
        -o "data/screen_collapse/square_zgpose_latest.json" \
        >> data/screen_square_zgpose.log 2>&1
    rc=$?
    "$PY" screen_collapse.py -c "$CKPT" -d "$GPU" --view-count-range 11,11 --random-init \
        -o "data/screen_collapse/square_zgpose_RANDOM_INIT.json" \
        >> data/screen_square_zgpose.log 2>&1
    rc2=$?
    say "SCREEN_DONE exit=$rc/$rc2"
    [ $rc -eq 0 ] && [ $rc2 -eq 0 ] || { say "SCREEN_FAILED -- see data/screen_square_zgpose.log"; exit 1; }
fi

# ---- stage 5: the relpose grid probe, IDENTICAL flags to the m4 pair (the
# NOTES *Latent probe* command scaled to this cell's K=11/range [1,11]).
if [ -f "data/probe_relpose_grid_square_zgpose/probe_relpose.json" ] \
   && [ "${FORCE_PROBE:-$FORCE_ALL}" != "1" ]; then
    say "PROBE_SKIPPED (already present)"
else
    say "PROBE_START (range 1,11; ~35 min/cell per NOTES *Traps*)"
    "$PY" -u probe_relpose.py -c "$CKPT" -o data/probe_relpose_grid_square_zgpose -d "$GPU" \
        --view-count-range 1,11 --stability-ranges "1,2;1,6;1,11;11,11" \
        --stability-states 256 --stability-repeats 6 --n-samples 2000 \
        --mlp-steps 2000 --num-threads 4 --random-init-control \
        >> data/probe_square_zgpose.log 2>&1
    rc=$?
    say "PROBE_DONE exit=$rc"
    [ $rc -eq 0 ] || { say "PROBE_FAILED -- see data/probe_square_zgpose.log"; exit 1; }
fi

# the documented comparability gate: the new cell's draws must MATCH the
# COMMITTED m4base probe's, range by range (identical flags make this
# checkable rather than asserted); mean_active must fit [1,11]
"$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import json
a = json.load(open('data/probe_relpose_grid_square_zgpose/probe_relpose.json'))
b = json.load(open('data/probe_relpose_grid_square_m4base/probe_relpose.json'))
fails = []
for r in sorted(set(a['latent_grid']) & set(b['latent_grid'])):
    fa = a['latent_grid'][r].get('draw_fingerprint')
    fb = b['latent_grid'][r].get('draw_fingerprint')
    print(f"PROBE_GATE range {r}: fingerprint zgpose={fa} m4base={fb} "
          f"{'MATCH' if fa == fb else 'MISMATCH'}")
    if fa != fb:
        fails.append(r)
ma = a['stats']['mean_active']
print(f"PROBE_GATE mean_active zgpose={ma:.3f} (want ~6.0 for [1,11])")
if not (5.5 < ma < 6.5):
    fails.append('mean_active')
if fails:
    raise SystemExit(f'PROBE_GATE_FAIL: {fails}')
print('PROBE_GATE PASS: fingerprints match the committed m4base probes, '
      'mean_active tracks [1,11]')
PYEOF
probe_rc=$?
say "PROBE_GATE_EXIT=$probe_rc"
[ $probe_rc -eq 0 ] || say "WARNING: probe comparability gate failed -- do not compare probe numbers across cells"

# ---- stage 6: the registered readouts, printed from the files -------------
say "REGISTERED READOUTS (criteria in data/zgpose_run.sh's header)"
"$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import json
import os
import numpy as np
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('eval', eval, replace=True)
from diffusion_policy.dataset.multiview_image_dataset import AUX_VIEW_POSE_KEY
d = json.load(open('data/analysis_zgpose/analyze_latent_series.json'))
a = np.array(d['fused']['n1_vs_full']['rel'])
tail = float(a[-25:].mean())
print(f'READOUT digest fused.rel: e200={a[200]:.5f} tail_mean(last 25)={tail:.5f} '
      f'max={a.max():.5f}@e{a.argmax()} (committed parents: tails 0.089 / 0.103; '
      f'peaks 0.128@e47 / 0.177@e20)')
print(f'READOUT instruments-moved (independent of aux): tail > 0.13 -> {tail > 0.13}')
p = json.load(open('data/probe_relpose_grid_square_zgpose/probe_relpose.json'))
s = p['latent_grid']['1,11']['zg_across_view_subsets']
print(f'READOUT probe zg_across_view_subsets(1,11)={s:.5f} '
      f'(committed parents: 0.0663 / 0.0780)')
print(f'READOUT instruments-moved (independent of aux): subsets > 0.09 -> {s > 0.09}')

# the registered aux bands: recompute the floor with the SAME dataset-only
# recipe as the run driver's CALIBRATION stage (seed 0, 64 samples) and read
# the run's final aux_loss, so the classification lands in committed files
with hydra.initialize_config_dir(
        config_dir=os.path.join(os.getcwd(), 'diffusion_policy', 'config'),
        version_base=None):
    cfg = hydra.compose(
        config_name='train_diffusion_unet_image_workspace_zgpose_latent')
dsd = hydra.utils.instantiate(cfg.task.dataset)
V = len(dsd.view_pool)
np.random.seed(0)
idxs = np.random.choice(len(dsd), size=64, replace=False)
tt = np.concatenate([dsd[int(i)][AUX_VIEW_POSE_KEY].numpy().reshape(-1, V, 9)
                     for i in idxs])
floor = float(((tt - tt.mean(0)) ** 2).mean())
rows = [json.loads(l) for l in open(
    'data/outputs/run_square_zgpose_s42_200ep/logs.json.txt') if l.strip()]
aux_e200 = [r['aux_loss'] for r in rows if 'aux_loss' in r][-1]
ratio = (aux_e200 / floor) if floor > 0 else float('inf')
band = ('<=0.5*floor (loss beaten)' if ratio <= 0.5 else
        'middle (PARTIAL)' if ratio < 0.9 else '>=0.9*floor (at floor)')
print(f'READOUT floor={floor:.6f} aux_e200={aux_e200:.6f} '
      f'ratio={ratio:.3f} -> band: {band}')
print('READOUT map (run header): aux beaten + instruments moved -> (i); aux '
      'beaten + flat -> (ii); at-floor + moved -> pressure reached the '
      'representation but the head did not fit; at-floor + flat -> falsifier')
PYEOF
say "READOUT_EXIT=$?"

# ---- stage 7: sizes + the digest, read from the files rather than assumed
say "SIZES $(du -h "$OUT/analyze_latent_series.json" 2>/dev/null | cut -f1) analysis json, \
$(du -h "$VIZ" 2>/dev/null | cut -f1) viz coords, \
$(du -sh "$SNAP" 2>/dev/null | cut -f1) of npz left under /data"
say "----- data/analysis_zgpose/summary.md -----"
cat "$OUT/summary.md" 2>/dev/null | tee -a "$LOG"
say "DONE -- outputs are UNCOMMITTED. Read them first, then commit:"
say "  git add $OUT data/screen_collapse/square_zgpose_latest.json \
data/screen_collapse/square_zgpose_RANDOM_INIT.json \
data/probe_relpose_grid_square_zgpose"
say "  # laptop, after committing: python3 visualize_latent_distribution.py plot \
-i $VIZ -d $OUT/analyze_latent_series.json -o $OUT/figures"
