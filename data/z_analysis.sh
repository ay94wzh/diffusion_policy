#!/bin/bash
# The z round's INSTRUMENT battery, on MIROC-SERVER (npz + checkpoints live
# under /data; this script must never run from the laptop). The same battery
# the m4base/zgpose/zvzg/prob cells got, so every comparison is against
# committed artifacts -- minus the prob round's variance stages (this round
# has no stochastic branch), plus the one reading the z round is the first to
# need: the SEMANTIC readout of the trained alignment head (risk R1's
# instrument -- L_align and the within/across cosines of P_s(z_v), against a
# re-initialized-head control; see analyze_semantic_head.py and data/z_run.sh).
#
# For each of cells a (zalign), b (zcons), c (zboth):
#   1. snapshot + npz gate -- layout, no gaps, arrays finite at both ends,
#      before any analysis is spent on a misconfigured run.
#   2. analyze_latent_series.py --strict over the 201 npz
#      -> data/analysis_z_<cell>/{analyze_latent_series.json,summary.md}
#   3. visualize_latent_distribution.py compute (6 epochs)
#      -> data/analysis_z_<cell>/latent_viz_coords.json
#      (laptop renders: python3 visualize_latent_distribution.py plot \
#         -i <coords> -d <digest> -o data/analysis_z_<cell>/figures)
#   4. collapse screen, matched range 11,11 + random-init, per cell
#   5. probe_relpose grid with IDENTICAL flags to m4base/m4clear/zgpose/zvzg/
#      prob_b/c/d, plus the comparability gate on draw fingerprints.
#   6. the four registered readouts, printed from the files against envelopes
#      COMPUTED FROM the committed grids, with the run header's margin
#      convention (moved iff outside the four-cell envelope by more than 10%
#      of its own width).
#   7. the SEMANTIC readout (analyze_semantic_head.py) per cell -- the trained
#      head vs the re-init control; the collapse signature is printed, not
#      judged (risk R1: report it, do not round it away).
#
# THE GIT POLICY: this script runs no git command. Outputs are committed only
# after a human has read them; the ~1.2-1.6 GB of npz per cell stay under /data.
#
# Run:  setsid bash -c 'bash data/z_analysis.sh' < /dev/null &
#   FORCE_ANALYSIS=1 / FORCE_VIZ=1 / FORCE_SCREEN=1 / FORCE_PROBE=1 /
#   FORCE_SEMANTIC=1 re-run one stage whose output exists (FORCE=1 means all).
#   CELLS=b GPU=cuda:1 split the battery across the two GPUs -- but never read
#   a contended timestamp as evidence of anything.
set -u
FORCE_ALL=${FORCE:-0}
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
GPU=${GPU:-cuda:0}
CELLS=${CELLS:-a b c}
EXPECT_NPZ=201
LOG=data/z_analysis.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: disk precondition (CLAUDE.md: check df on miroc-server)
say "DISK $(df -h / /data 2>/dev/null | tr '\n' ' ')"

# ---- stage 1: CPU gates. A guard must be able to fail AND print why.
for gate in test_analyze_latent_series test_visualize_latent_distribution \
            test_semantic_readout; do
    say "CPU gate: tests/${gate}.py"
    if ! "$PY" "tests/${gate}.py" > "/tmp/z_analysis_gate_${gate}.log" 2>&1; then
        say "CPU_GATE_FAILED (${gate}) -- not launching; last lines follow"
        tail -n 20 "/tmp/z_analysis_gate_${gate}.log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

for CELL in $CELLS; do
    RUN=/data/zihan/runs/run_square_z${CELL}_s42_200ep
    SNAP=$RUN/latent_snapshots
    CKPT=$RUN/checkpoints/latest.ckpt
    OUT=data/analysis_z_${CELL}
    GRID=data/probe_relpose_grid_square_z${CELL}
    [ -e "$CKPT" ] || { say "MISSING_CKPT $CKPT (cell $CELL)"; exit 1; }

    # ---- stage 2: snapshot + npz gate, read from the files rather than assumed
    say "NPZ_GATE_START cell=$CELL"
    if ! "$PY" - "$SNAP" "$EXPECT_NPZ" "$CELL" >> "$LOG" 2>&1 <<'PYEOF'
import glob, json, os, sys
import numpy as np

snap, expect, cell = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if not os.path.isdir(snap):
    raise SystemExit(f'no snapshots dir at {snap}')
eps = sorted(int(os.path.basename(p).split('_')[1].split('.')[0])
             for p in glob.glob(os.path.join(snap, 'epoch_*.npz')))
meta = json.load(open(os.path.join(snap, 'meta.json')))
have = set(eps)
gaps = [e for e in range(eps[0], eps[-1] + 1) if e not in have] if eps else []
ok = (len(eps) == expect and not gaps and meta['K'] == 11
      and meta['pool_ring'] == list(range(1, 12)) and meta['az0_pos'] == 5)
print(f'NPZ_GATE cell={cell} npz={len(eps)} first={eps[0] if eps else None} '
      f'last={eps[-1] if eps else None} gaps={gaps} K={meta["K"]} '
      f'pool={meta["pool_ring"][0]}..{meta["pool_ring"][-1]} az0_pos={meta["az0_pos"]}')
if not ok:
    raise SystemExit('NPZ_GATE FAILED: layout')

# the arrays themselves, at both ends of the run. The z rounds run the BASE
# latent probe (no Gaussian arrays): zv/zg/zg_n1/state_idx.
for ep in (eps[0], eps[-1]):
    z = np.load(os.path.join(snap, f'epoch_{ep:04d}.npz'), allow_pickle=False)
    keys = set(z.files)
    need = {'zv', 'zg', 'zg_n1', 'state_idx'}
    if not need <= keys:
        raise SystemExit(f'epoch {ep}: missing {sorted(need - keys)}')
    S, K, D = meta['n_states'], meta['K'], meta['fused_dim']
    if z['zv'].shape != (S, K, D) or not np.isfinite(z['zv']).all():
        raise SystemExit(f'epoch {ep}: zv shape/finite violation {z["zv"].shape}')
    for k in ('zg', 'zg_n1'):
        if z[k].shape != (S, D) or not np.isfinite(z[k]).all():
            raise SystemExit(f'epoch {ep}: {k} shape/finite violation {z[k].shape}')
    if z['state_idx'].shape != (S,):
        raise SystemExit(f'epoch {ep}: state_idx shape {z["state_idx"].shape}')
    print(f'NPZ_GATE cell={cell} epoch={ep} arrays OK '
          f'(zv {z["zv"].shape}, |zv|max {float(np.abs(z["zv"]).max()):.3f})')
print(f'NPZ_GATE PASS cell={cell}')
PYEOF
    then
        say "NPZ_GATE_FAILED cell=$CELL -- not launching this cell's battery"
        exit 1
    fi
    say "NPZ_GATE_PASSED cell=$CELL"

    # ---- stage 3: the digest
    if [ -f "$OUT/analyze_latent_series.json" ] && [ "${FORCE_ANALYSIS:-$FORCE_ALL}" != "1" ]; then
        say "ANALYSIS_SKIPPED cell=$CELL (already present; FORCE_ANALYSIS=1 to redo)"
    else
        say "ANALYSIS_START cell=$CELL -> $OUT"
        "$PY" -u analyze_latent_series.py -s "$SNAP" -l "$RUN/logs.json.txt" \
            -o "$OUT" --strict >> data/analyze_z_${CELL}.log 2>&1
        rc=$?
        say "ANALYSIS_EXIT=$rc cell=$CELL"
        if [ $rc -ne 0 ]; then
            say "ANALYSIS_FAILED -- see data/analyze_z_${CELL}.log"
            tail -n 20 data/analyze_z_${CELL}.log | tee -a "$LOG"
            exit $rc
        fi
    fi

    # ---- stage 4: latent-distribution coordinates (numpy-only compute)
    if [ -f "$OUT/latent_viz_coords.json" ] && [ "${FORCE_VIZ:-$FORCE_ALL}" != "1" ]; then
        say "VIZ_SKIPPED cell=$CELL (already present; FORCE_VIZ=1 to redo)"
    else
        say "VIZ_START cell=$CELL"
        "$PY" -u visualize_latent_distribution.py compute -s "$SNAP" \
            -o "$OUT/latent_viz_coords.json" --epochs 0,25,50,100,150,200 \
            --basis-epoch 200 >> data/compute_z_${CELL}_viz.log 2>&1
        rc=$?
        say "VIZ_EXIT=$rc cell=$CELL"
        [ $rc -eq 0 ] || { say "VIZ_FAILED -- see data/compute_z_${CELL}_viz.log"; exit $rc; }
    fi

    # ---- stage 5: collapse screen, matched range + random-init
    if [ -f "data/screen_collapse/square_z${CELL}_latest.json" ] \
       && [ "${FORCE_SCREEN:-$FORCE_ALL}" != "1" ]; then
        say "SCREEN_SKIPPED cell=$CELL (already present)"
    else
        say "SCREEN_START cell=$CELL (range 11,11)"
        "$PY" screen_collapse.py -c "$CKPT" -d "$GPU" --view-count-range 11,11 \
            -o "data/screen_collapse/square_z${CELL}_latest.json" \
            >> data/screen_square_z${CELL}.log 2>&1
        rc=$?
        "$PY" screen_collapse.py -c "$CKPT" -d "$GPU" --view-count-range 11,11 --random-init \
            -o "data/screen_collapse/square_z${CELL}_RANDOM_INIT.json" \
            >> data/screen_square_z${CELL}.log 2>&1
        rc2=$?
        say "SCREEN_DONE cell=$CELL exit=$rc/$rc2"
        [ $rc -eq 0 ] && [ $rc2 -eq 0 ] || { say "SCREEN_FAILED -- see data/screen_square_z${CELL}.log"; exit 1; }
    fi

    # ---- stage 6: the relpose grid probe, IDENTICAL flags to every K=11 arm
    if [ -f "$GRID/probe_relpose.json" ] && [ "${FORCE_PROBE:-$FORCE_ALL}" != "1" ]; then
        say "PROBE_SKIPPED cell=$CELL (already present)"
    else
        say "PROBE_START cell=$CELL (range 1,11; ~35 min/cell per NOTES *Traps*)"
        "$PY" -u probe_relpose.py -c "$CKPT" -o "$GRID" -d "$GPU" \
            --view-count-range 1,11 --stability-ranges "1,2;1,6;1,11;11,11" \
            --stability-states 256 --stability-repeats 6 --n-samples 2000 \
            --mlp-steps 2000 --num-threads 4 --random-init-control \
            >> data/probe_square_z${CELL}.log 2>&1
        rc=$?
        say "PROBE_DONE cell=$CELL exit=$rc"
        [ $rc -eq 0 ] || { say "PROBE_FAILED -- see data/probe_square_z${CELL}.log"; exit 1; }
    fi

    # ---- stage 7: the semantic readout -- the alignment mechanism's own
    # instrument. CPU-only (reads the last npz + the ckpt), ~1 min/cell.
    if [ -f "$OUT/semantic_readout.json" ] && [ "${FORCE_SEMANTIC:-$FORCE_ALL}" != "1" ]; then
        say "SEMANTIC_SKIPPED cell=$CELL (already present)"
    else
        say "SEMANTIC_START cell=$CELL"
        "$PY" -u analyze_semantic_head.py -c "$CKPT" -n "$SNAP" \
            -o "$OUT/semantic_readout.json" >> "$LOG" 2>&1
        rc=$?
        say "SEMANTIC_EXIT=$rc cell=$CELL"
        [ $rc -eq 0 ] || { say "SEMANTIC_FAILED cell=$CELL"; exit 1; }
    fi
done

# ---- the documented comparability gate: every cell's draws must MATCH all
# four committed references (m4base, m4clear, zgpose, zvzg) range by range --
# identical flags make this checkable rather than asserted
"$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import json

REFS = ('m4base', 'm4clear', 'zgpose', 'zvzg')
refs = {r: json.load(open(f'data/probe_relpose_grid_square_{r}/probe_relpose.json'))
        for r in REFS}
fails = []
for cell in 'abc':
    p = json.load(open(f'data/probe_relpose_grid_square_z{cell}/probe_relpose.json'))
    for rng in sorted(p['latent_grid']):
        fa = p['latent_grid'][rng].get('draw_fingerprint')
        line = [f'range {rng}: z{cell}={fa}']
        for name, ref in refs.items():
            match = fa == ref['latent_grid'].get(rng, {}).get('draw_fingerprint')
            line.append(f'{name}={"MATCH" if match else "MISMATCH"}')
            if not match:
                fails.append((cell, rng, name))
        print('PROBE_GATE ' + ' '.join(line))
    ma = p['stats']['mean_active']
    za = p['latent_grid']['1,11'].get('zg_abs_pose') or {}
    ridge = za.get('ridge') or {}
    print(f'PROBE_GATE z{cell} mean_active={ma:.3f} (want ~6.0 for [1,11]); '
          f'zg_abs_pose(1,11) n_train={za.get("n_train")} n_test={za.get("n_test")} '
          f'(committed refs: 110/28)')
    if not (5.5 < ma < 6.5):
        fails.append((cell, 'mean_active'))
print('PROBE_GATE PASS: fingerprints match ALL FOUR committed references'
      if not fails else f'PROBE_GATE_FAIL: {fails}')
if fails:
    raise SystemExit(1)
PYEOF
probe_rc=$?
say "PROBE_GATE_EXIT=$probe_rc"
[ $probe_rc -eq 0 ] || say "WARNING: probe comparability gate failed -- do not compare probe numbers across cells"

# ---- the registered readouts, printed from the files. The envelopes are
# computed FROM the committed grids (so they cannot go stale) and the margin
# is the run header's convention: 10% of each envelope's own width.
say "REGISTERED READOUTS (criteria in data/z_run.sh, section 2)"
"$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import json

REFS = ('m4base', 'm4clear', 'zgpose', 'zvzg')
refs = {r: json.load(open(f'data/probe_relpose_grid_square_{r}/probe_relpose.json'))
        for r in REFS}
# the digest readout's envelope comes from the DIGEST files, not the probe grid
DIGEST_DIRS = {'m4base': 'data/analysis_m4base', 'm4clear': 'data/analysis_clear',
               'zgpose': 'data/analysis_zgpose', 'zvzg': 'data/analysis_zvzg'}
digs = {r: json.load(open(f'{DIGEST_DIRS[r]}/analyze_latent_series.json'))
        for r in REFS}


def env(name, get, src=None):
    src = refs if src is None else src
    vals = [get(src[r]) for r in REFS]
    print(f'  envelope {name}: {[round(v, 4) if v is not None else None for v in vals]}')
    return min(vals), max(vals)


def rr(js):
    return js['latent_grid']['1,11']['zv_pair_ratio']


def ss(js):
    return js['latent_grid']['1,11']['zg_across_view_subsets']


def pose(js, key='r_mean_deg'):
    za = js['latent_grid']['1,11'].get('zg_abs_pose') or {}
    return ((za.get('ridge') or {}).get(key))


def tail(js):
    a = js['fused']['n1_vs_full']['rel']
    return sum(a[-25:]) / len(a[-25:])


def moved(value, lo, hi, margin):
    return value < lo - margin or value > hi + margin


CELLS = 'abc'
print('READOUT zv_pair_ratio(1,11) -- envelope and margin 0.20')
lo, hi = env('zv_pair_ratio', rr)
for cell in CELLS:
    p = json.load(open(f'data/probe_relpose_grid_square_z{cell}/probe_relpose.json'))
    v = p['latent_grid']['1,11']['zv_pair_ratio']
    print(f'READOUT zv_pair_ratio(1,11) z{cell}={v:.4f} '
          f'-> {"MOVED" if moved(v, lo, hi, 0.20) else "in-range"}')

print('READOUT zg_across_view_subsets(1,11) -- envelope and margin 0.03')
lo, hi = env('zg_across_view_subsets', ss)
for cell in CELLS:
    p = json.load(open(f'data/probe_relpose_grid_square_z{cell}/probe_relpose.json'))
    v = p['latent_grid']['1,11']['zg_across_view_subsets']
    print(f'READOUT zg_across_view_subsets(1,11) z{cell}={v:.4f} '
          f'-> {"MOVED" if moved(v, lo, hi, 0.03) else "in-range"}')

print('READOUT digest rel(zg_n1, zg_full) tail(25) -- envelope and margin 0.03')
lo, hi = env('digest_tail', tail, src=digs)
for cell in CELLS:
    d = json.load(open(f'data/analysis_z_{cell}/analyze_latent_series.json'))
    v = tail(d)
    print(f'READOUT digest tail z{cell}={v:.4f} '
          f'-> {"MOVED" if moved(v, lo, hi, 0.03) else "in-range"}')

print('READOUT probe N=1 zg pose readout (ridge, 1,11) -- envelope, '
      'margins 1.7 deg / 2.1 cm')
lo_r, hi_r = env('N1_zg_pose_rot_deg', lambda js: pose(js, 'r_mean_deg'))
lo_t, hi_t = env('N1_zg_pose_t_cm', lambda js: pose(js, 't_rmse_cm'))
for cell in CELLS:
    p = json.load(open(f'data/probe_relpose_grid_square_z{cell}/probe_relpose.json'))
    r, t = pose(p, 'r_mean_deg'), pose(p, 't_rmse_cm')
    tag = ('MOVED' if (moved(r, lo_r, hi_r, 1.7) or moved(t, lo_t, hi_t, 2.1))
           else 'in-range')
    print(f'READOUT N=1 zg pose z{cell}: rot {r:.3f} deg / t {t:.3f} cm -> {tag}')

print('READOUT summary lines are LEADs, not verdicts (the margins are a stated '
      'convention; these statistics\' across-seed noise is unmeasured)')

print('SEMANTIC readout (the round\'s own instrument; cell b\'s head never saw '
      'its loss -- read it as the untrained control):')
for cell in CELLS:
    s = json.load(open(f'data/analysis_z_{cell}/semantic_readout.json'))
    t, c = s['trained'], s['reinit_control']
    print(f'SEMANTIC z{cell}: trained L {t["L_align"]:.4f} cos_within '
          f'{t["cos_within"]:.4f} cos_across {t["cos_across"]:.4f} | reinit L '
          f'{c["L_align"]:.4f} cos_within {c["cos_within"]:.4f} cos_across '
          f'{c["cos_across"]:.4f} | collapse_signature='
          f'{s["collapse_signature"]}')
PYEOF
say "READOUT_EXIT=$?"

# ---- stage 9: sizes + the digests, read from the files rather than assumed
say "SIZES $(du -sh data/analysis_z_a data/analysis_z_b data/analysis_z_c \
data/probe_relpose_grid_square_z_a data/probe_relpose_grid_square_z_b \
data/probe_relpose_grid_square_z_c 2>/dev/null | tr '\n' ' ')"
for CELL in a c; do
    say "----- data/analysis_z_${CELL}/summary.md (head) -----"
    head -40 data/analysis_z_${CELL}/summary.md 2>/dev/null | tee -a "$LOG"
done
say "DONE -- outputs are UNCOMMITTED. Read them first, then commit:"
say "  git add data/analysis_z_a data/analysis_z_b data/analysis_z_c \
data/probe_relpose_grid_square_z_a data/probe_relpose_grid_square_z_b \
data/probe_relpose_grid_square_z_c \
data/screen_collapse/square_z{a,b,c}_latest.json data/screen_collapse/square_z{a,b,c}_RANDOM_INIT.json \
data/z_analysis.sh"
say "  # laptop, after committing: python3 visualize_latent_distribution.py plot \
-i data/analysis_z_<cell>/latent_viz_coords.json \
-d data/analysis_z_<cell>/analyze_latent_series.json \
-o data/analysis_z_<cell>/figures"
