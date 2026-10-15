#!/bin/bash
# The prob round's INSTRUMENT battery, on MIROC-SERVER (npz + checkpoints live
# under /data; this script must never run from the laptop). The same battery the
# m4base/zgpose/zvzg cells got, so every comparison is against committed
# artifacts -- plus the one reading those cells had no need for: the
# variance-quality read the run driver deferred (see stage 8).
#
# For each of cells B (MHA over the means), C (PoE) and D (mean of the means):
#   1. snapshot + npz gate -- the prob probe's own arrays are checked before
#      105 minutes of probing are spent on a misconfigured run. The gate is
#      deliberately strict in one direction: cells B and D must have a logvar
#      branch that is EXACTLY zero everywhere (that is the premise the round's
#      "do not read B/D's variance" caveat rests on) and cell C's must not.
#   2. analyze_latent_series.py --strict over the 201 npz
#      -> data/analysis_prob_<cell>/{analyze_latent_series.json,summary.md}
#      (read against data/analysis_m4base/ etc. for the registered readouts)
#   3. visualize_latent_distribution.py compute (6 epochs)
#      -> data/analysis_prob_<cell>/latent_viz_coords.json
#      (laptop renders: python3 visualize_latent_distribution.py plot \
#         -i <coords> -d <digest> -o data/analysis_prob_<cell>/figures)
#   4. collapse screen, matched range 11,11 + random-init, per cell (the screen
#      is only comparable against the same architecture's own baseline).
#   5. probe_relpose grid with IDENTICAL flags to m4base/m4clear/zgpose/zvzg,
#      plus a gate that the draw fingerprints match ALL FOUR committed grids --
#      the four are byte-identical, so this is checkable rather than asserted.
#   6. the four registered readouts, printed from the files against envelopes
#      COMPUTED FROM the committed grids (never hardcoded), with the run
#      header's margin convention: moved iff outside the four-cell envelope by
#      more than 10% of that envelope's own width.
#   7. stage 8: the variance-quality read (analyze_variance_quality.py) on cell
#      C, and the dead-logvar assertion on B and D. The script's header carries
#      the thresholds the run driver deferred to it (data/prob_run.sh:94-103).
#
# The four registered readouts and their committed envelopes (computed at
# runtime in stage 7 from data/probe_relpose_grid_square_{m4base,m4clear,
# zgpose,zvzg}/probe_relpose.json; the values below are what those files hold
# today, quoted from data/prob_run.sh:81-93):
#     probe  zv_pair_ratio(1,11)            envelope [0.267, 2.255]  margin 0.20
#     probe  zg_across_view_subsets(1,11)   envelope [0.066, 0.327]  margin 0.03
#     digest rel(zg_n1, zg_full) tail(25)   envelope [0.089, 0.433]  margin 0.03
#     probe  N=1 zg pose readout (ridge)    envelope [1.20 deg, 17.8 deg]
#                                            / [1.90 cm, 23.2 cm], margin 1.7 deg / 2.1 cm
# NOTE the N=1 pose readout comes from the SAME grid JSON (latent_grid["1,11"]
# .zg_abs_pose.ridge) -- the earlier cells' data/probe_zg_square_* runs are NOT
# needed here, and adding one would cost ~30 min/cell for a number this battery
# already has.
#
# Cross-arm reading traps (from data/prob_run.sh's header): this arm's logged
# `aux_loss` EQUALS its `aux_action_loss` (unweighted; unlike zvzg, where
# aux_loss is the weighted total); batch-0 train_loss is cell-specific (A/B fuse
# via MHA, C/D start at the masked mean); and all three cells are instrument-
# comparable to m4base (every ring view trains) but NOT ladder-comparable.
#
# THE GIT POLICY: this script runs no git command. Outputs are committed only
# after a human has read them; the ~1.2-1.6 GB of npz per cell stay under /data.
#
# Run:  setsid bash -c 'bash data/prob_analysis.sh' < /dev/null &
#   FORCE_ANALYSIS=1 / FORCE_VIZ=1 / FORCE_SCREEN=1 / FORCE_PROBE=1 /
#   FORCE_QUALITY=1 re-run one stage whose output exists (FORCE=1 means all).
#   CELLS=b / CELLS=c GPU=cuda:1 split the battery across the two GPUs -- but
#   never read a contended timestamp as evidence of anything.
set -u
FORCE_ALL=${FORCE:-0}
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
GPU=${GPU:-cuda:0}
CELLS=${CELLS:-b c d}
EXPECT_NPZ=201
VQOUT=data/analysis_prob_variance
LOG=data/prob_analysis.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: disk precondition (CLAUDE.md: check df on miroc-server, /data
# is where the run dirs live; the npz are ~1.2-1.6 GB per cell)
say "DISK $(df -h / /data 2>/dev/null | tr '\n' ' ')"

# ---- stage 1: CPU gates. A guard must be able to fail AND print why: status
# unpiped, log tail shown on failure.
for gate in test_variance_quality test_analyze_latent_series \
            test_visualize_latent_distribution; do
    say "CPU gate: tests/${gate}.py"
    if ! "$PY" "tests/${gate}.py" > "/tmp/prob_analysis_gate_${gate}.log" 2>&1; then
        say "CPU_GATE_FAILED (${gate}) -- not launching; last lines follow"
        tail -n 20 "/tmp/prob_analysis_gate_${gate}.log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

for CELL in $CELLS; do
    RUN=/data/zihan/runs/run_square_prob_${CELL}_s42_200ep
    SNAP=$RUN/latent_snapshots
    CKPT=$RUN/checkpoints/latest.ckpt
    OUT=data/analysis_prob_${CELL}
    GRID=data/probe_relpose_grid_square_prob_${CELL}
    [ -e "$CKPT" ] || { say "MISSING_CKPT $CKPT (cell $CELL)"; exit 1; }

    # ---- stage 2: snapshot + npz gate, read from the files rather than assumed
    say "NPZ_GATE_START cell=$CELL"
    if ! "$PY" - "$SNAP" "$EXPECT_NPZ" "$CELL" >> "$LOG" 2>&1 <<'PYEOF'
import glob, json, os, sys
import numpy as np

snap, expect, cell = sys.argv[1], int(sys.argv[2]), sys.argv[3]
POE = cell == 'c'
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

# the arrays themselves, at both ends of the run -- meta.json's `arrays` block
# is written only if absent, so the npz are the source of truth
for ep in (eps[0], eps[-1]):
    z = np.load(os.path.join(snap, f'epoch_{ep:04d}.npz'), allow_pickle=False)
    keys = set(z.files)
    need = {'mu', 'lv', 'lv_n1'}
    if not need <= keys:
        raise SystemExit(f'epoch {ep}: missing {sorted(need - keys)}')
    S, K = meta['n_states'], meta['K']
    for k in ('mu', 'lv', 'lv_n1'):
        a = z[k]
        if a.shape != (S, K, meta['fused_dim']) or not np.isfinite(a).all():
            raise SystemExit(f'epoch {ep}: {k} shape/finite violation {a.shape}')
    lv, lv_n1 = z['lv'], z['lv_n1']
    if POE:
        if 'lvg' not in keys or z['lvg'].shape != (S, meta['fused_dim']):
            raise SystemExit(f'epoch {ep}: lvg missing/wrong for the PoE cell')
        if float(np.abs(lv).max()) == 0.0:
            raise SystemExit(f'epoch {ep}: PoE cell carries an all-zero logvar')
    else:
        if 'lvg' in keys:
            raise SystemExit(f'epoch {ep}: non-PoE cell carries lvg')
        if float(np.abs(lv).max()) != 0.0:
            raise SystemExit(
                f'epoch {ep}: cell {cell} has a LIVE logvar branch '
                f'(max|lv|={float(np.abs(lv).max()):.3e}) -- the "do not read '
                f'B/D variance" premise from data/prob_run.sh is wrong')
    # the N=1 column must equal the full-draw column (the per-view-independence
    # premise the offline subset fusion rests on), and the dead columns zero
    az0 = meta['az0_pos']
    sel = [i for i in range(K) if i != az0]
    if float(np.abs(lv_n1[:, sel, :]).max()) != 0.0:
        raise SystemExit(f'epoch {ep}: lv_n1 has non-zero dead columns')
    # tolerance is a few fp16 ulps at |lv| ~ 1.3 (ulp ~ 1e-3): the two columns
    # are the same tensor computed at N=1 vs full draw, and cuDNN kernel choice
    # differs between those shapes -- an EXACT match is not the right gate here
    # (it is for the dead columns above, where the mask, not the fp16 cast, is
    # what decides the value)
    d = float(np.abs(lv_n1[:, az0, :] - lv[:, az0, :]).max())
    if d > 5e-3:
        raise SystemExit(f'epoch {ep}: lv_n1(az0) differs from lv(az0) by {d:.2e}')
    print(f'NPZ_GATE cell={cell} epoch={ep} arrays OK '
          f'({"poe" if POE else "dead-logvar"}, az0 lv delta {d:.2e}, '
          f'max|lv| {float(np.abs(lv).max()):.4g})')
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
            -o "$OUT" --strict >> data/analyze_prob_${CELL}.log 2>&1
        rc=$?
        say "ANALYSIS_EXIT=$rc cell=$CELL"
        if [ $rc -ne 0 ]; then
            say "ANALYSIS_FAILED -- see data/analyze_prob_${CELL}.log"
            tail -n 20 data/analyze_prob_${CELL}.log | tee -a "$LOG"
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
            --basis-epoch 200 >> data/compute_prob_${CELL}_viz.log 2>&1
        rc=$?
        say "VIZ_EXIT=$rc cell=$CELL"
        [ $rc -eq 0 ] || { say "VIZ_FAILED -- see data/compute_prob_${CELL}_viz.log"; exit $rc; }
    fi

    # ---- stage 5: collapse screen, matched range + random-init
    if [ -f "data/screen_collapse/square_prob_${CELL}_latest.json" ] \
       && [ "${FORCE_SCREEN:-$FORCE_ALL}" != "1" ]; then
        say "SCREEN_SKIPPED cell=$CELL (already present)"
    else
        say "SCREEN_START cell=$CELL (range 11,11)"
        "$PY" screen_collapse.py -c "$CKPT" -d "$GPU" --view-count-range 11,11 \
            -o "data/screen_collapse/square_prob_${CELL}_latest.json" \
            >> data/screen_square_prob_${CELL}.log 2>&1
        rc=$?
        "$PY" screen_collapse.py -c "$CKPT" -d "$GPU" --view-count-range 11,11 --random-init \
            -o "data/screen_collapse/square_prob_${CELL}_RANDOM_INIT.json" \
            >> data/screen_square_prob_${CELL}.log 2>&1
        rc2=$?
        say "SCREEN_DONE cell=$CELL exit=$rc/$rc2"
        [ $rc -eq 0 ] && [ $rc2 -eq 0 ] || { say "SCREEN_FAILED -- see data/screen_square_prob_${CELL}.log"; exit 1; }
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
            >> data/probe_square_prob_${CELL}.log 2>&1
        rc=$?
        say "PROBE_DONE cell=$CELL exit=$rc"
        [ $rc -eq 0 ] || { say "PROBE_FAILED -- see data/probe_square_prob_${CELL}.log"; exit 1; }
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
for cell in 'bcd':
    p = json.load(open(f'data/probe_relpose_grid_square_prob_{cell}/probe_relpose.json'))
    for rng in sorted(p['latent_grid']):
        fa = p['latent_grid'][rng].get('draw_fingerprint')
        line = [f'range {rng}: prob_{cell}={fa}']
        for name, ref in refs.items():
            match = fa == ref['latent_grid'].get(rng, {}).get('draw_fingerprint')
            line.append(f'{name}={"MATCH" if match else "MISMATCH"}')
            if not match:
                fails.append((cell, rng, name))
        print('PROBE_GATE ' + ' '.join(line))
    ma = p['stats']['mean_active']
    za = p['latent_grid']['1,11'].get('zg_abs_pose') or {}
    ridge = za.get('ridge') or {}
    print(f'PROBE_GATE prob_{cell} mean_active={ma:.3f} (want ~6.0 for [1,11]); '
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

# ---- stage 7: the registered readouts, printed from the files. The envelopes
# are computed FROM the committed grids (so they cannot go stale) and the
# margin is the run header's convention: 10% of each envelope's own width.
say "REGISTERED READOUTS (criteria in data/prob_run.sh:81-93)"
"$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import json

REFS = ('m4base', 'm4clear', 'zgpose', 'zvzg')
refs = {r: json.load(open(f'data/probe_relpose_grid_square_{r}/probe_relpose.json'))
        for r in REFS}
# the digest readout's envelope comes from the DIGEST files, not the probe grid
# -- and m4clear's digest lives under analysis_clear (the run's own name)
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


print('READOUT zv_pair_ratio(1,11) -- envelope and margin 0.20')
lo, hi = env('zv_pair_ratio', rr)
for cell in 'bcd':
    p = json.load(open(f'data/probe_relpose_grid_square_prob_{cell}/probe_relpose.json'))
    v = p['latent_grid']['1,11']['zv_pair_ratio']
    print(f'READOUT zv_pair_ratio(1,11) prob_{cell}={v:.4f} '
          f'-> {"MOVED" if moved(v, lo, hi, 0.20) else "in-range"}')

print('READOUT zg_across_view_subsets(1,11) -- envelope and margin 0.03')
lo, hi = env('zg_across_view_subsets', ss)
for cell in 'bcd':
    p = json.load(open(f'data/probe_relpose_grid_square_prob_{cell}/probe_relpose.json'))
    v = p['latent_grid']['1,11']['zg_across_view_subsets']
    print(f'READOUT zg_across_view_subsets(1,11) prob_{cell}={v:.4f} '
          f'-> {"MOVED" if moved(v, lo, hi, 0.03) else "in-range"}')

print('READOUT digest rel(zg_n1, zg_full) tail(25) -- envelope and margin 0.03')
lo, hi = env('digest_tail', tail, src=digs)
for cell in 'bcd':
    d = json.load(open(f'data/analysis_prob_{cell}/analyze_latent_series.json'))
    v = tail(d)
    print(f'READOUT digest tail prob_{cell}={v:.4f} '
          f'-> {"MOVED" if moved(v, lo, hi, 0.03) else "in-range"}')

print('READOUT probe N=1 zg pose readout (ridge, 1,11) -- envelope, '
      'margins 1.7 deg / 2.1 cm')
lo_r, hi_r = env('N1_zg_pose_rot_deg', lambda js: pose(js, 'r_mean_deg'))
lo_t, hi_t = env('N1_zg_pose_t_cm', lambda js: pose(js, 't_rmse_cm'))
for cell in 'bcd':
    p = json.load(open(f'data/probe_relpose_grid_square_prob_{cell}/probe_relpose.json'))
    r, t = pose(p, 'r_mean_deg'), pose(p, 't_rmse_cm')
    tag = ('MOVED' if (moved(r, lo_r, hi_r, 1.7) or moved(t, lo_t, hi_t, 2.1))
           else 'in-range')
    print(f'READOUT N=1 zg pose prob_{cell}: rot {r:.3f} deg / t {t:.3f} cm -> {tag}')

print('READOUT summary lines are LEADs, not verdicts (the margins are a stated '
      'convention; these statistics\' across-seed noise is unmeasured)')
PYEOF
say "READOUT_EXIT=$?"

# ---- stage 8: the variance-quality read (the run driver's deferred stage)
if [ -f "$VQOUT/variance_quality.json" ] && [ "${FORCE_QUALITY:-$FORCE_ALL}" != "1" ]; then
    say "VQ_SKIPPED (already present; FORCE_QUALITY=1 to redo)"
else
    RUN_C=/data/zihan/runs/run_square_prob_c_s42_200ep
    [ -e "$RUN_C/checkpoints/latest.ckpt" ] || { say "MISSING_CKPT for cell c"; exit 1; }
    say "VQ_START (cell c, full read)"
    "$PY" -u analyze_variance_quality.py -c "$RUN_C/checkpoints/latest.ckpt" \
        -o "$VQOUT" -d "$GPU" -l "$RUN_C/logs.json.txt" \
        >> data/analyze_prob_variance.log 2>&1
    rc=$?
    say "VQ_EXIT=$rc (cell c)"
    if [ $rc -ne 0 ]; then
        say "VQ_FAILED -- see data/analyze_prob_variance.log"
        tail -n 20 data/analyze_prob_variance.log | tee -a "$LOG"
        exit $rc
    fi
    for CELL in b d; do
        RUN_ANY=/data/zihan/runs/run_square_prob_${CELL}_s42_200ep
        say "VQ_DEAD_ASSERT cell=$CELL"
        "$PY" -u analyze_variance_quality.py -c "$RUN_ANY/checkpoints/latest.ckpt" \
            -o "$VQOUT/dead_${CELL}" --assert-dead-logvar \
            >> data/analyze_prob_variance.log 2>&1
        rc=$?
        say "VQ_DEAD_EXIT=$rc (cell $CELL)"
        if [ $rc -ne 0 ]; then
            say "VQ_DEAD_FAILED cell=$CELL -- the dead-logvar premise is WRONG; "
            say "  do not read this round's variance numbers in PROGRESS.md without fixing it"
            exit $rc
        fi
    done
fi

# ---- stage 9: sizes + the digest, read from the files rather than assumed
say "SIZES $(du -sh data/analysis_prob_b data/analysis_prob_c data/analysis_prob_d \
data/probe_relpose_grid_square_prob_b data/probe_relpose_grid_square_prob_c \
data/probe_relpose_grid_square_prob_d $VQOUT 2>/dev/null | tr '\n' ' ')"
say "----- data/analysis_prob_c/summary.md (head) -----"
head -40 data/analysis_prob_c/summary.md 2>/dev/null | tee -a "$LOG"
say "----- $VQOUT/summary.md -----"
cat "$VQOUT/summary.md" 2>/dev/null | tee -a "$LOG"
say "DONE -- outputs are UNCOMMITTED. Read them first, then commit:"
say "  git add data/analysis_prob_b data/analysis_prob_c data/analysis_prob_d \
$VQOUT data/probe_relpose_grid_square_prob_b data/probe_relpose_grid_square_prob_c \
data/probe_relpose_grid_square_prob_d \
data/screen_collapse/square_prob_b_latest.json data/screen_collapse/square_prob_b_RANDOM_INIT.json \
data/screen_collapse/square_prob_c_latest.json data/screen_collapse/square_prob_c_RANDOM_INIT.json \
data/screen_collapse/square_prob_d_latest.json data/screen_collapse/square_prob_d_RANDOM_INIT.json \
data/prob_analysis.sh"
say "  # laptop, after committing: python3 visualize_latent_distribution.py plot \
-i data/analysis_prob_<cell>/latent_viz_coords.json \
-d data/analysis_prob_<cell>/analyze_latent_series.json \
-o data/analysis_prob_<cell>/figures"
