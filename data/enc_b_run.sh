#!/bin/bash
# Encoder ladder stage B: the corrected EE-history protocol vs the legacy one.
#
# B0 = the new `enc_b` task with `task.dataset.eef_hist_window=sample` (the
#      legacy sample-window clip; with To=2 this is [B,B,B,B]/[B,B,B,B+1]).
# B1 = the same task with the config default `episode` (true past H poses
#      ending at each obs frame, clipped only at the episode start).
#
# ONE variable: the history window. Data, K=11 slots over pool [1..11],
# N in [1,11], the M3 policy (no aux head), the latent probe, seed 42 and all
# schedules are identical. The view-draw/crop RNG is unchanged by the flag
# (it changes only the history tensor values), so the pair is a clean test.
#
# ---- PRE-REGISTRATION (fixed before launch; read the result against this) ----
#
# The measured prior is the M3/M4 "history is inert" null, but it was
# confounded by the protocol: training history never read the true past. If
# the corrected history matters, B1 differs from B0 beyond the 0.15
# mean-over-viewpoints band on a registered mean; if it does not, the history
# null is now clean. Direction, if any, is NOT predicted: the branch now
# carries motion the model never saw, but every conditioning path so far has
# been behaviourally null. Registered means (50 paired episodes):
#   in-dist ring (11 vp), off-grid +-7.5..+-67.5 (10 vp),
#   past-ring +-82.5/+-90 (4 vp), elevation el_0/+-15 (descriptive cells).
# B1 becomes the S/R/RH/RHG ladder's internal baseline; m4base remains the
# external reference (0.796 / 0.774, committed).
#
# Mechanism-live read: the FiLM heads are zero-initialized, so growth of the
# per-stage gamma/beta norms off zero is the branch-activation gate. The
# per-epoch latent probe is the existing instrument; compare B0/B1 against
# the committed clear/m4base series with the same probe.
#
# Runtime reference: the clear run was 201 epochs in 3 h 32 m; two runs are
# ~7 h sequential. Check `df -h /data` on miroc-server first (a checkpoint is
# 4.62 GB and topk.k=1 roughly doubles a run's footprint).
#
# Run to completion with:
#   CELL=both setsid bash -c 'bash data/enc_b_run.sh' < /dev/null &
# or one cell at a time with CELL=b0 / CELL=b1.
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
DEV=${DEV:-cuda:0}
CELL=${CELL:-both}
LOG=data/enc_b_run.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates that apply to the no-aux B family.
for t in tests/test_multiview_dataset.py tests/test_latent_probe_hook.py; do
    say "CPU gate: $t"
    gate_log="/tmp/enc_b_cpu_gate_$(basename "$t").log"
    if ! "$PY" "$t" > "$gate_log" 2>&1; then
        say "CPU_GATE_FAILED ($t) -- not launching; last lines follow"
        tail -n 20 "$gate_log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

run_cell() {
    cell=$1
    window=$2
    run=$3
    say "TRAIN_START $cell (eef_hist_window=$window) -> $run ($DEV)"
    "$PY" train.py --config-name=train_diffusion_unet_image_workspace_enc_b_latent \
        task=enc_b_image_abs_multiview_az75 \
        task.dataset.eef_hist_window="$window" \
        training.seed=42 training.num_epochs=201 training.device="$DEV" \
        training.resume=false \
        dataloader.num_workers=14 val_dataloader.num_workers=2 \
        checkpoint.topk.k=1 \
        logging.project=diffusion_policy_view \
        hydra.run.dir="$run" >> "data/train_enc_b_${cell}.log" 2>&1 &
    pid=$!
    say "TRAIN_PID($cell)=$pid"

    # The launch gate: inspect the RESOLVED config, not a contended epoch time.
    sleep 90
    if [ -f "$run/.hydra/config.yaml" ]; then
        "$PY" - "$run/.hydra/config.yaml" "$window" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
want = sys.argv[2]
ds = c['task']['dataset']
pool, rng = ds['view_pool'], ds['view_count_range']
slots = c['task']['env_runner']['m3_slots']
probe = c.get('latent_probe', {})
ok = (sorted(pool) == list(range(1, 12)) and tuple(rng) == (1, 11)
      and slots == 11 and ds.get('eef_hist_window') == want
      and ds.get('emit_aux_action') is False
      and probe.get('n_states') == 128 and probe.get('az0_view') == 6)
print(f"OVERRIDE_GATE {'PASS' if ok else 'FAIL'}: pool={pool} range={rng} "
      f"slots={slots} window={ds.get('eef_hist_window')!r} "
      f"aux={ds.get('emit_aux_action')} "
      f"probe=({probe.get('n_states')}, az0_view={probe.get('az0_view')}) "
      f"(want pool 1..11, range [1,11], slots 11, window {want!r}, "
      f"aux False, probe 128/az0_view 6)")
PYEOF
    else
        say "OVERRIDE_GATE UNKNOWN: $run/.hydra/config.yaml not written yet"
    fi

    wait "$pid"
    say "TRAIN_EXIT($cell)=$?"
    mkdir -p "data/outputs/run_square_enc_${cell}_s42_200ep"
    cp -f "$run/logs.json.txt" \
        "data/outputs/run_square_enc_${cell}_s42_200ep/logs.json.txt"
    say "LOG_COPIED_BACK($cell)"
}

case "$CELL" in
    b0)   run_cell b0 sample  /data/zihan/runs/run_square_enc_b0_s42_200ep ;;
    b1)   run_cell b1 episode /data/zihan/runs/run_square_enc_b1_s42_200ep ;;
    both)
        run_cell b0 sample  /data/zihan/runs/run_square_enc_b0_s42_200ep
        run_cell b1 episode /data/zihan/runs/run_square_enc_b1_s42_200ep
        ;;
    *) echo "CELL must be b0, b1 or both (got $CELL)" >&2; exit 2 ;;
esac

say "DONE -- next: CELL=$CELL data/enc_b_eval.sh"
