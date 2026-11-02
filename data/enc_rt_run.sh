#!/bin/bash
# B1 residual token-interaction screen: A/B/C/D warm-start continuation.
#
# Same B1 checkpoint, same 30-epoch budget, fresh optimizer/scheduler in every
# cell. The only difference is `policy.obs_encoder.token_mixer_mode`:
#   a=none  b=single  c=intra  d=cross
#
# The workspace copies B1 model+EMA, restores the EMA continuation step from
# the checkpoint's global_step, and resets every RNG stream after
# env_runner/wandb construction. `skip_rollout=true` and `topk.k=0` keep the
# epoch-0 stream free of a model-dependent rollout; the fixed endpoint is the
# epoch-30 EMA saved as `latest.ckpt` by the workspace.
#
# Run to completion with:
#   CELLS="a b c d" setsid bash -c 'bash data/enc_rt_run.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
DEV=${DEV:-cuda:0}
CELLS=${CELLS:-"a b c d"}
WARM=${WARM:-/data/zihan/runs/run_square_enc_b1_s42_200ep/checkpoints/latest.ckpt}
LOG=data/enc_rt_run.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates that apply to the whole B1 family.
for t in tests/test_multiview_dataset.py tests/test_latent_probe_hook.py \
         tests/test_view_conditioned_obs_encoder.py \
         tests/test_prob_gaussian_fusion.py tests/test_aux_action_heads.py \
         tests/test_residual_token_view_conditioned_obs_encoder.py \
         tests/test_enc_rt_warmstart.py; do
    say "CPU gate: $t"
    gate_log="/tmp/enc_rt_cpu_gate_$(basename "$t").log"
    if ! "$PY" "$t" > "$gate_log" 2>&1; then
        say "CPU_GATE_FAILED ($t) -- not launching; last lines follow"
        tail -n 20 "$gate_log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

# ---- stage 1: the real B1 checkpoint must load into the new class exactly.
[ -e "$WARM" ] || { say "MISSING_WARMSTART $WARM"; exit 1; }
say "WARMSTART_CHECK $WARM"
if ! "$PY" check_enc_rt_warmstart.py --checkpoint "$WARM" \
        > /tmp/enc_rt_warmstart_check.log 2>&1; then
    say "WARMSTART_CHECK_FAILED"; tail -n 30 /tmp/enc_rt_warmstart_check.log | tee -a "$LOG"
    exit 1
fi
say "WARMSTART_CHECK_PASSED"

run_cell() {
    cell=$1
    mode=$2
    run="/data/zihan/runs/run_square_enc_rt_${cell}_s42_warm30"
    train_log="data/train_enc_rt_${cell}.log"
    if [ -e "$run" ]; then
        say "RUN_DIR_EXISTS($cell) $run -- refusing to resume/overwrite"
        return 1
    fi
    say "TRAIN_START $cell (mode=$mode) -> $run ($DEV)"
    "$PY" train.py \
        --config-name=train_diffusion_unet_image_workspace_enc_rt_latent \
        task=enc_b_image_abs_multiview_az75 \
        policy.obs_encoder.token_mixer_mode="$mode" \
        warmstart.checkpoint="$WARM" \
        training.seed=42 training.num_epochs=30 training.device="$DEV" \
        training.resume=false training.skip_rollout=true \
        dataloader.num_workers=14 val_dataloader.num_workers=2 \
        checkpoint.topk.k=0 \
        logging.project=diffusion_policy_view \
        logging.mode="${LOG_MODE:-offline}" \
        hydra.run.dir="$run" > "$train_log" 2>&1 &
    pid=$!
    say "TRAIN_PID($cell)=$pid"

    # The launch gate: wait for the resolved config, then inspect it.
    for _ in $(seq 1 30); do
        [ -f "$run/.hydra/config.yaml" ] && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 10
    done
    if [ ! -f "$run/.hydra/config.yaml" ]; then
        say "CONFIG_GATE_FAILED($cell): no .hydra/config.yaml"; kill "$pid" 2>/dev/null
        return 1
    fi
    if ! "$PY" - "$run/.hydra/config.yaml" "$mode" "$WARM" >> "$LOG" 2>&1 <<'PYEOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
mode, warm = sys.argv[2], sys.argv[3]
enc = c['policy']['obs_encoder']; ds = c['task']['dataset']
tr = c['training']; ck = c['checkpoint']; er = c['task']['env_runner']
ws = c.get('warmstart', {})
ok = (
    enc['token_mixer_mode'] == mode
    and enc['token_mixer_heads'] == 8
    and enc['token_mixer_ffn_dim'] == 1024
    and enc['fused_dim'] == 512
    and tuple(enc['crop_shape']) == (76, 76)
    and ds['dataset_path'].endswith('square_ph_ring13.zarr')
    and ds['eef_hist_window'] == 'episode'
    and sorted(ds['view_pool']) == list(range(1, 12))
    and tuple(ds['view_count_range']) == (1, 11)
    and ds['eef_hist_steps'] == 4
    and tr['num_epochs'] == 30
    and tr['seed'] == 42
    and tr['resume'] is False
    and tr['skip_rollout'] is True
    and tr['lr_warmup_steps'] == 500
    and c['optimizer']['lr'] == 1e-5
    and c['dataloader']['batch_size'] == 64
    and ck['topk']['k'] == 0
    and ck['save_last_ckpt'] is True
    and er['m3_slots'] == 11
    and ws.get('checkpoint') == warm
)
print(
    f"CONFIG_GATE {'PASS' if ok else 'FAIL'} mode={enc['token_mixer_mode']} "
    f"K={len(ds['view_pool'])} range={tuple(ds['view_count_range'])} "
    f"window={ds['eef_hist_window']!r} epochs={tr['num_epochs']} "
    f"lr={c['optimizer']['lr']} warmup={tr['lr_warmup_steps']} "
    f"skip_rollout={tr['skip_rollout']} "
    f"topk_k={ck['topk']['k']} "
    f"batch={c['dataloader']['batch_size']} slots={er['m3_slots']} "
    f"warm={ws.get('checkpoint')}")
sys.exit(0 if ok else 1)
PYEOF
    then
        say "CONFIG_GATE_FAILED($cell) -- see the CONFIG_GATE line above"
        kill "$pid" 2>/dev/null
        return 1
    fi

    # Wait for the warm-start load and the pre-loop RNG digest.
    for _ in $(seq 1 60); do
        grep -q 'WARMSTART_RNG' "$train_log" && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 10
    done
    if ! grep -q 'WARMSTART_OK' "$train_log"; then
        say "WARMSTART_OK_MISSING($cell)"; tail -n 30 "$train_log" | tee -a "$LOG"
        kill "$pid" 2>/dev/null
        return 1
    fi
    if ! grep -q 'WARMSTART_RNG' "$train_log"; then
        say "RNG_DIGEST_MISSING($cell)"; tail -n 30 "$train_log" | tee -a "$LOG"
        kill "$pid" 2>/dev/null
        return 1
    fi
    grep 'WARMSTART_RNG' "$train_log" | tail -n 1 > "data/enc_rt_rng_${cell}.txt"
    say "RNG_DIGEST($cell) $(cat "data/enc_rt_rng_${cell}.txt")"

    wait "$pid"
    rc=$?
    say "TRAIN_EXIT($cell)=$rc"
    if [ $rc -ne 0 ]; then
        tail -n 40 "$train_log" | tee -a "$LOG"
        return 1
    fi
    if ! grep -q 'FINAL_CHECKPOINT .*epoch=30' "$train_log"; then
        say "FINAL_CHECKPOINT_MISSING($cell)"; tail -n 30 "$train_log" | tee -a "$LOG"
        return 1
    fi
    [ -f "$run/checkpoints/latest.ckpt" ] || { say "LATEST_CKPT_MISSING($cell)"; return 1; }
    mkdir -p "data/outputs/run_square_enc_rt_${cell}_s42_warm30"
    cp -f "$run/logs.json.txt" \
        "data/outputs/run_square_enc_rt_${cell}_s42_warm30/logs.json.txt"
    say "LOG_COPIED_BACK($cell)"
    return 0
}

for cell in $CELLS; do
    case "$cell" in
        a) mode=none ;;
        b) mode=single ;;
        c) mode=intra ;;
        d) mode=cross ;;
        *) say "UNKNOWN_CELL($cell)"; exit 2 ;;
    esac
    if ! run_cell "$cell" "$mode"; then
        say "CELL_FAILED($cell) -- stopping"
        exit 1
    fi
done

# ---- stage 2: cross-cell pairing gates.
say "PAIRING_GATE"
if ! "$PY" - $CELLS >> "$LOG" 2>&1 <<'PYEOF'
import json, sys
cells = sys.argv[1:]
rng = []
for c in cells:
    p = f'data/enc_rt_rng_{c}.txt'
    try:
        rng.append(open(p).read().strip())
    except FileNotFoundError:
        print(f'PAIRING_GATE_FAIL missing {p}')
        sys.exit(1)
if len(set(rng)) != 1:
    print('PAIRING_GATE_FAIL RNG digests differ:')
    for c, line in zip(cells, rng):
        print(f'  {c}: {line}')
    sys.exit(1)
losses = []
for c in cells:
    p = f'data/outputs/run_square_enc_rt_{c}_s42_warm30/logs.json.txt'
    try:
        with open(p) as f:
            # The first JSON line is the first batch of epoch 0, not the
            # epoch-end average: its `train_loss` is the raw first-step loss.
            losses.append(float(json.loads(f.readline())['train_loss']))
    except Exception as exc:  # noqa: BLE001
        print(f'PAIRING_GATE_FAIL {p}: {exc}')
        sys.exit(1)
spread = max(losses) - min(losses)
print('first_batch_train_loss ' + ' '.join(
    f'{c}={v:.8f}' for c, v in zip(cells, losses)) + f' spread={spread:.3e}')
if spread > 1e-5:
    print('PAIRING_GATE_FAIL first-batch train_loss values differ by '
          f'{spread:.3e} > 1e-5')
    sys.exit(1)
print('PAIRING_GATE_PASS (identical RNG digests; first-batch loss paired)')
PYEOF
then
    say "PAIRING_GATE_FAILED -- inspect before evaluating"
    exit 1
fi
say "ALL_CELLS_DONE"
