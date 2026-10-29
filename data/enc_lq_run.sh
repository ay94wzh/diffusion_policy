#!/bin/bash
# Encoder ladder, stage LQ: LVSM RGB+Pluecker 8x8 patchify + EEF-history query
# tokens, one bidirectional transformer, only the query positions read out
# (ENCODER_LVSM_EEF_QUERY_V1.md v0.2).
#
# LQ is a NEW structural candidate, not another rung of B/S/R/RH/RHG. Each
# valid view's 84x84 RGB is cropped to 72x72; a per-pixel base-frame Pluecker
# map [d | m/L] is computed for the SAME crop window with the ORIGINAL
# intrinsics; the 9 channels [RGB | d | m/L] are patchified by a shared
# Conv2d(9, 256, 8, 8) (81 tokens/view). Four EEF-history query tokens are
# formed from [p/L, rot6d, (p-p_t)/L, age] and joined with all visual tokens in
# one bidirectional Pre-LN transformer (4 layers, 8 heads, FFN 1024, dropout 0).
# Only the 4 query outputs are kept, projected 256 -> 128, flattened
# oldest->current (512), concatenated with the current 9-d state (521), and two
# frames give the unchanged 1042-wide UNet condition.
#
# The crop change 76 -> 72 is REQUIRED because 76 is not divisible by 8. It is
# part of this structural comparison (the design doc section 6.1 warns that a
# tokenizer comparison needs a same-crop control); LQ vs B1/R/RH therefore
# differs in tokenizer AND crop. Do not attribute a delta to the tokenizer
# alone.
#
# ---- PRE-REGISTRATION (fixed before launch; read the result against this) ----
#
# Comparators available now (50 paired episodes, same presets):
#   B1 in-dist 0.813 / interp 0.784; S 0.653 / 0.618; R 0.602 / 0.604.
#   RH/RHG are coded but not trained as of this script. The band is 0.15
#   mean-over-viewpoints, pre-registered; direction NOT predicted.
#
# Primary comparisons (design section 8):
#   * LQ vs B1/R/RH: does the joint bidirectional visual/history information
#     flow recover the ring toward B1, or move it at all? Structural, with the
#     crop confound above.
#   * history_query=False: keep the same parameters, zero the 13-d query
#     features (learned/constant-query control). Is the sample-dependent EEF
#     history doing anything?
#   * visual_reads_history=False: block only the visual<-history attention
#     block. Does the history -> visual -> query feedback path help?
#   * 4 vs 2/6 layers: capacity / number of exchange rounds (later run).
# These controls are constructor flags; no code change is needed to run them.
#
# ---- MECHANISM-LIVE READS ----
#   * the CPU test pins the tokenizer geometry, the query feature contract,
#     the fixed K*G query slice, mask polarity, permutation invariance, the
#     parameter/RNG identity of both controls, the parameter count
#     (3,412,096), and the 521/1042 interface;
#   * the per-epoch LQ probe (same 128 states / az0_view 6) records z_views
#     (pre-transformer per-view means, 256-wide), z_global (512), query tokens
#     and post-transformer visual means; analyze_latent_series.py is
#     width-aware for this cell (the visualizer is not updated yet);
#   * train/eval both serve eef_hist_base, so a missing-key failure is caught
#     before the first epoch.
#
# RNG note: LQ is not RNG-lockstep with R/RH (different modules and
# parameter count); comparisons are structural.
#
# Runtime: S = 11*81 + 4 = 895 tokens at K=11 (RH is 279), so the joint
# attention is ~10x the token-pair count. Measure peak memory / throughput on
# the server BEFORE the full run; if memory is tight, lower
# dataloader.batch_size and raise training.gradient_accumulate_every. Check
# `df -h /data` first (a checkpoint is 4.62 GB and topk.k=1 roughly doubles a
# run's footprint).
#
# Run to completion with:
#   setsid bash -c 'bash data/enc_lq_run.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
DEV=${DEV:-cuda:0}
RUN=/data/zihan/runs/run_square_enc_lq_s42_200ep
LOG=data/enc_lq_run.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates for the LQ family.
for t in tests/test_multiview_dataset.py tests/test_latent_probe_hook.py \
         tests/test_lvsm_eef_query_obs_encoder.py; do
    say "CPU gate: $t"
    gate_log="/tmp/enc_lq_cpu_gate_$(basename "$t").log"
    if ! "$PY" "$t" > "$gate_log" 2>&1; then
        say "CPU_GATE_FAILED ($t) -- not launching; last lines follow"
        tail -n 20 "$gate_log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

say "TRAIN_START (stage LQ, LVSM patchify + EEF queries) -> $RUN ($DEV)"
# logging.mode: the installed wandb 0.15.12 cannot talk to the current server
# (`CommError: 'NoneType' object has no attribute 'get'` inside init(), before
# epoch 1). The workspace passes `mode` explicitly, so WANDB_MODE in the
# environment is ignored; the override has to come through the config.
# Default offline; same fix as data/z_run.sh / data/enc_{s,r,rh}_run.sh.
"$PY" train.py --config-name=train_diffusion_unet_image_workspace_enc_lq_latent \
    task=enc_lq_image_abs_multiview_az75 \
    training.seed=42 training.num_epochs=201 training.device="$DEV" \
    training.resume=false \
    dataloader.num_workers=14 val_dataloader.num_workers=2 \
    checkpoint.topk.k=1 \
    logging.project=diffusion_policy_view \
    logging.mode="${LOG_MODE:-offline}" \
    hydra.run.dir="$RUN" >> "data/train_enc_lq.log" 2>&1 &
pid=$!
say "TRAIN_PID=$pid"

# The launch gate: inspect the RESOLVED config, not a contended epoch time.
sleep 90
if [ -f "$RUN/.hydra/config.yaml" ]; then
    "$PY" - "$RUN/.hydra/config.yaml" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
ds = c['task']['dataset']
env = c['task']['env_runner']
probe = c.get('latent_probe', {})
obs = c['policy']['obs_encoder']
forbidden = [k for k in (
    'rgb_model', 'use_group_norm', 'share_rgb_model', 'fused_dim',
    'traj_hidden_dim', 'traj_layers', 'traj_heads', 'traj_ff_dim',
    'traj_mlp_hidden', 'use_geom_bias', 'geom_bias_heads',
    'geom_bias_sigma_m', 'ray_hidden_dim', 'use_ray_encoding',
    'cam_keys', 'cond_dim', 'use_plucker', 'use_eef_hist',
    'resize_shape') if k in obs]
ok = (sorted(ds['view_pool']) == list(range(1, 12))
      and tuple(ds['view_count_range']) == (1, 11)
      and env.get('m3_slots') == 11
      and 'cam_key_image_runner' in env.get('_target_', '')
      and ds.get('eef_hist_window') == 'episode'
      and ds.get('emit_aux_action') is False
      and ds.get('emit_eef_hist_base') is True
      and env.get('serve_eef_hist_base') is True
      and probe.get('n_states') == 128 and probe.get('az0_view') == 6
      and obs.get('_target_', '').endswith(
          'lvsm_eef_query_obs_encoder.LvsmEefQueryObsEncoder')
      and obs.get('d_model') == 256 and obs.get('n_heads') == 8
      and obs.get('n_layers') == 4 and obs.get('ffn_dim') == 1024
      and float(obs.get('dropout', -1)) == 0.0
      and obs.get('out_token_dim') == 128
      and obs.get('patch_size') == 8
      and tuple(obs.get('crop_shape', ())) == (72, 72)
      and bool(obs.get('random_crop', False)) is True
      and bool(obs.get('imagenet_norm', False)) is True
      and float(obs.get('ray_scale', 0)) == 1.0
      and obs.get('eef_hist_key') == 'eef_hist_base'
      and obs.get('eef_hist_steps') == 4
      and bool(obs.get('history_query', False)) is True
      and bool(obs.get('visual_reads_history', False)) is True
      and not forbidden
      and 'DiffusionUnetImagePolicy' in c['policy'].get('_target_', '')
      and 'Aux' not in c['policy'].get('_target_', '')
      and 'LQProbe' in c.get('_target_', '')
      and c['training'].get('seed') == 42
      and c['training'].get('num_epochs') == 201)
print(f"OVERRIDE_GATE {'PASS' if ok else 'FAIL'}: pool={ds.get('view_pool')} "
      f"range={ds.get('view_count_range')} slots={env.get('m3_slots')} "
      f"window={ds.get('eef_hist_window')!r} aux={ds.get('emit_aux_action')} "
      f"base_hist={ds.get('emit_eef_hist_base')} "
      f"serve_base={env.get('serve_eef_hist_base')} "
      f"probe=({probe.get('n_states')}, az0_view={probe.get('az0_view')}) "
      f"target={obs.get('_target_')} d={obs.get('d_model')} "
      f"heads={obs.get('n_heads')} layers={obs.get('n_layers')} "
      f"ffn={obs.get('ffn_dim')} out={obs.get('out_token_dim')} "
      f"patch={obs.get('patch_size')} crop={obs.get('crop_shape')} "
      f"ray_scale={obs.get('ray_scale')} hist_key={obs.get('eef_hist_key')} "
      f"H={obs.get('eef_hist_steps')} controls=({obs.get('history_query')}, "
      f"{obs.get('visual_reads_history')}) forbidden={forbidden} "
      f"workspace={c.get('_target_')} seed={c['training'].get('seed')} "
      f"epochs={c['training'].get('num_epochs')} "
      f"(want pool 1..11, range [1,11], slots 11, episode, aux False, "
      f"base_hist True, serve_base True, probe 128/az0_view 6, LQ target, "
      f"D 256, 8 heads, 4 layers, FFN 1024, out 128, patch 8, crop 72, "
      f"ray_scale 1.0, hist_key eef_hist_base, H=4, both controls True, "
      f"no RH/CNN keys, LQ probe, plain policy, seed 42, 201 epochs)")
PYEOF
else
    say "OVERRIDE_GATE UNKNOWN: $RUN/.hydra/config.yaml not written yet"
fi

wait "$pid"
say "TRAIN_EXIT=$?"
mkdir -p data/outputs/run_square_enc_lq_s42_200ep
cp -f "$RUN/logs.json.txt" \
    data/outputs/run_square_enc_lq_s42_200ep/logs.json.txt
say "LOG_COPIED_BACK"
say "DONE -- next: bash data/enc_lq_eval.sh"
