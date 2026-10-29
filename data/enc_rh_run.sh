#!/bin/bash
# Encoder ladder stage RH: R's ray-encoded spatial readout with the shared
# query conditioned on the FULL base-frame EE trajectory window
# (ENCODER_EXPERIMENT_PLAN.md section 4.3).
#
# RH is R with ONE structural addition: per image time, the 4-step window
# {(p_i, R_i)}_{i=t-3..t} in the ROBOT BASE frame goes through a small
# temporal Transformer and an MLP to h_t, and the ONE shared spatial query is
# q_t = q_learned + W_h h_t, reused by every valid view. The ray branch, the
# spatial readout, the cross-view MHA fusion and the 521-wide low-dim
# interface are R's. The only new obs key is `eef_hist_base` (per-sample,
# view-independent, (To, H, 9) = [pos, rot6d], oldest first, current last).
#
# Window length: H=4 consecutive observation frames. The buffer records one
# frame per 20 Hz control step, so the window covers 3 x 0.05 s = 0.15 s of true
# EE motion (plan section 4.1: "覆盖时长为 3x采样间隔").
#
# ---- PRE-REGISTRATION (fixed before launch; read the result against this) ----
#
# The measured prior: S replaced B's pooled encoder with a pure-RGB spatial
# readout and came back -0.160 in-distribution / -0.166 interpolation against
# B1 (just outside the 0.15 band, one seed); R added a feature-level ray
# address on the attention key and was the fifth behavioural null (R-S -0.051
# / -0.014, inside the band), not recovering B1 (-0.211 / -0.180). Every
# conditioning path tried so far (input Pluecker, FiLM history, aux heads,
# base-frame aux, pose-set supervision, probabilistic fusion, z supervision)
# is behaviourally inert; N>1 sampling and view quality carry the gains.
#
# RH's question is narrow: does conditioning WHERE the shared query looks on
# the full EE trajectory (not just the current frame) recover the ring toward
# B1, or move it at all?
#
# Registered means (50 paired episodes, same presets as B/S/R):
#   in-dist ring (11 vp), off-grid +-7.5..+-67.5 (10 vp),
#   past-ring +-82.5/+-90 (4 vp), elevation el_0/+-15 (descriptive cells).
# Band 0.15 mean-over-viewpoints, pre-registered; direction NOT predicted.
# Internal comparators: R (`run_square_enc_r_s42_200ep`, 0.602 / 0.604), S
# (0.653 / 0.618), B1 (`run_square_enc_b1_s42_200ep`, 0.813 / 0.784). If RH
# sits within 0.15 of R on both registered means, the trajectory condition is
# a sixth behavioural null; if it recovers the ring toward B1, the full
# trajectory (rather than the pooled encoder) is what the readout lacked.
#
# Mechanism-live reads (why a null is not a black box if it is one):
#   * the CPU tests pin the trajectory feature contract, q = q_learned + W_h h_t,
#     the activation order, the parameter count and the dataset/runner/probe
#     pass-through of `eef_hist_base`;
#   * the per-epoch latent probe (same 128 states / az0_view 6) gives RH's
#     z_v/z_g series against R's;
#   * the probe carries the view-independent base window in both draws, so a
#     missing-key failure would be caught before the first epoch.
#
# RNG note: RH and R are NOT RNG-lockstep (the trajectory modules consume
# different amounts of global torch RNG at construction; RH adds 922,752
# encoder parameters), so seed 42 does not pair them draw-for-draw. The
# comparison is structural, like R - S and S - B.
#
# Runtime reference: R completed 201 epochs in 3 h 11 m and its sweep in
# 1 h 55 m on cuda:1. RH adds ~0.92 M encoder parameters and one small
# 4-step Transformer per frame, negligible next to the backbone. Check
# `df -h /data` on miroc-server first (a checkpoint is 4.62 GB and topk.k=1
# roughly doubles a run's footprint).
#
# Run to completion with:
#   setsid bash -c 'bash data/enc_rh_run.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
DEV=${DEV:-cuda:0}
RUN=/data/zihan/runs/run_square_enc_rh_s42_200ep
LOG=data/enc_rh_run.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates for the RH family.
for t in tests/test_multiview_dataset.py tests/test_latent_probe_hook.py \
         tests/test_traj_spatial_readout_obs_encoder.py; do
    say "CPU gate: $t"
    gate_log="/tmp/enc_rh_cpu_gate_$(basename "$t").log"
    if ! "$PY" "$t" > "$gate_log" 2>&1; then
        say "CPU_GATE_FAILED ($t) -- not launching; last lines follow"
        tail -n 20 "$gate_log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

say "TRAIN_START (stage RH, trajectory-conditioned spatial readout) -> $RUN ($DEV)"
# logging.mode: the installed wandb 0.15.12 cannot talk to the current server
# (`CommError: 'NoneType' object has no attribute 'get'` inside init(), before
# epoch 1). The workspace passes `mode` explicitly, so WANDB_MODE in the
# environment is ignored; the override has to come through the config.
# Default offline; same fix as data/z_run.sh / data/enc_{s,r}_run.sh.
"$PY" train.py --config-name=train_diffusion_unet_image_workspace_enc_rh_latent \
    task=enc_rh_image_abs_multiview_az75 \
    training.seed=42 training.num_epochs=201 training.device="$DEV" \
    training.resume=false \
    dataloader.num_workers=14 val_dataloader.num_workers=2 \
    checkpoint.topk.k=1 \
    logging.project=diffusion_policy_view \
    logging.mode="${LOG_MODE:-offline}" \
    hydra.run.dir="$RUN" >> "data/train_enc_rh.log" 2>&1 &
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
forbidden = [k for k in ('cond_dim', 'use_plucker', 'use_eef_hist',
                          'resize_shape', 'cam_keys') if k in obs]
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
          'traj_spatial_readout_obs_encoder.TrajSpatialReadoutObsEncoder')
      and obs.get('fused_dim') == 512 and obs.get('n_heads') == 8
      and obs.get('ray_hidden_dim') == 512
      and float(obs.get('ray_scale', 0)) == 1.0
      and bool(obs.get('use_ray_encoding', False)) is True
      and obs.get('eef_hist_key') == 'eef_hist_base'
      and obs.get('eef_hist_steps') == 4
      and obs.get('traj_hidden_dim') == 128
      and obs.get('traj_layers') == 1
      and obs.get('traj_heads') == 4
      and obs.get('traj_ff_dim') == 256
      and obs.get('traj_mlp_hidden') == 512
      and bool(obs.get('use_geom_bias', True)) is False
      and tuple(obs.get('crop_shape', ())) == (76, 76)
      and not forbidden
      and 'DiffusionUnetImagePolicy' in c['policy'].get('_target_', '')
      and 'Aux' not in c['policy'].get('_target_', '')
      and c['training'].get('seed') == 42
      and c['training'].get('num_epochs') == 201)
print(f"OVERRIDE_GATE {'PASS' if ok else 'FAIL'}: pool={ds.get('view_pool')} "
      f"range={ds.get('view_count_range')} slots={env.get('m3_slots')} "
      f"window={ds.get('eef_hist_window')!r} aux={ds.get('emit_aux_action')} "
      f"base_hist={ds.get('emit_eef_hist_base')} "
      f"serve_base={env.get('serve_eef_hist_base')} "
      f"probe=({probe.get('n_states')}, az0_view={probe.get('az0_view')}) "
      f"target={obs.get('_target_')} fused={obs.get('fused_dim')} "
      f"heads={obs.get('n_heads')} ray_hidden={obs.get('ray_hidden_dim')} "
      f"ray_scale={obs.get('ray_scale')} use_rays={obs.get('use_ray_encoding')} "
      f"hist_key={obs.get('eef_hist_key')} traj=({obs.get('traj_hidden_dim')}, "
      f"{obs.get('traj_layers')}, {obs.get('traj_heads')}, "
      f"{obs.get('traj_ff_dim')}, {obs.get('traj_mlp_hidden')}) "
      f"geom_bias={obs.get('use_geom_bias')} crop={obs.get('crop_shape')} "
      f"forbidden={forbidden} seed={c['training'].get('seed')} "
      f"epochs={c['training'].get('num_epochs')} "
      f"(want pool 1..11, range [1,11], slots 11, episode, aux False, "
      f"base_hist True, serve_base True, probe 128/az0_view 6, RH target, "
      f"fused 512, 8 heads, ray_hidden 512, ray_scale 1.0, rays on, "
      f"hist_key eef_hist_base, H=4, traj 128/1/4/256/512, no geom bias, "
      f"crop 76, plain policy, seed 42, 201 epochs)")
PYEOF
else
    say "OVERRIDE_GATE UNKNOWN: $RUN/.hydra/config.yaml not written yet"
fi

wait "$pid"
say "TRAIN_EXIT=$?"
mkdir -p data/outputs/run_square_enc_rh_s42_200ep
cp -f "$RUN/logs.json.txt" \
    data/outputs/run_square_enc_rh_s42_200ep/logs.json.txt
say "LOG_COPIED_BACK"
say "DONE -- next: bash data/enc_rh_eval.sh"
