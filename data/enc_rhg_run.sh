#!/bin/bash
# Encoder ladder stage RHG: RH plus R07's point-to-ray geometric bias on the
# intra-image attention logits (ENCODER_EXPERIMENT_PLAN.md section 4.4).
#
# RHG is RH with ONE flag changed: `use_geom_bias: True`, plus the pre-launch
# calibrated, LOCKED sigma. The bias adds, on the first 4 of the 8 attention
# heads,
#     b_{v,j} = -alpha * min( || p_t x d_{v,j} - m_{v,j} ||^2 / (2 sigma^2), b_max )
# (alpha = 1, b_max = 4; rho in metres), and is switched OFF for a view whose
# current EE point is behind the camera or projects outside the sampled crop.
# Window length: H=4 consecutive observation frames. The buffer
# records one frame per 20 Hz control step, so the window covers 3 x 0.05 s
# = 0.15 s of true EE motion (plan section 4.1: "覆盖时长为 3x采样间隔").
#
# `use_geom_bias` adds no parameter and consumes no RNG at construction, so
# RH and RHG start from BIT-IDENTICAL parameters and see the identical data
# stream: RHG - RH is a clean one-variable comparison.
#
# ---- PRE-REGISTRATION (fixed before launch; read the result against this) ----
#
# The measured prior: every explicit geometric conditioning path tried in this
# project has been behaviourally inert -- input Pluecker channels (M3), FiLM
# history (M3, B0/B1), per-view aux heads (M4, both frames), the pose-set
# supervision, the z supervisions -- and R's feature-level ray address was the
# fifth null (R-S -0.051 in-dist / -0.014 interp, inside 0.15; not recovering
# B1's -0.211 / -0.180). The bias is the FIRST conditioning term that changes
# the attention RETRIEVAL distribution directly, with a physically meaningful
# anchor (the current EE point) rather than a learned one.
#
# Registered means (50 paired episodes, same presets as B/S/R):
#   in-dist ring (11 vp), off-grid +-7.5..+-67.5 (10 vp),
#   past-ring +-82.5/+-90 (4 vp), elevation el_0/+-15 (descriptive cells).
# Band 0.15 mean-over-viewpoints, pre-registered; direction NOT predicted.
# Comparator: RH (the same cell without the bias; its committed numbers are
# read from its own sweep, not assumed here). If RHG sits within 0.15 of RH on
# both registered means, the bias is a sixth behavioural null; if it moves the
# ring, the explicit point-to-ray relation is what the retrieval lacked.
#
# Sigma: calibrated ONCE on the training zarr by calibrate_geom_bias.py
# (median point-to-ray distance at a 1-cell feature-grid neighbour of the EE
# projection, centre crop), written to data/enc_rhg_calib_square.json and
# data/enc_rhg_calib.env, and REUSED by any later cell (delete the env file or
# FORCE_CALIB=1 to recalibrate). The value and the implied bias at 1/2 cells
# are printed into the log BEFORE the launch; the launch gate asserts the
# resolved config carries exactly that value.
#
# Mechanism-live reads (why a null is not a black box if it is one):
#   * the CPU tests pin the point-to-ray convention, the formula, the validity
#     gate and the 4/8-head wiring against a manual attention;
#   * the per-epoch latent probe gives RHG's z_v/z_g series against RH's;
#   * the diagnostic ablation `use_geom_bias=False` IS RH (same parameters,
#     already trained) -- no extra cell is needed for the bias-off control.
#
# Runtime reference: R completed 201 epochs in 3 h 11 m and its sweep in
# 1 h 55 m on cuda:1; RH/RHG add ~0.92 M encoder parameters over R. The sigma
# calibration reads 256 states (11 views x To) and takes minutes. Check
# `df -h /data` on miroc-server first (a checkpoint is 4.62 GB and topk.k=1
# roughly doubles a run's footprint).
#
# Run to completion with:
#   setsid bash -c 'bash data/enc_rhg_run.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
DEV=${DEV:-cuda:0}
RUN=/data/zihan/runs/run_square_enc_rhg_s42_200ep
LOG=data/enc_rhg_run.log
CALIB_JSON=data/enc_rhg_calib_square.json
CALIB_ENV=data/enc_rhg_calib.env
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates for the RH/RHG family.
for t in tests/test_multiview_dataset.py tests/test_latent_probe_hook.py \
         tests/test_traj_spatial_readout_obs_encoder.py; do
    say "CPU gate: $t"
    gate_log="/tmp/enc_rhg_cpu_gate_$(basename "$t").log"
    if ! "$PY" "$t" > "$gate_log" 2>&1; then
        say "CPU_GATE_FAILED ($t) -- not launching; last lines follow"
        tail -n 20 "$gate_log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

# ---- stage 1: sigma calibration (executable proof + the locked value).
# Reuse across reruns/cells so every RHG cell shares one sigma; delete the
# env file or FORCE_CALIB=1 to recalibrate.
if [ -f "$CALIB_ENV" ] && [ "${FORCE_CALIB:-0}" != "1" ]; then
    . "$CALIB_ENV"
    : "${GEOM_BIAS_SIGMA:?$CALIB_ENV did not define GEOM_BIAS_SIGMA}"
    say "CALIBRATION_REUSED from $CALIB_ENV: GEOM_BIAS_SIGMA=$GEOM_BIAS_SIGMA"
else
    say "CALIBRATION_START (sigma from the training zarr, centre crop) -> $CALIB_JSON"
    if ! "$PY" calibrate_geom_bias.py \
            -c train_diffusion_unet_image_workspace_enc_rhg_latent \
            --n-samples 256 --seed 42 --out "$CALIB_JSON" \
            > /tmp/enc_rhg_calib.log 2>&1; then
        say "CALIBRATION_FAILED -- not launching; last lines follow"
        tail -n 30 /tmp/enc_rhg_calib.log | tee -a "$LOG"
        exit 1
    fi
    cp -f /tmp/enc_rhg_calib.log data/enc_rhg_calib.log
    GEOM_BIAS_SIGMA=$("$PY" -c "import json;d=json.load(open('$CALIB_JSON'));print('%.6f'%d['sigma_m'])")
    echo "GEOM_BIAS_SIGMA=$GEOM_BIAS_SIGMA" > "$CALIB_ENV"
    say "CALIBRATION_DONE sigma=$GEOM_BIAS_SIGMA (wrote $CALIB_JSON and $CALIB_ENV)"
fi
# gate: a finite, positive sigma; print the calibration's own evidence.
# The status is captured BEFORE any pipe (NOTES.md *A guard that discards its
# own error*): `... | tee` would report tee's status, not the gate's.
"$PY" - "$CALIB_JSON" "$GEOM_BIAS_SIGMA" > /tmp/enc_rhg_calib_gate.log 2>&1 <<'PYEOF'
import json, math, sys
d = json.load(open(sys.argv[1]))
sigma = float(sys.argv[2])
ok = (math.isfinite(sigma) and sigma > 0.0
      and abs(float(d['sigma_m']) - sigma) < 1e-9
      and d['n_valid_view_frames'] > 0)
print(f"CALIB_GATE {'PASS' if ok else 'FAIL'}: sigma={sigma:.6f} m "
      f"valid_views={d['n_valid_view_frames']}/{d['n_view_frames']} "
      f"rho_1cell_median={d['rho_1cell']['median']:.4f} "
      f"rho_2cell_median={(d['rho_2cell'] or {}).get('median')} "
      f"bias_1cell={d['implied_bias_1cell']} "
      f"bias_2cell={d['implied_bias_2cell']} alpha={d['alpha']} "
      f"b_max={d['b_max']} crop={d['crop_shape']} grid={d['token_grid']}")
sys.exit(0 if ok else 1)
PYEOF
calib_rc=$?
cat /tmp/enc_rhg_calib_gate.log | tee -a "$LOG"
if [ $calib_rc -ne 0 ]; then
    say "CALIB_GATE_FAILED -- not launching"; exit 1
fi

say "TRAIN_START (stage RHG, geometric point-to-ray bias, sigma=$GEOM_BIAS_SIGMA) -> $RUN ($DEV)"
# logging.mode: the installed wandb 0.15.12 cannot talk to the current server;
# default offline, same fix as data/z_run.sh / data/enc_{s,r,rh}_run.sh.
"$PY" train.py --config-name=train_diffusion_unet_image_workspace_enc_rhg_latent \
    task=enc_rh_image_abs_multiview_az75 \
    policy.obs_encoder.geom_bias_sigma_m="$GEOM_BIAS_SIGMA" \
    training.seed=42 training.num_epochs=201 training.device="$DEV" \
    training.resume=false \
    dataloader.num_workers=14 val_dataloader.num_workers=2 \
    checkpoint.topk.k=1 \
    logging.project=diffusion_policy_view \
    logging.mode="${LOG_MODE:-offline}" \
    hydra.run.dir="$RUN" >> "data/train_enc_rhg.log" 2>&1 &
pid=$!
say "TRAIN_PID=$pid"

# The launch gate: inspect the RESOLVED config, not a contended epoch time.
sleep 90
if [ -f "$RUN/.hydra/config.yaml" ]; then
    "$PY" - "$RUN/.hydra/config.yaml" "$GEOM_BIAS_SIGMA" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
sigma = float(sys.argv[2])
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
      and bool(obs.get('use_geom_bias', False)) is True
      and int(obs.get('geom_bias_heads', 0)) == 4
      and float(obs.get('geom_bias_alpha', 0)) == 1.0
      and float(obs.get('geom_bias_max', 0)) == 4.0
      and abs(float(obs.get('geom_bias_sigma_m') or 0.0) - sigma) < 1e-9
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
      f"geom_bias={obs.get('use_geom_bias')} heads={obs.get('geom_bias_heads')} "
      f"alpha={obs.get('geom_bias_alpha')} max={obs.get('geom_bias_max')} "
      f"sigma={obs.get('geom_bias_sigma_m')} (want {sigma}) "
      f"crop={obs.get('crop_shape')} forbidden={forbidden} "
      f"seed={c['training'].get('seed')} "
      f"epochs={c['training'].get('num_epochs')} "
      f"(want pool 1..11, range [1,11], slots 11, episode, aux False, "
      f"base_hist True, serve_base True, probe 128/az0_view 6, RH target, "
      f"fused 512, 8 heads, ray_hidden 512, ray_scale 1.0, rays on, "
      f"hist_key eef_hist_base, H=4, traj 128/1/4/256/512, geom bias on "
      f"4/8 heads alpha 1 max 4, sigma=calibrated, crop 76, plain policy, "
      f"seed 42, 201 epochs)")
PYEOF
else
    say "OVERRIDE_GATE UNKNOWN: $RUN/.hydra/config.yaml not written yet"
fi

wait "$pid"
say "TRAIN_EXIT=$?"
mkdir -p data/outputs/run_square_enc_rhg_s42_200ep
cp -f "$RUN/logs.json.txt" \
    data/outputs/run_square_enc_rhg_s42_200ep/logs.json.txt
say "LOG_COPIED_BACK"
say "DONE -- next: bash data/enc_rhg_eval.sh"
