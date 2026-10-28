#!/bin/bash
# Encoder ladder stage R: S's spatial readout + feature-level ray encoding.
#
# R is S with ONE structural addition (ENCODER_EXPERIMENT_PLAN.md section 4.2):
# each layer3 feature cell's Pluecker ray -- unit world-frame direction d and
# moment m = o x d, evaluated at the cell's receptive-field centre with the
# SAME crop window as the image -- is MLP-encoded to 512-d and added to the
# attention KEY; the VALUE stays visual: k = W_K LN(f + e), v = W_V LN(f).
# Everything else is S: pure-RGB ResNet18 layer3, 1x1 conv 256->512, 25
# spatial tokens, one shared 512-d query, 8-head intra-image attention, S's
# masked cross-view MHA fusion, 521-wide action interface.
#
# ---- PRE-REGISTRATION (fixed before launch; read the result against this) ----
#
# The measured prior: every conditioning path tried so far (input Pluecker
# channels, pre-fusion FiLM, aux heads, base-frame aux, pose-set supervision,
# probabilistic fusion, z supervision) is behaviourally inert, while N>1
# sampling and view quality carry the gains. S replaced B's pooled encoder
# with a pure-RGB spatial readout and came back -0.160 in-distribution /
# -0.166 interpolation against B1 -- just outside the 0.15 band, one seed.
# R adds a geometric ADDRESS to the same readout, so the question is narrow:
# does telling the shared query WHERE each token is seen recover S's ring
# deficit, or move the ring at all?
#
# Registered means (50 paired episodes, same presets as B/S):
#   in-dist ring (11 vp), off-grid +-7.5..+-67.5 (10 vp),
#   past-ring +-82.5/+-90 (4 vp), elevation el_0/+-15 (descriptive cells).
# Band 0.15 mean-over-viewpoints, pre-registered; direction NOT predicted.
# Internal comparators: S (`run_square_enc_s_s42_200ep`, 0.653 / 0.618) and
# B1 (`run_square_enc_b1_s42_200ep`, 0.813 / 0.784). If R sits within 0.15 of
# S on both, the ray address is a fifth behavioural null; if R recovers the
# ring toward B1, the address is what the pooled encoder had and the readout
# lacked.
#
# Mechanism-live reads (why this is not a black-box null if it is null):
#   * the CPU tests pin the ray convention (full-map sampling at
#     offset + 16*i, unit d, translation-carrying m), crop synchronisation,
#     the 5,284,928-parameter count, and the activation order;
#   * the per-epoch latent probe (same 128 states / az0_view 6) gives R's
#     z_v/z_g series against S's, so a null can be read as "the address did
#     not reshape the latent" or "it did and behaviour did not follow";
#   * `use_ray_encoding: False` is the registered diagnostic ablation (rays
#     severed, same parameters); launch it only if the main cell moves.
#
# RNG note: R and S are NOT RNG-lockstep (module construction and the extra
# parameters consume different amounts of global torch RNG), so seed 42 does
# not pair them draw-for-draw. The comparison is structural, like S - B.
#
# Runtime reference: S completed 201 epochs; R adds ~0.3 M encoder parameters
# and the 25x6 ray placement per view, which is negligible next to the
# backbone. Check `df -h /data` on miroc-server first (a checkpoint is
# 4.62 GB and topk.k=1 roughly doubles a run's footprint).
#
# Run to completion with:
#   setsid bash -c 'bash data/enc_r_run.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
DEV=${DEV:-cuda:0}
RUN=/data/zihan/runs/run_square_enc_r_s42_200ep
LOG=data/enc_r_run.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates that apply to the no-aux R family.
for t in tests/test_multiview_dataset.py tests/test_latent_probe_hook.py \
         tests/test_ray_spatial_readout_obs_encoder.py; do
    say "CPU gate: $t"
    gate_log="/tmp/enc_r_cpu_gate_$(basename "$t").log"
    if ! "$PY" "$t" > "$gate_log" 2>&1; then
        say "CPU_GATE_FAILED ($t) -- not launching; last lines follow"
        tail -n 20 "$gate_log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

say "TRAIN_START (stage R, ray-encoded spatial readout) -> $RUN ($DEV)"
# logging.mode: the installed wandb 0.15.12 cannot talk to the current server
# (`CommError: 'NoneType' object has no attribute 'get'` inside init(), before
# epoch 1). The workspace passes `mode` explicitly, so WANDB_MODE in the
# environment is ignored; the override has to come through the config.
# Default offline; same fix as data/z_run.sh / data/enc_s_run.sh.
"$PY" train.py --config-name=train_diffusion_unet_image_workspace_enc_r_latent \
    task=enc_r_image_abs_multiview_az75 \
    training.seed=42 training.num_epochs=201 training.device="$DEV" \
    training.resume=false \
    dataloader.num_workers=14 val_dataloader.num_workers=2 \
    checkpoint.topk.k=1 \
    logging.project=diffusion_policy_view \
    logging.mode="${LOG_MODE:-offline}" \
    hydra.run.dir="$RUN" >> "data/train_enc_r.log" 2>&1 &
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
pool, rng = ds['view_pool'], ds['view_count_range']
probe = c.get('latent_probe', {})
obs = c['policy']['obs_encoder']
forbidden = [k for k in ('cond_dim', 'use_plucker', 'use_eef_hist',
                          'eef_hist_steps', 'eef_hist_key',
                          'resize_shape', 'cam_keys') if k in obs]
ok = (sorted(pool) == list(range(1, 12)) and tuple(rng) == (1, 11)
      and env.get('m3_slots') == 11
      and 'cam_key_image_runner' in env.get('_target_', '')
      and ds.get('eef_hist_window') == 'episode'
      and ds.get('emit_aux_action') is False
      and probe.get('n_states') == 128 and probe.get('az0_view') == 6
      and obs.get('_target_', '').endswith(
          'ray_spatial_readout_obs_encoder.RaySpatialReadoutObsEncoder')
      and obs.get('fused_dim') == 512 and obs.get('n_heads') == 8
      and obs.get('ray_hidden_dim') == 512
      and float(obs.get('ray_scale', 0)) == 1.0
      and bool(obs.get('use_ray_encoding', False)) is True
      and tuple(obs.get('crop_shape', ())) == (76, 76)
      and not forbidden
      and 'DiffusionUnetImagePolicy' in c['policy'].get('_target_', '')
      and 'Aux' not in c['policy'].get('_target_', '')
      and c['training'].get('seed') == 42
      and c['training'].get('num_epochs') == 201)
print(f"OVERRIDE_GATE {'PASS' if ok else 'FAIL'}: pool={pool} range={rng} "
      f"slots={env.get('m3_slots')} window={ds.get('eef_hist_window')!r} "
      f"aux={ds.get('emit_aux_action')} "
      f"probe=({probe.get('n_states')}, az0_view={probe.get('az0_view')}) "
      f"target={obs.get('_target_')} fused={obs.get('fused_dim')} "
      f"heads={obs.get('n_heads')} ray_hidden={obs.get('ray_hidden_dim')} "
      f"ray_scale={obs.get('ray_scale')} use_rays={obs.get('use_ray_encoding')} "
      f"crop={obs.get('crop_shape')} forbidden={forbidden} "
      f"seed={c['training'].get('seed')} "
      f"epochs={c['training'].get('num_epochs')} "
      f"(want pool 1..11, range [1,11], slots 11, episode, aux False, "
      f"probe 128/az0_view 6, R target, fused 512, 8 heads, ray_hidden 512, "
      f"ray_scale 1.0, rays on, crop 76, no history/cam keys, plain policy, "
      f"seed 42, 201 epochs)")
PYEOF
else
    say "OVERRIDE_GATE UNKNOWN: $RUN/.hydra/config.yaml not written yet"
fi

wait "$pid"
say "TRAIN_EXIT=$?"
mkdir -p data/outputs/run_square_enc_r_s42_200ep
cp -f "$RUN/logs.json.txt" \
    data/outputs/run_square_enc_r_s42_200ep/logs.json.txt
say "LOG_COPIED_BACK"
say "DONE -- next: bash data/enc_r_eval.sh"
