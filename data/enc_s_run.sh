#!/bin/bash
# Encoder ladder stage S: the pure-RGB spatial-readout encoder.
#
# S vs B is a STRUCTURAL comparison, not a paired-seed ablation. Same square
# data, K=11 slots over ring indices 1..11, N in [1,11], seed 42, 201 epochs,
# the plain M3 policy (no aux head) and the latent probe -- but S replaces B's
# 9-channel conv1 (RGB+Pluecker), per-stage FiLM from the camera-frame EE
# history and global average pool with: pure-RGB ResNet18 layer3 -> 1x1 conv
# 256->512 -> 25 spatial tokens -> one shared zero-init 512-d query, 8-head
# intra-image cross-attention -> one 512-d token per view -> the SAME masked
# cross-view MHA fusion as M3. output_shape() stays 521, so the action UNet is
# unchanged (5,016,640 encoder params vs B's 12,532,928).
#
# ---- PRE-REGISTRATION (fixed before launch; read the result against this) ----
#
# The measured prior is that every conditioning path tried so far is
# behaviourally inert, while N>1 sampling and view quality carry the gains.
# S asks a different question: does replacing the global pool with a spatial
# query readout move behaviour? If the readout matters, S differs from B1
# beyond the 0.15 mean-over-viewpoints band on a registered mean; if it does
# not, the B/S comparison is a clean structural null. Direction, if any, is
# NOT predicted.
#
# Registered means (50 paired episodes, same presets as B):
#   in-dist ring (11 vp), off-grid +-7.5..+-67.5 (10 vp),
#   past-ring +-82.5/+-90 (4 vp), elevation el_0/+-15 (descriptive cells).
# Internal baseline: B1 (`run_square_enc_b1_s42_200ep`, corrected-history
# `eef_hist_window=episode`). External reference: m4base (0.796 / 0.774).
# B1's run/eval must exist for the S-B1 comparison; S's own sweeps are
# independent and can be launched first.
#
# Mechanism-live reads: the architecture test pins the 3-channel conv1, the
# absence of layer4, the 5x5 token grid, the two 8-head attentions, and the
# exact 5,016,640 parameter count; the gradient test pins that the readout is
# live. The per-epoch latent probe is the existing instrument; compare S's
# z_v/z_g series against B1/m4base with the same probe.
#
# RNG note: the runs are NOT RNG-lockstep. get_resnet and the module
# construction consume different amounts of global torch RNG, so seed 42 does
# not pair S and B draw-for-draw. The per-slot crop loop consumes crop draws
# in the same order for a given view draw, but that does not make the runs
# paired.
#
# Runtime reference: the clear run was 201 epochs in 3 h 32 m; S drops layer4
# and should be no slower. Check `df -h /data` on miroc-server first (a
# checkpoint is 4.62 GB and topk.k=1 roughly doubles a run's footprint).
#
# Run to completion with:
#   setsid bash -c 'bash data/enc_s_run.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
DEV=${DEV:-cuda:0}
RUN=/data/zihan/runs/run_square_enc_s_s42_200ep
LOG=data/enc_s_run.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates that apply to the no-aux S family.
for t in tests/test_multiview_dataset.py tests/test_latent_probe_hook.py \
         tests/test_spatial_readout_obs_encoder.py; do
    say "CPU gate: $t"
    gate_log="/tmp/enc_s_cpu_gate_$(basename "$t").log"
    if ! "$PY" "$t" > "$gate_log" 2>&1; then
        say "CPU_GATE_FAILED ($t) -- not launching; last lines follow"
        tail -n 20 "$gate_log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

say "TRAIN_START (stage S, pure-RGB spatial readout) -> $RUN ($DEV)"
"$PY" train.py --config-name=train_diffusion_unet_image_workspace_enc_s_latent \
    task=enc_s_image_abs_multiview_az75 \
    training.seed=42 training.num_epochs=201 training.device="$DEV" \
    training.resume=false \
    dataloader.num_workers=14 val_dataloader.num_workers=2 \
    checkpoint.topk.k=1 \
    logging.project=diffusion_policy_view \
    hydra.run.dir="$RUN" >> "data/train_enc_s.log" 2>&1 &
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
                          'eef_hist_steps', 'cam_keys', 'eef_hist_key',
                          'resize_shape') if k in obs]
ok = (sorted(pool) == list(range(1, 12)) and tuple(rng) == (1, 11)
      and env.get('m3_slots') == 11
      and 'cam_key_image_runner' in env.get('_target_', '')
      and ds.get('eef_hist_window') == 'episode'
      and ds.get('emit_aux_action') is False
      and probe.get('n_states') == 128 and probe.get('az0_view') == 6
      and obs.get('_target_', '').endswith(
          'spatial_readout_obs_encoder.SpatialReadoutObsEncoder')
      and obs.get('fused_dim') == 512 and obs.get('n_heads') == 8
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
      f"heads={obs.get('n_heads')} crop={obs.get('crop_shape')} "
      f"forbidden={forbidden} seed={c['training'].get('seed')} "
      f"epochs={c['training'].get('num_epochs')} "
      f"(want pool 1..11, range [1,11], slots 11, episode, aux False, "
      f"probe 128/az0_view 6, S target, fused 512, 8 heads, crop 76, "
      f"no forbidden keys, plain policy, seed 42, 201 epochs)")
PYEOF
else
    say "OVERRIDE_GATE UNKNOWN: $RUN/.hydra/config.yaml not written yet"
fi

wait "$pid"
say "TRAIN_EXIT=$?"
mkdir -p data/outputs/run_square_enc_s_s42_200ep
cp -f "$RUN/logs.json.txt" \
    data/outputs/run_square_enc_s_s42_200ep/logs.json.txt
say "LOG_COPIED_BACK"
say "DONE -- next: bash data/enc_s_eval.sh"
