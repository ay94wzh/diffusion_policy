#!/bin/bash
# The z_g pose-set supervision run (PLAN candidate 2), on the clear run's
# recipe. ONE design-level variable vs the committed `m4base` / `m4clear`
# (run_square_m4{base,clear}_s42_200ep): instead of M4's per-slot camera-frame
# action chunk supervised on z_v, the FUSED latent z_global is supervised on
# the POOLED CAMERA-POSE SET of the live (drawn) views -- a fixed (V=11, 9)
# output, row i == pool position i, [pos(3) | rot6d(6)] of that view's camera,
# masked to the live views. Data (square ph ring13), K=11 slots over pool
# [1..11], N ~ [1,11], seed 42, 201 epochs, optimizer/EMA and the per-epoch
# latent probe (128 states, az0_view 6) are all m4base's.
#
# NOT CLAIMED: draw-level RNG lockstep with m4base. The new head is built
# after every parent module, so all shared initial WEIGHTS match m4base's
# seed-42 stream -- but constructing it consumes RNG, so the DataLoader
# base_seed (shuffle order, crops, view draws, noise) differs. The
# one-variable comparison is at the design level.
#
# ---- PRE-REGISTRATION (fixed before launch; read the result against this) ----
#
# 1. BEHAVIOURAL. No direction predicted. Reference points (m4base / m4clear
#    registered means, 50 paired episodes): in-dist ring 0.796 / 0.747;
#    off-grid +-7.5..+-67.5 0.774 / 0.716; past the ring +-82.5/+-90
#    0.310 / 0.310; elevation el_0 0.840 / 0.700, el_+-15 0 / 0. A move
#    beyond the documented 0.15 mean-over-viewpoints band is notable; within
#    band reads as another behavioural null.
# 2. THE INSTRUMENT (this run's distinguishing diagnostic). Committed parents
#    at e200: digest `fused.n1_vs_full.rel` tail-mean(last 25) 0.089 (m4base)
#    / 0.103 (m4clear), tail std <= 0.005 (transient peaks 0.128 @e47 /
#    0.177 @e20 -- read the TAIL, not the peak); probe
#    latent_grid["1,11"].zg_across_view_subsets 0.0663 / 0.0780.
#    (i) THE SUPERVISION BITES THE FUSION: tail rel > 0.13 and/or
#        zg_across_view_subsets(1,11) > 0.09 -> z_g became view-set-aware;
#        the candidate-2 headroom premise is confirmed at the representation,
#        and M5's distillation premise is re-opened (state that explicitly).
#    (ii) IT DOES NOT, while aux_loss decays below this driver's CALIB
#        constant-predictor floor -> the head extracted the live-set
#        information from the existing z_g (post-hoc fitting, M4's lesson at
#        the z_g locus) -- which itself says z_g already carries the set in a
#        form the linear probe and the rel-distance digest cannot see despite
#        the 0.998 N-invariance.
#    FALSIFIER: both diagnostics flat AND aux_loss plateaus at/above the
#    floor (nothing beyond the target's marginals is extractable) ->
#    candidate 2's premise fails for this target; the remaining route is
#    candidate 1 (pose tokens). The floor is the honest reference: the view
#    draw is re-randomized every epoch, so the head cannot memorize
#    (sample -> draw); a below-floor loss means z_g carries the set.
# 3. MECHANISM GATES (must hold or the run is void). Batch-0 aux/diff ~ 0.33
#    by construction: the weight is W = 0.33 * diff0 / aux0 with diff0 =
#    1.0828 (the batch-0 diffusion loss, measured identical on BOTH parents'
#    committed logs) and aux0 = the masked mean target-square under a
#    zero-init head -- a formula fixed here, not chosen after the fact; the
#    CALIBRATION stage below computes aux0 (dataset-only: a zero head outputs
#    zero) at np.random.seed(0) over 16 samples and records the floor. The
#    launch gate proves the override. aux_loss must decay well below its
#    batch-0 value (m4base reference: 74x, 0.361 -> 0.0049); the diffusion
#    term at e200 (train_loss - W*aux) must stay in m4base's range; the
#    in-training rollout must not collapse. Secondary read: the loss can
#    reach z_v only through the fusion, so zv_pair_ratio / zv_abs_pose
#    movement would show whether the pressure propagated.
#    NOTE: val_loss now includes W*aux -- NOT comparable to m4base's.
#
# Runtime reference (clear run / m4base): 201 epochs in ~3h32m at ~58 s/epoch.
# An epoch time near 43 s means the 11-view pool override was silently ignored
# (NOTES.md *Traps*).
#
# Run to completion with:  setsid bash -c 'bash data/zgpose_run.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
RUN=/data/zihan/runs/run_square_zgpose_s42_200ep
DEV=${DEV:-cuda:0}
LOG=data/zgpose_run.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates. The new cell's suite plus the three the shared
# dataset seam could break (M4's targets, the dataset contract, the probe
# instrument). Statuses unpiped, log tail printed on failure.
for gate in test_zg_pose_supervision test_aux_action_heads \
            test_multiview_dataset test_latent_probe_hook; do
    say "CPU gate: tests/${gate}.py"
    if ! "$PY" "tests/${gate}.py" > "/tmp/zgpose_cpu_gate_${gate}.log" 2>&1; then
        say "CPU_GATE_FAILED (${gate}) -- not launching; last lines follow"
        tail -n 20 "/tmp/zgpose_cpu_gate_${gate}.log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

# ---- stage 1: weight calibration (pre-registered formula; dataset-only).
# A zero-init head outputs exactly zero, so aux0 -- the batch-0 aux value --
# is the masked mean target-square, computable without a model. W = 0.33 *
# diff0 / aux0. The stage prints CALIB_* and CALIB_W=, and CAN fail.
say "CALIBRATION: aux0 and the constant-predictor floor (16 samples, np.random.seed(0))"
if ! "$PY" - > /tmp/zgpose_calib.log 2>&1 <<'PYEOF'
import os
import numpy as np
import hydra
from omegaconf import OmegaConf

# the task yaml uses ${eval:...}; train.py registers this resolver at import,
# so the standalone stage must do it itself
OmegaConf.register_new_resolver('eval', eval, replace=True)
from diffusion_policy.dataset.multiview_image_dataset import (
    MultiViewImageDataset, AUX_VIEW_POSE_KEY, AUX_VIEW_POSE_MASK_KEY,
    VIEW_POSE_DIM)

with hydra.initialize_config_dir(
        config_dir=os.path.join(os.getcwd(), 'diffusion_policy', 'config'),
        version_base=None):
    cfg = hydra.compose(
        config_name='train_diffusion_unet_image_workspace_zgpose_latent')
ds = hydra.utils.instantiate(cfg.task.dataset)
assert isinstance(ds, MultiViewImageDataset)
assert ds.emit_aux_view_pose and not ds.emit_aux_action, 'wrong task flags'

norm = ds.get_normalizer()[AUX_VIEW_POSE_KEY]
assert bool((norm.params_dict['scale'] == 1).all()), 'pose normalizer not identity'
assert bool((norm.params_dict['offset'] == 0).all()), 'pose normalizer not identity'

V, P = len(ds.view_pool), VIEW_POSE_DIM
np.random.seed(0)                                   # the registered seed
idxs = np.random.choice(len(ds), size=16, replace=False)
t, m = [], []
for i in idxs:
    s = ds[int(i)]
    t.append(s[AUX_VIEW_POSE_KEY].numpy().reshape(-1, V, P))
    m.append(s[AUX_VIEW_POSE_MASK_KEY].numpy().reshape(-1, V))
t = np.concatenate(t)                               # (M, V, P)
m = np.concatenate(m) > 0.5                         # (M, V)
assert bool(m.any(axis=1).all()) and int(m.sum()) > 0
aux0 = float((t ** 2).mean(-1)[m].mean())           # zero-init head predicts 0
n_live = m.sum(0).astype(np.float64)
assert bool((n_live > 0).all()), 'a pool view never drawn in 16 samples'
c = (t * m[..., None]).sum(0) / n_live[:, None]     # best constant predictor
floor = float(((t - c) ** 2).mean(-1)[m].mean())
assert np.isfinite(aux0) and aux0 > 0, aux0
diff0 = 1.0828
W = round(0.33 * diff0 / aux0, 4)
print(f'CALIB samples={len(idxs)} frames={t.shape[0]} live_pairs={int(m.sum())} V={V} P={P}')
print(f'CALIB aux0_zero_head={aux0:.6f}')
print(f'CALIB floor_constant_predictor={floor:.6f} ratio_aux0_floor={aux0 / max(floor, 1e-12):.3f}')
print(f'CALIB diff0={diff0} W=0.33*{diff0}/{aux0:.6f} = {W} (init aux share {W * aux0 / diff0:.3f})')
print(f'CALIB_W={W}')
if W > 3:
    print(f'CALIB_WARN: W={W} > 3 -- the aux term would dominate the diffusion loss; read before launching')
PYEOF
then
    say "CALIBRATION_FAILED -- not launching; log tail follows"
    tail -n 20 /tmp/zgpose_calib.log | tee -a "$LOG"
    exit 1
fi
cat /tmp/zgpose_calib.log | tee -a "$LOG"
W=$(sed -n 's/^CALIB_W=//p' /tmp/zgpose_calib.log)
[ -n "$W" ] || { say "CALIBRATION_FAILED: no CALIB_W line"; exit 1; }
say "CALIBRATION_OK W=$W"

# 201 epochs, not 200: checkpoint and rollout fire on epoch % 50 == 0 and there
# is no save at the end, so 201 makes epoch 200 fire.
# training.resume=false: the config default is True and it SILENTLY resumes.
say "TRAIN_START -> $RUN ($DEV, W=$W)"
"$PY" train.py --config-name=train_diffusion_unet_image_workspace_zgpose_latent \
  task=zgpose_image_abs_multiview_az75 \
  training.seed=42 training.num_epochs=201 training.device="$DEV" \
  training.resume=false \
  dataloader.num_workers=14 val_dataloader.num_workers=2 checkpoint.topk.k=1 \
  policy.aux_loss_weight="$W" \
  logging.project=diffusion_policy_view \
  hydra.run.dir="$RUN" >> data/train_zgpose.log 2>&1 &
TRAIN_PID=$!
say "TRAIN_PID=$TRAIN_PID"

# The launch gate, done by INSPECTING the resolved config rather than by timing
# the first epochs (NOTES.md: a contended epoch-time reading is ambiguous by
# construction). It must be able to fail AND print its evidence.
sleep 90
if [ -f "$RUN/.hydra/config.yaml" ]; then
    "$PY" - "$RUN/.hydra/config.yaml" "$W" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
ds = c['task']['dataset']
pool, rng = ds['view_pool'], ds['view_count_range']
slots = c['task']['env_runner']['m3_slots']
probe = c.get('latent_probe', {})
w = float(c['policy']['aux_loss_weight'])
want_w = float(sys.argv[2])
ok = (sorted(pool) == list(range(1, 12)) and tuple(rng) == (1, 11)
      and slots == 11
      and ds.get('emit_aux_view_pose') is True
      and ds.get('emit_aux_action') in (False, None)
      and int(c['policy']['aux_n_views']) == 11
      and abs(w - want_w) < 5e-5
      and probe.get('n_states') == 128 and probe.get('az0_view') == 6
      and probe.get('out_dir') == 'latent_snapshots')
print(f"OVERRIDE_GATE {'PASS' if ok else 'FAIL'}: pool={pool} range={rng} "
      f"slots={slots} emit_aux_view_pose={ds.get('emit_aux_view_pose')} "
      f"emit_aux_action={ds.get('emit_aux_action')} "
      f"aux_n_views={c['policy'].get('aux_n_views')} W={w} (want {want_w}) "
      f"probe=({probe.get('n_states')}, az0_view={probe.get('az0_view')}, "
      f"{probe.get('out_dir')})  mean-N={(rng[0]+rng[1])/2}")
PYEOF
else
    say "OVERRIDE_GATE UNKNOWN: $RUN/.hydra/config.yaml not written yet"
fi

wait $TRAIN_PID
say "TRAIN_EXIT=$?"

# Copy the log back to the git-tracked location (weights stay under /data).
mkdir -p data/outputs/run_square_zgpose_s42_200ep
cp -f "$RUN/logs.json.txt" data/outputs/run_square_zgpose_s42_200ep/logs.json.txt
say "LOG_COPIED_BACK"

# Post-run reads, from the log rather than assumed:
#   * epoch time ~58 s (43 s means the pool override was ignored);
#   * a rollout at epoch 0 (N=1 at az_0, served by CamKeyImageRunner);
#   * latent_snapshots/ appearing (~1.7-3.7 MB/epoch; 201 npz by the end);
#   * batch-0 train_loss - W*aux vs the CALIB aux0, and the aux decay vs the
#     CALIB constant-predictor floor (the mechanism gates of the header).
say "DONE -- next: data/zgpose_eval.sh, then data/zgpose_analysis.sh"
