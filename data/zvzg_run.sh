#!/bin/bash
# The COMBINED-supervision run (`zvzg`): BOTH auxiliary lines on at once, on
# the clear run's recipe. ONE design-level variable vs EACH parent:
#   vs `m4base` (run_square_m4base_s42_200ep): PLUS the zgpose pose-set term
#     (ViewPoseSetHead on the fused latent, weight W_POSE calibrated below);
#   vs `zgpose` (run_square_zgpose_s42_200ep): PLUS m4base's per-view
#     BASE-frame action head on z_v at its own registered weight w_v = 1.0.
# Data (square ph ring13), K=11 slots over pool [1..11], N ~ [1,11], seed 42,
# 201 epochs, optimizer/EMA and the per-epoch latent probe (128 states,
# az0_view 6) are identical to both parents. The two aux targets are emitted
# together from the ONE view draw (neither flag consumes RNG; pinned by
# tests/test_zvzg_supervision.py).
#
# NOT CLAIMED: draw-level RNG lockstep with m4base -- the pose head is built
# after every parent module, so all shared initial WEIGHTS match m4base's
# seed-42 stream, but constructing it consumes RNG, so the DataLoader
# base_seed (shuffle order, crops, view draws, noise) differs. Against zgpose
# the constructor sequence is identical in shape (action head 512->256->80,
# then pose head 512->256->99), so the shared trunk AND the pose head's init
# are expected -- not registered -- to match zgpose's stream. Both
# one-variable comparisons are at the design level.
#
# The `aux_loss` log line for THIS run is the WEIGHTED TOTAL w_v*L_a + w_g*L_p
# (the policy documents this); it is NOT comparable to either parent's
# `aux_loss` (an unweighted single term) or to any floor. The per-term keys
# `aux_action_loss` / `aux_pose_loss` are the comparable quantities -- the
# workspace seam logs both, unweighted, alongside.
#
# ---- PRE-REGISTRATION (fixed before launch; read the result against this) ----
#
# 1. BEHAVIOURAL. Both parents were null on the registered means; if the two
#    supervisions are additive this run is a THIRD null. Registered bands
#    (0.15 mean-over-viewpoints, 50 paired episodes):
#      in-dist ring    -- within 0.15 of BOTH m4base 0.796 and zgpose 0.735;
#      off-grid +-7.5..+-67.5 -- within 0.15 of BOTH 0.774 / 0.714;
#      past the ring +-82.5/+-90 -- REGISTERED SEPARATELY, no band: zgpose's
#        one beyond-band move was a COST here (0.140 vs parents' 0.310); does
#        the action term recover it toward 0.310, or does the pose term's cost
#        persist?
#      elevation el_0 / el_m15 / el_p15 -- three single cells, descriptive
#        only (m4base 0.840/0/0; m4clear 0.700/0/0; zgpose 0.820/0/0).
# 2. THE INSTRUMENTS (which pressure wins at the representation). Committed
#    parents at e200:
#      digest `fused.n1_vs_full.rel` tail-mean(last 25) 0.0889 (m4base) /
#        0.1031 (m4clear) / 0.199 (zgpose); moved threshold tail > 0.13
#        (zgpose's registered line).
#      probe latent_grid["1,11"].zg_across_view_subsets 0.0663 / 0.0780 /
#        0.1434; moved threshold > 0.09.
#      probe latent_grid["1,11"].zv_pair_ratio -- THE TENSION READOUT (the
#        base-frame term halved it: 0.267 m4base vs 0.544 m4clear; the pose
#        term pushed it up: 2.255 zgpose). No direction predicted; classify:
#        < 0.5 -> base-frame pressure wins; 0.5..1.5 -> intermediate;
#        > 1.5 -> pose pressure wins.
# 3. MECHANISM GATES (must hold or the run is void).
#      pose term: batch-0 `aux_pose_loss` ~= the CALIB aux0_pose (zero-init
#        head); the e200 value is read against the floor-relative bands (the
#        same tree as zgpose: <=0.5*floor beaten / 0.5-0.9 partial / >=0.9 at
#        floor; a big decay is NOT evidence of anything -- a per-row constant
#        drops it ~2.3x by itself).
#      action term: batch-0 `aux_action_loss` ~= the CALIB aux0_active, and
#        e200 far below it (m4base's committed reference: 0.361 -> 0.0049,
#        74x, against a best-global-constant score of 0.2692 -- PROGRESS.md
#        *M4* and *The base-frame arm*). Mechanism-live only; no registered
#        band.
#      weights: W_POSE = 0.33 * diff0 / aux0_pose with diff0 = 1.0828 (the
#        batch-0 diffusion loss on BOTH parents' committed logs) and aux0_pose
#        = the mean target-square over ALL rows under a zero-init head -- the
#        zgpose formula, recomputed dataset-only below (seed 0, 64 samples).
#        The CALIBRATION also asserts the committed values (aux0_pose
#        0.268672, floor 0.108586) -- the same-recipe proof, since the
#        combined emission consumes no extra RNG. The action term stays at its
#        registered w_v = 1.0. The launch gate proves the override.
#      the in-training rollout must not collapse.
# 4. DOCUMENTED AMBIGUITY. If behaviour moves beyond band while BOTH
#    instrument readouts stay at their parents' values, the combination
#    bought something neither instrument sees -- report the movement, do NOT
#    attribute a mechanism.
#
# Runtime reference: 201 epochs in ~3h32m at ~58 s/epoch (m4base/zgpose).
# An epoch time near 43 s means the 11-view pool override was silently ignored
# (NOTES.md *Traps*).
#
# Run to completion with:  setsid bash -c 'bash data/zvzg_run.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
RUN=/data/zihan/runs/run_square_zvzg_s42_200ep
DEV=${DEV:-cuda:0}
LOG=data/zvzg_run.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates. The new cell's suite plus the ones the shared
# dataset/policy/workspace seams could break. Statuses unpiped, log tail
# printed on failure.
for gate in test_zvzg_supervision test_aux_action_heads test_zg_pose_supervision \
            test_multiview_dataset test_latent_probe_hook; do
    say "CPU gate: tests/${gate}.py"
    if ! "$PY" "tests/${gate}.py" > "/tmp/zvzg_cpu_gate_${gate}.log" 2>&1; then
        say "CPU_GATE_FAILED (${gate}) -- not launching; last lines follow"
        tail -n 20 "/tmp/zvzg_cpu_gate_${gate}.log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

# ---- stage 1: weight calibration (the zgpose formula) + both terms'
# zero-init references, dataset-only (a zero-init head outputs exactly zero,
# so the batch-0 values are computable without a model). CAN fail.
say "CALIBRATION: pose aux0+floor (asserted vs zgpose's committed values) and the action term's references"
if ! "$PY" - > /tmp/zvzg_calib.log 2>&1 <<'PYEOF'
import os
import numpy as np
import hydra
from omegaconf import OmegaConf

# the task yaml uses ${eval:...}; train.py registers this resolver at import,
# so the standalone stage must do it itself
OmegaConf.register_new_resolver('eval', eval, replace=True)
from diffusion_policy.dataset.multiview_image_dataset import (
    MultiViewImageDataset, AUX_ACTION_KEY, AUX_VIEW_POSE_KEY,
    AUX_VIEW_POSE_MASK_KEY, VIEW_POSE_DIM, ACTION_DIM)

with hydra.initialize_config_dir(
        config_dir=os.path.join(os.getcwd(), 'diffusion_policy', 'config'),
        version_base=None):
    cfg = hydra.compose(
        config_name='train_diffusion_unet_image_workspace_zvzg_latent')
ds = hydra.utils.instantiate(cfg.task.dataset)
assert isinstance(ds, MultiViewImageDataset)
# BOTH targets on, action target in the base frame (this arm's premise)
assert ds.emit_aux_action and ds.emit_aux_view_pose, 'wrong task flags'
assert ds.aux_action_frame == 'base', ds.aux_action_frame

# the pose target's normalizer is identity; the base-frame action target's
# normalizer IS the action key's (exact, not an approximation)
norm = ds.get_normalizer()
pn = norm[AUX_VIEW_POSE_KEY]
assert bool((pn.params_dict['scale'] == 1).all()), 'pose normalizer not identity'
assert bool((pn.params_dict['offset'] == 0).all()), 'pose normalizer not identity'
an = norm[AUX_ACTION_KEY]
assert all(bool((an.params_dict[k] == norm['action'].params_dict[k]).all())
           for k in ('scale', 'offset')), \
    'base-frame aux normalizer must equal the action normalizer'

V, P = len(ds.view_pool), VIEW_POSE_DIM
C = ds.aux_n_steps
K = len(ds.rgb_keys)
N_SAMPLES = 64                                      # floor noise: rel sd ~8%
np.random.seed(0)                                   # the registered seed
idxs = np.random.choice(len(ds), size=N_SAMPLES, replace=False)
t_pose, m_pose, t_act, v_mask = [], [], [], []
for i in idxs:
    s = ds[int(i)]
    t_pose.append(s[AUX_VIEW_POSE_KEY].numpy().reshape(-1, V, P))
    m_pose.append(s[AUX_VIEW_POSE_MASK_KEY].numpy().reshape(-1, V))
    t_act.append(s[AUX_ACTION_KEY].numpy().reshape(-1, K, C * ACTION_DIM))
    v_mask.append(s['obs'][ds.view_mask_key].numpy().reshape(-1, K))

# --- the pose term (w_g, calibrated) ---------------------------------------
t_pose = np.concatenate(t_pose)                     # (M, V, P)
m_pose = np.concatenate(m_pose) > 0.5               # (M, V)
M = t_pose.shape[0]
assert bool(m_pose.any(axis=1).all()) and int(m_pose.sum()) > 0
# zero-init head predicts 0 and ALL rows are scored -> aux0 is the plain mean
# target-square (dead rows contribute their exact zeros)
aux0_pose = float((t_pose ** 2).mean())
n_live = m_pose.sum(0)
assert bool(((n_live > 0) & (n_live < M)).any()), \
    'no row is partially live over these samples -- the floor would be 0'
floor_pose = float(((t_pose - t_pose.mean(0)) ** 2).mean())
assert np.isfinite(aux0_pose) and aux0_pose > 0, aux0_pose
# same-recipe proof: the joint emission consumes no extra RNG, so this must
# reproduce zgpose's committed calibration exactly (its CALIB log)
assert abs(aux0_pose - 0.268672) < 1e-4, aux0_pose
assert abs(floor_pose - 0.108586) < 1e-4, floor_pose
diff0 = 1.0828
W_POSE = round(0.33 * diff0 / aux0_pose, 4)

# --- the action term (w_v = 1.0, registered) --------------------------------
t_act = np.concatenate(t_act)                       # (M, K, C*Da)
v_mask = np.concatenate(v_mask) > 0.5               # (M, K)
ta = norm[AUX_ACTION_KEY].normalize(
    t_act.reshape(M * K, C * ACTION_DIM)).numpy()    # normalized units
am = v_mask.reshape(M * K)
assert am.any() and not am.all()
aux0_action = float((ta[am] ** 2).mean())
floor_action = float(((ta[am] - ta[am].mean(0)) ** 2).mean())
assert np.isfinite(aux0_action) and aux0_action > 0, aux0_action

print(f'CALIB samples={len(idxs)} frames={M} live_pairs={int(m_pose.sum())} V={V} P={P}')
print(f'CALIB n_live min={int(n_live.min())} max={int(n_live.max())} (of {M} frames)')
print(f'CALIB aux0_pose_zero_head={aux0_pose:.6f}')
print(f'CALIB floor_constant_predictor={floor_pose:.6f} ratio_aux0_floor={aux0_pose / max(floor_pose, 1e-12):.3f}')
print(f'CALIB pose reproduced zgpose: aux0 delta {aux0_pose - 0.268672:+.2e}, floor delta {floor_pose - 0.108586:+.2e}')
print(f'CALIB action aux0_active={aux0_action:.6f} floor_const={floor_action:.6f} '
      f'(m4base reference: batch-0 0.36095, floor 0.2692)')
print(f'CALIB diff0={diff0} W_POSE=0.33*{diff0}/{aux0_pose:.6f} = {W_POSE} '
      f'(init pose share {W_POSE * aux0_pose / diff0:.3f}; init action share {1.0 * aux0_action / diff0:.3f})')
print(f'CALIB_W_POSE={W_POSE}')
if W_POSE > 3:
    print(f'CALIB_WARN: W_POSE={W_POSE} > 3 -- the aux term would dominate the diffusion loss; read before launching')
PYEOF
then
    say "CALIBRATION_FAILED -- not launching; log tail follows"
    tail -n 20 /tmp/zvzg_calib.log | tee -a "$LOG"
    exit 1
fi
cat /tmp/zvzg_calib.log | tee -a "$LOG"
W_POSE=$(sed -n 's/^CALIB_W_POSE=//p' /tmp/zvzg_calib.log)
[ -n "$W_POSE" ] || { say "CALIBRATION_FAILED: no CALIB_W_POSE line"; exit 1; }
say "CALIBRATION_OK W_POSE=$W_POSE"

# 201 epochs, not 200: checkpoint and rollout fire on epoch % 50 == 0 and there
# is no save at the end, so 201 makes epoch 200 fire.
# training.resume=false: the config default is True and it SILENTLY resumes.
# w_v stays at its registered 1.0 (the config value); only w_g is overridden.
say "TRAIN_START -> $RUN ($DEV, W_POSE=$W_POSE)"
"$PY" train.py --config-name=train_diffusion_unet_image_workspace_zvzg_latent \
  task=zvzg_base_image_abs_multiview_az75 \
  training.seed=42 training.num_epochs=201 training.device="$DEV" \
  training.resume=false \
  dataloader.num_workers=14 val_dataloader.num_workers=2 checkpoint.topk.k=1 \
  policy.aux_pose_weight="$W_POSE" \
  logging.project=diffusion_policy_view \
  hydra.run.dir="$RUN" >> data/train_zvzg.log 2>&1 &
TRAIN_PID=$!
say "TRAIN_PID=$TRAIN_PID"

# The launch gate, done by INSPECTING the resolved config rather than by timing
# the first epochs (NOTES.md: a contended epoch-time reading is ambiguous by
# construction). It must be able to fail AND print its evidence.
sleep 90
if [ -f "$RUN/.hydra/config.yaml" ]; then
    "$PY" - "$RUN/.hydra/config.yaml" "$W_POSE" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
ds = c['task']['dataset']
pool, rng = ds['view_pool'], ds['view_count_range']
slots = c['task']['env_runner']['m3_slots']
probe = c.get('latent_probe', {})
pol = c['policy']
w_a = float(pol['aux_loss_weight'])
w_p = float(pol['aux_pose_weight'])
want_p = float(sys.argv[2])
ok = (sorted(pool) == list(range(1, 12)) and tuple(rng) == (1, 11)
      and slots == 11
      and ds.get('emit_aux_action') is True
      and ds.get('emit_aux_view_pose') is True
      and ds.get('aux_action_frame') == 'base'
      and int(ds.get('aux_n_steps')) == 8
      and int(pol['aux_n_steps']) == 8
      and int(pol['aux_n_views']) == 11
      and abs(w_a - 1.0) < 5e-5
      and abs(w_p - want_p) < 5e-5
      and probe.get('n_states') == 128 and probe.get('az0_view') == 6
      and probe.get('out_dir') == 'latent_snapshots')
print(f"OVERRIDE_GATE {'PASS' if ok else 'FAIL'}: pool={pool} range={rng} "
      f"slots={slots} emit_aux_action={ds.get('emit_aux_action')} "
      f"emit_aux_view_pose={ds.get('emit_aux_view_pose')} "
      f"aux_action_frame={ds.get('aux_action_frame')} "
      f"aux_n_steps={ds.get('aux_n_steps')}/{pol.get('aux_n_steps')} "
      f"aux_n_views={pol.get('aux_n_views')} w_v={w_a} w_g={w_p} "
      f"(want {want_p}) probe=({probe.get('n_states')}, "
      f"az0_view={probe.get('az0_view')}, {probe.get('out_dir')})  "
      f"mean-N={(rng[0] + rng[1]) / 2}")
PYEOF
else
    say "OVERRIDE_GATE UNKNOWN: $RUN/.hydra/config.yaml not written yet"
fi

wait $TRAIN_PID
say "TRAIN_EXIT=$?"

# Copy the log back to the git-tracked location (weights stay under /data).
mkdir -p data/outputs/run_square_zvzg_s42_200ep
cp -f "$RUN/logs.json.txt" data/outputs/run_square_zvzg_s42_200ep/logs.json.txt
say "LOG_COPIED_BACK"

# Post-run reads, from the log rather than assumed:
#   * epoch time ~58 s (43 s means the pool override was ignored);
#   * a rollout at epoch 0 (N=1 at az_0, served by CamKeyImageRunner);
#   * latent_snapshots/ appearing (~1.7-3.7 MB/epoch; 201 npz by the end);
#   * batch-0 `aux_action_loss` vs the CALIB aux0_active, and the final value
#     vs m4base's 0.00487 (its floor is 0.2692);
#   * `aux_pose_loss` final value vs the CALIB floor bands (the registered
#     mapping is in the header) -- NOT the `aux_loss` line, which is the
#     weighted total here.
say "DONE -- next: data/zvzg_eval.sh, then data/zvzg_analysis.sh"
