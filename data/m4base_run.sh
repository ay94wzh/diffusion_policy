#!/bin/bash
# The z_v-supervision run: the clear run's recipe with the aux target's FRAME
# swapped camera -> base (view-specific -> view-invariant).
#
# ONE VARIABLE vs `m4clear` (run_square_m4clear_s42_200ep, committed 2026-10-01):
# the per-view auxiliary head's target. The clear run supervises each slot's
# z_v with the action chunk in that slot's CAMERA frame (a different target per
# slot); this run supervises with the raw stored absolute action -- identical
# for every live slot, i.e. the explicit cross-view consistency target. Data,
# slots, pool, range, seed, aux weight and the per-epoch latent probe are all
# identical (config train_diffusion_unet_image_workspace_m4base_latent).
#
# ---- PRE-REGISTRATION (fixed before launch; read the result against this) ----
#
# 1. BEHAVIOURAL. If the target's frame is the variable that matters, m4base
#    differs from m4clear beyond the 0.15 mean-over-viewpoints band on a
#    registered mean: in-dist ring 0.747, off-grid +-7.5..+-67.5 0.716,
#    +-82.5/+-90 0.310, elevation el_0/+-15 = 0.700/0/0. M4-informed
#    expectation, stated in advance: WITHIN BAND on all three -- the base
#    target is also largely decodable per view (probe_relpose), so the prior
#    is another null. Direction, if any, is NOT predicted (consistency
#    pressure could help; per-view solo-sufficiency could hurt).
# 2. LATENT SERIES (the instrument). Against the clear run's committed digest:
#    does the base-frame target change (a) the settle epochs of the
#    view-structure statistics (zv_pair_ratio 126, zv_pr 149), (b) the final
#    view-vs-state balance (zv_pair_ratio 0.485), (c) the z_g N-invariance
#    (0.53% at e200)? No direction registered -- these discriminate "the
#    supervision re-organized z_v" from "no effect anywhere", including where
#    the behaviour cannot see it.
# 3. MECHANISM-LIVE GATE. aux_loss falls well below its batch-0 value (M4
#    reference: 52x) and the batch-0 aux/diff ratio is recorded (M4 read
#    ~0.316 camera-frame; base-frame is re-measured here). M4's rule --
#    proceed unless the ratio exceeds 1.0 -- sets the weight; it does not
#    interpret the result.
#
# Runtime reference: the clear run was 201 epochs in 3 h 32 m at ~58 s/epoch
# (7.3 s/epoch per mean active view + ~14 s fixed at mean-N 6.0). An epoch
# time near 43 s means the 11-view pool override was silently ignored -- the
# exact failure this project has hit before (NOTES.md *Traps*).
#
# Run to completion with:  setsid bash -c 'bash data/m4base_run.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
RUN=/data/zihan/runs/run_square_m4base_s42_200ep
DEV=${DEV:-cuda:0}
LOG=data/m4base_run.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates. The aux test now covers the base-frame target
# (exact stored-action equality, slot identity, live-vs-camera mutation, RNG
# neutrality, normalizer==action); the probe-hook gate guards the instrument
# this run carries. Statuses unpiped, log tail printed on failure.
say "CPU gate: tests/test_aux_action_heads.py"
if ! "$PY" tests/test_aux_action_heads.py > /tmp/m4base_cpu_gate_aux.log 2>&1; then
    say "CPU_GATE_FAILED (aux) -- not launching; last lines follow"
    tail -n 20 /tmp/m4base_cpu_gate_aux.log | tee -a "$LOG"
    exit 1
fi
say "CPU gate: tests/test_latent_probe_hook.py"
if ! "$PY" tests/test_latent_probe_hook.py > /tmp/m4base_cpu_gate_probe.log 2>&1; then
    say "CPU_GATE_FAILED (probe) -- not launching; last lines follow"
    tail -n 20 /tmp/m4base_cpu_gate_probe.log | tee -a "$LOG"
    exit 1
fi
say "CPU_GATES_PASSED"

# 201 epochs, not 200: checkpoint and rollout fire on epoch % 50 == 0 and there
# is no save at the end, so 201 makes epoch 200 fire.
# training.resume=false: the config default is True and it SILENTLY resumes.
say "TRAIN_START -> $RUN ($DEV)"
"$PY" train.py --config-name=train_diffusion_unet_image_workspace_m4base_latent \
  task=m4_base_image_abs_multiview_az75 \
  training.seed=42 training.num_epochs=201 training.device="$DEV" \
  training.resume=false \
  dataloader.num_workers=14 val_dataloader.num_workers=2 checkpoint.topk.k=1 \
  logging.project=diffusion_policy_view \
  hydra.run.dir="$RUN" >> data/train_m4base.log 2>&1 &
TRAIN_PID=$!
say "TRAIN_PID=$TRAIN_PID"

# The launch gate, done by INSPECTING the resolved config rather than by timing
# the first epochs (NOTES.md: a contended epoch-time reading is ambiguous by
# construction). It must be able to fail AND print its evidence.
sleep 90
if [ -f "$RUN/.hydra/config.yaml" ]; then
    "$PY" - "$RUN/.hydra/config.yaml" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
ds = c['task']['dataset']
pool, rng = ds['view_pool'], ds['view_count_range']
slots = c['task']['env_runner']['m3_slots']
frame = ds.get('aux_action_frame')
probe = c.get('latent_probe', {})
ok = (sorted(pool) == list(range(1, 12)) and tuple(rng) == (1, 11)
      and slots == 11 and frame == 'base'
      and probe.get('n_states') == 128 and probe.get('az0_view') == 6)
print(f"OVERRIDE_GATE {'PASS' if ok else 'FAIL'}: pool={pool} range={rng} "
      f"slots={slots} aux_action_frame={frame!r} "
      f"probe=({probe.get('n_states')}, az0_view={probe.get('az0_view')}) "
      f"(want pool 1..11, range [1,11], slots 11, frame 'base', "
      f"probe 128/az0_view 6)  mean-N={(rng[0]+rng[1])/2}")
PYEOF
else
    say "OVERRIDE_GATE UNKNOWN: $RUN/.hydra/config.yaml not written yet"
fi

wait $TRAIN_PID
say "TRAIN_EXIT=$?"

# Copy the log back to the git-tracked location, beside the symlinked weights.
mkdir -p data/outputs/run_square_m4base_s42_200ep
cp -f "$RUN/logs.json.txt" data/outputs/run_square_m4base_s42_200ep/logs.json.txt
say "LOG_COPIED_BACK"

# Post-run reads, from the log rather than assumed:
#   * epoch time ~58 s (43 s means the pool override was ignored);
#   * a rollout at epoch 0 (N=1 at az_0, served by CamKeyImageRunner);
#   * latent_snapshots/ appearing (~1.7-3.7 MB/epoch; 201 npz by the end);
#   * train_loss - aux_loss at batch 0 (the aux/diff ratio recorded for the
#     pre-registration's mechanism-live gate).
say "DONE -- next: data/m4base_eval.sh, then data/m4base_analysis.sh"
