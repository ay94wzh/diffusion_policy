#!/bin/bash
# Phase 3: the "clear" run -- 11-view all-on training with a per-epoch latent hook.
#
# The only coded-but-never-launched artifact of the project (committed 918f4a2).
# Its purpose is a PER-EPOCH LATENT SERIES, which no earlier run has: they save
# only topk + latest, so nothing can order encoder collapse against the
# behavioural failure -- the project's stated open question.
#
# Task m4_aux_image_abs_multiview_az75: K=11 slots over ring indices 1..11
# (az -75..+75 every 15 deg), N drawn from [1,11] so mean active N is 6.0, every
# module on. NOTE: nothing is held out here (all 11 views train), so this cell is
# a NEW ARTIFACT for the latent study, NOT a rung of the committed ladder, and
# its behaviour numbers are not comparable to the committed cells.
#
# Run to completion with:  setsid bash -c 'bash data/clear_run.sh' < /dev/null &
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
RUN=/data/zihan/runs/run_square_m4clear_s42_200ep
LOG=data/clear_run.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# CPU gate first: RNG neutrality, bit-identical draws across two epochs, config
# compose, artifact layout, mutation check. All six passed 2026-10-01.
say "CPU gate: tests/test_latent_probe_hook.py"
if ! "$PY" tests/test_latent_probe_hook.py > /tmp/clear_cpu_gate.log 2>&1; then
    say "CPU_GATE_FAILED -- not launching (see /tmp/clear_cpu_gate.log)"; exit 1
fi
say "CPU_GATE_PASSED"

# 201 epochs, not 200: checkpoint and rollout fire on epoch % 50 == 0 and there
# is no save at the end, so 201 makes epoch 200 fire.
# training.resume=false: the config default is True and it SILENTLY resumes.
say "TRAIN_START -> $RUN"
"$PY" train.py --config-name=train_diffusion_unet_image_workspace_m4latent \
  task=m4_aux_image_abs_multiview_az75 \
  training.seed=42 training.num_epochs=201 training.device=cuda:1 \
  training.resume=false \
  dataloader.num_workers=14 val_dataloader.num_workers=2 checkpoint.topk.k=1 \
  logging.project=diffusion_policy_view \
  hydra.run.dir="$RUN" >> data/train_clear.log 2>&1 &
TRAIN_PID=$!
say "TRAIN_PID=$TRAIN_PID"

# The launch gate, done by INSPECTING the resolved config rather than by timing
# the first epochs. An epoch-time check only infers "was the override ignored?"
# from a wall-clock number, and this box's load swings ~5 to ~30 (NOTES.md
# *Machine and timing*), so a contended reading is ambiguous by construction.
# Reading config.yaml + the dataset's own draw is direct and contention-proof.
sleep 90
if [ -f "$RUN/.hydra/config.yaml" ]; then
    "$PY" - "$RUN/.hydra/config.yaml" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
ds = c['task']['dataset']
pool, rng = ds['view_pool'], ds['view_count_range']
slots = c['task']['env_runner']['m3_slots']
ok = sorted(pool) == list(range(1, 12)) and tuple(rng) == (1, 11) and slots == 11
print(f"OVERRIDE_GATE {'PASS' if ok else 'FAIL'}: pool={pool} range={rng} slots={slots} "
      f"(want pool 1..11, range [1,11], slots 11)  mean-N={(rng[0]+rng[1])/2}")
PYEOF
else
    say "OVERRIDE_GATE UNKNOWN: $RUN/.hydra/config.yaml not written yet"
fi

wait $TRAIN_PID
say "TRAIN_EXIT=$?"

# Copy the log back to the git-tracked location, beside the symlinked weights.
mkdir -p data/outputs/run_square_m4clear_s42_200ep
cp -f "$RUN/logs.json.txt" data/outputs/run_square_m4clear_s42_200ep/logs.json.txt
say "LOG_COPIED_BACK"

# Launch gates, read from the log rather than assumed:
#   * epoch time ~58 s (7.3 s/epoch per mean active view + 14 s fixed at mean-N
#     6.0). A value near 43 s means the 11-view pool override was silently
#     ignored -- the exact failure this project has hit before.
#   * a rollout at epoch 0 (N=1 at az_0, served by CamKeyImageRunner).
#   * latent_snapshots/ appearing, ~3.7 MB/epoch.
say "DONE"
