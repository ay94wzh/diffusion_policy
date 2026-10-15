#!/bin/bash
# The probabilistic-fusion runs (PLAN_G.md stage 1): per-view Gaussian heads
# (mu, clamped logvar) + a selectable fusion, on the m4base recipe. THREE cells,
# one code path, ONE variable each in the comparison against cell A:
#   A = the committed m4base run (MHA over z_v, aux on z_v) -- NOT retrained;
#   B = policy.obs_encoder.fusion_mode=mha  -- MHA over mu, aux on mu (the
#       deterministic-projection control: isolates the projection + the
#       supervision position from any probabilistic mechanism);
#   C = policy.obs_encoder.fusion_mode=poe  -- Product of Experts over
#       (mu_v, logvar_v): uncertainty-weighted fusion;
#   D = policy.obs_encoder.fusion_mode=mean -- the uniform mean of the means
#       (PoE vs plain averaging; at init C == D BITWISE, pinned in the test).
# Data (square ph ring13), K=11 slots over pool [1..11], N ~ [1,11], seed 42,
# 201 epochs, optimizer/EMA and the per-epoch latent probe are identical to
# m4base. The task yaml is a name-only clone of m4base's; the calibration stage
# below PROVES the clone changed no data (dataset-side references agree to
# <1e-6). The aux target stays in the BASE frame at the registered w_v = 1.0;
# no camera-pose supervision; the diffusion head is untouched and consumes mu_g
# (the fused mean) exactly as before (PLAN_G sections 3-4).
#
# RNG CONTRACT (claimed and CPU-pinned in tests/test_prob_gaussian_fusion.py,
# unlike zvzg's design-level non-claim): the Gaussian head is built from a
# PRIVATE RNG stream (save -> reseed a fresh CPU generator -> build -> restore
# in prob_view_conditioned_obs_encoder.py), so constructing it consumes nothing
# from the global stream. Every shared parameter -- and every later draw (UNet
# init, aux-head init, DataLoader shuffle seed, crops, noise, timesteps) -- is
# therefore bit-identical to m4base's. B/C/D differ from A by the fusion branch
# and the aux head's input, and from each other by one string.
#
# NOT CLAIMED: identical parameter COUNT (+2 x 262 k in the shared Gaussian
# head, ~0.18% of the 290 M policy) and therefore the optimizer's parameter
# list; and anything about the variance's statistical meaning -- round 1 tests
# whether the fused MEAN changes behaviour, not whether sigma is calibrated.
#
# Storage: the prob probe adds mu/lv/lv_n1/lvg to the per-epoch snapshots,
# ~+4.5 MB/epoch => ~+0.9 GB per cell on /data. Check `df -h` on miroc-server
# before the campaign (CLAUDE.md). Runtime reference ~3h35m/cell at ~58 s/epoch
# (m4base/zgpose); an epoch time near 43 s means an 11-view-pool override was
# silently ignored (NOTES.md *Traps*) -- but the pool lives in the task yaml
# here, and the launch gate proves the per-cell fusion_mode override landed.
#
# ---- PRE-REGISTRATION (fixed before launch; read the result against this).
# The two open choices are settled: off-grid is a TWO-WAY band on both
# contrasts, and the representation readouts are the fixed list in section 2 --
# every listed readout is reported for every cell, no post-hoc selection.
#
# 1. BEHAVIOURAL. success_rate, 50 paired episodes, means over viewpoints; the
#    family's band is 0.15 (single cells are never read individually). Cell A's
#    committed means: in-dist ring 0.796, off-grid +-7.5..+-67.5 0.774,
#    past the ring +-82.5/+-90 0.310, elevation el_0/el_m15/el_p15 0.840/0/0;
#    m4clear (0.747) / zgpose (0.735) / zvzg (0.751) are context, not the
#    comparison.
#      in-dist ring (11 vp, trained) -- NO directional prediction: no healthy
#        arm in the lineage has broken this band (all landed 0.735-0.796), so
#        the registered claim is only "within 0.15 of A" = the projection +
#        supervision move did no harm; beyond band either way is the finding.
#      off-grid (10 vp, held out) -- H2's axis, and the round's headline
#        question. Registered: BOTH contrasts, TWO-WAY, on the same 0.15 band
#        -- C-vs-B and D-vs-C, each read INSIDE band as UNRESOLVED (one seed,
#        50 episodes, with a measured mean-scale spread of 0.05-0.08 cannot
#        resolve less) and OUTSIDE band in either direction as evidence of a
#        real difference -- a C < B result is a finding, not a null to round
#        off.
#      past the ring (4 vp) -- REGISTERED SEPARATELY, no band: does the PoE
#        recover toward A's 0.310, or do we get a third sub-0.15 cell
#        (zgpose 0.140, zvzg 0.100)? Recovery = positive; another floor = an
#        extrapolation cost now with three arms behind it.
#      elevation el_0 / el_m15 / el_p15 -- three single cells, descriptive
#        only (every arm: 0.70-0.84 / 0 / 0).
#    Read the grid, not single contrasts: B-vs-A bounds the projection and the
#    supervision move; C-vs-B is the probabilistic contrast (precision-weighted
#    vs plain MHA over the same mu); D-vs-C asks whether anything beyond plain
#    averaging was gained.
# 2. THE INSTRUMENTS. In this arm the logged zg_* readouts describe mu_g (the
#    fused mean); z_views stays raw z_v, so the per-view statistics remain
#    comparable across all cells. Committed e200 values to move against:
#    probe zv_pair_ratio(1,11) 0.267(A) / 0.544(m4clear) / 2.255(zgpose) /
#    0.857(zvzg); digest rel(zg_n1, zg_full) tail 0.089 / 0.103 / 0.199 /
#    0.433 (same order). REGISTERED READOUTS AND MOVED LINES -- fixed now,
#    before any prob number exists; every readout below is reported for EVERY
#    cell, in both directions, and no readout outside this list enters a
#    mechanism claim (no post-hoc selection among instruments):
#      probe zv_pair_ratio(1,11)           moved iff outside [0.267, 2.255]
#      probe zg_across_view_subsets(1,11)  moved iff outside [0.066, 0.327]
#      digest rel(zg_n1, zg_full) tail(25) moved iff outside [0.089, 0.433]
#      probe N=1 zg pose readout (ridge)   moved iff outside
#        [1.20 deg, 17.8 deg] / [1.90 cm, 23.2 cm]
#    Each range is the committed four-cell envelope (A / m4clear / zgpose /
#    zvzg) and "outside" means beyond it by more than 10% of the range's own
#    width (0.20 / 0.03 / 0.03 / 1.7 deg / 2.1 cm). The margins are a stated
#    convention -- the across-seed noise of these statistics is unmeasured --
#    so a moved reading is a lead to follow up, not a verdict, and an
#    in-range value is a bound, not a demonstrated invariance.
#    Variance-quality readings (PLAN_G section 4) and their pre-fixed
#    thresholds:
#      lv_view_spread > 0 at e200 -- the views disagree about their own
#        uncertainty; at exactly 0 the PoE weights are uniform across slots
#        and C is a reparameterized D.
#      lv_at_min_frac / lv_at_max_frac < 0.9 -- else the clamp is binding and
#        PoE is railed toward winner-take-all.
#      corr(per-view lv, per-view error) and the fused-lv calibration --
#        stage-2 readings; registered here only as: the (deferred) uncertainty
#        script fixes those thresholds BEFORE those numbers are looked at.
# 3. MECHANISM GATES (must hold or the run is void):
#      mu_resid_rel == 0.0 at e0 (zero-init residual head) and > 0 later; a
#        head that stays at 0 makes the run a re-seeded m4base and void.
#      lv_* scalars finite every epoch; the clamp fractions are REPORTED, not
#        acted on: if lv_at_max_frac > 0.9 at e200 the bound is binding ->
#        report it as a finding, do not change the bounds mid-campaign.
#      D-cell lv_std == 0.0 EXACTLY (the logvar branch has no gradient path
#        there) -- an instrument check, and the reason D's uncertainty numbers
#        are not to be read as uncertainty.
#      the in-training rollout (N=1 at az_0, every 50 epochs) must not collapse.
# 4. DOCUMENTED AMBIGUITY. If a behavioural mean moves beyond band while the
#    variance instrument reads inert for that cell (lv_view_spread == 0 at
#    e200, or the D-cell caveat applies), the movement cannot be attributed to
#    uncertainty weighting: report the movement, do NOT attribute a mechanism.
# 5. RNG. The B/C/D-vs-A claim in the header above is registered as CLAIMED and
#    CPU-pinned; if the run-level draws diverge (e.g. a future module starts
#    consuming global RNG) the test suite is what catches it, not this driver.
#
# Run to completion with:  setsid bash -c 'bash data/prob_run.sh' < /dev/null &
# Two cells in parallel (2 GPUs):  CELLS=b DEV=cuda:0 bash data/prob_run.sh   in
# one shell, CELLS=c DEV=cuda:1 bash data/prob_run.sh in the other (do NOT read
# a contended epoch time as evidence of anything).
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
DEV=${DEV:-cuda:0}
CELLS=${CELLS:-b c d}
LOG=data/prob_run.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates. The new cell's suite plus the ones the shared
# workspace seam / probe seam could break. Statuses unpiped, log tail printed
# on failure.
for gate in test_prob_gaussian_fusion test_aux_action_heads \
            test_view_conditioned_obs_encoder test_latent_probe_hook \
            test_multiview_dataset; do
    say "CPU gate: tests/${gate}.py"
    if ! "$PY" "tests/${gate}.py" > "/tmp/prob_cpu_gate_${gate}.log" 2>&1; then
        say "CPU_GATE_FAILED (${gate}) -- not launching; last lines follow"
        tail -n 20 "/tmp/prob_cpu_gate_${gate}.log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

# ---- stage 1: calibration (dataset-only, can fail). The task yaml is a
# name-only clone of m4base's; this is the EXECUTABLE proof: the two datasets
# must agree to <1e-6 on the normalized aux targets' batch-0 references
# (aux0_active, the constant-predictor floor) at the registered seed/sample
# count. NOTE: diff0 is CELL-SPECIFIC here (A/B fuse via MHA, C/D start at the
# masked mean), so m4base's batch-0 1.0828 must NOT be reused -- read each
# cell's batch-0 diff from its own run log.
say "CALIBRATION: prob task vs m4base task, dataset-side references (must agree)"
if ! "$PY" - > /tmp/prob_calib.log 2>&1 <<'PYEOF'
import os
import numpy as np
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver('eval', eval, replace=True)
from diffusion_policy.dataset.multiview_image_dataset import (
    MultiViewImageDataset, AUX_ACTION_KEY, ACTION_DIM)

with hydra.initialize_config_dir(
        config_dir=os.path.join(os.getcwd(), 'diffusion_policy', 'config'),
        version_base=None):
    cfg_p = hydra.compose(config_name='train_diffusion_unet_image_workspace_prob_latent')
    cfg_b = hydra.compose(config_name='train_diffusion_unet_image_workspace_m4base_latent')

N_SAMPLES = 64        # the family's registered calibration draw

def refs(cfg, tag):
    ds = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(ds, MultiViewImageDataset), type(ds)
    K = len(ds.rgb_keys)
    C = ds.aux_n_steps
    norm = ds.get_normalizer()
    np.random.seed(0)
    idxs = np.random.choice(len(ds), size=N_SAMPLES, replace=False)
    t_act, v_mask = [], []
    for i in idxs:
        s = ds[int(i)]
        t_act.append(s[AUX_ACTION_KEY].numpy().reshape(-1, K, C * ACTION_DIM))
        v_mask.append(s['obs'][ds.view_mask_key].numpy().reshape(-1, K))
    t_act = np.concatenate(t_act)
    v_mask = np.concatenate(v_mask) > 0.5
    ta = norm[AUX_ACTION_KEY].normalize(
        t_act.reshape(-1, C * ACTION_DIM)).detach().numpy()
    am = v_mask.reshape(-1)
    assert am.any() and not am.all()
    aux0 = float((ta[am] ** 2).mean())
    floor = float(((ta[am] - ta[am].mean(0)) ** 2).mean())
    print(f'CALIB[{tag}] K={K} n={len(ds)} mean_active={v_mask.sum(-1).mean():.3f} '
          f'aux0_active={aux0:.6f} floor_const={floor:.6f}')
    return aux0, floor

a_p = refs(cfg_p, 'prob')
a_b = refs(cfg_b, 'm4base')
assert abs(a_p[0] - a_b[0]) < 1e-6, (a_p, a_b)
assert abs(a_p[1] - a_b[1]) < 1e-6, (a_p, a_b)
print(f'CALIB task-clone inert: aux0 delta {a_p[0] - a_b[0]:+.2e}, '
      f'floor delta {a_p[1] - a_b[1]:+.2e}')
PYEOF
then
    say "CALIBRATION_FAILED -- not launching; log tail follows"
    tail -n 20 /tmp/prob_calib.log | tee -a "$LOG"
    exit 1
fi
cat /tmp/prob_calib.log | tee -a "$LOG"
say "CALIBRATION_OK"

# ---- stage 2: the cells ------------------------------------------------------
# The config file's committed default IS cell B; c/d are the one-line override.
for CELL in $CELLS; do
    case "$CELL" in
        b) MODE=mha ;;
        c) MODE=poe ;;
        d) MODE=mean ;;
        *) say "UNKNOWN CELL '$CELL' (want b|c|d)"; exit 1 ;;
    esac
    RUN=/data/zihan/runs/run_square_prob_${CELL}_s42_200ep
    if [ -e "$RUN" ]; then
        say "RUN_DIR_EXISTS $RUN -- refusing to start (no silent resume)"
        exit 1
    fi
    say "TRAIN_START cell=$CELL fusion_mode=$MODE -> $RUN ($DEV)"

    # 201 epochs, not 200: checkpoint and rollout fire on epoch % 50 == 0 and
    # there is no save at the end, so 201 makes epoch 200 fire.
    # training.resume=false: the config default is True and it SILENTLY resumes.
    "$PY" train.py --config-name=train_diffusion_unet_image_workspace_prob_latent \
      task=prob_base_image_abs_multiview_az75 \
      training.seed=42 training.num_epochs=201 training.device="$DEV" \
      training.resume=false \
      dataloader.num_workers=14 val_dataloader.num_workers=2 checkpoint.topk.k=1 \
      policy.obs_encoder.fusion_mode="$MODE" \
      logging.project=diffusion_policy_view \
      hydra.run.dir="$RUN" >> data/train_prob_${CELL}.log 2>&1 &
    TRAIN_PID=$!
    say "TRAIN_PID=$TRAIN_PID"

    # The launch gate, done by INSPECTING the resolved config rather than by
    # timing the first epochs (NOTES.md: a contended epoch-time reading is
    # ambiguous by construction). It must be able to fail AND print evidence --
    # the fusion_mode override is the one thing that could silently no-op.
    sleep 90
    if [ -f "$RUN/.hydra/config.yaml" ]; then
        "$PY" - "$RUN/.hydra/config.yaml" "$MODE" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
want = sys.argv[2]
ds, pol, enc = c['task']['dataset'], c['policy'], c['policy']['obs_encoder']
probe = c.get('latent_probe', {})
ok = (enc.get('fusion_mode') == want
      and 'prob_view_conditioned_obs_encoder' in enc['_target_']
      and 'DiffusionUnetImagePolicyProb' in pol['_target_']
      and pol.get('aux_input') == 'mu_v'
      and abs(float(pol['aux_loss_weight']) - 1.0) < 5e-5
      and int(pol['aux_n_steps']) == 8 == int(ds.get('aux_n_steps'))
      and sorted(ds['view_pool']) == list(range(1, 12))
      and tuple(ds['view_count_range']) == (1, 11)
      and int(c['task']['env_runner']['m3_slots']) == 11
      and probe.get('n_states') == 128 and probe.get('az0_view') == 6
      and probe.get('save_gaussian') is True)
print(f"OVERRIDE_GATE {'PASS' if ok else 'FAIL'}: fusion_mode={enc.get('fusion_mode')} "
      f"(want {want}) encoder={enc['_target_'].rsplit('.', 1)[-1]} "
      f"aux_input={pol.get('aux_input')} w_v={pol.get('aux_loss_weight')} "
      f"aux_n_steps={pol.get('aux_n_steps')}/{ds.get('aux_n_steps')} "
      f"pool={ds['view_pool'][0]}..{ds['view_pool'][-1]} "
      f"range={tuple(ds['view_count_range'])} "
      f"slots={c['task']['env_runner']['m3_slots']} "
      f"probe=({probe.get('n_states')}, az0_view={probe.get('az0_view')}, "
      f"save_gaussian={probe.get('save_gaussian')})")
PYEOF
    else
        say "OVERRIDE_GATE UNKNOWN: $RUN/.hydra/config.yaml not written yet"
    fi

    wait $TRAIN_PID
    say "TRAIN_EXIT=$?"

    # Copy the log back to the git-tracked location (weights stay under /data).
    mkdir -p "data/outputs/run_square_prob_${CELL}_s42_200ep"
    cp -f "$RUN/logs.json.txt" "data/outputs/run_square_prob_${CELL}_s42_200ep/logs.json.txt"
    say "LOG_COPIED_BACK cell=$CELL"
done

# Post-run reads, from the log rather than assumed:
#   * epoch time ~58 s (43 s means an override was silently ignored);
#   * a rollout at epoch 0 (N=1 at az_0, served by CamKeyImageRunner);
#   * latent_snapshots/ appearing (~1.7-3.7 MB/epoch + ~4.5 MB of mu/lv/lvg);
#   * `latent_probe/mu_resid_rel` EXACTLY 0.0 at e0 (zero-init head), > 0 later;
#   * `latent_probe/lv_mean` ~ 0 early, and in cell `d` `lv_std` pinned at 0.0
#     exactly (no gradient path -- an instrument check, not a finding);
#   * `aux_action_loss` (the unweighted per-term key) vs the CALIB aux0_active
#     (0.362935, the run's own printed value) and vs m4base's committed
#     0.361 -> 0.0049 (74x, against the BASE-frame floor 0.308614 -- the
#     CALIB floor_const above, reproduced exactly by zvzg's 2026-10-12
#     calibration). CORRECTION (2026-10-15, pre-launch, comment-only): the
#     0.2692 this line used to cite is the CAMERA-frame target's floor from the
#     m4on/m4off milestone (PROGRESS.md, M4; its batch-0 was 0.2389), not this
#     campaign's. Reading a base-frame loss against it overstates convergence
#     by ~14%; do not reuse it here.
#     (aux0_active 0.362935 vs m4base's batch-0 0.36095 is a DRAW difference,
#     not a data change -- m4base's is its first shuffled training batch, this
#     is the registered seed-0 64-sample draw; the dataset is provably the
#     same, which stage 1 asserts and the 0.00e+00 deltas confirm.)
#   * batch-0 `train_loss` is CELL-SPECIFIC (A/B fuse via MHA, C/D start at
#     the masked mean): read each cell's own, never reuse m4base's 1.0828.
say "DONE -- next: data/prob_eval.sh (cell by cell), then the uncertainty read"
