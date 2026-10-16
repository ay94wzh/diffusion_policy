#!/bin/bash
# The z-supervision runs (PLAN_Z.md): per-view shared-semantics alignment
# (L_align on z_v via a shared projection head P_s) and cross-view action
# consistency (L_cons on z_g: two independent view draws, one noisy action /
# timestep / noise through the UNet twice). THREE cells, one class, CLI
# overrides only:
#   a zalign -- align_weight=W_ALIGN, cons_weight=0,      emit_second_draw=false
#   b zcons  -- align_weight=0,      cons_weight=W_CONS,  emit_second_draw=true
#   c zboth  -- align_weight=W_ALIGN, cons_weight=W_CONS, emit_second_draw=true
# Baseline = the committed m4base run (cell A of the prob round), NOT retrained.
# The SAME (W_ALIGN, W_CONS) go into all three cells, so C decomposes as A + B.
# Data (square ph ring13), K=11 slots over pool [1..11], N ~ [1,11], seed 42,
# 201 epochs, optimizer/EMA and the per-epoch latent probe identical to m4base.
# The task yaml is a name-only clone of m4base's + `emit_second_draw` (default
# false there, flipped on the CLI for b/c) -- the calibration PROVES the clone
# changed no data and that the flag leaves draw 1 and the global numpy stream
# bit-identical.
#
# RNG CONTRACT (claimed and CPU-pinned in tests/test_z_supervision.py):
#   * cell a: construction, the dataset draw, the crop indices, the noise and
#     the timestep draws are ALL bit-identical to m4base's -- the align term
#     consumes no RNG and adds no forward, so the arm differs from the control
#     only by the parameter trajectory and the ~262k extra head parameters.
#   * cells b/c: the second encoder pass runs under a torch CPU+CUDA RNG
#     save/restore, so every shared draw keeps its stream position too; the
#     second draw's view subset rides a stateless per-sample private stream
#     (`RandomState(seed + idx)`) -- a pure function of (seed, idx).
#   * the enc2 crop windows are a REUSE of draw-1's saved stream state (not
#     fresh entropy) -- deliberate, so the cons residual reflects the view set,
#     not crop noise.
# NOT CLAIMED: identical parameter count / optimizer parameter list (the shared
# semantic head is +~262k, ~0.09% of the ~290M policy), and anything about the
# variance machinery (there is none in this round).
#
# ---- PRE-REGISTRATION (fixed before launch; read the results against this).
# 1. BEHAVIOURAL. success_rate, 50 paired episodes, means over viewpoints; the
#    family's band is 0.15 (single cells are never read individually). Cell A's
#    committed means: in-dist ring 0.796, off-grid +-7.5..+-67.5 0.774, past
#    the ring +-82.5/+-90 0.310, elevation el_0/el_m15/el_p15 0.840/0/0;
#    m4clear (0.747) / zgpose (0.735) / zvzg (0.751) are context, not the
#    comparison.
#      in-dist ring (11 vp, trained) -- NO directional prediction: no healthy
#        arm in the lineage has broken this band (all landed 0.735-0.796), so
#        the registered claim is only "within 0.15 of A"; beyond band either
#        way is the finding.
#      off-grid (10 vp, held out) -- the round's generalization axis. By
#        PLAN_Z's order A, B, C are the three cells; NO directional prediction
#        is registered for any of them (the supervision arms' record: every
#        registered mean inside band, twice outside as a cost). Two-way band
#        0.15 vs A: inside = unresolved at this budget, outside either way is
#        evidence of a real difference.
#      past the ring (4 vp) -- REGISTERED SEPARATELY, no band: does any cell
#        recover toward A's 0.310, or repeat the pose arms' floor (zgpose 0.140,
#        zvzg 0.100)?
#      elevation el_0 / el_m15 / el_p15 -- three single cells, descriptive only
#        (every arm: 0.70-0.92 / 0 / 0).
# 2. THE INSTRUMENTS. The same battery as the prob round (digest, probes,
#    screens -- data/z_analysis.sh), read against envelopes COMPUTED from the
#    committed grids (m4base/m4clear/zgpose/zvzg; margins = 10% of the
#    envelope's width): probe zv_pair_ratio(1,11), probe
#    zg_across_view_subsets(1,11), digest rel(zg_n1, zg_full) tail(25), probe
#    N=1 zg pose readout. PLUS one new readout this round is the first to have
#    a mechanism for -- the SEMANTIC readout (analyze_semantic_head.py on the
#    probe set): L_align and the mean pairwise cosine of P_s(z_v) at e200,
#    against a re-initialized-head control. The registered head-collapse
#    signature (risk R1): align_loss ~ 0 with cos ~ 1 while zv_pair_ratio
#    stays at m4base's values -- the pressure was consumed by the head, not
#    the encoder. If that signature appears, report it; do not round it away.
# 3. MECHANISM GATES (must hold or the run is void):
#      align_loss logged (cells a/c) and falling below its batch-0 value;
#      cons_loss logged (cells b/c) and falling below its batch-0 value;
#      per-cell batch-0 `diff_loss` == m4base's derived 1.082832 (its
#        train_loss 1.4437798 minus its aux_loss 0.360948) -- the RNG-lockstep
#        claim read from the artifacts;
#      `diff_loss` must NOT rise above the family's late-training band
#        (0.0028-0.0069 at e200) -- the instrument for risk R2 (a cons term
#        that shrinks the UNet's dependence on the conditioning instead of
#        aligning views would show up here);
#      the in-training rollout (N=1 at az_0, every 50 epochs) must not collapse;
#      201 latent npz present and finite in every cell (analysis stage).
# 4. DOCUMENTED AMBIGUITY. If a behavioural mean moves beyond band while the
#    instruments read inert for that cell, the movement cannot be attributed to
#    either mechanism: report the movement, do NOT attribute a mechanism.
# 5. WEIGHTS. The two lambdas are calibrated pre-launch by the family
#    convention W = 0.33 * diff0 / loss0 from ONE train-mode forward at init on
#    the registered 64-sample seed-0 draw (stage 1 below), printed with the
#    realized shares, warned if W > 3, and registered HERE in the log before
#    any run starts. The 0.33 share holds only at init and only for that draw
#    (the family convention's own caveat).
#
# Run to completion with:  setsid bash -c 'bash data/z_run.sh' < /dev/null &
# Two cells in parallel (2 GPUs):  CELLS=a DEV=cuda:0 bash data/z_run.sh   in
# one shell, CELLS=b DEV=cuda:1 bash data/z_run.sh in the other. The FIRST
# shell writes the calibrated (W_ALIGN, W_CONS) to data/z_calib_env.sh; the
# second REUSES that file (shadowed vars would break the "C = A + B" claim --
# each shell must not calibrate independently). Delete the file (or
# FORCE_CALIB=1) to recalibrate. Never read a contended epoch time as evidence
# of anything.
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
DEV=${DEV:-cuda:0}
CELLS=${CELLS:-a b c}
LOG=data/z_run.log
CALIB_ENV=data/z_calib_env.sh
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- stage 0: CPU gates. The new cell's suite plus the ones the shared
# workspace seam / dataset seam could break. Statuses unpiped, log tail printed
# on failure.
for gate in test_z_supervision test_aux_action_heads test_zvzg_supervision \
            test_view_conditioned_obs_encoder test_latent_probe_hook \
            test_multiview_dataset; do
    say "CPU gate: tests/${gate}.py"
    if ! "$PY" "tests/${gate}.py" > "/tmp/z_cpu_gate_${gate}.log" 2>&1; then
        say "CPU_GATE_FAILED (${gate}) -- not launching; last lines follow"
        tail -n 20 "/tmp/z_cpu_gate_${gate}.log" | tee -a "$LOG"
        exit 1
    fi
done
say "CPU_GATES_PASSED"

# ---- stage 1: calibration (executable proof + the two lambdas).
#  (i) the task clone is inert: dataset-side references (aux0_active, the
#      constant-predictor floor) agree with m4base's to <1e-6 at the registered
#      seed/sample count;
#  (ii) the emit_second_draw flag is RNG-neutral for draw 1 and the global
#      numpy stream (fetch the same idxs with the flag on/off after the same
#      np.random.seed, compare view_mask + cam keys + the post-call state);
#  (iii) ONE train-mode forward of the z policy at init -> diff0, align0,
#      cons0; W_ALIGN = round(0.33*diff0/align0, 4), W_CONS =
#      round(0.33*diff0/cons0, 4); CALIB_WARN if either exceeds 3 (W_CONS can
#      legitimately be large -- cons0 is a difference of small UNet outputs).
# NOTE the calibration forward's diff0 is draw-specific (a 64-sample CPU/GPU
# batch), so it is PRINTED next to the family constant 1.082832 as a
# cross-check but the W's are built from the measured value, stated as such.
say "CALIBRATION: z task vs m4base task + flag neutrality + lambda calibration"
if [ -f "$CALIB_ENV" ] && [ "${FORCE_CALIB:-0}" != "1" ]; then
    say "CALIBRATION_REUSED from $CALIB_ENV (delete it or FORCE_CALIB=1 to redo)"
    . "$CALIB_ENV"
else
if ! "$PY" - "$DEV" > /tmp/z_calib.log 2>&1 <<'PYEOF'
import os, sys
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

OmegaConf.register_new_resolver('eval', eval, replace=True)
from diffusion_policy.dataset.multiview_image_dataset import (
    MultiViewImageDataset, AUX_ACTION_KEY, SECOND_OBS_KEY)
from diffusion_policy.common.pytorch_util import dict_apply

dev = sys.argv[1]
with hydra.initialize_config_dir(
        config_dir=os.path.join(os.getcwd(), 'diffusion_policy', 'config'),
        version_base=None):
    cfg_z = hydra.compose(
        config_name='train_diffusion_unet_image_workspace_z_latent')
    cfg_b = hydra.compose(
        config_name='train_diffusion_unet_image_workspace_m4base_latent')

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
        t_act.append(s[AUX_ACTION_KEY].numpy().reshape(-1, K, C * 10))
        v_mask.append(s['obs'][ds.view_mask_key].numpy().reshape(-1, K))
    t_act = np.concatenate(t_act)
    v_mask = np.concatenate(v_mask) > 0.5
    ta = norm[AUX_ACTION_KEY].normalize(
        t_act.reshape(-1, C * 10)).detach().numpy()
    am = v_mask.reshape(-1)
    assert am.any() and not am.all()
    aux0 = float((ta[am] ** 2).mean())
    floor = float(((ta[am] - ta[am].mean(0)) ** 2).mean())
    print(f'CALIB[{tag}] K={K} n={len(ds)} mean_active={v_mask.sum(-1).mean():.3f} '
          f'aux0_active={aux0:.6f} floor_const={floor:.6f}')
    return aux0, floor

a_z = refs(cfg_z, 'z')
a_b = refs(cfg_b, 'm4base')
assert abs(a_z[0] - a_b[0]) < 1e-6, (a_z, a_b)
assert abs(a_z[1] - a_b[1]) < 1e-6, (a_z, a_b)
print(f'CALIB task-clone inert: aux0 delta {a_z[0] - a_b[0]:+.2e}, '
      f'floor delta {a_z[1] - a_b[1]:+.2e}')

# (ii) the flag is draw-1/global-RNG neutral
ds_off = hydra.utils.instantiate(cfg_z.task.dataset)
ds_on = hydra.utils.instantiate(
    cfg_z.task.dataset, emit_second_draw=True)
np.random.seed(7)
s_off = ds_off[3]
st_off = np.random.get_state()
np.random.seed(7)
s_on = ds_on[3]
st_on = np.random.get_state()
for k in s_off['obs']:
    assert torch.equal(s_on['obs'][k], s_off['obs'][k]), k
assert st_off[0] == st_on[0] and (st_off[1] == st_on[1]).all() \
    and st_off[2] == st_on[2], 'the second draw consumed global numpy RNG'
assert SECOND_OBS_KEY in s_on and SECOND_OBS_KEY not in s_off
print('CALIB flag-neutral: draw 1 + global numpy state identical with '
      'emit_second_draw on/off')

# (iii) one train-mode forward -> the two lambda weights
np.random.seed(0)
idxs = list(np.random.choice(len(ds_on), size=N_SAMPLES, replace=False))
samples = [ds_on[int(i)] for i in idxs]
batch = {
    'obs': {k: torch.stack([s['obs'][k] for s in samples])
            for k in samples[0]['obs']},
    'action': torch.stack([s['action'] for s in samples]),
    AUX_ACTION_KEY: torch.stack([s[AUX_ACTION_KEY] for s in samples]),
    SECOND_OBS_KEY: {k: torch.stack([s[SECOND_OBS_KEY][k] for s in samples])
                     for k in samples[0][SECOND_OBS_KEY]},
}
device = torch.device(dev if torch.cuda.is_available() else 'cpu')
batch = dict_apply(batch, lambda x: x.to(device))

torch.manual_seed(42)
pol = hydra.utils.instantiate(cfg_z.policy)
pol.set_normalizer(ds_on.get_normalizer())
pol.to(device)
pol.train()          # match the run's crop regime (random_crop consumes RNG)
with torch.no_grad():
    loss = pol.compute_loss(batch)
diff0 = float(pol.last_diff_loss)
align0 = float(pol.last_align_loss)
cons0 = float(pol.last_cons_loss)
assert align0 > 0.0, 'align0 == 0 at init: the head is dead (zero-init?)'
assert cons0 > 0.0, 'cons0 == 0 at init: the two draws are indistinguishable'
print(f"CALIB forward (device={device}, init, train-mode crops): "
      f"diff0={diff0:.6f} align0={align0:.6f} cons0={cons0:.6f} "
      f"(total {float(loss):.6f})")
FAMILY_DIFF0 = 1.082832   # m4base: 1.4437798 - 0.360948
print(f'CALIB diff0 vs the family constant {FAMILY_DIFF0}: '
      f'{diff0 - FAMILY_DIFF0:+.4f} (a DIFFERENT draw -- informational only)')
w_a = round(0.33 * diff0 / align0, 4)
w_c = round(0.33 * diff0 / cons0, 4)
print(f'CALIB shares at init: align 0.330, cons 0.330 (by construction)')
for name, w in (('W_ALIGN', w_a), ('W_CONS', w_c)):
    if w > 3.0:
        print(f'CALIB_WARN {name}={w} exceeds 3 -- the 0.33 share may pull '
              f'hard on this axis; registered, NOT adjusted')
print(f'CALIB_W_ALIGN={w_a}')
print(f'CALIB_W_CONS={w_c}')
PYEOF
then
    say "CALIBRATION_FAILED -- not launching; log tail follows"
    tail -n 25 /tmp/z_calib.log | tee -a "$LOG"
    exit 1
fi
cat /tmp/z_calib.log | tee -a "$LOG"
eval "$(grep -E '^CALIB_W_' /tmp/z_calib.log)"
[ -n "${CALIB_W_ALIGN:-}" ] && [ -n "${CALIB_W_CONS:-}" ] \
    || { say "CALIB weights missing from the log -- aborting"; exit 1; }
printf 'CALIB_W_ALIGN=%s\nCALIB_W_CONS=%s\n' \
    "$CALIB_W_ALIGN" "$CALIB_W_CONS" > "$CALIB_ENV"
say "CALIBRATION_OK (W_ALIGN=$CALIB_W_ALIGN W_CONS=$CALIB_W_CONS, written to $CALIB_ENV)"
fi

# ---- stage 2: the cells ------------------------------------------------------
# The config's committed defaults are both weights 0.0 (the aux control);
# every cell is a CLI override, so the launch gate can catch a no-op.
for CELL in $CELLS; do
    case "$CELL" in
        a) W_AL=$CALIB_W_ALIGN; W_CO=0.0;          FS=false ;;
        b) W_AL=0.0;            W_CO=$CALIB_W_CONS; FS=true  ;;
        c) W_AL=$CALIB_W_ALIGN; W_CO=$CALIB_W_CONS; FS=true  ;;
        *) say "UNKNOWN CELL '$CELL' (want a|b|c)"; exit 1 ;;
    esac
    RUN=/data/zihan/runs/run_square_z${CELL}_s42_200ep
    if [ -e "$RUN" ]; then
        say "RUN_DIR_EXISTS $RUN -- refusing to start (no silent resume)"
        exit 1
    fi
    say "TRAIN_START cell=z${CELL} align=$W_AL cons=$W_CO second_draw=$FS -> $RUN ($DEV)"

    # 201 epochs, not 200: checkpoint and rollout fire on epoch % 50 == 0 and
    # there is no save at the end, so 201 makes epoch 200 fire.
    # training.resume=false: the config default is True and it SILENTLY resumes.
    "$PY" train.py --config-name=train_diffusion_unet_image_workspace_z_latent \
      task=z_base_image_abs_multiview_az75 \
      training.seed=42 training.num_epochs=201 training.device="$DEV" \
      training.resume=false \
      dataloader.num_workers=14 val_dataloader.num_workers=2 checkpoint.topk.k=1 \
      policy.align_weight="$W_AL" policy.cons_weight="$W_CO" \
      task.dataset.emit_second_draw="$FS" \
      logging.project=diffusion_policy_view \
      hydra.run.dir="$RUN" >> data/train_z_${CELL}.log 2>&1 &
    TRAIN_PID=$!
    say "TRAIN_PID=$TRAIN_PID"

    # The launch gate, by INSPECTING the resolved config rather than timing the
    # first epochs. It must be able to fail in BOTH directions: an off-weight
    # must be exactly 0.0, an on-weight must equal the calibrated value, and
    # emit_second_draw must match whether cons is on.
    sleep 90
    if [ -f "$RUN/.hydra/config.yaml" ]; then
        "$PY" - "$RUN/.hydra/config.yaml" "$W_AL" "$W_CO" "$FS" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
want_al, want_co, want_fs = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
ds, pol, enc = c['task']['dataset'], c['policy'], c['policy']['obs_encoder']
probe = c.get('latent_probe', {})
ok = ('DiffusionUnetImagePolicyZSup' in pol['_target_']
      and abs(float(pol['align_weight']) - want_al) < 5e-5
      and abs(float(pol['cons_weight']) - want_co) < 5e-5
      and (ds.get('emit_second_draw') is True) == (want_fs == 'true')
      and 'ViewConditionedObsEncoder' in enc['_target_']
      and abs(float(pol['aux_loss_weight']) - 1.0) < 5e-5
      and int(pol['aux_n_steps']) == 8 == int(ds.get('aux_n_steps'))
      and sorted(ds['view_pool']) == list(range(1, 12))
      and tuple(ds['view_count_range']) == (1, 11)
      and int(c['task']['env_runner']['m3_slots']) == 11
      and probe.get('n_states') == 128 and probe.get('az0_view') == 6)
print(f"OVERRIDE_GATE {'PASS' if ok else 'FAIL'}: "
      f"align={pol.get('align_weight')} (want {want_al}) "
      f"cons={pol.get('cons_weight')} (want {want_co}) "
      f"emit_second_draw={ds.get('emit_second_draw')} (want {want_fs}) "
      f"pol={pol['_target_'].rsplit('.', 1)[-1]} "
      f"aux_w={pol.get('aux_loss_weight')} aux_n_steps={pol.get('aux_n_steps')}/{ds.get('aux_n_steps')} "
      f"pool={ds['view_pool'][0]}..{ds['view_pool'][-1]} "
      f"range={tuple(ds['view_count_range'])} "
      f"slots={c['task']['env_runner']['m3_slots']} "
      f"probe=({probe.get('n_states')}, az0_view={probe.get('az0_view')})")
PYEOF
    else
        say "OVERRIDE_GATE UNKNOWN: $RUN/.hydra/config.yaml not written yet"
    fi

    wait $TRAIN_PID
    say "TRAIN_EXIT=$?"

    # Copy the log back to the git-tracked location (weights stay under /data).
    mkdir -p "data/outputs/run_square_z${CELL}_s42_200ep"
    cp -f "$RUN/logs.json.txt" "data/outputs/run_square_z${CELL}_s42_200ep/logs.json.txt"
    say "LOG_COPIED_BACK cell=z${CELL}"
done

# Post-run reads, from the log rather than assumed:
#   * epoch time ~58 s for cell a; ~110-120 s for b/c -- a b/c epoch time near
#     58 s means `emit_second_draw` did not land (the physical proof the second
#     encoder+UNet pass ran; the inverse of the 43 s trap in NOTES.md);
#   * batch-0 `diff_loss` == 1.082832 (the RNG-lockstep claim from the
#     artifacts; every cell, because the save/restores keep the stream);
#   * batch-0 `align_loss` (cells a/c) vs the CALIB align0 -- same order, not
#     equality (different draws); `align_loss` must fall below it and keep
#     falling; same for `cons_loss` in b/c vs CALIB cons0;
#   * `align_loss` ABSENT-or-0.0 in cell b and `cons_loss` in cell a (the
#     delegation path logs the attribute as 0.0);
#   * `diff_loss` late-training at the family band (0.0028-0.0069 at e200);
#     a rise would be risk R2 (the cons term fighting the diffusion objective);
#   * a rollout at epoch 0 (N=1 at az_0, served by CamKeyImageRunner);
#   * latent_snapshots/ appearing (~1.7-3.7 MB/epoch, all finite).
say "DONE -- next: data/z_eval.sh (cell by cell), then data/z_analysis.sh"
