"""Does the predicted uncertainty mean anything? (PLAN_G.md sections 5-stage-4 / 7.2)

The probabilistic-fusion round (cells B/C/D) gave every view a Gaussian
``q_v = N(mu_v, diag(exp(logvar_v)))`` and fused C's by Product of Experts. The
behavioural read and the four registered representation readouts are separate
tools; THIS script answers the round's own remaining question -- PLAN_G section
7.2: *is the variance the model predicts a real uncertainty, or just two extra
output branches?* The run driver (`data/prob_run.sh`) checked the two readings it
could fix in advance (``lv_view_spread > 0``, clamp fractions ``< 0.9``) and
deferred everything else to this script to fix BEFORE the numbers are looked at.
That pre-registration is section *Registered thresholds*, below.

What is measured, on a frozen checkpoint (EMA weights -- the ones the sweeps and
the per-epoch probe use), one forward pass per state over a full 11-view draw:

  R1  do views disagree about their own uncertainty? -- per-state std of the
      per-view scalar logvar across the live views, averaged (the probe's own
      ``lv_view_spread`` statistic, recomputed as a cross-check).
  R2  do the *worse* views claim more uncertainty? -- per ring view, mean scalar
      logvar vs mean error, rank-correlated over the 11 ring poses, plus the
      direct contrast of the azimuth-0 view against the ring's edge views.
  R3  do high-variance (state, view) pairs have higher error? -- rank correlation
      over live pairs between scalar logvar and TWO error readings: the
      auxiliary action head's own per-pair MSE (its training target), and the
      residual of a ridge decode from ``mu_v`` to the dataset's true low-dim
      state (``robot0_eef_pos``), which the model was never trained to predict.
  R4  does the FUSED variance track real error? -- per-pair ``lvg`` vs (a) the
      same ridge decode run from ``mu_g`` (= the digest's ``zg``), and (b) the
      per-sample diffusion loss, read off ``policy.last_diff_loss`` one sample
      at a time (the policy reduces that loss over the batch, so B=1 is the
      only way to get it per sample without re-implementing the loss body).
  R5  does adding/removing a view change the fusion sensibly? -- offline
      re-fusion with the PRODUCTION ``poe_fuse`` (no extra forward pass) over
      view subsets of the full draw: ``lvg`` and the drift of ``mu_g`` as N goes
      1..11, plus a leave-one-out table. The arrays are the checkpoint's own
      fp32 outputs (not the fp16 snapshots), so the storage floor does not
      enter; the only assumption is that ``mu_v``/``logvar_v`` do not depend on
      which other views share the draw, which the probe checks on every run
      (``zv_n_agnostic_max_abs_diff``).
  R6  overconfidence (PLAN_G section 8, risk 2) -- the PoE-implied per-dim
      sigma against the between-view spread of the means: a product over
      correlated experts claims more certainty than their agreement supports.

What this script does NOT do: it never touches the simulator, and it makes no
claim about the *statistical* correctness of the Gaussian -- there is no ground
truth for the latent state, so "calibration" here can only mean "the variance
ranks states/views the way their errors are ranked", which is exactly what the
correlations above test. A null here is a bound on the variance's usefulness at
this draw size, not a proof that no signal exists.

Instrument gates, each of which can fail and prints evidence: the checkpoint
must be a PoE cell (B/D are refused -- use ``--assert-dead-logvar``, which
asserts the logvar branch is EXACTLY zero, not merely small); the full-draw
override must actually take (``mean_active == K``); and the per-pair aux error,
which this script re-derives from the policy's normalizer, encoder and head,
must agree with the policy's own ``last_aux_action_loss`` at B=1 to 1e-3
relative -- the one place a silent divergence from the training reduction could
hide, reported in the JSON under ``gates.aux_mirror``.

Registered thresholds (fixed 2026-10-16, BEFORE any number from this script was
computed; every value is read against them, with no post-hoc selection):

  R1  informative iff value > 0 AND the cross-check against the run log's e200
      ``lv_view_spread`` holds within a factor of 2 either way (a different
      state draw of the same statistic). Exactly 0 -> the PoE weights are
      uniform and cell C is a re-parameterized D; reported as FAILED, not
      rounded.
  R2  supported iff Spearman rho(mean lv, mean error) >= 0.5 with one-sided
      permutation p < 0.05, for BOTH error readings; rho <= 0 for both is
      reported as a finding AGAINST the model (the uncertainty is anti-
      correlated with which views are informative), not as a null; anything
      between is "weak".
  R3  supported iff rho >= 0.2 AND above the 95th percentile of the shuffle
      control, per error reading; rho <= 0 reported as a null.
  R4  same form as R3, each sub-reading judged separately; the diffusion-loss
      sub-reading carries the pre-stated caveat that the loss reflects the
      UNet's fit as much as the fusion's uncertainty.
  R5  registered expectation: ``lvg`` falls with N and the drift decays; a
      violation of EITHER is a finding. Leave-one-out: removing the azimuth-0
      view (M1's own camera pose) must move the fused mean more than removing an
      edge view -- the ring pose at +-75, which is the pool's outermost trained
      view (the +-90 poses are NOT in the pool [1..11]); a violation is a
      finding, and the whole 11-row table is reported so no single row is read
      alone.
  R6  no verdict: the median ratio and the fraction of (state, dim) cells with
      implied sigma below the views' own disagreement are REPORTED under the
      stated convention that a high fraction means overconfident-relative-to-
      disagreement. A diagnostic, not a hypothesis test.

Cell scope: B (MHA fusion) and D (mean fusion) have no gradient path to the
log-variance branch -- ``lv_std`` is exactly 0.0 in their training logs -- so
their uncertainty readings are meaningless by construction. Run the full read
only on a PoE cell (the script refuses otherwise); use ``--assert-dead-logvar``
for B and D, which verifies the dead branch (logvar exactly 0 on live slots, no
fused logvar) and exits nonzero if it is live, i.e. if the round's "do not read
their variance" premise is wrong.

Usage (on miroc-server -- needs the checkpoint, the multi-view zarr, a GPU):

    python analyze_variance_quality.py -c <run>/checkpoints/latest.ckpt \\
        -o data/analysis_prob_c/variance_quality -d cuda:0 \\
        -l data/outputs/run_square_prob_c_s42_200ep/logs.json.txt

    # the B/D sanity path (fast; no full read)
    python analyze_variance_quality.py -c <run>/checkpoints/latest.ckpt \\
        -o /tmp/dead_logvar_prob_b --assert-dead-logvar
"""
import os
import sys
import json
import time
import pathlib
import tempfile

# bootstrap: make repo root importable regardless of cwd
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import click
import hydra
import numpy as np
import torch
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.multiview_image_dataset import AUX_ACTION_KEY
from diffusion_policy.model.vision.per_view_gaussian import poe_fuse

# reuse rather than re-implement: the checkpoint loader, the draw-range
# override, the slot-to-ring recovery and the ridge are the probe's, pinned by
# its own test suite -- and this script's whole point is that its numbers sit
# next to the probe's.
from probe_relpose import (
    load_policy, _apply_range, _slot_view_indices, ridge_fit, _standardize,
    _with_bias)

# ---------------------------------------------------------------------------
# pure analysis core (numpy/torch-CPU; pinned by tests/test_variance_quality.py)
# ---------------------------------------------------------------------------


def scalar_lv(logvar):
    """``(M, K, D)`` per-pair log-variance -> ``(M, K)`` mean over latent dims.

    The scalar form is what every "how uncertain is this view" question needs;
    averaging over dims is the reduction the probe's own ``lv_view_spread`` is
    built from, and matching it is what makes the R1 cross-check a cross-check.
    """
    return logvar.mean(dim=-1)


def lv_view_spread(logvar, active):
    """Per-state std of the per-view scalar logvar across LIVE views, averaged.

    At exactly 0 the views agree about their own uncertainty, the PoE weights
    are uniform across slots, and cell C is a re-parameterized D.
    """
    s = scalar_lv(logvar).float()
    out = []
    for row, act in zip(s, active):
        if int(act.sum()) < 2:
            continue
        out.append(row[act].std(unbiased=False))
    return float(torch.stack(out).mean()) if out else float('nan')


def _rankdata(x):
    """Average ranks, 1-based -- the only tie handling Spearman needs.

    Implemented rather than imported (the repo's probe tools carry no scipy
    dependency): sort, then average the ranks inside every run of equal values.
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x, y):
    """Rank correlation; NaN when either side is constant (no ranking exists)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float('nan')


def perm_pvalue(x, y, n_perm=10000, seed=0):
    """One-sided upper-tail p for Spearman(x, y): P(rho_perm >= rho_obs).

    One-sided because the registered direction is "higher variance means higher
    error"; a strongly NEGATIVE rho is a finding (routed through the verdict
    function), not evidence for the hypothesis, and must not read as significant
    just because the two-sided test would also be small.
    """
    rho = spearman(x, y)
    if not np.isfinite(rho):
        return rho, float('nan')
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    hits = 0
    for _ in range(n_perm):
        if spearman(x, rng.permutation(y)) >= rho:
            hits += 1
    return rho, float((hits + 1) / (n_perm + 1))


def shuffle_control(x, y, n_shuffles=1000, seed=0):
    """95th percentile of |rho| over row shuffles -- the statistic's own floor.

    ONE side is shuffled: that destroys the pairing while keeping both
    marginals, which is the null the observed rho must beat. (Shuffling both
    sides is the same distribution and would only add noise.)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    vals = [abs(spearman(rng.permutation(x), y)) for _ in range(n_shuffles)]
    return float(np.percentile(vals, 95))


def verdict_corr(rho, p, rho_min=0.2, floor=None, p_max=0.05):
    """The registered decision rule, in one place so the test can mutate it.

    ``p`` may be NaN (a constant input) and ``floor`` None (the n=11 per-view
    correlation uses the permutation p instead of a shuffle floor).
    """
    if not np.isfinite(rho):
        return 'undefined (constant input)'
    if rho <= 0:
        return f'null/finding-against (rho={rho:+.3f} <= 0)'
    ok_floor = floor is None or rho > floor
    ok_p = (not np.isfinite(p)) or p < p_max
    if rho >= rho_min and ok_floor and ok_p:
        return f'supported (rho={rho:+.3f} >= {rho_min})'
    return f'weak (rho={rho:+.3f} below threshold or floor)'


def subset_fuse(mu, logvar, live_rows):
    """Re-fuse a chosen subset with the PRODUCTION ``poe_fuse`` (offline).

    ``live_rows`` is an ``(M, K)`` bool mask; returns ``(mu_g, logvar_g)``. This
    is the same code path the model runs at inference, so R5 measures the fusion
    the cells actually deploy rather than a re-derivation of it.
    """
    if live_rows.dtype != torch.bool:
        raise ValueError(f'live_rows must be bool, got {live_rows.dtype}')
    if not bool(live_rows.any(dim=1).all()):
        raise ValueError('every row needs at least one live view')
    return poe_fuse(mu, logvar, live_rows)


def n_sensitivity(mu, logvar, n_levels=None, n_draws=100, seed=0):
    """``lvg`` level and ``mu_g`` drift vs N, over random view subsets.

    The drift reference is the FULL-draw fused mean (every slot live), which is
    what the trained model deploys; the drift is relative
    (``||mu_g(sub) - mu_g(full)|| / ||mu_g(full)||``) so it is scale-free and
    comparable across cells.
    """
    S, K = mu.shape[0], mu.shape[1]
    n_levels = list(range(1, K + 1)) if n_levels is None else list(n_levels)
    full = torch.ones(S, K, dtype=torch.bool, device=mu.device)
    mu_full, _ = subset_fuse(mu, logvar, full)
    ref = mu_full.norm(dim=-1).clamp_min(1e-6)
    rng = np.random.default_rng(seed)
    rows = torch.arange(S).unsqueeze(1)
    out = {}
    for n in n_levels:
        lvgs, drifts = [], []
        for _ in range(n_draws):
            pick = np.argsort(rng.random((S, K)), axis=1)[:, :n]
            mask = torch.zeros(S, K, dtype=torch.bool, device=mu.device)
            mask[rows, torch.from_numpy(pick)] = True
            mu_g, lv_g = subset_fuse(mu, logvar, mask)
            drifts.append(float(((mu_g - mu_full).norm(dim=-1) / ref).mean()))
            lvgs.append(float(lv_g.mean()))
        out[str(n)] = dict(lvg_mean=float(np.mean(lvgs)),
                           drift_mean=float(np.mean(drifts)),
                           n_draws=int(n_draws))
    return out


def leave_one_out(mu, logvar):
    """Drop one ring view at a time: how far does the fused mean move?

    One row per removed slot (slot order), so the registered ``az_0 > +-90``
    ordering claim can be read off directly once the caller annotates slots with
    their ring azimuths.
    """
    S, K = mu.shape[0], mu.shape[1]
    full = torch.ones(S, K, dtype=torch.bool, device=mu.device)
    mu_full, lv_full = subset_fuse(mu, logvar, full)
    ref = mu_full.norm(dim=-1).clamp_min(1e-6)
    rows = []
    for k in range(K):
        mask = full.clone()
        mask[:, k] = False
        mu_g, lv_g = subset_fuse(mu, logvar, mask)
        rows.append(dict(slot=k,
                         drop_drift=float(((mu_g - mu_full).norm(dim=-1) / ref).mean()),
                         drop_lvg_shift=float((lv_g - lv_full).mean())))
    return rows


def overconfidence(mu, logvar, active):
    """PoE-implied sigma vs the between-view spread of the means, per (state, dim).

    Three dimensionless numbers, all reported (none is a verdict):
      * ``frac_sigma_below_between`` -- share of (state, dim) cells whose fused
        sigma is below the live views' own disagreement about that dim;
      * ``w_max`` -- the mean largest precision share, ``max_v p_v / sum_v p_v``
        (uniform over K views = 1/K, so anything much above that says the fusion
        is not an average wearing a probability);
      * ``double_count_factor`` = ``sqrt(sum_v p_v / max_v p_v)`` -- 1.0 means
        one expert dominates (no double counting), ``sqrt(K)`` means every view
        was counted as if independent, which is PLAN_G section 8's risk 2 in a
        single number.

    What this CANNOT say: "the variance is miscalibrated". There is no ground
    truth for the latent state; the between-view spread of the means contains
    each view's systematic view-dependence as well as its uncertainty, and a
    product over correlated experts is expected to shrink. These are lower
    bounds on overconfidence, and only R3/R4's error correlations carry
    evidence about whether the variance tracks anything real.
    """
    _, lv_g = subset_fuse(mu, logvar, active)
    m = mu.float()
    a = active.unsqueeze(-1)
    count = a.sum(dim=1).clamp_min(1)                      # (M, 1)
    mean = (m * a).sum(1) / count
    var = (((m - mean.unsqueeze(1)) ** 2) * a).sum(1) / count
    between = var.sqrt()
    ratio = torch.exp(lv_g.float() / 2.0) / between.clamp_min(1e-12)
    p = torch.exp(-logvar.float())
    w = torch.where(a, p, torch.zeros_like(p))
    total = w.sum(1).clamp_min(1e-12)
    top = w.amax(dim=1)
    return dict(
        frac_sigma_below_between=float((ratio < 1.0).float().mean()),
        median_ratio=float(ratio.median()),
        p10_ratio=float(ratio.flatten().quantile(0.10)),
        p90_ratio=float(ratio.flatten().quantile(0.90)),
        w_max=float((top / total).mean()),
        w_max_uniform=1.0 / float(mu.shape[1]),
        double_count_factor=float((total / top.clamp_min(1e-12)).sqrt().mean()),
    )


def ridge_residuals(X, Y, lam=1e-4, train_frac=0.8):
    """Fit ridge on a prefix split, return per-row residual norms on the rest.

    The probe's ``ridge_fit`` / ``_standardize`` / ``_with_bias``, used here to
    give a per-(frame, view) STATE error where training has no such target: the
    residual of a decode from the latent to the dataset's own low-dim state.
    Returns ``(residuals, n_train)`` in the caller's units.
    """
    n = X.shape[0]
    n_tr = int(n * train_frac)
    X_tr, X_te = _standardize(X[:n_tr].float(), X[n_tr:].float())
    w = ridge_fit(_with_bias(X_tr), Y[:n_tr].float(), lam=lam)
    pred = _with_bias(X_te) @ w
    return (pred - Y[n_tr:].float()).norm(dim=-1), n_tr


def per_view_table(ring_of_slot, active, lv_scalar, err_aux, err_state, azimuths):
    """Per ring view: mean scalar logvar and mean error, both readings.

    Keyed by RING index (not slot) -- slots are containers filled by the draw,
    so the ring index is the only view identity stable across cells.

    ``err_state`` carries NaN on the rows the ridge decoder was FIT on (a
    decoded error only exists out-of-sample), so its mean is taken over the
    finite rows and its count reported separately -- a NaN-blind mean would read
    NaN for every view while the aux column looked fine.
    """
    rows = {}
    for slot, ring in ring_of_slot.items():
        m = active[:, slot]
        if not bool(m.any()):
            continue
        r = rows.setdefault(int(ring), dict(n=0, n_state=0, lv=0.0, e_aux=0.0,
                                            e_state=0.0))
        r['n'] += int(m.sum())
        r['lv'] += float(lv_scalar[m, slot].sum())
        r['e_aux'] += float(err_aux[m, slot].sum())
        es = err_state[m, slot]
        r['n_state'] += int(torch.isfinite(es).sum())
        r['e_state'] += float(torch.nan_to_num(es).sum())
    out = []
    for ring in sorted(rows):
        r = rows[ring]
        n = max(r['n'], 1)
        ns = max(r['n_state'], 1)
        out.append(dict(ring=ring, azimuth_deg=float(azimuths[ring]),
                        n_pairs=int(r['n']), n_state_pairs=int(r['n_state']),
                        mean_lv=r['lv'] / n, mean_err_aux=r['e_aux'] / n,
                        mean_err_state_cm=r['e_state'] / ns))
    return out


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def _settings(policy):
    """Which fusion the checkpoint deploys, and its Gaussian clamp bounds."""
    enc = policy.obs_encoder
    return dict(
        fusion_mode=str(getattr(enc, 'fusion_mode', None)),
        emits_mu=bool(getattr(enc, 'emits_mu', False)),
        logvar_min=float(getattr(enc, 'logvar_min', float('nan'))),
        logvar_max=float(getattr(enc, 'logvar_max', float('nan'))),
        aux_input=str(getattr(policy, 'aux_input', None)),
        aux_key=str(getattr(policy, 'aux_key', None)),
    )


def collect(policy, cfg, device, n_states):
    """One forward per state at a FULL 11-view draw, with both error readings.

    The full draw is forced for one reason: the per-view statistics need every
    ring view observed at every state, or each view's mean logvar/error would
    average over a different state subset and the R2 table would compare
    different things. ``z_v`` is N-agnostic by construction (the probe checks it
    on every run), so forcing the draw does not change what the encoder computes
    per view -- it only changes what is served alongside.
    """
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    cell_range = tuple(int(x) for x in dataset.view_count_range)
    _apply_range(dataset, (int(dataset.n_slots), int(dataset.n_slots)))
    To = policy.n_obs_steps
    K = int(dataset.n_slots)
    eef_keys = [k for k in policy.obs_encoder.low_dim_keys if k.endswith('eef_pos')]
    if len(eef_keys) != 1:
        raise SystemExit(f'expected exactly one *eef_pos low-dim key, got {eef_keys}')
    eef_key = eef_keys[0]

    n = min(int(n_states), len(dataset))
    MU, LV, ACT, ERR_AUX, EEF, ZG, LVG, AUX_MEAN = [], [], [], [], [], [], [], []
    ring_of_slot = None
    for i in range(n):
        s = dataset[i]
        obs = dict_apply(s['obs'], lambda x: x.unsqueeze(0).to(device))
        nobs = policy.normalizer.normalize(obs)
        this_nobs = dict_apply(nobs, lambda x: x[:, :To].reshape(-1, *x.shape[2:]))
        with torch.no_grad():
            enc = policy.obs_encoder.forward_full(this_nobs)
            mu = enc[policy.aux_key].float()                 # (To, K, D)
            lv = enc['logvar_views'].float()                 # (To, K, D)
            active = enc['view_active'].bool()               # (To, K)
            # the aux head's own per-pair error, the reduction the policy uses.
            # s[AUX_ACTION_KEY] is already (To, K, aux_n_steps * ACTION_DIM) --
            # exactly the shape `mu[active]` is indexed against -- and the
            # normalizer applies over the last dim, so no reshape is needed.
            # FIXED 2026-10-16: this line read
            #   `.unsqueeze(0).to(device).reshape(-1, K, -1)`
            # which can never work -- `reshape` infers at most ONE dimension, so
            # the second -1 raised `RuntimeError: only one dimension can be
            # inferred` the moment stage 8 first ran on a real checkpoint. The
            # CPU test suite did not catch it: tests/test_variance_quality.py
            # covers the dead-logvar path and the reductions, not this collect()
            # line against a live PoE checkpoint.
            aux_tgt = policy.normalizer[AUX_ACTION_KEY].normalize(
                s[AUX_ACTION_KEY].to(device))
            pair = F.mse_loss(policy.aux_head(mu[active]), aux_tgt[active],
                              reduction='none').mean(dim=-1)
            err_aux = torch.full(active.shape, float('nan'), device=device)
            err_aux[active] = pair
        if ring_of_slot is None:
            ring_of_slot = dict(_slot_view_indices(dataset, s['obs'], active[0]))
        # the per-sample mean over live pairs -- the quantity the policy's own
        # `last_aux_action_loss` reports at B=1, kept for the mirror gate
        AUX_MEAN.append(float(pair.mean()))
        MU.append(mu.cpu()); LV.append(lv.cpu()); ACT.append(active.cpu())
        ERR_AUX.append(err_aux.cpu())
        EEF.append(s['obs'][eef_key].float())                # (To, 3)
        ZG.append(enc['z_global'].float().cpu())
        LVG.append(None if enc['z_global_logvar'] is None
                   else enc['z_global_logvar'].float().cpu())

    stats = dict(
        n_states=int(n), n_frames=int(MU[0].shape[0]) * len(MU), K=K,
        dim=int(MU[0].shape[-1]),
        mean_active=float(torch.cat(ACT, 0).sum(-1).float().mean()),
        n_dataset=len(dataset), effective_range=[K, K],
        cell_view_count_range=list(cell_range),
        ring_of_slot={int(k): int(v) for k, v in sorted(ring_of_slot.items())},
        state_idx=list(range(n)))
    return dict(mu=torch.cat(MU, 0), lv=torch.cat(LV, 0), active=torch.cat(ACT, 0),
                err_aux=torch.cat(ERR_AUX, 0), eef=torch.cat(EEF, 0),
                zg=torch.cat(ZG, 0),
                lvg=None if LVG[0] is None else torch.cat(LVG, 0),
                aux_mean=torch.tensor(AUX_MEAN),
                dataset=dataset, stats=stats)


def diffusion_losses(policy, dataset, device, n_states, seed, aux_mean=None):
    """Per-sample diffusion loss, one B=1 forward at a time.

    The policy reduces the diffusion term over the batch before storing it, so a
    per-sample reading cannot come out of a batch call; the alternative --
    re-implementing the loss body -- would be a second copy of the training
    objective that could drift from it. B=1 calls are the honest route: each
    sample gets its own noise/timestep draw, exactly as in training, and the
    draw is seeded so the reading is reproducible.

    The same calls also carry the AUX-MIRROR GATE: this script re-derives the
    per-pair aux error from the policy's own normalizer, encoder and head, and
    that re-derivation is the one place a silent divergence from the training
    reduction could hide. ``policy.last_aux_action_loss`` -- the policy's own
    unweighted per-sample aux term, set during the same forward -- is compared
    against this script's per-sample mean over live pairs; returns the worst
    relative difference so the caller can gate on it.
    """
    torch.manual_seed(int(seed))
    out = []
    worst = 0.0
    for i in range(min(int(n_states), len(dataset))):
        s = dataset[i]
        if AUX_ACTION_KEY not in s:
            raise SystemExit(
                f'the dataset emits no {AUX_ACTION_KEY}: this checkpoint has an aux '
                f'head, so compute_loss cannot be evaluated and the R4b reading '
                f'would be meaningless')
        # the aux target MUST ride along: this policy's compute_loss calls its
        # own aux_loss, which reads batch[AUX_ACTION_KEY] and would KeyError
        # without it (the training dataloader always supplies it)
        batch = dict_apply({'obs': s['obs'], 'action': s['action'],
                            AUX_ACTION_KEY: s[AUX_ACTION_KEY]},
                           lambda x: x.unsqueeze(0).to(device))
        with torch.no_grad():
            policy.compute_loss(batch)
        out.append(float(policy.last_diff_loss))
        if aux_mean is not None:
            ref = float(aux_mean[i])
            got = float(policy.last_aux_action_loss)
            if ref > 0:
                worst = max(worst, abs(got - ref) / ref)
    return torch.tensor(out), worst


def register_gates(policy):
    """Refuse a checkpoint whose uncertainty cannot be read -- with the reason."""
    enc = policy.obs_encoder
    if not getattr(enc, 'emits_mu', False):
        raise SystemExit(
            f'{type(enc).__name__} emits no per-view Gaussians: nothing to read')
    mode = str(getattr(enc, 'fusion_mode', ''))
    if mode != 'poe':
        raise SystemExit(
            f'fusion_mode={mode!r}: the full variance read is defined for the PoE '
            f"cell (B/D's logvar branch has no gradient path). Use "
            f'--assert-dead-logvar for cells B and D.')


# ---------------------------------------------------------------------------
# the dead-branch path (cells B and D)
# ---------------------------------------------------------------------------


def assert_dead_logvar(policy, cfg, device, n_states=4):
    """Verify the premise that makes B/D's variance unreadable: it is all zeros.

    Exact equality, not a tolerance: the branch is zero-initialized and nothing
    in these cells' losses depends on it, so any nonzero value means a gradient
    path exists after all and the round's "do not read their uncertainty"
    caveat -- already written into PROGRESS.md -- would be wrong.
    """
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    To = policy.n_obs_steps
    worst = 0.0
    fused_none = True
    for i in range(min(int(n_states), len(dataset))):
        s = dataset[i]
        obs = dict_apply(s['obs'], lambda x: x.unsqueeze(0).to(device))
        nobs = policy.normalizer.normalize(obs)
        this = dict_apply(nobs, lambda x: x[:, :To].reshape(-1, *x.shape[2:]))
        with torch.no_grad():
            enc = policy.obs_encoder.forward_full(this)
        lv = enc['logvar_views']
        live = enc['view_active'].bool().unsqueeze(-1).expand_as(lv)
        worst = max(worst, float(lv[live].abs().max()))
        fused_none = fused_none and enc['z_global_logvar'] is None
    return dict(max_abs_logvar_live=worst, fused_logvar_absent=bool(fused_none),
                dead=bool(worst == 0.0 and fused_none))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def build_readings(c, log_scalars):
    """Assemble R1..R6 from the collected tensors -- the script's whole output."""
    mu, lv, active = c['mu'], c['lv'], c['active']
    lv_s = scalar_lv(lv)
    eef = c['eef']
    D = mu.shape[-1]

    # ---- error readings -------------------------------------------------------
    # The ridge is fit on ALL live pairs pooled: the question is how well a
    # view's own estimate predicts the state, not how well a per-view model
    # fits, and pooling is what makes the per-view ranking fair.
    #
    # The state target is per FRAME, the latents are per (frame, view) PAIR --
    # so the target has to be broadcast across views before the pair mask is
    # applied. Indexing the (S, 3) target with the (S*K,) pair mask raises; a
    # version that quietly reshaped instead would mis-pair states with views.
    #
    # The residuals exist only on the rows the decoder did NOT see, so every
    # state-error reading below runs on that same held-out set (``test_idx``);
    # the aux error is the head's own out-of-sample-free training objective and
    # covers all live pairs. The two counts are reported separately for exactly
    # that reason.
    sel = active.reshape(-1)                                  # (S*K,) bool, cpu
    idx = sel.nonzero(as_tuple=True)[0]                       # (M,) live pairs
    X = mu.reshape(-1, D)[idx]
    Y = eef.unsqueeze(1).expand(-1, active.shape[1], -1).reshape(
        -1, eef.shape[-1])[idx]
    resid_cm, n_tr = ridge_residuals(X, Y)
    resid_cm = resid_cm * 100.0                               # metres -> cm
    test_idx = idx[n_tr:]
    err_state = torch.full((active.numel(),), float('nan'))
    err_state[test_idx] = resid_cm
    err_state = err_state.reshape(active.shape)

    lv_flat = lv_s.reshape(-1)[idx].numpy()
    ea_flat = c['err_aux'].reshape(-1)[idx].numpy()
    lv_test = lv_s.reshape(-1)[test_idx].numpy()
    es_test = resid_cm.numpy()

    readings = {}
    # ---- R1 -------------------------------------------------------------------
    r1 = lv_view_spread(lv, active)
    r1_ref = log_scalars.get('lv_view_spread')
    if r1 == 0.0:
        r1_verdict = 'FAILED (zero spread -- C would be a re-parameterized D)'
    elif r1_ref is not None and not (0.5 * r1_ref <= r1 <= 2.0 * r1_ref):
        r1_verdict = f'informative but off the log cross-check (log e200 {r1_ref})'
    else:
        r1_verdict = 'informative'
    readings['r1_lv_view_spread'] = dict(
        value=r1, log_e200_value=r1_ref,
        threshold='> 0 and within 2x of the log cross-check',
        verdict=r1_verdict)

    # ---- R2 -------------------------------------------------------------------
    az = c['azimuths']
    table = per_view_table(c['stats']['ring_of_slot'], active, lv_s,
                           c['err_aux'], err_state, az)
    rho_a, p_a = perm_pvalue([r['mean_lv'] for r in table],
                             [r['mean_err_aux'] for r in table],
                             n_perm=c['config']['n_perm'], seed=c['config']['seed'])
    rho_s, p_s = perm_pvalue([r['mean_lv'] for r in table],
                             [r['mean_err_state_cm'] for r in table],
                             n_perm=c['config']['n_perm'], seed=c['config']['seed'])
    zero_row = next((r for r in table if abs(r['azimuth_deg']) < 1e-6), None)
    edges = [r for r in table if abs(r['azimuth_deg']) >= 75.0]
    readings['r2_view_quality'] = dict(
        per_view=table,
        rho_lv_vs_err_aux=dict(rho=rho_a, p=p_a,
                               verdict=verdict_corr(rho_a, p_a, rho_min=0.5)),
        rho_lv_vs_err_state=dict(rho=rho_s, p=p_s,
                                 verdict=verdict_corr(rho_s, p_s, rho_min=0.5)),
        contrast_az0_vs_edges=dict(
            az0_lv=None if zero_row is None else zero_row['mean_lv'],
            edge_lv_mean=None if not edges else float(np.mean(
                [r['mean_lv'] for r in edges])),
            az0_err_aux=None if zero_row is None else zero_row['mean_err_aux'],
            edge_err_aux_mean=None if not edges else float(np.mean(
                [r['mean_err_aux'] for r in edges]))),
        verdict='supported only if BOTH rho >= 0.5 with one-sided p < 0.05')

    # ---- R3 -------------------------------------------------------------------
    def _corr_block(x, y):
        rho, p = perm_pvalue(x, y, n_perm=c['config']['n_perm'],
                             seed=c['config']['seed'])
        floor = shuffle_control(x, y, n_shuffles=c['config']['n_shuffles'],
                                seed=c['config']['seed'])
        return dict(rho=rho, p=p, shuffle_p95=floor,
                    verdict=verdict_corr(rho, p, rho_min=0.2, floor=floor))
    readings['r3_pair_error'] = dict(
        n_pairs=int(idx.numel()), n_test_pairs=int(test_idx.numel()),
        n_train_pairs=int(n_tr),
        rho_lv_vs_err_aux=_corr_block(lv_flat, ea_flat),
        rho_lv_vs_err_state=_corr_block(lv_test, es_test))

    # ---- R4 -------------------------------------------------------------------
    if c['lvg'] is None:
        readings['r4_fused_vs_error'] = dict(
            verdict='NOT MEASURABLE: this checkpoint has no fused logvar (not PoE)')
    else:
        # the FUSED latent is a per-FRAME quantity while the pair index space is
        # frame x slot -- so it broadcasts across a frame's live views rather
        # than being indexed by the pair. Indexing it with the pair index raises
        # for K > 1; a version that reshaped instead would mis-align rows.
        K = mu.shape[1]
        lvg_test = c['lvg'][test_idx // K].mean(dim=-1)             # (M_test,)
        zg_sel = c['zg'][idx // K]
        zg_resid, n_tr_zg = ridge_residuals(zg_sel, Y)
        if n_tr_zg != n_tr:
            raise RuntimeError(
                f'the fused decode split ({n_tr_zg}) disagrees with the per-view one '
                f'({n_tr}); the two readings would not be on the same rows')
        block = dict(
            n_test_pairs=int(test_idx.numel()),
            rho_lvg_vs_state_residual=_corr_block(
                lvg_test.numpy(), (zg_resid * 100.0).numpy()))
        diff = c.get('diff_loss')
        if diff is not None:
            S, To = len(diff), lv.shape[0] // len(diff)
            lvg_by_state = c['lvg'].reshape(S, To, D).mean(dim=(1, 2)).numpy()
            block['rho_lvg_vs_diffusion_loss'] = dict(
                **_corr_block(lvg_by_state, diff.numpy()),
                caveat='the diffusion loss reflects the UNet fit too (pre-stated)')
        readings['r4_fused_vs_error'] = block

    # ---- R5 -------------------------------------------------------------------
    cfg_r5 = c['config']
    curve = n_sensitivity(mu, lv, n_draws=cfg_r5['n_draws_per_level'],
                          seed=cfg_r5['seed'])
    loo = leave_one_out(mu, lv)
    order = sorted(curve, key=int)
    lvg_seq = [curve[n]['lvg_mean'] for n in order]
    drift_seq = [curve[n]['drift_mean'] for n in order]
    mono_lvg = all(b <= a + 1e-6 for a, b in zip(lvg_seq, lvg_seq[1:]))
    mono_drift = all(b <= a + 1e-6 for a, b in zip(drift_seq, drift_seq[1:]))
    ring_of_slot = c['stats']['ring_of_slot']
    slot_of_ring = {v: k for k, v in ring_of_slot.items()}
    az0_ring = next((r['ring'] for r in table if abs(r['azimuth_deg']) < 1e-6), None)
    edge_rings = [r['ring'] for r in table if abs(r['azimuth_deg']) >= 75.0]
    az0_drift = (loo[slot_of_ring[az0_ring]]['drop_drift']
                 if az0_ring in slot_of_ring else None)
    edge_drift = max((loo[slot_of_ring[r]]['drop_drift'] for r in edge_rings
                      if r in slot_of_ring), default=None)
    readings['r5_n_sensitivity'] = dict(
        lvg_by_n={n: curve[n]['lvg_mean'] for n in order},
        drift_by_n={n: curve[n]['drift_mean'] for n in order},
        leave_one_out=[dict(ring=ring_of_slot[r['slot']],
                            azimuth_deg=az[ring_of_slot[r['slot']]], **r)
                       for r in loo],
        lvg_monotone_nonincreasing=bool(mono_lvg),
        drift_monotone_nonincreasing=bool(mono_drift),
        az0_removal_drift=az0_drift,
        worst_edge_removal_drift=edge_drift,
        registered_ordering_holds=(None if az0_drift is None or edge_drift is None
                                   else bool(az0_drift > edge_drift)),
        verdict='registered expectation: lvg falls with N, drift decays, '
                'az_0 removal moves the fusion more than any edge view')

    # ---- R6 -------------------------------------------------------------------
    readings['r6_overconfidence'] = overconfidence(mu, lv, active)
    return readings


def summary_markdown(out):
    """The human-readable digest -- what the driver prints and PROGRESS quotes."""
    r = out['readings']
    L = [f"# variance quality -- {out['checkpoint']}", '',
         f"fusion={out['settings']['fusion_mode']} aux_input="
         f"{out['settings']['aux_input']} n_states={out['config']['n_states']} "
         f"seed={out['config']['seed']} (full 11-view draw)", '']
    R1 = r['r1_lv_view_spread']
    L += [f"## R1 view spread -- {R1['value']:.4f} "
          f"(log e200 {R1['log_e200_value']}) -- {R1['verdict']}", '',]

    L += ['## R2 per-view (ring) -- mean lv vs mean error',
          '| ring | az | n | mean lv | err aux | err state (cm) |',
          '|---|---|---|---|---|---|']
    for row in r['r2_view_quality']['per_view']:
        L.append(f"| {row['ring']} | {row['azimuth_deg']:+.1f} | {row['n_pairs']} | "
                 f"{row['mean_lv']:+.4f} | {row['mean_err_aux']:.5f} | "
                 f"{row['mean_err_state_cm']:.4f} |")
    q = r['r2_view_quality']
    for k in ('rho_lv_vs_err_aux', 'rho_lv_vs_err_state'):
        L.append(f"- {k}: rho={q[k]['rho']:+.3f} p={q[k]['p']:.4f} -- {q[k]['verdict']}")
    L += ['', f"## R3 pair-level ({r['r3_pair_error']['n_pairs']} live pairs; "
              f"the state reading on {r['r3_pair_error']['n_test_pairs']} held-out rows)"]
    for k in ('rho_lv_vs_err_aux', 'rho_lv_vs_err_state'):
        e = r['r3_pair_error'][k]
        L.append(f"- {k}: rho={e['rho']:+.3f} p={e['p']:.4f} "
                 f"shuffle95={e['shuffle_p95']:.3f} -- {e['verdict']}")
    L += ['', '## R4 fused']
    for k, e in r['r4_fused_vs_error'].items():
        if isinstance(e, dict):
            L.append(f"- {k}: rho={e['rho']:+.3f} p={e['p']:.4f} "
                     f"shuffle95={e['shuffle_p95']:.3f} -- {e['verdict']}")
        else:
            L.append(f"- {k}: {e}")
    r5 = r['r5_n_sensitivity']
    order = sorted(r5['lvg_by_n'], key=int)
    L += ['', '## R5 N-sensitivity',
          '| N | ' + ' | '.join(order) + ' |',
          '|---|' + '---|' * len(order),
          '| lvg | ' + ' | '.join(f"{r5['lvg_by_n'][n]:.2f}" for n in order) + ' |',
          '| drift | ' + ' | '.join(f"{r5['drift_by_n'][n]:.4f}" for n in order) + ' |',
          f"lvg non-increasing: {r5['lvg_monotone_nonincreasing']}  "
          f"drift non-increasing: {r5['drift_monotone_nonincreasing']}",
          f"az_0 removal drift {r5['az0_removal_drift']} vs worst edge "
          f"{r5['worst_edge_removal_drift']} -> registered ordering "
          f"{r5['registered_ordering_holds']}", '']
    o = r['r6_overconfidence']
    L += ['## R6 overconfidence',
          f"frac(sigma_g < between-view sigma) = {o['frac_sigma_below_between']:.3f}; "
          f"median ratio {o['median_ratio']:.4f} "
          f"(p10 {o['p10_ratio']:.4f}, p90 {o['p90_ratio']:.4f})",
          f"w_max = {o['w_max']:.4f} (uniform {o['w_max_uniform']:.4f}); "
          f"double-count factor {o['double_count_factor']:.4f} "
          f"(1 = one expert dominates, sqrt(K) = all counted independent)"]
    return '\n'.join(L) + '\n'


def log_scalars_at_epoch(log_path, epoch=200):
    """The run log's e200 latent-probe scalars, for the cross-checks."""
    if not log_path or not os.path.exists(log_path):
        return {}
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    hits = [r for r in rows
            if r.get('epoch') == epoch and 'latent_probe/zv_pair_ratio' in r]
    if not hits:
        return {}
    r = hits[-1]
    return {k.split('/', 1)[1]: v for k, v in r.items()
            if k.startswith('latent_probe/')}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-l', '--log_path', default=None,
              help='the run log, for the e200 cross-checks (optional)')
@click.option('--n-states', default=256, help='dataset indices (each = To frames)')
@click.option('--n-shuffles', default=1000, help='shuffle-control draws (R3/R4 floor)')
@click.option('--n-perm', default=10000, help='permutation draws (p-values)')
@click.option('--n-draws-per-level', default=100, help='R5 draws per N level')
@click.option('--seed', default=0)
@click.option('--num-threads', default=4)
@click.option('--no-diffusion-loss', is_flag=True,
              help='skip R4b (n_states B=1 UNet forward passes)')
@click.option('--assert-dead-logvar', 'dead_logvar', is_flag=True,
              help='cell B/D path: verify the logvar branch is exactly zero, then stop')
def main(checkpoint, output_dir, device, log_path, n_states, n_shuffles, n_perm,
         n_draws_per_level, seed, num_threads, no_diffusion_loss, dead_logvar):
    torch.set_num_threads(int(num_threads))
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    t_start = time.time()

    policy, cfg = load_policy(checkpoint, device)
    settings = _settings(policy)
    print(f'checkpoint {checkpoint}\n  fusion_mode={settings["fusion_mode"]} '
          f'aux_input={settings["aux_input"]} emits_mu={settings["emits_mu"]}')

    if dead_logvar:
        res = assert_dead_logvar(policy, cfg, device, n_states=min(4, n_states))
        with open(os.path.join(output_dir, 'dead_logvar.json'), 'w') as f:
            json.dump(dict(schema=1, tool='analyze_variance_quality.py',
                           mode='assert_dead_logvar', checkpoint=checkpoint,
                           settings=settings, **res), f, indent=2)
        print(f'max |logvar| over live slots = {res["max_abs_logvar_live"]}  '
              f'fused logvar absent = {res["fused_logvar_absent"]}  -> '
              f'{"DEAD (as registered)" if res["dead"] else "LIVE (premise wrong!)"}')
        if not res['dead']:
            raise SystemExit(
                'the logvar branch is NOT dead -- the "do not read B/D variance" '
                'premise from data/prob_run.sh is wrong; report before proceeding')
        print(f'wrote {os.path.join(output_dir, "dead_logvar.json")}')
        return

    register_gates(policy)

    t0 = time.time()
    c = collect(policy, cfg, device, n_states)
    print(f'collected {c["stats"]["n_frames"]} frames x {c["stats"]["K"]} live views '
          f'({time.time() - t0:.0f}s)')
    assert c['stats']['mean_active'] == c['stats']['K'], (
        'the full-draw override did not take: mean_active '
        f'{c["stats"]["mean_active"]} != K {c["stats"]["K"]}')

    # view azimuths keyed by RING index (cam_table is the ring's own table)
    c['azimuths'] = {i: float(a)
                     for i, a in enumerate(c['dataset'].camera_params['azimuth_deg'])}
    if c['lvg'] is None:
        raise SystemExit('no fused logvar from the encoder (not a PoE checkpoint)')
    c['config'] = dict(n_perm=int(n_perm), n_shuffles=int(n_shuffles),
                       n_draws_per_level=int(n_draws_per_level), seed=int(seed))
    gates = {}
    if not no_diffusion_loss:
        t0 = time.time()
        c['diff_loss'], aux_gate = diffusion_losses(
            policy, c['dataset'], device, c['stats']['n_states'], seed,
            aux_mean=c['aux_mean'])
        # registered: the script's re-derived per-pair aux error must match the
        # policy's own per-sample aux term to 1e-3 relative, else the R2/R3 aux
        # columns measure a drifted copy
        gates['aux_mirror'] = dict(
            max_rel_diff=float(aux_gate), threshold=1e-3,
            pass_=bool(aux_gate <= 1e-3),
            note='script mean-over-live-pairs vs policy.last_aux_action_loss at B=1')
        print(f'per-sample diffusion losses done ({time.time() - t0:.0f}s); '
              f'aux-mirror max rel diff {aux_gate:.2e}')
        if not gates['aux_mirror']['pass_']:
            print('WARNING: aux-mirror gate FAILED -- the aux error readings below '
                  'are not the policy\'s own; do not quote them')

    logs = log_scalars_at_epoch(log_path, epoch=200)
    readings = build_readings(c, logs)

    out = dict(
        schema=1, tool='analyze_variance_quality.py', checkpoint=checkpoint,
        checkpoint_bytes=os.path.getsize(checkpoint),
        checkpoint_mtime=time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(os.path.getmtime(checkpoint))),
        created_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
        settings=settings,
        config=dict(n_states=int(n_states), seed=int(seed), n_shuffles=int(n_shuffles),
                    n_perm=int(n_perm), n_draws_per_level=int(n_draws_per_level),
                    diffusion_loss=not no_diffusion_loss),
        stats={k: v for k, v in c['stats'].items() if k != 'state_idx'},
        state_idx=c['stats']['state_idx'],
        gates=gates,
        log_e200=logs, readings=readings,
        cost=dict(wall_s=round(time.time() - t_start, 1)))
    dest = os.path.join(output_dir, 'variance_quality.json')
    with open(dest, 'w') as f:
        json.dump(out, f, indent=2)
    md = summary_markdown(out)
    with open(os.path.join(output_dir, 'summary.md'), 'w') as f:
        f.write(md)
    print('\n' + md)
    print(f'wrote {dest}')


if __name__ == '__main__':
    main()
