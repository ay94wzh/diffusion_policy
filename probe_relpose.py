"""Probe: what geometry is already in `z_v`, on FROZEN checkpoints, with no training.

Why this exists
---------------
M4 ended as a mechanically-live, behaviourally-null A/B (`aux_loss` fell 52x, the head
reached 6.4% of a mean-collapsing floor, and the rollout number did not move), and the
explanation that survived was **the information was already in `z_v`**. So before adding
a relational head to `z_v`, measure what is already decodable from it.

Four measurements, all from one forward pass per sample:

  1. **absolute camera pose from one view's `z_v`** -- (pos, rot6d) of that view's own
     camera. Trivially decodable when `use_plucker=True` (the ray map is an *input*), so
     the informative comparison is against `use_plucker=False`.
  2. **camera-frame EE position from `z_v`** -- `R_c^T (p_eef - cam_pos)`, the quantity
     M4's aux head predicts actions from.
  3. **relative pose from a PAIR of views' `z_v`** -- the relational target, derived from
     the per-slot camera vectors already in the batch (no dataset change needed).
  4. **how stable `z_g` is across different view subsets of the SAME state**, against how
     much `z_v` moves across views. That is the proposal's core claim (per-view latent is
     view-aware, fused latent is view-invariant), measured rather than asserted.

Each target is fit two ways -- a closed-form ridge probe (linear decodability) and
optionally a small MLP (nonlinear decodability) -- against two baselines that make the
numbers meaningful: the **mean predictor** (the collapse floor a head must beat) and a
**shuffled-target control** (must fail, or the probe is reading something other than
geometry).

Running it on `m3on` and on `m3off` answers a question this project has never asked:
*does the Pluecker path change how much geometry is decodable from `z_v`?*

Geometry helpers are numpy, in the dtype convention of the dataset module's own helpers
(`quat_wxyz_to_mat` / `rot6d_to_mat`): float64 in, and the caller converts to torch. The
rot6d path goes through the dataset's pinned `rot6d_to_mat`, never a 6x6 shortcut.

Usage (on the box -- needs the multi-view zarr and the checkpoint):

    python probe_relpose.py \\
        -c data/outputs/run_square_m3on_s42_200ep/checkpoints/latest.ckpt \\
        -o data/probe_relpose_square_m3on -d cuda:0

    # smoke, ~1 minute
    python probe_relpose.py -c <ckpt> -o /tmp/probe_smoke --n-samples 64 --stability-states 8
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
import dill
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.multiview_image_dataset import (
    quat_wxyz_to_mat, rot6d_to_mat)

# ---------------------------------------------------------------------------
# geometry helpers (pure, CPU-testable -- see tests/test_relpose_probe.py)
# ---------------------------------------------------------------------------


def relative_pose_from_cams(cam_i, cam_j):
    """Relative pose carrying a point from view i's frame into view j's frame.

    `cam` is the per-slot 10-vector ``[pos(3), quat_wxyz(4), fovy(1), h(1), w(1)]``.
    With ``world = R_c x_c + p_c`` (the convention the dataset's helpers use), the
    transform is::

        R = R_j^T R_i        t = R_j^T (p_i - p_j)

    so that ``x_j = R x_i + t``. Returns ``(R (3,3), t (3,))`` as float64 numpy.

    The direction is "where view i is, expressed in view j's frame" -- the stereo-baseline
    form, and it is NOT symmetric: the inverse is not the negated translation. A probe fit
    in one direction cannot be read as the other, which is why the test pins the convention
    against a brute-force composition and a mutation that swaps it.
    """
    cam_i = np.asarray(cam_i, dtype=np.float64)
    cam_j = np.asarray(cam_j, dtype=np.float64)
    R_i = quat_wxyz_to_mat(cam_i[3:7])
    R_j = quat_wxyz_to_mat(cam_j[3:7])
    R = R_j.T @ R_i
    t = R_j.T @ (cam_i[:3] - cam_j[:3])
    return R, t


def rot6d_from_mat(R):
    """Rows 0 and 1 of R, concatenated -- the 6d encoding pytorch3d uses."""
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


def mat_from_rot6d(d6):
    """``(..., 6)`` rot6d -> ``(..., 3, 3)``, via the dataset's pinned conversion.

    Deliberately goes through the full 3x3 matrix rather than a 6x6 linear map: the
    shortcut is exactly right at the identity and wrong everywhere else (M4's lesson).
    """
    arr = d6.detach().cpu().numpy()
    shape = arr.shape[:-1]
    return torch.from_numpy(rot6d_to_mat(arr.reshape(-1, 6)).reshape(*shape, 3, 3))


def geodesic_deg(R_pred, R_true):
    """Rotation error in degrees: the angle of ``R_pred^T R_true``, per element.

    Casts to float64 first: matmul does not type-promote, so a float32 prediction
    against a float64 target would raise rather than compute -- and the two sides come
    from different code paths here (a fitted model vs the numpy camera helpers).
    """
    R = R_pred.double().transpose(-1, -2) @ R_true.double()
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def ridge_fit(X, Y, lam=1e-4, chunk=8192):
    """Closed-form ridge on (X^T X + lam I) w = X^T Y, accumulated in float64.

    Chunked so a 30k x 1.5k design matrix never has to exist in float64 all at once.
    """
    d = X.shape[1]
    XtX = torch.zeros(d, d, dtype=torch.float64)
    XtY = torch.zeros(d, Y.shape[1], dtype=torch.float64)
    for s in range(0, X.shape[0], chunk):
        xb = X[s:s + chunk].double()
        XtX += xb.transpose(-1, -2) @ xb
        XtY += xb.transpose(-1, -2) @ Y[s:s + chunk].double()
    XtX += lam * torch.eye(d, dtype=torch.float64)
    return torch.linalg.solve(XtX, XtY).float()


def _standardize(X_tr, X_te):
    mu = X_tr.mean(0, keepdim=True)
    sd = X_tr.std(0, keepdim=True).clamp_min(1e-6)
    return (X_tr - mu) / sd, (X_te - mu) / sd


def _with_bias(X):
    return torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)


class MLPProbe(torch.nn.Module):
    """Small nonlinear readout: separates 'not linearly decodable' from 'not there'."""

    def __init__(self, d_in, d_out, hidden=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp(X_tr, Y_tr, X_te, steps, device, seed=42, bs=4096, lr=1e-3):
    """Train the nonlinear readout.

    The target is cast to float32 to match the readout's own dtype. Targets here are
    float64 -- they come from the numpy camera math -- and while ``mse_loss`` *forward*
    type-promotes, the backward pass through a float32 ``nn.Linear`` does not: it dies at
    ``loss.backward()`` with ``Found dtype Double but expected Float``. The same class of
    trap as the one ``geodesic_deg`` documents, in the opposite direction, and it is why
    the cast is here rather than left to promotion.
    """
    torch.manual_seed(seed)
    model = MLPProbe(X_tr.shape[1], Y_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr, Ytr = X_tr.to(device), Y_tr.to(device).float()
    n = Xtr.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), device=device)
        loss = torch.nn.functional.mse_loss(model(Xtr[idx]), Ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return model(X_te.to(device)).cpu()


def score(pred_t, pred_R, tgt_t, tgt_R, name, out):
    """Translation error (cm) + rotation error (deg), next to the target's own scale.

    ``tgt_R is None`` is the translation-only target (``cam_eef``): the rotation keys are
    omitted rather than filled with a placeholder, so an absent measurement cannot be read
    off the JSON as a measured null.
    """
    t_err = (pred_t - tgt_t).norm(dim=-1) * 100.0          # metres -> cm
    entry = dict(
        t_rmse_cm=float(t_err.pow(2).mean().sqrt()),
        t_median_cm=float(t_err.median()),
        target_median_cm=float(tgt_t.norm(dim=-1).median() * 100.0),
    )
    if tgt_R is not None:
        r_err = geodesic_deg(pred_R, tgt_R)
        entry['r_mean_deg'] = float(r_err.mean())
        entry['r_median_deg'] = float(r_err.median())
    out[name] = entry
    return entry


def report_target(X, tgt_t, tgt_R, name, args, device, out):
    """Fit ridge (+ optional MLP) on one target and score it against both baselines.

    ``tgt_R is None`` (the ``cam_eef`` target -- the quantity M4's aux head read) drops the
    rotation half of the target, of the prediction split and of the metrics *together*. The
    three have to move as one: splitting a translation-only prediction at column 3 yields an
    empty rotation block, and ``mat_from_rot6d`` on it fails rather than returning anything
    meaningful. This path is exactly what the smoke run caught, and it is why the check
    below is paired with a regression test rather than left to the fit to fail on.
    """
    n = X.shape[0]
    n_tr = int(n * 0.8)
    has_rot = tgt_R is not None
    Y = tgt_t if not has_rot else torch.cat(
        [tgt_t, torch.from_numpy(rot6d_from_mat(tgt_R.numpy()))], dim=-1)

    X_tr, X_te = _standardize(X[:n_tr].float(), X[n_tr:].float())
    Y_tr, Y_te = Y[:n_tr], Y[n_tr:]
    Xb_tr, Xb_te = _with_bias(X_tr), _with_bias(X_te)

    def _split(pred):
        """Prediction columns -> (t, R). R is None when the target carries no rotation."""
        if not has_rot:
            return pred[:, :3], None
        return pred[:, :3], mat_from_rot6d(pred[:, 3:])

    def _tgt(perm=None):
        if not has_rot:
            return None
        return tgt_R[n_tr:] if perm is None else tgt_R[n_tr:][perm]

    entry = {}
    pred = Xb_te @ ridge_fit(Xb_tr, Y_tr)
    score(*_split(pred), tgt_t[n_tr:], _tgt(), 'ridge', entry)

    # baseline 1: the mean predictor -- the collapse floor any head must beat
    mean_pred = Y_tr.mean(0, keepdim=True).expand(Y_te.shape[0], -1)
    score(*_split(mean_pred), tgt_t[n_tr:], _tgt(), 'mean_predictor', entry)

    # baseline 2: shuffled targets -- must fail, or the probe reads something else.
    # Permute the TEST targets among themselves: same marginal distribution, broken
    # correspondence. (Permuting training rows in would both be the wrong control and
    # index out of bounds, which is how this was caught.)
    perm = torch.randperm(len(X_te))
    score(*_split(pred), tgt_t[n_tr:][perm], _tgt(perm), 'shuffled_target', entry)

    if args.mlp_steps > 0:
        pred_mlp = fit_mlp(X_tr, Y_tr, X_te, args.mlp_steps, device)
        score(*_split(pred_mlp), tgt_t[n_tr:], _tgt(), 'mlp', entry)

    entry['n_train'] = n_tr
    entry['n_test'] = n - n_tr
    out[name] = entry
    return entry


# ---------------------------------------------------------------------------
# data collection
# ---------------------------------------------------------------------------


def load_policy(checkpoint, device):
    """Mirror eval.py / eval_novel_view.py: rebuild the workspace from the payload."""
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    tmp = tempfile.mkdtemp(prefix='probe_relpose_')
    workspace = cls(cfg, output_dir=tmp)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    return policy, cfg


def _apply_range(dataset, view_count_range):
    """Apply a draw-range override to an instantiated dataset, in place.

    Set on the INSTANCE rather than the cfg: a checkpoint's payload cfg comes back in
    struct mode, so `cfg.task.dataset.view_count_range = ...` raises, and relaxing it
    with `OmegaConf.set_struct(False)` would have to be undone to keep the rest of the
    cfg read-only. Setting the attribute bypasses `__init__`, so its guard is
    re-implemented here -- and it is the guard that must hold, because `hi` has to fit
    in the pool or `np.random.choice(..., replace=False)` in `_m3_slots` raises later.
    """
    if view_count_range is None:
        # `view_count_range` exists only in slot mode (view_pool). A `view_subset`
        # dataset -- L1's -- has no such attribute, so this has to be a getattr or the
        # screen cannot run on that whole encoder family at all.
        cur = getattr(dataset, 'view_count_range', None)
        return tuple(int(x) for x in cur) if cur is not None else None
    lo, hi = (int(x) for x in view_count_range)
    n_slots = int(dataset.n_slots)
    if not (1 <= lo <= hi <= n_slots):
        raise ValueError(
            f'view_count_range [{lo}, {hi}] must satisfy 1 <= lo <= hi <= n_slots ({n_slots})')
    if hi > len(dataset.view_pool):
        raise ValueError(
            f'view_count_range hi={hi} exceeds the pool size {len(dataset.view_pool)}')
    dataset.view_count_range = (lo, hi)
    return (lo, hi)


def _slot_view_indices(dataset, obs, active_row):
    """``[(slot, ring index), ...]`` for the active slots of one frame.

    Recovered by EXACT match of the slot's camera vector against `cam_table` (the
    dataset serves `cam_table[v]` verbatim, so equality holds); an ambiguous match
    raises rather than picking a nearest neighbour, because a wrong pairing would
    silently understate every per-view statistic instead of failing.
    """
    out = []
    for k in range(int(dataset.n_slots)):
        if not bool(active_row[k]):
            continue
        cam = obs[dataset.cam_keys[k]][0].numpy()
        hit = np.nonzero(np.all(dataset.cam_table == cam, axis=1))[0]
        if len(hit) != 1:
            raise RuntimeError(
                f'slot {k}: camera vector matches {len(hit)} cam_table rows, expected 1')
        out.append((k, int(hit[0])))
    return out


def _rel_dist(a, b):
    """Mean relative L2 between two matched row-sets: ``||a-b|| / (0.5(||a||+||b||))``.

    Scale-free, so the number is comparable across checkpoints -- which is the whole
    point, since the thing under test is a cross-cell difference.
    """
    return float(((a - b).norm(dim=-1) /
                  (0.5 * (a.norm(dim=-1) + b.norm(dim=-1))).clamp_min(1e-6)).mean())


def collect(policy, cfg, device, args, view_count_range=None):
    """One forward pass per sample -> per-view latents, camera vectors, EE position.

    Camera vectors are read in the DATASET's slot order, because the dataset is what
    emitted this obs dict; the encoder's own `cam_keys` must agree or the pairing of
    latent to camera would be silently wrong (and a wrong pairing would make every
    probe here understate the geometry rather than fail).
    """
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    cell_range = tuple(int(x) for x in dataset.view_count_range)
    _apply_range(dataset, view_count_range)
    To = policy.n_obs_steps
    cam_keys = list(getattr(dataset, 'cam_keys', None) or policy.obs_encoder.cam_keys)
    enc_cam_keys = list(policy.obs_encoder.cam_keys)
    if enc_cam_keys and cam_keys != enc_cam_keys:
        raise RuntimeError(
            f'dataset cam_keys {cam_keys} != encoder cam_keys {enc_cam_keys}; the '
            f'slot-to-camera pairing would be wrong for every probe below')
    eef_keys = [k for k in policy.obs_encoder.low_dim_keys if k.endswith('eef_pos')]
    if len(eef_keys) != 1:
        raise RuntimeError(f'expected exactly one *eef_pos low-dim key, got {eef_keys}')
    eef_key = eef_keys[0]

    n = min(args.n_samples, len(dataset))
    Z, ACT, CAM, EEF = [], [], [], []
    for i in range(n):
        s = dataset[i]
        obs = dict_apply(s['obs'], lambda x: x.unsqueeze(0).to(device))
        nobs = policy.normalizer.normalize(obs)
        this_nobs = dict_apply(nobs, lambda x: x[:, :To].reshape(-1, *x.shape[2:]))
        with torch.no_grad():
            enc = policy.obs_encoder.forward_full(this_nobs)
        Z.append(enc['z_views'].float().cpu())            # (To, K, D)
        ACT.append(enc['view_active'].bool().cpu())       # (To, K)
        CAM.append(torch.stack([s['obs'][k] for k in cam_keys], dim=1))  # (To, K, 10)
        EEF.append(s['obs'][eef_key])                     # (To, 3)

    Zs, ACTs = torch.cat(Z, 0), torch.cat(ACT, 0)
    CAMs, EEFs = torch.cat(CAM, 0), torch.cat(EEF, 0)
    stats = dict(n_frames=int(Zs.shape[0]), n_slots=int(Zs.shape[1]),
                 dim=int(Zs.shape[2]), mean_active=float(ACTs.sum(-1).float().mean()),
                 n_dataset=len(dataset),
                 # Gate: mean_active must track the EFFECTIVE range (a [1,7] override
                 # reads ~4.0; 1.5 means the override was silently ignored), and both
                 # ranges are recorded so a reader can tell them apart.
                 effective_range=[int(x) for x in dataset.view_count_range],
                 cell_view_count_range=[int(x) for x in cell_range])
    return Zs, ACTs, CAMs, EEFs, stats


def build_probe_sets(Zs, ACTs, CAMs, EEFs, args, seed=42):
    """Absolute-pose, camera-frame-EE and pair-relative features + targets."""
    g = torch.Generator().manual_seed(seed)
    abs_X, abs_t, abs_R = [], [], []
    eef_X, eef_t = [], []
    pair_X, pair_t, pair_R = [], [], []
    for f in range(Zs.shape[0]):
        act = ACTs[f].nonzero().flatten()
        if len(act) == 0:
            continue
        for k in act.tolist():
            cam = CAMs[f, k].numpy().astype(np.float64)
            R_c = quat_wxyz_to_mat(cam[3:7])
            # 1. absolute pose of this view's own camera, from its own latent
            abs_X.append(Zs[f, k])
            abs_t.append(torch.from_numpy(cam[:3]))
            abs_R.append(torch.from_numpy(R_c))
            # 2. EE position expressed in this camera's frame
            eef_world = EEFs[f].numpy().astype(np.float64)
            eef_X.append(Zs[f, k])
            eef_t.append(torch.from_numpy(R_c.T @ (eef_world - cam[:3])))
        # 3. relative pose over sampled ordered pairs of ACTIVE slots. Same frame = same
        #    state, different view -- the pairing that makes the target well defined.
        if len(act) >= 2:
            n_pairs = min(args.pairs_per_sample, len(act) * (len(act) - 1))
            for _ in range(n_pairs):
                pick = torch.randperm(len(act), generator=g)[:2]
                i, j = act[pick[0]].item(), act[pick[1]].item()
                R, t = relative_pose_from_cams(CAMs[f, i].numpy(), CAMs[f, j].numpy())
                pair_X.append(torch.cat([Zs[f, i], Zs[f, j], Zs[f, i] - Zs[f, j]]))
                pair_t.append(torch.from_numpy(t))
                pair_R.append(torch.from_numpy(R))
    def _t3(xs, ts, Rs):
        """``None`` -- absent -- when a target has no rows, rather than crashing.

        A collect range whose `hi` is 1 produces no view PAIRS, so `rel_pose` is
        empty; `torch.stack([])` raises a message that says nothing about the cause,
        which is how this was found. Absent has to be distinguishable from a
        placeholder, here as everywhere else in this probe.
        """
        if len(xs) == 0:
            return None
        return (torch.stack(xs), torch.stack(ts),
                torch.stack(Rs) if Rs is not None else None)

    return dict(
        abs_pose=_t3(abs_X, abs_t, abs_R),
        cam_eef=_t3(eef_X, eef_t, None),
        rel_pose=_t3(pair_X, pair_t, pair_R),
    )


def _mean_or_none(vals):
    """``None`` -- never 0.0 and never NaN -- when there is nothing to average.

    ``float(np.mean([]))`` is NaN, which is invalid strict JSON and, read back as a
    number, is indistinguishable from a measurement. That path is not hypothetical: at
    a range whose draws are all N=1, every within-draw statistic is empty.
    """
    return float(np.mean(vals)) if len(vals) else None


def _full_draw_block(dataset, draws, n_slots):
    """The **fusion-free** statistic: pairwise ``z_v`` distances over the view POOL at a
    full draw, against the ``z_v`` distance *between states* at a fixed view.

    ``z_v`` is N-agnostic by construction -- GroupNorm rather than BatchNorm, a
    deterministic centre crop in eval, no dropout -- so one view encodes to the same
    tensor at any N. That is what licenses this comparison. It matters because at N=1
    the fusion is exactly ``z_g = A z_v + b`` with ``A`` trained *per cell*, so a raw
    ``z_g`` comparison across cells conflates the backbone with that cell's
    value-projection gain; this block is the one that does not.

    The half-split gap is the statistic's own noise floor, and is what makes a
    confirm/refute call quantitative instead of eyeballed.
    """
    # the FIRST full draw per state, not just `per[0]`: a state whose first draw was
    # small may still contribute, and at [1,7] most first draws are not full. Any full
    # draw serves the whole pool, so which repeat supplies it does not matter -- and the
    # 21 view pairs are the same set every time, so extra repeats add no information.
    full = []
    for per in draws:
        hit = next((p for p in per if len(p[3]) == n_slots), None)
        if hit is not None:
            full.append(hit)
    if not full:
        return dict(zv_pair_matrix=None, zv_across_states=None,
                    zv_pair_ratio=None, half_split=None)
    by_view = [{v: z[:, slot] for slot, v in pairs} for (_, z, _, pairs) in full]
    views = sorted(by_view[0].keys())

    def _ratio(ds):
        pair = [_rel_dist(d[i], d[j]) for d in ds
                for a_i, i in enumerate(views) for j in views[a_i + 1:]
                if i in d and j in d]
        cross = [_rel_dist(ds[i][v], ds[j][v])
                 for v in views for i in range(len(ds)) for j in range(i + 1, len(ds))
                 if v in ds[i] and v in ds[j]]
        pm, cm = _mean_or_none(pair), _mean_or_none(cross)
        return pm, cm, (pm / cm if pm is not None and cm else None)

    pair_mean, cross_mean, ratio = _ratio(by_view)
    half = len(by_view) // 2
    _, _, ra = _ratio(by_view[:half]) if half else (None, None, None)
    _, _, rb = _ratio(by_view[half:]) if half else (None, None, None)
    return dict(
        zv_pair_matrix=dict(mean=pair_mean, n=len(by_view)),
        zv_across_states=cross_mean,
        zv_pair_ratio=ratio,
        half_split=dict(a=ra, b=rb,
                        gap=(abs(ra - rb) if ra is not None and rb is not None else None)),
    )


def _stability_entry(policy, dataset, device, args, To, rng):
    """One draw-range's worth of latent statistics."""
    n_states = min(int(args.stability_states), len(dataset))
    # seeded per block, so the stability draws do not depend on how much of the global
    # stream `collect` consumed -- i.e. on --n-samples. Two cells are then drawn
    # IDENTICALLY, which is what makes the fingerprint below checkable rather than
    # merely asserted.
    np.random.seed(int(args.seed))

    draws, size_hist, heads = [], {}, []
    for i in range(n_states):
        per = []
        for _ in range(int(args.stability_repeats)):
            s = dataset[i]
            obs = dict_apply(s['obs'], lambda x: x.unsqueeze(0).to(device))
            nobs = policy.normalizer.normalize(obs)
            this_nobs = dict_apply(nobs, lambda x: x[:, :To].reshape(-1, *x.shape[2:]))
            with torch.no_grad():
                enc = policy.obs_encoder.forward_full(this_nobs)
            active = enc['view_active'].bool().cpu()
            per.append((enc['z_global'].float().cpu(),
                        enc['z_views'].float().cpu(),
                        active,
                        _slot_view_indices(dataset, s['obs'], active[0])))
        draws.append(per)
        n_act = len(per[0][3])
        size_hist[str(n_act)] = size_hist.get(str(n_act), 0) + 1
        if len(heads) < 4:
            heads.append(sorted(v for _, v in per[0][3]))

    zg_by_size = {}
    for per in draws:
        for a in range(len(per)):
            for b in range(a + 1, len(per)):
                ka, kb = len(per[a][3]), len(per[b][3])
                zg_by_size.setdefault(f'{min(ka, kb)}|{max(ka, kb)}', []).append(
                    _rel_dist(per[a][0], per[b][0]))

    zv_by_size = {}
    for per in draws:
        for (_, z, act, _) in per:
            for f in range(z.shape[0]):
                idx = act[f].nonzero().flatten()
                if len(idx) < 2:
                    continue
                zv = z[f, idx]
                zv_by_size.setdefault(str(len(idx)), []).extend(
                    _rel_dist(zv[p], zv[q])
                    for p in range(len(idx)) for q in range(p + 1, len(idx)))

    # --- does the FUSION pass the geometry through? -------------------------------
    # The policy never sees z_v: `forward` hands the UNet `z_g` alone. So a healthy,
    # decodable z_v does NOT imply a decodable z_g, and that gap is exactly the
    # "is the failure downstream of the encoder?" question -- which is unanswerable
    # from the per-view latents alone. Measured at draws with ONE active view, since
    # that is both the inference condition and the case where z_g is an unambiguous
    # function of a single view's z_v.
    zg_X, zg_t, zg_R = [], [], []
    for per in draws:
        for (zg, _z, _act, pairs) in per:
            if len(pairs) != 1:
                continue
            cam = np.asarray(dataset.cam_table[pairs[0][1]], dtype=np.float64)
            zg_X.append(zg[0])                       # first obs step's fused latent
            zg_t.append(torch.from_numpy(cam[:3]))
            zg_R.append(torch.from_numpy(quat_wxyz_to_mat(cam[3:7])))

    entry = dict(
        n_states=n_states,
        n_repeats=int(args.stability_repeats),
        draw_fingerprint=dict(size_hist=size_hist, first_views=heads),
        zg_across_view_subsets=_mean_or_none([v for vs in zg_by_size.values() for v in vs]),
        zg_by_size={k: dict(mean=_mean_or_none(v), n=len(v))
                    for k, v in sorted(zg_by_size.items())},
        zv_within_draw_by_size={k: dict(mean=_mean_or_none(v), n=len(v))
                                for k, v in sorted(zv_by_size.items())},
        # At a full draw every repeat serves the same view SET re-permuted, and the
        # fusion is permutation-invariant, so this bucket is the pipeline's noise floor
        # for the entire z_g statistic -- for free, and it CAN fail: anything above
        # float noise means something non-deterministic is live and nothing else here
        # is trustworthy.
        zg_permutation_floor=_mean_or_none(
            zg_by_size.get(f'{int(dataset.n_slots)}|{int(dataset.n_slots)}', [])),
    )
    entry.update(_full_draw_block(dataset, draws, n_slots=int(dataset.n_slots)))
    # decode the camera pose from the FUSED latent, to be compared against the same
    # decode from z_v (`targets.abs_pose`). `None` when the range never produces a
    # single-view draw -- absent, never a placeholder.
    entry['zg_abs_pose'] = None
    if len(zg_X) >= 50:
        sub = {}
        report_target(torch.stack(zg_X), torch.stack(zg_t), torch.stack(zg_R),
                      'zg_abs_pose', args, device, sub)
        entry['zg_abs_pose'] = sub['zg_abs_pose']
    return entry


def measure_stability(policy, cfg, device, args, ranges):
    """Probe the latent statistics on a GRID of draw ranges.

    A grid rather than one range, because the draw composition is otherwise governed by
    the checkpoint's own training range -- `[1,2]` compares mostly 1-vs-2 view draws
    while `[1,7]` compares up to 1-vs-7 -- so a single number is not comparable across
    cells. Every cell runs through one code path on the same grid, and the fingerprints
    it emits are what let two cells' draws be checked as identical rather than assumed.
    """
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    cell_range = tuple(int(x) for x in dataset.view_count_range)
    To = policy.n_obs_steps
    grid = {}
    for rng in ranges:
        _apply_range(dataset, rng)
        entry = _stability_entry(policy, dataset, device, args, To, rng)
        z11 = (entry['zg_by_size'].get('1|1') or {}).get('mean')
        # the fuser's amplification, carried explicitly as a nuisance parameter: a
        # difference here with no difference in the fusion-free ratio means the FUSER
        # differs, not the backbone.
        entry['amp'] = (z11 / entry['zv_pair_ratio']
                        if z11 is not None and entry.get('zv_pair_ratio') else None)
        grid[f'{rng[0]},{rng[1]}'] = entry
    return cell_range, grid


def _check_zv_n_agnostic(policy, dataset, device, args, To):
    """The single assumption every per-view statistic here rests on: that ``z_v`` does not
    depend on how many OTHER views share the batch.

    Encodes one view inside a full 7-view draw, then the same view alone, and requires
    bit-identical output. It should hold by construction (GroupNorm, not BatchNorm; a
    deterministic eval-mode centre crop; no dropout) -- but it is an assumption about the
    encoder, and if it is not ~0 then every per-view number in the JSON means something
    other than what it says. Cheap enough to check on every run.
    """
    _apply_range(dataset, (int(dataset.n_slots), int(dataset.n_slots)))
    np.random.seed(int(args.seed))
    obs = dict_apply(dataset[0]['obs'], lambda x: x.unsqueeze(0).to(device))
    nobs = policy.normalizer.normalize(obs)

    def _enc(o):
        this = dict_apply(o, lambda x: x[:, :To].reshape(-1, *x.shape[2:]))
        with torch.no_grad():
            return policy.obs_encoder.forward_full(this)

    full = _enc(nobs)
    alone = dict(nobs)
    mask = nobs[dataset.view_mask_key].clone()
    mask[:, :, 1:] = 0.0                       # keep slot 0 only
    alone[dataset.view_mask_key] = mask
    one = _enc(alone)
    return float((full['z_views'][:, 0] - one['z_views'][:, 0]).abs().max())


def _pair(spec):
    lo, hi = (int(x) for x in str(spec).split(','))
    return (lo, hi)


@click.command()
@click.option('-c', '--checkpoint', required=True, help='M3/M4 checkpoint (latest.ckpt)')
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n-samples', default=2000, help='dataset indices (each = To frames)')
@click.option('--pairs-per-sample', default=8, help='relative-pose pairs per frame')
@click.option('--stability-states', default=64)
@click.option('--stability-repeats', default=4, help='view draws per state')
@click.option('--stability-ranges', default=None,
              help='"LO,HI;LO,HI;..." grid. Default: the checkpoint\'s own range only.')
@click.option('--view-count-range', default=None,
              help='LO,HI override for the geometry collect pass (default: first grid entry)')
@click.option('--random-init-control', is_flag=True,
              help='also run the grid on a re-initialised encoder -- the ceiling control')
@click.option('--num-threads', default=4, help='torch CPU threads')
@click.option('--mlp-steps', default=0, help='>0 trains a small MLP readout too')
@click.option('--seed', default=42)
def main(checkpoint, output_dir, device, n_samples, pairs_per_sample,
         stability_states, stability_repeats, stability_ranges, view_count_range,
         random_init_control, num_threads, mlp_steps, seed):
    torch.set_num_threads(int(num_threads))
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = click.get_current_context().params
    args = type('A', (), args)()

    policy, cfg = load_policy(checkpoint, device)
    enc_cfg = cfg.policy.obs_encoder
    # Guards, so a wrong checkpoint fails here rather than after minutes of collection
    # or, worse, produces a probe of a quantity the model never had.
    if not getattr(policy.obs_encoder, 'cam_keys', None):
        raise SystemExit(
            'this checkpoint\'s encoder has no per-slot cam keys: the probe needs an '
            'M3/M4 slot-mode encoder (view slots + camera vectors)')
    if getattr(cfg.task.dataset, 'view_pool', None) is None:
        raise SystemExit(
            'the checkpoint\'s dataset config is not in M3 slot mode (no view_pool): '
            'relative-pose targets are only well defined when several views of the '
            'same state are served together')
    print(f'checkpoint: {checkpoint}')
    print(f'  use_plucker={enc_cfg.use_plucker} use_eef_hist={enc_cfg.use_eef_hist} '
          f'n_slots={len(policy.obs_encoder.cam_keys)}')

    grid_ranges = ([_pair(p) for p in str(stability_ranges).split(';') if p.strip()]
                   if stability_ranges else None)
    if grid_ranges is None:
        grid_ranges = [tuple(int(x) for x in cfg.task.dataset.view_count_range)]
    collect_range = _pair(view_count_range) if view_count_range else grid_ranges[0]
    print(f'  grid={grid_ranges}  collect_range={collect_range}')

    t0 = time.time()
    Zs, ACTs, CAMs, EEFs, stats = collect(policy, cfg, device, args, collect_range)
    print(f'collected {stats["n_frames"]} frames, mean active slots '
          f'{stats["mean_active"]:.2f} of {stats["n_slots"]}  '
          f'[cell {stats["cell_view_count_range"]} -> effective '
          f'{stats["effective_range"]}]  ({time.time() - t0:.0f}s)')

    sets = build_probe_sets(Zs, ACTs, CAMs, EEFs, args, seed=seed)
    out = dict(schema=2, checkpoint=checkpoint,
               checkpoint_bytes=os.path.getsize(checkpoint),
               checkpoint_mtime=time.strftime(
                   '%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(checkpoint))),
               stats=stats,
               use_plucker=bool(enc_cfg.use_plucker),
               use_eef_hist=bool(enc_cfg.use_eef_hist), targets={})
    for name in ('abs_pose', 'cam_eef', 'rel_pose'):
        got = sets[name]
        if got is None:
            print(f'  {name}: no rows at collect range {collect_range}, skipped')
            out['targets'][name] = None
            continue
        X, t, R = got
        report_target(X, t, R, name, args, device, out['targets'])

    t0 = time.time()
    cell_range, grid = measure_stability(policy, cfg, device, args, grid_ranges)
    out['cell_view_count_range'] = list(cell_range)
    out['grid_ranges'] = [list(r) for r in grid_ranges]
    out['latent_grid'] = grid
    out['latent_stats'] = grid[f'{grid_ranges[0][0]},{grid_ranges[0][1]}']
    print(f'stability grid done ({time.time() - t0:.0f}s)')

    ds = hydra.utils.instantiate(cfg.task.dataset)
    out['zv_n_agnostic_max_abs_diff'] = _check_zv_n_agnostic(
        policy, ds, device, args, policy.n_obs_steps)
    print(f'  z_v N-agnostic max|diff| = {out["zv_n_agnostic_max_abs_diff"]:.3e}')

    out['latent_controls'] = None
    if random_init_control:
        enc = policy.obs_encoder
        saved = {k: v.detach().clone() for k, v in enc.state_dict().items()}
        try:
            enc.apply(lambda m: m.reset_parameters()
                      if hasattr(m, 'reset_parameters') else None)
            _, ctrl = measure_stability(policy, cfg, device, args, grid_ranges)
        finally:
            enc.load_state_dict(saved)
        out['latent_controls'] = dict(random_init=ctrl)
        print('  random-init control done')

    dest = os.path.join(output_dir, 'probe_relpose.json')
    with open(dest, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {dest}')


if __name__ == '__main__':
    main()
