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


def collect(policy, cfg, device, args):
    """One forward pass per sample -> per-view latents, camera vectors, EE position.

    Camera vectors are read in the DATASET's slot order, because the dataset is what
    emitted this obs dict; the encoder's own `cam_keys` must agree or the pairing of
    latent to camera would be silently wrong (and a wrong pairing would make every
    probe here understate the geometry rather than fail).
    """
    dataset = hydra.utils.instantiate(cfg.task.dataset)
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
                 n_dataset=len(dataset))
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
    return dict(
        abs_pose=(torch.stack(abs_X), torch.stack(abs_t), torch.stack(abs_R)),
        cam_eef=(torch.stack(eef_X), torch.stack(eef_t), None),
        rel_pose=(torch.stack(pair_X), torch.stack(pair_t), torch.stack(pair_R)),
    )


def measure_stability(policy, cfg, device, args):
    """z_g across view subsets of the SAME state, vs z_v across views of one draw."""
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    To = policy.n_obs_steps
    zg_draws, zv_spread = [], []
    for i in range(min(args.stability_states, len(dataset))):
        per_draw = []
        for _ in range(args.stability_repeats):
            s = dataset[i]
            obs = dict_apply(s['obs'], lambda x: x.unsqueeze(0).to(device))
            nobs = policy.normalizer.normalize(obs)
            this_nobs = dict_apply(nobs, lambda x: x[:, :To].reshape(-1, *x.shape[2:]))
            with torch.no_grad():
                enc = policy.obs_encoder.forward_full(this_nobs)
            per_draw.append((enc['z_global'].float().cpu(),
                             enc['z_views'].float().cpu(),
                             enc['view_active'].bool().cpu()))
        # how far z_g moves when the VIEW SET changes for a fixed state, normalised so
        # the number is comparable across checkpoints
        for a in range(len(per_draw)):
            for b in range(a + 1, len(per_draw)):
                za, zb = per_draw[a][0], per_draw[b][0]
                zg_draws.append(float(((za - zb).norm(dim=-1) /
                                       (0.5 * (za.norm(dim=-1) + zb.norm(dim=-1)))).mean()))
        # how far the live views' z_v are from each other within one draw
        z, act = per_draw[0][1], per_draw[0][2]
        for f in range(z.shape[0]):
            idx = act[f].nonzero().flatten()
            if len(idx) < 2:
                continue
            zv = z[f, idx]
            m = zv.mean(0, keepdim=True)
            zv_spread.append(float(((zv - m).norm(dim=-1) /
                                    m.norm().clamp_min(1e-6)).mean()))
    return dict(
        z_g_across_view_subsets=float(np.mean(zg_draws)),
        z_v_across_views_within_draw=float(np.mean(zv_spread)),
        n_zg_pairs=len(zg_draws), n_zv_frames=len(zv_spread))


@click.command()
@click.option('-c', '--checkpoint', required=True, help='M3/M4 checkpoint (latest.ckpt)')
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n-samples', default=2000, help='dataset indices (each = To frames)')
@click.option('--pairs-per-sample', default=8, help='relative-pose pairs per frame')
@click.option('--stability-states', default=64)
@click.option('--stability-repeats', default=4, help='view draws per state')
@click.option('--mlp-steps', default=0, help='>0 trains a small MLP readout too')
@click.option('--seed', default=42)
def main(checkpoint, output_dir, device, n_samples, pairs_per_sample,
         stability_states, stability_repeats, mlp_steps, seed):
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

    Zs, ACTs, CAMs, EEFs, stats = collect(policy, cfg, device, args)
    print(f'collected {stats["n_frames"]} frames, mean active slots '
          f'{stats["mean_active"]:.2f} of {stats["n_slots"]}')

    sets = build_probe_sets(Zs, ACTs, CAMs, EEFs, args, seed=seed)
    out = dict(checkpoint=checkpoint, stats=stats,
               use_plucker=bool(enc_cfg.use_plucker),
               use_eef_hist=bool(enc_cfg.use_eef_hist), targets={})
    for name in ('abs_pose', 'cam_eef', 'rel_pose'):
        X, t, R = sets[name]
        report_target(X, t, R, name, args, device, out['targets'])
    out['latent_stats'] = measure_stability(policy, cfg, device, args)

    print(json.dumps(out, indent=2))
    dest = os.path.join(output_dir, 'probe_relpose.json')
    with open(dest, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {dest}')


if __name__ == '__main__':
    main()
