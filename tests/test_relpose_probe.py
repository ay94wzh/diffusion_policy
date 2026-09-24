"""CPU-only test for probe_relpose.py (no simulator, no checkpoint, no zarr, no GPU).

Run from the repo root:
    python tests/test_relpose_probe.py

What this pins, and why each check has a mutation next to it
------------------------------------------------------------
`probe_relpose.py` decides whether a relational head is worth training, so a silent
error in it would send the whole plan the wrong way and never crash. Two lessons from
this repo's history drive the shape of this file:

  * "A check that cannot fail proves nothing" (NOTES.md). Every convention assertion
    below is paired with a mutation that MUST fail, and with the degenerate case in
    which that mutation wrongly *passes* -- which is exactly how a broken check gets
    shipped. For relative pose, the degenerate case is two identical camera poses:
    every wrong chaining convention agrees there.
  * The 6d rotation shortcut is correct at the identity (M4). The rot6d round-trip is
    therefore checked at non-identity rotations, and the shortcut is asserted to fail
    away from the identity.

What is NOT tested here: whether the real `z_v` latents carry this geometry. That is
the measurement the script exists to make, and it needs the box.
"""
import os
import sys
import math

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from types import SimpleNamespace

import numpy as np
import torch

from probe_relpose import (
    relative_pose_from_cams, rot6d_from_mat, mat_from_rot6d, geodesic_deg,
    ridge_fit, _standardize, _with_bias, report_target)
from diffusion_policy.dataset.multiview_image_dataset import quat_wxyz_to_mat
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def _rand_quat_wxyz(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def _rand_cam(rng, radius=1.0):
    """A camera 10-vector in the dataset's layout, with a non-degenerate look direction."""
    pos = rng.normal(size=3) * radius
    return np.concatenate([pos, _rand_quat_wxyz(rng), [45.0, 84.0, 84.0]])


def test_relative_pose_convention():
    """x_j == R x_i + t, checked against an independently written projection."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        cam_i, cam_j = _rand_cam(rng), _rand_cam(rng)
        R, t = relative_pose_from_cams(cam_i, cam_j)
        X = rng.normal(size=3)                      # a world point
        R_i, R_j = quat_wxyz_to_mat(cam_i[3:7]), quat_wxyz_to_mat(cam_j[3:7])
        # camera coords straight from the definition x_c = R_c^T (X - p_c)
        x_i = R_i.T @ (X - cam_i[:3])
        x_j = R_j.T @ (X - cam_j[:3])
        worst = max(worst, float(np.abs(x_j - (R @ x_i + t)).max()))
    assert worst < 1e-10, f'relative-pose chaining wrong by {worst:.2e}'
    print(f'  relative pose: x_j == R x_i + t to {worst:.2e} over 200 random poses')

    # the inverse relation, as a second independent property
    rng = np.random.default_rng(1)
    worst_inv = 0.0
    for _ in range(200):
        cam_i, cam_j = _rand_cam(rng), _rand_cam(rng)
        R, t = relative_pose_from_cams(cam_i, cam_j)
        R2, t2 = relative_pose_from_cams(cam_j, cam_i)
        worst_inv = max(worst_inv,
                        float(np.abs(R2 - R.T).max()),
                        float(np.abs(t2 - (-R.T @ t)).max()))
    assert worst_inv < 1e-10, f'inverse relation wrong by {worst_inv:.2e}'
    print(f'  inverse relation R_ji = R_ij^T holds to {worst_inv:.2e}')


def test_relative_pose_mutation_power():
    """Every wrong convention must FAIL -- and must wrongly pass at identical poses."""
    rng = np.random.default_rng(2)
    cam_i, cam_j = _rand_cam(rng), _rand_cam(rng)
    R, t = relative_pose_from_cams(cam_i, cam_j)
    X = rng.normal(size=3)
    R_i, R_j = quat_wxyz_to_mat(cam_i[3:7]), quat_wxyz_to_mat(cam_j[3:7])
    x_i, x_j = R_i.T @ (X - cam_i[:3]), R_j.T @ (X - cam_j[:3])

    def residual(Ra, ta, cam_a, cam_b):
        Ra_i, Ra_j = quat_wxyz_to_mat(cam_a[3:7]), quat_wxyz_to_mat(cam_b[3:7])
        xa_i = Ra_i.T @ (X - cam_a[:3])
        xa_j = Ra_j.T @ (X - cam_b[:3])
        return float(np.abs(xa_j - (Ra @ xa_i + ta)).max())

    assert residual(R, t, cam_i, cam_j) < 1e-12

    # mutation 1: swapped direction (R_i^T R_j instead of R_j^T R_i)
    R_wrong = R_i.T @ R_j
    assert residual(R_wrong, t, cam_i, cam_j) > 1e-3, 'swapped direction was not caught'
    # mutation 2: unrotated translation (p_i - p_j)
    assert residual(R, cam_i[:3] - cam_j[:3], cam_i, cam_j) > 1e-3, \
        'unrotated translation was not caught'

    # ...and both mutations PASS at two identical poses, which is how a check that
    # cannot fail would have been written
    same = _rand_cam(rng)
    R0, t0 = relative_pose_from_cams(same, same)
    assert residual(R0, t0, same, same) < 1e-12
    assert residual(quat_wxyz_to_mat(same[3:7]).T @ quat_wxyz_to_mat(same[3:7]),
                    np.zeros(3), same, same) < 1e-12
    print('  mutation power: swapped direction and unrotated translation both caught, '
          'and both wrongly pass at identical poses (as predicted)')


def test_rot6d_round_trip():
    """rot6d <-> matrix at NON-identity rotations, plus the 6x6-shortcut mutation."""
    rng = np.random.default_rng(3)
    to6d = RotationTransformer(from_rep='matrix', to_rep='rotation_6d')
    worst = 0.0
    for _ in range(200):
        R = quat_wxyz_to_mat(_rand_quat_wxyz(rng))
        d6 = rot6d_from_mat(R)
        worst = max(worst, float(np.abs(mat_from_rot6d(torch.from_numpy(d6)).numpy() - R).max()))
        # the encoding itself matches the one that produced the stored actions
        d6_ref = to6d.forward(torch.from_numpy(R).float().unsqueeze(0))[0].numpy()
        worst = max(worst, float(np.abs(d6 - d6_ref).max()))
    assert worst < 1e-5, f'rot6d round-trip / convention wrong by {worst:.2e}'
    print(f'  rot6d round-trip + pytorch3d convention agree to {worst:.2e}')

    # geodesic_deg measures a real angle: 30 deg about z against the identity
    ang = math.radians(30.0)
    Rz = torch.tensor([[math.cos(ang), -math.sin(ang), 0],
                       [math.sin(ang), math.cos(ang), 0],
                       [0.0, 0.0, 1.0]])
    err = geodesic_deg(Rz, torch.eye(3))
    assert abs(float(err) - 30.0) < 1e-3, f'geodesic error gave {float(err):.3f} deg'
    print(f'  geodesic_deg recovers a known 30 deg separation ({float(err):.4f} deg)')

    # mutation: encoding COLUMNS 0/1 instead of ROWS 0/1 must be caught at a general
    # rotation -- and, as always, wrongly passes at the identity, where they coincide
    R = quat_wxyz_to_mat(_rand_quat_wxyz(np.random.default_rng(4)))
    cols = np.concatenate([R[:, 0], R[:, 1]])
    err_cols = float(geodesic_deg(mat_from_rot6d(torch.from_numpy(cols)),
                                  torch.from_numpy(R)))
    eye_cols = np.concatenate([np.eye(3)[:, 0], np.eye(3)[:, 1]])
    err_cols_id = float(geodesic_deg(mat_from_rot6d(torch.from_numpy(eye_cols)),
                                     torch.eye(3)))
    assert err_cols > 1.0, f'columns-instead-of-rows was not caught ({err_cols:.3f} deg)'
    assert err_cols_id < 1e-6, 'the column/row mutation should be invisible at the identity'
    print(f'  columns-instead-of-rows: {err_cols:.2f} deg wrong at a general rotation, '
          f'{err_cols_id:.1e} deg at the identity (the trap it sets)')


def test_ridge_recovers_known_weights():
    """The fitter must recover a known linear map, and beat the mean predictor."""
    torch.manual_seed(0)
    n, d, k = 4000, 64, 3
    X = torch.randn(n, d)
    W = torch.randn(d, k)
    Y = X @ W + 0.01 * torch.randn(n, k)
    Xb = _with_bias(_standardize(X[:3200], X[3200:])[0])
    Xb_te = _with_bias(_standardize(X[:3200], X[3200:])[1])
    Wfit = ridge_fit(Xb, Y[:3200])
    err = float((Xb_te @ Wfit - Y[3200:]).pow(2).mean().sqrt())
    mean_err = float((Y[:3200].mean(0) - Y[3200:]).pow(2).mean().sqrt())
    assert err < 0.02, f'ridge did not recover the map (rmse {err:.4f})'
    assert err < 0.5 * mean_err, 'ridge should clearly beat the mean predictor'
    print(f'  ridge recovers a known linear map: rmse {err:.4f} vs mean-predictor '
          f'{mean_err:.4f}')


def test_report_target_detects_decodable_geometry():
    """End-to-end plumbing check where the answer IS in the features, by construction.

    This validates the reporting path -- the train/test split, the rot6d target
    assembly, `mat_from_rot6d`, the geodesic scoring and both baselines -- on a problem
    whose answer is known. It says nothing about whether any real representation carries
    geometry; that is the measurement the probe script exists to make.

    Note what an earlier version of this test got wrong, because it is a real property
    of the method: with the two cameras as raw features, a LINEAR probe cannot recover
    the relative pose at all -- relative pose is bilinear in the two camera poses (a
    product of rotations and a rotated difference), and the ridge fit scored *worse*
    than the mean predictor (250.6 cm vs 248.2 cm). So a linear probe can legitimately
    fail on geometry that is present, which is exactly why `--mlp-steps` exists and why
    a null from the ridge probe must not be read as "the information is absent".
    """
    rng = np.random.default_rng(5)
    X, t_all, R_all = [], [], []
    for _ in range(1500):
        cam_i, cam_j = _rand_cam(rng), _rand_cam(rng)
        R, t = relative_pose_from_cams(cam_i, cam_j)
        y = np.concatenate([t, rot6d_from_mat(R)])
        X.append(torch.from_numpy(y + 0.05 * rng.normal(size=y.shape)).float())
        t_all.append(torch.from_numpy(t))
        R_all.append(torch.from_numpy(R))
    out = {}
    args = SimpleNamespace(mlp_steps=0)
    entry = report_target(torch.stack(X), torch.stack(t_all).float(),
                          torch.stack(R_all), 'rel_pose', args,
                          torch.device('cpu'), out)
    mean = out['rel_pose']['mean_predictor']
    shuf = out['rel_pose']['shuffled_target']
    assert entry['ridge']['t_rmse_cm'] < 0.5 * mean['t_rmse_cm'], \
        f"ridge {entry['ridge']['t_rmse_cm']:.3f} vs mean {mean['t_rmse_cm']:.3f}"
    assert entry['ridge']['r_mean_deg'] < 0.5 * mean['r_mean_deg'], \
        f"ridge {entry['ridge']['r_mean_deg']:.3f} vs mean {mean['r_mean_deg']:.3f}"
    assert shuf['t_rmse_cm'] > entry['ridge']['t_rmse_cm'], \
        'the shuffled control should be worse than the real fit'
    print(f"  report_target detects decodable geometry: ridge {entry['ridge']['r_mean_deg']:.2f} deg "
          f"vs mean predictor {mean['r_mean_deg']:.2f} deg, shuffled {shuf['r_mean_deg']:.2f} deg")


def test():
    print('=' * 70)
    print('relpose probe -- CPU checks')
    print('=' * 70)
    test_relative_pose_convention()
    test_relative_pose_mutation_power()
    test_rot6d_round_trip()
    test_ridge_recovers_known_weights()
    test_report_target_detects_decodable_geometry()
    print('=' * 70)
    print('ALL RELPOSE PROBE CHECKS PASSED')
    print('=' * 70)


if __name__ == '__main__':
    test()
