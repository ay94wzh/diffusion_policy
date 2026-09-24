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
import json
import math
import shutil
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from types import SimpleNamespace

import numpy as np
import torch

from probe_relpose import (
    relative_pose_from_cams, rot6d_from_mat, mat_from_rot6d, geodesic_deg,
    ridge_fit, _standardize, _with_bias, report_target,
    _apply_range, _mean_or_none, _rel_dist, _slot_view_indices, _pair)
from diffusion_policy.dataset.multiview_image_dataset import (
    quat_wxyz_to_mat, MultiViewImageDataset)
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


def test_report_target_translation_only():
    """`cam_eef` is a translation-only target (`tgt_R is None`) -- the path the smoke run
    caught crashing, and the quantity M4's aux head read, so it is the column that doubles
    as the probe's positive control.

    A check that only asserted 'it does not raise' could not fail, so this pins the two
    things that make the number meaningful: the fit beats the mean predictor on a target
    whose answer is present by construction, and the rotation metrics are *absent* rather
    than silently zero -- an implementation that filled in `r_mean_deg = 0.0` would pass a
    crash check and read as a perfect rotation fit.
    """
    rng = np.random.default_rng(6)
    X, t_all = [], []
    for _ in range(1200):
        cam = _rand_cam(rng)
        R_c = quat_wxyz_to_mat(cam[3:7])
        p_eef = rng.normal(size=3) * 0.3
        y = R_c.T @ (p_eef - cam[:3])                  # EE position, camera frame
        X.append(torch.from_numpy(y + 0.05 * rng.normal(size=3)).float())
        t_all.append(torch.from_numpy(y))
    out = {}
    args = SimpleNamespace(mlp_steps=0)
    entry = report_target(torch.stack(X), torch.stack(t_all).float(), None,
                          'cam_eef', args, torch.device('cpu'), out)
    # the rotation metric must be absent, not zero, in every arm
    for arm in ('ridge', 'mean_predictor', 'shuffled_target'):
        assert 'r_mean_deg' not in out['cam_eef'][arm], \
            f"translation-only target reported a rotation error in '{arm}'"
    assert 'r_mean_deg' not in entry
    mean = out['cam_eef']['mean_predictor']
    assert entry['ridge']['t_rmse_cm'] < 0.5 * mean['t_rmse_cm'], \
        f"ridge {entry['ridge']['t_rmse_cm']:.3f} vs mean {mean['t_rmse_cm']:.3f}"
    print(f"  translation-only target: ridge {entry['ridge']['t_rmse_cm']:.3f} cm vs mean "
          f"predictor {mean['t_rmse_cm']:.3f} cm, no rotation key reported")


def test_mlp_probe_path_runs():
    """`fit_mlp` is the column that decides step 1a on `rel_pose`, and until now nothing had
    ever executed it -- not the CPU suite, not the smoke run, which used `--mlp-steps 0`.

    It is also where a dtype trap lives, so the dtypes here are the real ones and not
    convenient ones: **float32 features against float64 targets**, which is what `z_v` and
    the numpy camera math actually produce. `mse_loss` type-promotes in the forward pass,
    so the run gets all the way to `loss.backward()` before failing, and only at full scale
    (`--mlp-steps 2000`) did that surface.

    Both halves of the target split are exercised, because they take different branches:
    `cam_eef` carries no rotation and must report no rotation metric, while `abs_pose` does.
    The assertion is the meaningful one -- the readout must beat the mean predictor on a
    target whose answer is present by construction -- so this cannot pass on a code path
    that merely declines to raise.
    """
    torch.manual_seed(0)
    n, d = 800, 64
    X = torch.randn(n, d)                                  # float32, as z_v is
    t64 = X.double() @ torch.randn(d, 3, dtype=torch.float64) \
        + 0.05 * torch.randn(n, 3, dtype=torch.float64)     # float64, as the cams are
    R = mat_from_rot6d(X.double() @ torch.randn(d, 6, dtype=torch.float64))

    args = SimpleNamespace(mlp_steps=500)
    dev = torch.device('cpu')
    out = {}
    e1 = report_target(X, t64, None, 'cam_eef', args, dev, out)
    assert 'mlp' in e1, 'the MLP arm was not produced for a translation-only target'
    assert 'r_mean_deg' not in e1, 'translation-only target reported a rotation error'
    mean = out['cam_eef']['mean_predictor']['t_rmse_cm']
    assert e1['mlp']['t_rmse_cm'] < 0.5 * mean, \
        f"mlp {e1['mlp']['t_rmse_cm']:.3f} vs mean predictor {mean:.3f} cm"

    e2 = report_target(X, t64, R, 'abs_pose', args, dev, out)
    assert 'mlp' in e2, 'the MLP arm was not produced for a rotation target'
    # the metric lives inside each arm, not at the target level
    assert 'r_mean_deg' in e2['mlp'], 'the rotation arm lost its rotation metric'
    print(f"  mlp readout: {e1['mlp']['t_rmse_cm']:.3f} cm vs mean predictor "
          f"{mean:.3f} cm, float64 targets against float32 features")


def _slot_dataset(path, n_views, hw, view_count_range, steps=3):
    """A slot-mode synthetic dataset -- the fixture the override/statistic tests share."""
    obs_meta = {f'view_slot_{k:02d}_image': {'shape': [3, hw, hw], 'type': 'rgb'}
                for k in range(n_views)}
    obs_meta['robot0_eef_pos'] = {'shape': [3]}
    obs_meta['robot0_eef_quat'] = {'shape': [4]}
    obs_meta['robot0_gripper_qpos'] = {'shape': [2]}
    return MultiViewImageDataset(
        shape_meta={'obs': obs_meta, 'action': {'shape': [10]}},
        dataset_path=path, horizon=4, pad_before=1, pad_after=7, n_obs_steps=2,
        abs_action=True, view_pool=list(range(n_views)),
        view_count_range=view_count_range, eef_hist_steps=steps, seed=0, val_ratio=0.0)


def test_range_override_is_applied():
    """The draw-range override must actually reach the dataset's draw.

    It is what makes a cross-cell comparison legitimate: without it the probe measures
    whatever range the CHECKPOINT trained with, so `[1,2]` and `[1,7]` would be compared
    on different draws -- precisely the confound it exists to remove. It **cannot pass
    against the pre-change code**, where `_apply_range` did not exist and there was no
    override at all.

    Mutation power: an override that is silently ignored leaves the cell's own range in
    place, so the observed draw sizes below report that range rather than the requested
    one. The unsatisfiable cases matter for the same reason -- `hi` must fit the pool or
    `np.random.choice(..., replace=False)` raises deeper in `_m3_slots`, where the cause
    would be much harder to see.
    """
    import test_multiview_dataset as tmd
    tmp = tempfile.mkdtemp(prefix='probe_range_')
    try:
        path = os.path.join(tmp, 'synth.zarr')
        tmd.build_synthetic_zarr(path)
        n_views, hw = tmd.N_VIEWS, tmd.H
        ds = _slot_dataset(path, n_views, hw, (1, n_views))

        assert _pair('1,2') == (1, 2)
        assert _apply_range(ds, (1, 1)) == (1, 1) and ds.view_count_range == (1, 1)
        np.random.seed(0)
        assert {int(ds[i % len(ds)]['obs']['view_mask'][0].sum())
                for i in range(30)} == {1}, 'override to [1,1] did not take'

        assert _apply_range(ds, (1, n_views)) == (1, n_views)
        np.random.seed(0)
        seen = {int(ds[i % len(ds)]['obs']['view_mask'][0].sum()) for i in range(60)}
        assert seen == set(range(1, n_views + 1)), seen

        for bad in ((0, 1), (2, 1), (1, n_views + 1)):
            try:
                _apply_range(ds, bad)
            except ValueError:
                continue
            raise AssertionError(f'view_count_range {bad} was accepted')

        ds.view_count_range = (1, n_views)
        assert _apply_range(ds, None) == (1, n_views), 'None must be a no-op'
        print(f'  range override applied and gated (refuses (0,1), (2,1), (1,{n_views + 1}))')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_empty_metric_is_absent_not_nan():
    """An empty statistic must be `None` -- not NaN, and not 0.0.

    At a range whose draws are all N=1 every within-draw statistic is empty, and the old
    code returned `float(np.mean([]))`. NaN is invalid strict JSON -- Python writes the
    bare token `NaN`, which no strict parser accepts -- and read back as a number it is
    indistinguishable from a measurement. The tempting repair, returning 0.0, is worse:
    0.0 reads as PERFECT view-invariance, which is exactly the false confirmation this
    probe must never produce.
    """
    nan = float(np.mean([]))
    assert nan != nan, 'the old path is expected to yield NaN'
    assert 'NaN' in json.dumps(dict(x=nan)), 'and json writes it as invalid strict JSON'

    assert _mean_or_none([]) is None, 'an empty statistic must be absent'
    assert 'NaN' not in json.dumps(dict(x=_mean_or_none([])))
    assert _mean_or_none([0.0]) == 0.0, 'a MEASURED zero must survive as a zero'
    assert abs(_mean_or_none([1.0, 3.0]) - 2.0) < 1e-12
    print('  empty statistic -> null (not NaN, not 0.0); a measured 0.0 is preserved')


def test_rel_dist_is_scale_free():
    """`_rel_dist` must divide out scale, or a cross-cell comparison compares gains."""
    a, b = torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0, 0.0]])
    assert abs(_rel_dist(a, b) - (1.0 / 1.5)) < 1e-6, _rel_dist(a, b)
    # mutation: dropping the denominator reports 100x more at 100x the magnitude. The
    # whole point is that the statistic is comparable between models of different size.
    assert abs(_rel_dist(100 * a, 100 * b) - _rel_dist(a, b)) < 1e-6, 'not scale-free'
    assert abs(_rel_dist(a, a)) < 1e-6
    print('  _rel_dist is scale-free (identical at 1x and 100x magnitude)')


def test_slot_view_indices_recovers_views_exactly():
    """Slot -> ring index recovery must be exact, and must refuse an ambiguous match.

    A nearest-neighbour match would silently pair the wrong view with a latent, which
    understates every per-view statistic rather than failing -- the same failure shape as
    the dataset's own cam/slot pairing, and the reason this raises instead of guessing.
    """
    import test_multiview_dataset as tmd
    tmp = tempfile.mkdtemp(prefix='probe_views_')
    try:
        path = os.path.join(tmp, 'synth.zarr')
        tmd.build_synthetic_zarr(path)
        n_views, hw = tmd.N_VIEWS, tmd.H
        ds = _slot_dataset(path, n_views, hw, (1, n_views))
        # The synthetic fixture's cam_table repeats a row (views 0 and 2 share a pose),
        # which NO recovery-by-exact-match can disambiguate -- and raising there is the
        # correct behaviour, so the fixture is not wrong, it is simply degenerate for this
        # purpose. The real ring has 13 distinct rows (verified against the zarr), so give
        # the test a distinct table and let it exercise the recovery rather than the
        # fixture's geometry.
        tbl = np.zeros((n_views, 10), dtype=np.float32)
        tbl[:, 0] = np.arange(n_views)
        tbl[:, 3] = 1.0
        tbl[:, 7] = 45.0
        tbl[:, 8] = tbl[:, 9] = hw
        ds.cam_table = tbl
        np.random.seed(0)
        obs = ds[0]['obs']
        active = obs['view_mask'][0] > 0.5
        pairs = _slot_view_indices(ds, obs, active)
        slots = [s for s, _ in pairs]
        vals = [v for _, v in pairs]
        assert slots == list(range(int(active.sum()))), slots
        assert len(set(vals)) == len(vals), f'views must be distinct: {vals}'
        assert all(0 <= v < n_views for v in vals), vals

        # mutation: duplicate the row of a view that IS active, so two table rows match.
        # This is the degenerate-fixture case above, induced deliberately.
        v0 = vals[0]
        bad = ds.cam_table.copy()
        bad[v0] = bad[(v0 + 1) % n_views]
        ds.cam_table = bad
        try:
            _slot_view_indices(ds, obs, active)
        except RuntimeError:
            pass
        else:
            raise AssertionError('an ambiguous camera match was accepted')
        print(f'  slot->view recovery exact ({vals}); a duplicated table row is refused')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test():
    print('=' * 70)
    print('relpose probe -- CPU checks')
    print('=' * 70)
    test_relative_pose_convention()
    test_relative_pose_mutation_power()
    test_rot6d_round_trip()
    test_ridge_recovers_known_weights()
    test_report_target_detects_decodable_geometry()
    test_report_target_translation_only()
    test_mlp_probe_path_runs()
    test_range_override_is_applied()
    test_empty_metric_is_absent_not_nan()
    test_rel_dist_is_scale_free()
    test_slot_view_indices_recovers_views_exactly()
    print('=' * 70)
    print('ALL RELPOSE PROBE CHECKS PASSED')
    print('=' * 70)


if __name__ == '__main__':
    test()
