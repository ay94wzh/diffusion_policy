"""CPU-only test for stage RH/RHG's trajectory-conditioned spatial readout.

Verifies, in order:
  1. the architecture: R's pure-RGB layer3 trunk, 256->512 projection, two
     8-head 512-wide attentions, the ray MLP, the trajectory branch
     (13 -> 128 -> 1-layer transformer -> 512 MLP -> 512), the zero-init
     `query_proj` and the two zero-init query parameters -- ONE learned
     spatial query, not one per view;
  2. the trajectory feature contract: [p/L, rot6d, (p - p_t)/L, age], oldest
     first, p_t = the LAST window step, age -1..0;
  3. the query: q = q_learned + W_h h_t, exactly q_learned at init, and a
     full-window dependency once W_h moves;
  4. the geometric bias math: point-to-ray distance in metres, the
     -alpha*min(rho^2/2sigma^2, b_max) formula, and the validity gate
     (behind the camera / outside the crop -> zero);
  5. the per-head wiring: the bias acts on heads 1..4 only, and an
     end-to-end manual attention reproduces `_encode_views` exactly;
  6. output_shape (521,) for K=1 and K=11, train and eval, RNG-neutral;
  7. the drop-in contract (forward == cat(z_global, low-dims); B*To slicing);
  8. masking: inactive slots stay exactly zero and cannot affect the output;
  9. permutation invariance of the view tokens and the fusion;
 10. N=1 and slot-agnosticism;
 11. camera / trajectory validation: missing or malformed keys fail loudly;
 12. the activation order: the trajectory branch is bit-inert at init, then
     live after `query_proj` moves; the key/ray chain is R's;
 13. gradients reach every live path, in the documented order;
 14. the exact parameter count: RH == RHG == 6,207,680 (R + 922,752), and
     RH/RHG are construction-identical (the flag consumes no RNG);
 15. the 521/1042 width interface shared with the stock encoder;
 16. the real RH and RHG configs compose; RHG without a sigma refuses to
     instantiate.

Run from the repo root:
    python tests/test_traj_spatial_readout_obs_encoder.py
"""
import os
import sys

# repo convention: tests run from the repo root
this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(this_dir)
os.chdir(repo_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, this_dir)

import torch
import torch.nn.functional as F

from diffusion_policy.model.vision.traj_spatial_readout_obs_encoder import (
    TrajSpatialReadoutObsEncoder, TRAJ_STEP_DIM, TRAJ_STEP_FEAT_DIM,
    point_to_ray_distances, project_point_to_pixel,
)
from diffusion_policy.model.vision.ray_spatial_readout_obs_encoder import (
    RaySpatialReadoutObsEncoder,
)
from diffusion_policy.model.vision.spatial_readout_obs_encoder import (
    SpatialReadoutObsEncoder,
)
from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    ViewConditionedObsEncoder,
)
from diffusion_policy.model.vision.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)
from diffusion_policy.model.vision.plucker import (
    plucker_ray_map, plucker_ray_map_at)
from diffusion_policy.model.vision.model_getter import get_resnet

H = W = 84
CROP = 76
STRIDE = 16
GRID = 5
HIST_STEPS = 4
HIST_DIM = 11 * HIST_STEPS
RH_PARAMS = 6_207_680
R_PARAMS = 5_284_928
S_PARAMS = 5_016_640
TRAJ_DELTA = RH_PARAMS - R_PARAMS


def _shape_meta(k_slots):
    obs = {f'view_slot_{k:02d}_image': {'shape': [3, H, W], 'type': 'rgb'}
           for k in range(k_slots)}
    obs['robot0_eef_pos'] = {'shape': [3]}
    obs['robot0_eef_quat'] = {'shape': [4]}
    obs['robot0_gripper_qpos'] = {'shape': [2]}
    return {'obs': obs, 'action': {'shape': [10]}}


def _build_encoder(k_slots, **kw):
    kw.setdefault('crop_shape', [CROP, CROP])
    kw.setdefault('random_crop', True)
    kw.setdefault('use_group_norm', True)
    kw.setdefault('imagenet_norm', True)
    kw.setdefault('share_rgb_model', True)
    return TrajSpatialReadoutObsEncoder(
        shape_meta=_shape_meta(k_slots),
        rgb_model=get_resnet('resnet18', weights=None), **kw)


def _cam(n, g, seed=0, fovy=45.0):
    """A batch of valid camera vectors [pos(3), quat_wxyz(4), fovy, h, w]."""
    g = torch.Generator().manual_seed(seed) if g is None else g
    cam = torch.zeros(n, 10)
    cam[:, :3] = torch.randn(n, 3, generator=g) * 0.3
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    cam[:, 3:7] = q
    cam[:, 7] = fovy
    cam[:, 8] = H
    cam[:, 9] = W
    return cam


def _hist(n, seed=0):
    """A finite (n, H, 9) base-frame window (not a physically valid ring, but
    every entry is a real number -- the encoder treats it as features)."""
    g = torch.Generator().manual_seed(seed)
    hist = torch.randn(n, HIST_STEPS, TRAJ_STEP_DIM, generator=g) * 0.1
    hist[..., :3] += 0.4                      # positions around the scene
    hist[..., 3] = 1.0                        # rot6d of the identity
    hist[..., 7] = 1.0
    return hist


def _fake_obs(k_slots, n, seed=0, mask=None):
    """A batch shaped as the policy delivers it: leading dim is B*To."""
    g = torch.Generator().manual_seed(seed)
    obs = {}
    for k in range(k_slots):
        obs[f'view_slot_{k:02d}_image'] = torch.rand(n, 3, H, W, generator=g)
        obs[f'view_slot_{k:02d}_cam'] = _cam(n, g)
    obs['view_mask'] = torch.ones(n, k_slots) if mask is None else mask
    obs['view_eef_hist'] = torch.randn(n, k_slots, HIST_DIM, generator=g)
    obs['eef_hist_base'] = _hist(n, seed=seed + 7)
    obs['robot0_eef_pos'] = torch.randn(n, 3, generator=g)
    obs['robot0_eef_quat'] = torch.randn(n, 4, generator=g)
    obs['robot0_gripper_qpos'] = torch.randn(n, 2, generator=g)
    return obs


# ---------------------------------------------------------------------------
# 1. architecture
# ---------------------------------------------------------------------------
def test_architecture():
    enc = _build_encoder(5)
    assert tuple(enc.backbone.conv1.weight.shape) == (64, 3, 7, 7), (
        enc.backbone.conv1.weight.shape)
    for name in ('layer4', 'avgpool', 'fc'):
        assert not hasattr(enc.backbone, name), name
    assert enc.proj.in_channels == 256 and enc.proj.out_channels == 512
    assert tuple(enc.proj.kernel_size) == (1, 1)
    mhas = [m for m in enc.modules()
            if isinstance(m, torch.nn.MultiheadAttention)]
    # the two 512-wide attentions plus one per temporal-Transformer layer
    assert len(mhas) == 2 + len(enc.traj_encoder.layers), len(mhas)
    for m in (m for m in mhas if m.embed_dim == 512):
        assert m.num_heads == 8, (m.embed_dim, m.num_heads)
    # exactly ONE learnable spatial query and ONE fusion query
    assert tuple(enc.spatial_query.shape) == (1, 1, 512)
    assert tuple(enc.fusion_query.shape) == (1, 1, 512)
    assert bool((enc.spatial_query == 0).all())
    assert bool((enc.fusion_query == 0).all())
    assert enc.spatial_query is not enc.fusion_query

    # R's ray branch
    assert isinstance(enc.ray_mlp[0], torch.nn.Linear)
    assert enc.ray_mlp[0].in_features == 6 and enc.ray_mlp[0].out_features == 512
    assert isinstance(enc.ray_mlp[-1], torch.nn.Linear)
    assert bool((enc.ray_mlp[-1].weight == 0).all())
    assert bool((enc.ray_mlp[-1].bias == 0).all())
    assert enc.key_norm.normalized_shape == (512,)
    assert enc.value_norm.normalized_shape == (512,)

    # RH's trajectory branch
    assert enc.eef_hist_key == 'eef_hist_base'
    assert enc.eef_hist_steps == HIST_STEPS
    assert enc.traj_in.in_features == TRAJ_STEP_FEAT_DIM == 13
    assert enc.traj_in.out_features == 128
    assert tuple(enc.traj_pos.shape) == (1, HIST_STEPS, 128)
    assert len(enc.traj_encoder.layers) == 1
    assert enc.traj_encoder.layers[0].self_attn.num_heads == 4
    assert enc.traj_out[0].in_features == HIST_STEPS * 128
    assert enc.traj_out[-1].out_features == 512
    # W_h is zero-init (weight AND bias) so q_t == q_learned at init
    assert enc.query_proj.in_features == enc.query_proj.out_features == 512
    assert bool((enc.query_proj.weight == 0).all())
    assert bool((enc.query_proj.bias == 0).all())
    assert enc.token_grid == (GRID, GRID), enc.token_grid
    assert enc.total_stride == STRIDE, enc.total_stride

    # RHG consumes the SAME parameters; the flag adds none
    rh = _build_encoder(2)
    rhg = _build_encoder(2, use_geom_bias=True, geom_bias_sigma_m=0.05)
    assert (sum(p.numel() for p in rh.parameters())
            == sum(p.numel() for p in rhg.parameters()))
    assert rhg.use_geom_bias is True and rh.use_geom_bias is False
    assert rhg.geom_bias_heads == 4 and rhg.geom_bias_alpha == 1.0
    assert rhg.geom_bias_max == 4.0 and rhg.geom_bias_sigma_m == 0.05

    try:
        _build_encoder(2, share_rgb_model=False)
        raise AssertionError('share_rgb_model=False should raise')
    except NotImplementedError:
        pass
    # an uncalibrated sigma must refuse to build, not silently run
    try:
        _build_encoder(2, use_geom_bias=True)
        raise AssertionError('use_geom_bias without sigma should raise')
    except ValueError:
        pass
    for bad in ('cond_dim', 'use_plucker', 'use_eef_hist', 'resize_shape'):
        try:
            _build_encoder(2, **{bad: 1})
            raise AssertionError(f'{bad} should be rejected by RH')
        except TypeError:
            pass
    try:
        _build_encoder(2, use_geom_bias=True, geom_bias_sigma_m=0.05,
                       geom_bias_heads=9)
        raise AssertionError('more bias heads than heads should raise')
    except ValueError:
        pass
    print('architecture OK (R trunk + ray MLP + traj branch 13->128->512, '
          'one spatial query, zero-init W_h, RHG adds no parameter)')


# ---------------------------------------------------------------------------
# 1b. the copied ray placement is R's
# ---------------------------------------------------------------------------
def test_ray_convention():
    """`_feature_grid_rays` must equal the full Pluecker map sampled at the
    receptive-field centres of the SAME crop window -- the check that catches a
    drift in this file's copy of R's geometry."""
    enc = _build_encoder(1, random_crop=False)
    enc.eval()
    cam = _cam(3, None, seed=41)
    inds = torch.tensor([[4, 4], [0, 7], [5, 1]])
    rays = enc._feature_grid_rays(cam, inds, H, W)
    assert tuple(rays.shape) == (3, GRID * GRID, 6)
    full = plucker_ray_map(cam[:, :3], cam[:, 3:7], cam[:, 7], H, W)
    step = torch.arange(GRID) * STRIDE
    for r in range(3):
        want = full[r][:, inds[r, 0] + step, :][:, :, inds[r, 1] + step]
        want = want.reshape(6, GRID * GRID).transpose(0, 1).clone()
        want[:, 3:] = want[:, 3:] / enc.ray_scale
        assert torch.allclose(rays[r], want, atol=1e-6), r
    # unit directions; the moment is translation-carrying (not camera-frame 0)
    assert torch.allclose(rays[..., :3].norm(dim=-1), torch.ones(3, GRID * GRID),
                          atol=1e-5)
    assert rays[..., 3:].abs().max() > 0
    print('ray convention OK (_feature_grid_rays == the full map at the crop '
          'cells; unit d, non-zero world-frame moment)')


# ---------------------------------------------------------------------------
# 2. trajectory features
# ---------------------------------------------------------------------------
def test_traj_features():
    enc = _build_encoder(1)
    enc.eval()
    hist = _hist(3, seed=5)
    feat = enc._traj_step_features(hist)
    assert tuple(feat.shape) == (3, HIST_STEPS, 13), feat.shape
    L = enc.ray_scale
    assert torch.allclose(feat[..., :3], hist[..., :3] / L, atol=1e-6)
    assert torch.allclose(feat[..., 3:9], hist[..., 3:], atol=1e-6)
    # relative position is against the LAST step (the observation frame)
    p_t = hist[:, -1, :3] / L
    assert torch.allclose(
        feat[..., 9:12], hist[..., :3] / L - p_t[:, None, :], atol=1e-6)
    # age: oldest -1, current 0
    age = feat[..., 12]
    assert torch.allclose(age[0], torch.linspace(-1.0, 0.0, HIST_STEPS))
    assert age[:, -1].abs().max() == 0.0

    # the vector depends on EVERY step, including the oldest and the newest
    h = enc._traj_vector(hist)
    assert tuple(h.shape) == (3, 512)
    for step in (0, 1, 2, 3):
        h2 = hist.clone()
        h2[:, step, :3] += 0.05
        assert not torch.allclose(h, enc._traj_vector(h2)), step
    # ...and on the orientation, not only position
    h3 = hist.clone()
    h3[:, 1, 3] += 0.1
    assert not torch.allclose(h, enc._traj_vector(h3))
    # malformed windows fail loudly
    for bad in (torch.zeros(2, HIST_STEPS, 8), torch.zeros(2, HIST_STEPS + 1, 9)):
        try:
            enc._traj_vector(bad)
            raise AssertionError(f'bad window {tuple(bad.shape)} should raise')
        except ValueError:
            pass
    hbad = hist.clone()
    hbad[0, 1, 0] = float('nan')
    try:
        enc._traj_vector(hbad)
        raise AssertionError('non-finite window should raise')
    except ValueError:
        pass
    print('trajectory features OK ([p/L, rot6d, (p-p_t)/L, age]; all steps '
          'and orientations matter; malformed windows raise)')


# ---------------------------------------------------------------------------
# 3. query conditioning
# ---------------------------------------------------------------------------
def test_query():
    enc = _build_encoder(3)
    enc.eval()
    hist = _hist(2, seed=11)
    q0 = enc._query(hist)
    assert tuple(q0.shape) == (2, 1, 512)
    # at init W_h == 0, so q is exactly the learned query for ANY window
    assert torch.equal(q0, enc.spatial_query.expand(2, -1, -1))
    other = _hist(2, seed=99)
    assert torch.equal(enc._query(other), q0)

    # once W_h moves, the query is trajectory-dependent and no longer the
    # learned vector alone
    with torch.no_grad():
        enc.query_proj.weight.normal_(0.0, 0.02)
    q1, q2 = enc._query(hist), enc._query(other)
    assert not torch.allclose(q1, q2)
    assert not torch.allclose(q1, enc.spatial_query.expand(2, -1, -1))
    # one query per frame, shared by every view: the query has no slot axis
    assert q1.shape[1] == 1
    print('query OK (q = q_learned + W_h h_t exactly; inert at init, '
          'whole-window-dependent once W_h moves, one per frame)')


# ---------------------------------------------------------------------------
# 4. geometric bias math
# ---------------------------------------------------------------------------
def test_geom_bias_math():
    # a ray through the origin along -z: point on it -> 0, offset -> delta
    cam = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 45., 84., 84.]])
    u = torch.full((1, 1), 42.0)
    v = torch.full((1, 1), 42.0)
    rays = plucker_ray_map_at(
        cam[:, :3], cam[:, 3:7], cam[:, 7], 84, 84, u, v).transpose(1, 2)
    for delta in (0.0, 0.03, 0.1):
        p = torch.tensor([[delta, 0.0, -1.0]])
        assert torch.allclose(
            point_to_ray_distances(p, rays),
            torch.tensor([[delta]]), atol=1e-5), delta
    # ray_scale multiplies the stored moment back to metres
    rays_scaled = rays.clone()
    rays_scaled[..., 3:] /= 2.0
    p = torch.tensor([[0.1, 0.0, -1.0]])
    assert torch.allclose(
        point_to_ray_distances(p, rays_scaled, ray_scale=2.0),
        point_to_ray_distances(p, rays), atol=1e-6)

    enc = _build_encoder(2, use_geom_bias=True, geom_bias_sigma_m=0.05)
    enc.eval()
    # inside the crop: p at the centre of the view, in front
    cams = torch.tensor([
        [0., 0., 0., 1., 0., 0., 0., 45., 84., 84.],
        [0., 0., 0., 1., 0., 0., 0., 45., 84., 84.],
        [0., 0., 0., 1., 0., 0., 0., 45., 84., 84.],
    ])
    inds = torch.tensor([[4, 4], [4, 4], [4, 4]])
    ps = torch.tensor([
        [0.0, 0.0, -1.0],     # projects to the image centre -> valid
        [0.0, 0.0, 1.0],      # behind the camera -> gate off
        [0.5, 0.0, -1.0],     # projects outside the 84-px fov -> gate off
    ])
    # one ray grid per view: the token ray at the crop-cell centres,
    # row-major (i = k // 5, j = k % 5)
    uu = (torch.arange(GRID).repeat(GRID) * 16 + 4 + 0.5)[None]
    vv = (torch.arange(GRID).repeat_interleave(GRID) * 16 + 4 + 0.5)[None]
    full = torch.zeros(3, GRID * GRID, 6)
    for i in range(3):
        r = plucker_ray_map_at(
            cams[i:i + 1, :3], cams[i:i + 1, 3:7], cams[i:i + 1, 7],
            84, 84, uu, vv)
        full[i] = r.transpose(1, 2)[0]
    bias = enc._geom_bias(cams, inds, ps, full, 84, 84, (CROP, CROP))
    assert tuple(bias.shape) == (3, GRID * GRID)
    assert bool((bias[1] == 0).all()), 'behind-camera view must be unbiased'
    assert bool((bias[2] == 0).all()), 'outside-crop view must be unbiased'
    assert bool((bias[0] <= 0).all()), 'inside view must carry non-positive bias'
    assert int((bias[0] < -0.01).sum()) > GRID * GRID // 2, (
        'the bias must strongly suppress off-ray tokens')
    # formula, hand-computed on the valid view
    rho = point_to_ray_distances(ps[:1], full[:1])
    want = -1.0 * torch.clamp(rho ** 2 / (2 * 0.05 ** 2), max=4.0)
    assert torch.allclose(bias[:1], want, atol=1e-6)
    print('geometric bias OK (point-to-ray in metres, formula, validity gate '
          'for behind-camera and outside-crop views)')


# ---------------------------------------------------------------------------
# 5. per-head wiring
# ---------------------------------------------------------------------------
def _manual_readout(enc, x, rays, q, bias, n_bias_heads):
    """Reference attention on `intra_attn`'s own weights, head by head."""
    fmap = enc._backbone_feature_map(x)
    tokens = enc.proj(fmap).flatten(2).transpose(1, 2)
    e = enc.ray_mlp(rays) if enc.use_ray_encoding else torch.zeros_like(tokens)
    keys = enc.key_norm(tokens + e)
    values = enc.value_norm(tokens)
    D, Hn = enc.fused_dim, enc.n_heads
    Dh = D // Hn
    W, B = enc.intra_attn.in_proj_weight, enc.intra_attn.in_proj_bias
    qp = F.linear(q, W[:D], B[:D]).reshape(-1, 1, Hn, Dh).transpose(1, 2)
    kp = F.linear(keys, W[D:2 * D], B[D:2 * D]).reshape(
        -1, tokens.shape[1], Hn, Dh).transpose(1, 2)
    vp = F.linear(values, W[2 * D:], B[2 * D:]).reshape(
        -1, tokens.shape[1], Hn, Dh).transpose(1, 2)
    logits = qp @ kp.transpose(-1, -2) / (Dh ** 0.5)
    if bias is not None and n_bias_heads > 0:
        logits[:, :n_bias_heads] = logits[:, :n_bias_heads] + bias[:, None, None, :]
    att = torch.softmax(logits, dim=-1)
    out = (att @ vp).transpose(1, 2).reshape(-1, 1, D)
    return enc.intra_attn.out_proj(out)[:, 0]


def test_geom_bias_heads():
    enc = _build_encoder(3, use_geom_bias=True, geom_bias_sigma_m=0.05)
    enc.eval()
    obs = _fake_obs(3, 2, seed=3)
    x, rays, _ = enc._prepare_slot(
        obs['view_slot_00_image'], obs['view_slot_00_cam'], train=False)
    hist = obs['eef_hist_base']
    q = enc._query(hist)
    g = torch.Generator().manual_seed(31)
    bias = -torch.rand(x.shape[0], rays.shape[1], generator=g) * 2.0

    # no bias: the module call and the manual reference agree
    assert torch.allclose(enc._encode_views(x, rays, q),
                          _manual_readout(enc, x, rays, q, None, 0),
                          atol=1e-5)
    # the bias reaches exactly `geom_bias_heads` heads
    for heads in (0, 1, 4, 8):
        enc.geom_bias_heads = heads
        got = enc._encode_views(x, rays, q, bias=bias)
        want = _manual_readout(enc, x, rays, q, bias, heads)
        assert torch.allclose(got, want, atol=1e-5), heads
    enc.geom_bias_heads = 4

    # a huge negative bias on one token makes that token's value the readout:
    # an end-to-end proof that the sign and the softmax placement are right
    with torch.no_grad():
        enc.query_proj.weight.normal_(0.0, 0.02)
        enc.geom_bias_heads = 8
        pick = 12
        huge = torch.full((x.shape[0], rays.shape[1]), -1e4)
        huge[:, pick] = 0.0
        # the plan's b_max caps the REAL bias, so feed the huge one directly
        got = enc._encode_views(x, rays, q, bias=huge)
        # manual: softmax over a single surviving key -> v of that token
        fmap = enc._backbone_feature_map(x)
        tokens = enc.proj(fmap).flatten(2).transpose(1, 2)
        values = enc.value_norm(tokens)
        D = enc.fused_dim
        vp = F.linear(values[:, pick], enc.intra_attn.in_proj_weight[2 * D:],
                      enc.intra_attn.in_proj_bias[2 * D:]).reshape(-1, 1, D)
        want = enc.intra_attn.out_proj(vp)[:, 0]
        assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()
    print('per-head wiring OK (bias on heads 1..N only; manual attention '
          'matches; a dominating bias selects exactly that token)')


# ---------------------------------------------------------------------------
# 6. output_shape
# ---------------------------------------------------------------------------
def test_output_shape():
    for k in (1, 11):
        for geom in (False, True):
            kw = dict(use_geom_bias=True, geom_bias_sigma_m=0.05) if geom else {}
            enc = _build_encoder(k, **kw)
            before = torch.get_rng_state().clone()
            shape = enc.output_shape()
            after = torch.get_rng_state()
            assert tuple(shape) == (521,), shape
            assert torch.equal(before, after), 'output_shape consumed RNG'
            enc.train()
            assert tuple(enc.output_shape()) == (521,)
    print('output_shape OK ((521,) for K=1/11, RH/RHG, train/eval, RNG-neutral)')


# ---------------------------------------------------------------------------
# 7-10. contract, masking, permutation, N=1
# ---------------------------------------------------------------------------
def test_forward_contract():
    k, n = 3, 4
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, n, seed=7)
    with torch.no_grad():
        out = enc(obs)
        full = enc.forward_full(obs)
    assert tuple(out.shape) == (n, 521)
    low = torch.cat([obs['robot0_eef_pos'], obs['robot0_eef_quat'],
                     obs['robot0_gripper_qpos']], dim=-1)
    assert torch.equal(out, torch.cat([full['z_global'], low], dim=-1))

    # B*To flattening: per-frame rows equal the batched call
    B, To = 2, 2
    flat = _fake_obs(k, B * To, seed=9)
    with torch.no_grad():
        batched = enc(flat)
        rows = torch.cat([enc({kk: vv[i:i + 1] for kk, vv in flat.items()})
                          for i in range(B * To)], dim=0)
    assert torch.allclose(batched, rows, atol=1e-6)
    print('forward contract OK (z_global + low-dims = 521; B*To rows)')


def test_inactive_and_independence():
    k, n = 4, 3
    mask = torch.ones(n, k)
    mask[:, 2] = 0.0
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, n, seed=13)
    obs['view_mask'] = mask
    with torch.no_grad():
        a = enc.forward_full(obs)
    assert bool((a['z_views'][:, 2] == 0).all()), 'inactive row not exactly 0'
    obs2 = {kk: vv.clone() for kk, vv in obs.items()}
    obs2['view_slot_02_image'] = torch.rand_like(obs2['view_slot_02_image'])
    obs2['view_slot_02_cam'] = _cam(n, None, seed=99)
    with torch.no_grad():
        b = enc.forward_full(obs2)
    assert torch.equal(a['z_global'], b['z_global'])
    assert torch.equal(a['z_views'][:, :2], b['z_views'][:, :2])
    print('masking OK (inactive slots exactly zero and causally inert)')


def test_permutation():
    k, n = 3, 2
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, n, seed=17)
    perm = [2, 0, 1]
    obs2 = dict(obs)
    for name in list(obs):
        if not name.startswith('view_slot_'):
            continue
        slot = int(name.split('_')[2])
        suffix = name[len('view_slot_XX'):]
        obs2[f'view_slot_{slot:02d}{suffix}'] = \
            obs[f'view_slot_{perm[slot]:02d}{suffix}']
    with torch.no_grad():
        a = enc.forward_full(obs)
        b = enc.forward_full(obs2)
    assert torch.allclose(a['z_global'], b['z_global'], atol=1e-6)
    for dst, src in enumerate(perm):
        assert torch.allclose(a['z_views'][:, src], b['z_views'][:, dst],
                              atol=1e-6)
    print('permutation OK (slots have no identity; the fusion is invariant)')


def test_n1():
    k, n = 3, 2
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, n, seed=19)
    # identical content under every slot label: N=1 must be slot-agnostic
    for slot in range(1, k):
        obs[f'view_slot_{slot:02d}_image'] = obs['view_slot_00_image'].clone()
        obs[f'view_slot_{slot:02d}_cam'] = obs['view_slot_00_cam'].clone()
    outs = []
    for slot in range(k):
        mask = torch.zeros(n, k)
        mask[:, slot] = 1.0
        o = {kk: vv.clone() for kk, vv in obs.items()}
        o['view_mask'] = mask
        with torch.no_grad():
            full = enc.forward_full(o)
        outs.append(full['z_global'])
        # one unmasked key: fusion is exactly the query over that token
        token = full['z_views'][:, slot:slot + 1]
        z, _ = enc.fusion(enc.fusion_query.expand(n, 1, 512), token, token,
                          need_weights=False)
        assert torch.allclose(full['z_global'], z.reshape(n, 512), atol=1e-6)
    assert torch.allclose(outs[0], outs[1], atol=1e-6)
    assert torch.allclose(outs[0], outs[2], atol=1e-6)
    print('N=1 OK (single live slot, slot-agnostic)')


# ---------------------------------------------------------------------------
# 11. validation
# ---------------------------------------------------------------------------
def test_validation():
    k = 2
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, 2, seed=23)

    def expect(ob, exc=(KeyError, ValueError, RuntimeError)):
        try:
            enc(ob)
        except exc:
            return
        raise AssertionError('should have raised')

    o = dict(obs); del o['view_slot_01_cam']
    expect(o)
    o = dict(obs); del o['eef_hist_base']
    expect(o)
    o = dict(obs); del o['view_mask']
    expect(o)
    o = dict(obs); o['eef_hist_base'] = torch.zeros(2, HIST_STEPS, 8)
    expect(o)
    o = dict(obs); o['eef_hist_base'] = torch.zeros(3, HIST_STEPS, 9)
    expect(o)
    o = dict(obs); o['view_slot_00_image'] = torch.zeros(2, 3, 8, 8)
    expect(o)
    o = dict(obs)
    o['view_slot_00_cam'] = _cam(2, None, seed=1)
    o['view_slot_00_cam'][0, 3:7] = 0.0         # degenerate quaternion
    expect(o)
    o = dict(obs); o['view_mask'] = torch.full((2, k), 0.5)
    expect(o)
    print('validation OK (missing/malformed keys and degenerate poses raise)')


# ---------------------------------------------------------------------------
# 12. activation order
# ---------------------------------------------------------------------------
def test_traj_inert_then_live():
    k = 3
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, 2, seed=29)
    obs2 = {kk: vv.clone() for kk, vv in obs.items()}
    obs2['eef_hist_base'] = _hist(2, seed=77)

    with torch.no_grad():
        assert torch.equal(enc(obs), enc(obs2)), (
            'the trajectory branch must be bit-inert at initialisation')
        # the spatial query moves first (step 1), the trajectory branch still
        # has W_h == 0
        enc.spatial_query.normal_(0.0, 0.02)
        assert torch.equal(enc(obs), enc(obs2))
        # W_h moves on the next step's gradient (h != 0); now the window acts
        enc.query_proj.weight.normal_(0.0, 0.02)
        assert not torch.allclose(enc(obs), enc(obs2))
    print('activation order OK (trajectory inert at init -> live once W_h '
          'moves; the spatial-query step alone leaves it dead)')


def test_ray_branch_order():
    """R's chain still holds inside RH: zero query -> key path dead."""
    k = 2
    enc = _build_encoder(k)
    enc.eval()
    obs = _fake_obs(k, 2, seed=31)
    obs2 = {kk: vv.clone() for kk, vv in obs.items()}
    obs2['view_slot_00_cam'] = _cam(2, None, seed=33)
    with torch.no_grad():
        assert torch.equal(enc(obs), enc(obs2))          # q == 0 at init
        enc.spatial_query.normal_(0.0, 0.02)             # key path live...
        assert torch.equal(enc(obs), enc(obs2))          # ...but ray out == 0
        enc.ray_mlp[-1].weight.normal_(0.0, 0.02)        # ray out moves
        assert not torch.allclose(enc(obs), enc(obs2))
    print('ray chain OK (query -> ray output layer -> camera matters)')


# ---------------------------------------------------------------------------
# 13. gradients
# ---------------------------------------------------------------------------
def test_gradients():
    k = 3
    enc = _build_encoder(k)
    enc.train()
    obs = _fake_obs(k, 2, seed=37)

    def grads():
        enc.zero_grad(set_to_none=True)
        enc(obs).sum().backward()
        return {name: p.grad for name, p in (
            ('conv1', enc.backbone.conv1.weight),
            ('layer3', enc.backbone.layer3[0].conv1.weight),
            ('proj', enc.proj.weight),
            ('value_norm', enc.value_norm.weight),
            ('key_norm', enc.key_norm.weight),
            ('ray_mlp_out', enc.ray_mlp[-1].weight),
            ('ray_mlp_first', enc.ray_mlp[0].weight),
            ('traj_in', enc.traj_in.weight),
            ('traj_encoder', enc.traj_encoder.layers[0].self_attn.in_proj_weight),
            ('traj_out', enc.traj_out[-1].weight),
            ('traj_pos', enc.traj_pos),
            ('query_proj', enc.query_proj.weight),
            ('spatial_query', enc.spatial_query),
            ('intra_attn', enc.intra_attn.in_proj_weight),
            ('fusion', enc.fusion.in_proj_weight),
            ('fusion_query', enc.fusion_query),
        )}

    g = grads()
    for name in ('conv1', 'layer3', 'proj', 'value_norm', 'intra_attn',
                 'spatial_query', 'fusion', 'fusion_query', 'query_proj'):
        p = g[name]
        assert p is not None and bool(torch.isfinite(p).all()), name
        assert float(p.abs().sum()) > 0, name
    # q == 0 at init: key path dead, and with it the whole trajectory branch
    for name in ('key_norm', 'ray_mlp_out', 'ray_mlp_first',
                 'traj_in', 'traj_encoder', 'traj_out', 'traj_pos'):
        p = g[name]
        assert p is not None and bool((p == 0).all()), (
            f'{name} should be exactly zero at init')

    with torch.no_grad():
        enc.spatial_query.normal_(0.0, 0.02)
    g = grads()
    for name in ('key_norm', 'ray_mlp_out'):
        assert float(g[name].abs().sum()) > 0, name
    assert bool((g['ray_mlp_first'] == 0).all())
    # W_h still zero -> the trajectory encoder is still dead
    for name in ('traj_in', 'traj_encoder', 'traj_out', 'traj_pos'):
        assert bool((g[name] == 0).all()), name

    with torch.no_grad():
        enc.query_proj.weight.normal_(0.0, 0.02)
    g = grads()
    for name in ('traj_in', 'traj_encoder', 'traj_out', 'traj_pos',
                 'query_proj'):
        p = g[name]
        assert p is not None and bool(torch.isfinite(p).all()), name
        assert float(p.abs().sum()) > 0, name
    with torch.no_grad():
        enc.ray_mlp[-1].weight.normal_(0.0, 0.02)
    g = grads()
    assert float(g['ray_mlp_first'].abs().sum()) > 0
    print('gradients OK (live paths at init; key/ray chain and the trajectory '
          'branch activate in the documented order)')


# ---------------------------------------------------------------------------
# 14-15. parameter count and width
# ---------------------------------------------------------------------------
def test_parameter_count():
    rh = _build_encoder(11)
    rhg = _build_encoder(11, use_geom_bias=True, geom_bias_sigma_m=0.05)
    n_rh = sum(p.numel() for p in rh.parameters())
    n_rhg = sum(p.numel() for p in rhg.parameters())
    assert n_rh == n_rhg == RH_PARAMS, (n_rh, n_rhg)
    r = RaySpatialReadoutObsEncoder(
        shape_meta=_shape_meta(1),
        rgb_model=get_resnet('resnet18', weights=None),
        fused_dim=512, n_heads=8, ray_hidden_dim=512, ray_scale=1.0,
        use_ray_encoding=True, crop_shape=[CROP, CROP], random_crop=True,
        use_group_norm=True, share_rgb_model=True, imagenet_norm=True)
    n_r = sum(p.numel() for p in r.parameters())
    assert n_r == R_PARAMS, n_r
    s = SpatialReadoutObsEncoder(
        shape_meta=_shape_meta(1),
        rgb_model=get_resnet('resnet18', weights=None),
        fused_dim=512, n_heads=8, crop_shape=[CROP, CROP], random_crop=True,
        use_group_norm=True, share_rgb_model=True, imagenet_norm=True)
    n_s = sum(p.numel() for p in s.parameters())
    assert n_s == S_PARAMS, n_s
    # construction is bit-identical between RH and RHG (the flag is inert on
    # the RNG stream), so RHG - RH is a flag-only comparison
    torch.manual_seed(0)
    a = _build_encoder(2)
    torch.manual_seed(0)
    b = _build_encoder(2, use_geom_bias=True, geom_bias_sigma_m=0.05)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    print(f'parameter count OK (RH = RHG = {RH_PARAMS:,} = R {R_PARAMS:,} + '
          f'{TRAJ_DELTA:,}; S {S_PARAMS:,}; RH/RHG construction identical)')


def test_width_interface():
    stock = MultiImageObsEncoder(
        shape_meta=_shape_meta(1),
        rgb_model=get_resnet('resnet18', weights=None),
        resize_shape=None, crop_shape=[CROP, CROP], random_crop=True,
        use_group_norm=True, share_rgb_model=False, imagenet_norm=True)
    assert tuple(stock.output_shape()) == (521,)
    for k in (1, 11):
        enc = _build_encoder(k)
        assert tuple(enc.output_shape()) == (521,)
    assert 521 * 2 == 1042
    print('width interface OK (521 per frame, 1042 for To=2, same as stock)')


# ---------------------------------------------------------------------------
# 16. configs
# ---------------------------------------------------------------------------
def test_config():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        rh = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_enc_rh_latent')
        rhg = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_enc_rhg_latent')
    for cfg, name in ((rh, 'rh'), (rhg, 'rhg')):
        assert cfg.task.name == 'enc_rh_image_abs_multiview_az75'
        assert 'DiffusionUnetImagePolicy' in cfg.policy._target_
        assert 'LatentProbe' in cfg._target_
        assert sorted(cfg.task.dataset.view_pool) == list(range(1, 12))
        assert tuple(cfg.task.dataset.view_count_range) == (1, 11)
        assert cfg.task.env_runner.m3_slots == 11
        assert 'cam_key_image_runner' in cfg.task.env_runner._target_
        assert bool(cfg.task.dataset.emit_aux_action) is False
        assert bool(cfg.task.dataset.emit_eef_hist_base) is True
        assert bool(cfg.task.env_runner.serve_eef_hist_base) is True
        assert cfg.task.dataset.eef_hist_window == 'episode'
        assert cfg.task.dataset.eef_hist_steps == 4
        assert cfg.task.env_runner.eef_hist_steps == 4
        assert cfg.latent_probe.n_states == 128
        assert cfg.latent_probe.seed == 12345
        assert cfg.latent_probe.az0_view == 6
        obs_cfg = cfg.policy.obs_encoder
        assert obs_cfg._target_.endswith(
            'traj_spatial_readout_obs_encoder.TrajSpatialReadoutObsEncoder')
        assert obs_cfg.fused_dim == 512 and obs_cfg.n_heads == 8
        assert obs_cfg.ray_hidden_dim == 512
        assert float(obs_cfg.ray_scale) == 1.0
        assert bool(obs_cfg.use_ray_encoding) is True
        assert obs_cfg.eef_hist_key == 'eef_hist_base'
        assert obs_cfg.eef_hist_steps == 4
        assert obs_cfg.traj_hidden_dim == 128
        assert obs_cfg.traj_layers == 1
        assert obs_cfg.traj_heads == 4
        assert tuple(obs_cfg.crop_shape) == (76, 76)
        assert bool(obs_cfg.share_rgb_model) is True
        assert bool(obs_cfg.use_group_norm) is True
        assert bool(obs_cfg.imagenet_norm) is True
    assert bool(rh.policy.obs_encoder.use_geom_bias) is False
    assert 'geom_bias_sigma_m' not in rh.policy.obs_encoder
    assert bool(rhg.policy.obs_encoder.use_geom_bias) is True
    assert rhg.policy.obs_encoder.geom_bias_heads == 4
    assert float(rhg.policy.obs_encoder.geom_bias_alpha) == 1.0
    assert float(rhg.policy.obs_encoder.geom_bias_max) == 4.0
    assert rhg.policy.obs_encoder.geom_bias_sigma_m is None

    enc = hydra.utils.instantiate(rh.policy.obs_encoder)
    assert tuple(enc.output_shape()) == (521,)
    assert enc.n_slots == 11 and enc.token_grid == (GRID, GRID)
    assert enc.cam_keys == [f'view_slot_{k:02d}_cam' for k in range(11)]
    # RHG without the calibrated sigma must refuse to instantiate
    from hydra.errors import InstantiationException
    try:
        hydra.utils.instantiate(rhg.policy.obs_encoder)
        raise AssertionError('RHG config without sigma should not instantiate')
    except InstantiationException:
        pass
    encg = hydra.utils.instantiate(
        rhg.policy.obs_encoder, geom_bias_sigma_m=0.05)
    assert tuple(encg.output_shape()) == (521,)
    assert encg.geom_bias_sigma_m == 0.05
    OmegaConf.resolve(rh)
    assert rh.task.dataset.dataset_path == (
        'data/multiview/square_ph_ring13.zarr')
    print('config OK (RH/RHG share the task; sigma required for RHG; '
          'K=11 pool [1..11], probe 128/az0=6, 521 wide)')


def test():
    test_architecture()
    test_ray_convention()
    test_traj_features()
    test_query()
    test_geom_bias_math()
    test_geom_bias_heads()
    test_output_shape()
    test_forward_contract()
    test_inactive_and_independence()
    test_permutation()
    test_n1()
    test_validation()
    test_traj_inert_then_live()
    test_ray_branch_order()
    test_gradients()
    test_parameter_count()
    test_width_interface()
    test_config()
    print('ALL TRAJECTORY SPATIAL-READOUT ENCODER CHECKS PASSED')


if __name__ == '__main__':
    test()
