"""CPU-only test for stage LQ's LVSM-patchify + EEF-query joint transformer.

Verifies, in order:
  1. the architecture: 9 -> 256 8x8 patch embed, 9x9 = 81 tokens/view, four
     independently initialised Pre-LN layers (8 heads, FFN 1024), type/step
     embeddings, LN -> Linear(256 -> 128) output head, no slot-ID embedding,
     and the exact parameter count;
  2. the tokenizer: the 84 -> 72 crop, the per-pixel Pluecker map at the SAME
     crop window with the ORIGINAL intrinsics, the 9-channel order
     [RGB | d | m/L], and the patch-embed layout (patch (i, j) covers pixels
     [8i, 8i+8) x [8j, 8j+8), flattening in C,H,W order);
  3. the query feature contract [p/L, rot6d, (p-p_t)/L, age], oldest first,
     p_t = the LAST window step, age -1..0;
  4. the joint layout: X is slot-major (K, G), queries live at the fixed
     K*G slice, padding expands each dead slot G times, and only the query
     positions are read out;
  5. dead slots: never encoded (NaN image / zero camera are inert), their
     z_views rows are exactly zero, an all-dead row raises, and the mask must
     be binary;
  6. permutation invariance of the view slots and slot-agnosticism at N=1;
  7. N=1 and N=11 forward, output_shape (521,), train/eval, RNG neutrality;
  8. gradients reach every live path -- discarding the visual OUTPUTS must not
     detach the visual keys/values;
  9. the two registered controls: history_query=False keeps the parameters
     identical and changes the output; visual_reads_history=False makes the
     visual outputs independent of the history while the query outputs still
     read it;
 10. standard (non-zero) init; the four layers are not bit-identical; the
     query tokens are not identical;
 11. loud validation of malformed inputs and invalid constructor arguments;
 12. the real LQ task/workspace configs compose and instantiate, and the
     policy's global_cond_dim is 1042.

Run from the repo root:
    python tests/test_lvsm_eef_query_obs_encoder.py
"""
import copy
import math
import os
import sys

# repo convention: tests run from the repo root
this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(this_dir)
os.chdir(repo_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, this_dir)

import torch

from diffusion_policy.model.vision.lvsm_eef_query_obs_encoder import (
    LvsmEefQueryObsEncoder, EEF_STEP_DIM, QUERY_FEAT_DIM,
    PATCH_IN_CHANNELS, CONDITION_WIDTH)
from diffusion_policy.model.vision.plucker import plucker_ray_map_at

H = W = 84
CROP = 72
PATCH = 8
GRID = 9
TOKENS = GRID * GRID
HIST_STEPS = 4
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
FFN_DIM = 1024
OUT_DIM = 128
LQ_PARAMS = 3_412_096


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
    kw.setdefault('imagenet_norm', True)
    return LvsmEefQueryObsEncoder(shape_meta=_shape_meta(k_slots), **kw)


def _cam(n, g=None, seed=0, fovy=45.0):
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
    hist = torch.randn(n, HIST_STEPS, EEF_STEP_DIM, generator=g) * 0.1
    hist[..., :3] += 0.4                      # positions around the scene
    hist[..., 3] = 1.0                        # rot6d of the identity
    hist[..., 7] = 1.0
    return hist


def _fake_obs(k_slots, n, seed=0, mask=None, hist=None):
    """A batch shaped as the policy delivers it: leading dim is B*To."""
    g = torch.Generator().manual_seed(seed)
    obs = {}
    for k in range(k_slots):
        obs[f'view_slot_{k:02d}_image'] = torch.rand(n, 3, H, W, generator=g)
        obs[f'view_slot_{k:02d}_cam'] = _cam(n, g)
    obs['view_mask'] = (torch.ones(n, k_slots) if mask is None
                        else torch.as_tensor(mask, dtype=torch.float32))
    obs['view_eef_hist'] = torch.randn(
        n, k_slots, 11 * HIST_STEPS, generator=g)
    obs['eef_hist_base'] = _hist(n, seed=seed + 7) if hist is None else hist
    obs['robot0_eef_pos'] = torch.randn(n, 3, generator=g)
    obs['robot0_eef_quat'] = torch.randn(n, 4, generator=g)
    obs['robot0_gripper_qpos'] = torch.randn(n, 2, generator=g)
    return obs


# ---------------------------------------------------------------------------
# 1. architecture
# ---------------------------------------------------------------------------
def test_architecture():
    enc = _build_encoder(1)
    assert enc.token_grid == (GRID, GRID)
    assert enc.n_tokens == TOKENS and enc.n_tokens == 81
    assert enc.seq_visual == TOKENS
    assert enc.seq_len == TOKENS + HIST_STEPS
    assert enc.n_slots == 1
    assert isinstance(enc.patch_embed, torch.nn.Conv2d)
    assert enc.patch_embed.in_channels == PATCH_IN_CHANNELS == 9
    assert enc.patch_embed.out_channels == D_MODEL == 256
    assert enc.patch_embed.kernel_size == (PATCH, PATCH)
    assert enc.patch_embed.stride == (PATCH, PATCH)
    assert len(enc.layers) == N_LAYERS == 4
    for layer in enc.layers:
        assert isinstance(layer, torch.nn.TransformerEncoderLayer)
        # torch 1.12 exposes these through the submodules, not as attributes
        assert layer.self_attn.num_heads == N_HEADS
        assert layer.linear1.out_features == FFN_DIM
        assert layer.norm_first is True
        assert layer.dropout.p == 0.0
    embs = [m for m in enc.modules() if isinstance(m, torch.nn.Embedding)]
    assert sorted(m.num_embeddings for m in embs) == [2, HIST_STEPS]
    assert enc.type_emb.embedding_dim == D_MODEL
    assert enc.step_emb.embedding_dim == D_MODEL
    assert enc.out_proj.in_features == D_MODEL
    assert enc.out_proj.out_features == OUT_DIM == 128
    assert enc.history_query is True
    assert enc.visual_reads_history is True
    # no slot-ID embedding: the only nn.Embedding tables are type and step
    assert len(embs) == 2
    n = sum(p.numel() for p in enc.parameters())
    assert n == LQ_PARAMS, (n, LQ_PARAMS)
    print(f'architecture OK (G={TOKENS}/view, S={enc.seq_len}, '
          f'{n} params)')


# ---------------------------------------------------------------------------
# 2. tokenizer: crop, Pluecker map, patch layout
# ---------------------------------------------------------------------------
def test_tokenizer():
    enc = _build_encoder(1)
    enc.eval()                                   # centre crop, deterministic
    img = torch.rand(1, 3, H, W, generator=torch.Generator().manual_seed(3))
    cam = _cam(1, seed=4)

    x = enc._prepare_slot(img, cam)
    assert tuple(x.shape) == (1, 9, CROP, CROP)

    off = (H - CROP) // 2                        # 84 -> 72 centre crop
    assert off == 6
    rgb = img[:, :, off:off + CROP, off:off + CROP]
    mean = rgb.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = rgb.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(x[:, :3], (rgb - mean) / std, atol=1e-6)

    # the Pluecker map is evaluated at the crop's pixel centres with the
    # ORIGINAL 84x84 intrinsics
    jj = torch.arange(CROP, dtype=cam.dtype) + 0.5 + off
    ii = torch.arange(CROP, dtype=cam.dtype) + 0.5 + off
    u = jj[None, :].expand(CROP, CROP).reshape(1, -1)
    v = ii[:, None].expand(CROP, CROP).reshape(1, -1)
    rays = plucker_ray_map_at(
        cam[:, :3], cam[:, 3:7], cam[:, 7], H, W, u, v)
    rays = rays.reshape(1, 6, CROP, CROP)        # m/L with L=1
    assert torch.allclose(x[:, 3:], rays, atol=1e-6)
    # directions are unit length; the moment is the world-frame moment / L
    assert torch.allclose(
        x[:, 3:6].norm(dim=1), torch.ones(1, CROP, CROP), atol=1e-5)

    # patch embed is a per-patch linear map in C,H,W order
    fx = enc.patch_embed(x)
    assert tuple(fx.shape) == (1, D_MODEL, GRID, GRID)
    i, j = 2, 5
    patch = x[0, :, PATCH * i:PATCH * (i + 1),
              PATCH * j:PATCH * (j + 1)].reshape(-1)
    manual = enc.patch_embed.weight.reshape(D_MODEL, -1) @ patch
    manual = manual + enc.patch_embed.bias
    assert torch.allclose(fx[0, :, i, j], manual, atol=1e-5)
    toks = fx.flatten(2).transpose(1, 2)
    assert torch.allclose(toks[0, i * GRID + j], fx[0, :, i, j], atol=0)
    print('tokenizer OK (84->72 crop, per-pixel Pluecker, patch (i,j) '
          'covers [8i,8i+8)x[8j,8j+8))')


# ---------------------------------------------------------------------------
# 3. query features
# ---------------------------------------------------------------------------
def test_query_features():
    enc = _build_encoder(1)
    hist = torch.zeros(1, HIST_STEPS, EEF_STEP_DIM)
    px = torch.tensor([0.1, 0.2, 0.3, 0.4])
    hist[0, :, 0] = px
    hist[0, :, 1] = 0.5
    hist[0, :, 3] = 1.0
    hist[0, :, 7] = 1.0
    feat = enc._query_features(hist)
    assert tuple(feat.shape) == (1, HIST_STEPS, QUERY_FEAT_DIM)
    assert torch.allclose(feat[0, :, 0], px)
    assert torch.allclose(feat[0, :, 3:9], hist[0, :, 3:9])
    p_t = hist[0, -1, :3]
    assert torch.allclose(feat[0, :, 9:12], hist[0, :, :3] - p_t)
    assert torch.allclose(
        feat[0, :, 12], torch.tensor([-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0]))
    # the LAST row is the current frame, so its relative term is exactly 0
    assert torch.equal(feat[0, -1, 9:12], torch.zeros(3))
    print('query features OK ([p/L, rot6d, (p-p_t)/L, age], oldest->current)')


# ---------------------------------------------------------------------------
# 4. layout and forward contract
# ---------------------------------------------------------------------------
def test_layout_and_contract():
    enc = _build_encoder(3)
    enc.eval()                      # deterministic centre crop across calls
    obs = _fake_obs(3, 2, seed=5, mask=[[1, 0, 1], [0, 1, 0]])
    out = enc(obs)
    assert tuple(out.shape) == (2, 521)
    full = enc.forward_full(obs)
    assert tuple(full['z_global'].shape) == (2, 512)
    assert tuple(full['z_views'].shape) == (2, 3, D_MODEL)
    assert tuple(full['query_tokens'].shape) == (2, HIST_STEPS, D_MODEL)
    assert tuple(full['z_visual_out'].shape) == (2, 3, D_MODEL)
    assert tuple(full['condition_512'].shape) == (2, 512)
    assert torch.equal(full['z_global'], full['condition_512'])
    low = torch.cat([obs['robot0_eef_pos'], obs['robot0_eef_quat'],
                     obs['robot0_gripper_qpos']], dim=-1)
    assert torch.allclose(out, torch.cat([full['z_global'], low], dim=-1),
                          atol=0, rtol=0)
    # the readout is exactly out_proj(ln_out(query positions))
    manual = enc.out_proj(enc.ln_out(full['query_tokens'])).reshape(2, -1)
    assert torch.allclose(manual, full['condition_512'], atol=1e-6)
    # dead slots have exactly zero probe rows
    assert torch.count_nonzero(full['z_views'][0, 1]) == 0
    assert torch.count_nonzero(full['z_views'][1, 0]) == 0
    assert torch.count_nonzero(full['z_views'][1, 2]) == 0
    print('layout/contract OK (X slot-major, queries at K*G, 521 out)')


# ---------------------------------------------------------------------------
# 5. dead slots are inert
# ---------------------------------------------------------------------------
def test_dead_slots_inert():
    enc = _build_encoder(2)
    enc.eval()                      # deterministic centre crop across calls
    mask = [[1, 0], [0, 1]]
    obs = _fake_obs(2, 2, seed=6, mask=mask)
    obs2 = copy.deepcopy(obs)
    # poison the DEAD rows only: NaN image and an all-zero (invalid) camera
    obs2['view_slot_01_image'][0] = float('nan')
    obs2['view_slot_01_cam'][0] = 0.0
    obs2['view_slot_00_image'][1] = float('nan')
    obs2['view_slot_00_cam'][1] = 0.0
    with torch.no_grad():
        a = enc.forward_full(obs)
        b = enc.forward_full(obs2)
    for key in ('z_global', 'z_views', 'query_tokens', 'z_visual_out'):
        assert torch.allclose(a[key], b[key], atol=0, rtol=0), key
    assert torch.count_nonzero(a['z_views'][0, 1]) == 0
    assert torch.count_nonzero(a['z_views'][1, 0]) == 0
    try:
        enc.forward_full(_fake_obs(2, 1, mask=[[0, 0]]))
        raise AssertionError('all-dead row should raise')
    except ValueError:
        pass
    bad = _fake_obs(2, 1, mask=[[1, 0]])
    bad['view_mask'] = torch.tensor([[0.5, 0.5]])
    try:
        enc.forward_full(bad)
        raise AssertionError('non-binary mask should raise')
    except ValueError:
        pass
    print('dead slots OK (never encoded, zero probe rows, loud gates)')


# ---------------------------------------------------------------------------
# 6. permutation invariance / slot-agnosticism
# ---------------------------------------------------------------------------
def test_permutation():
    enc = _build_encoder(4)
    enc.eval()
    obs = _fake_obs(4, 2, seed=7)
    perm = [2, 0, 3, 1]
    obs_p = dict(obs)
    for k in range(4):
        obs_p[f'view_slot_{k:02d}_image'] = obs[f'view_slot_{perm[k]:02d}_image']
        obs_p[f'view_slot_{k:02d}_cam'] = obs[f'view_slot_{perm[k]:02d}_cam']
    with torch.no_grad():
        a = enc.forward_full(obs)
        b = enc.forward_full(obs_p)
    assert torch.allclose(a['condition_512'], b['condition_512'],
                          atol=1e-4, rtol=1e-4)
    assert torch.allclose(a['query_tokens'], b['query_tokens'],
                          atol=1e-4, rtol=1e-4)

    # N=1: the live slot's position carries no identity
    enc1 = _build_encoder(3)
    enc1.eval()
    base = _fake_obs(3, 2, seed=8, mask=[[1, 0, 0], [1, 0, 0]])
    moved = _fake_obs(3, 2, seed=8, mask=[[0, 0, 1], [0, 1, 0]])
    for n in range(2):
        for k in range(3):
            moved[f'view_slot_{k:02d}_image'][n] = base['view_slot_00_image'][n]
            moved[f'view_slot_{k:02d}_cam'][n] = base['view_slot_00_cam'][n]
    with torch.no_grad():
        c = enc1.forward_full(base)
        d = enc1.forward_full(moved)
    assert torch.allclose(c['condition_512'], d['condition_512'],
                          atol=1e-4, rtol=1e-4)
    print('permutation OK (slot order and N=1 position carry no identity)')


# ---------------------------------------------------------------------------
# 7. N=1 / N=11, output_shape, RNG neutrality
# ---------------------------------------------------------------------------
def test_output_shape_and_n():
    for k in (1, 11):
        enc = _build_encoder(k)
        assert tuple(enc.output_shape()) == (521,)
        obs = _fake_obs(k, 1, seed=9)
        out = enc(obs)
        assert tuple(out.shape) == (1, 521)
        enc.eval()
        with torch.no_grad():
            assert tuple(enc(obs).shape) == (1, 521)
        assert bool(torch.isfinite(out).all())
        enc.train()
        state = torch.get_rng_state().clone()
        enc.output_shape()
        assert torch.equal(state, torch.get_rng_state())
        assert enc.training is True
    print('output_shape OK ((521,) for K=1/11, eval/no-grad, RNG-neutral)')


# ---------------------------------------------------------------------------
# 8. gradients
# ---------------------------------------------------------------------------
def test_gradients():
    enc = _build_encoder(2)
    enc.train()
    obs = _fake_obs(2, 2, seed=10, mask=[[1, 0], [0, 1]])
    full = enc.forward_full(obs)
    loss = full['condition_512'].pow(2).mean()
    loss.backward()
    named = dict(enc.named_parameters())
    checks = {
        'patch_embed.weight': 'visual tokenizer',
        'query_mlp.0.weight': 'query MLP',
        'query_mlp.2.weight': 'query MLP out',
        'step_emb.weight': 'step embedding',
        'type_emb.weight': 'type embedding',
        'layers.0.self_attn.in_proj_weight': 'joint attention',
        'layers.0.linear1.weight': 'joint FFN',
        'out_proj.weight': 'output projection',
    }
    for name, what in checks.items():
        p = named[name]
        assert p.grad is not None, f'{name} ({what}) has no gradient'
        assert bool(torch.isfinite(p.grad).all()), f'{name} grad not finite'
        assert float(p.grad.abs().sum()) > 0.0, f'{name} ({what}) grad is zero'
    for name, p in named.items():
        if p.grad is not None:
            assert bool(torch.isfinite(p.grad).all()), f'{name} grad not finite'
    print('gradients OK (visual keys/values stay attached after discarding '
          'visual outputs)')


# ---------------------------------------------------------------------------
# 9. controls
# ---------------------------------------------------------------------------
def test_controls():
    def build(seed, **kw):
        torch.manual_seed(seed)
        return _build_encoder(1, **kw)

    base = build(0)
    const = build(0, history_query=False)
    blocked = build(0, visual_reads_history=False)
    base.eval(); const.eval(); blocked.eval()   # deterministic centre crop
    sd = base.state_dict()
    for other in (const, blocked):
        for key, val in other.state_dict().items():
            assert torch.equal(val, sd[key]), key

    obs = _fake_obs(1, 2, seed=11)
    with torch.no_grad():
        a = base.forward_full(obs)['condition_512']
        b = const.forward_full(obs)['condition_512']
    assert not torch.allclose(a, b), 'constant-query control must move output'

    # the feedback control: visual outputs stop depending on the history,
    # while the query outputs still read it directly
    obs2 = copy.deepcopy(obs)
    obs2['eef_hist_base'] = _hist(2, seed=99)
    assert not torch.allclose(obs['eef_hist_base'], obs2['eef_hist_base'])
    with torch.no_grad():
        f1 = base.forward_full(obs)
        f2 = base.forward_full(obs2)
        b1 = blocked.forward_full(obs)
        b2 = blocked.forward_full(obs2)
    assert not torch.allclose(f1['z_visual_out'], f2['z_visual_out'])
    assert torch.allclose(b1['z_visual_out'], b2['z_visual_out'],
                          atol=0, rtol=0)
    assert not torch.allclose(b1['query_tokens'], b2['query_tokens'])
    assert not torch.allclose(b1['condition_512'], b2['condition_512'])
    print('controls OK (history vs constant; visual<-history block isolates '
          'the feedback path)')


# ---------------------------------------------------------------------------
# 10. init
# ---------------------------------------------------------------------------
def test_init():
    enc = _build_encoder(1)
    for name in ('patch_embed.weight', 'query_mlp.0.weight',
                 'type_emb.weight', 'step_emb.weight'):
        p = dict(enc.named_parameters())[name]
        assert torch.count_nonzero(p) > 0, f'{name} is zero-initialised'
    w0 = enc.layers[0].self_attn.in_proj_weight
    w1 = enc.layers[1].self_attn.in_proj_weight
    assert not torch.equal(w0, w1), 'cloned layers share initial weights'
    enc.eval()
    with torch.no_grad():
        full = enc.forward_full(_fake_obs(1, 1, seed=12))
    q = full['query_tokens'][0]
    for i in range(HIST_STEPS - 1):
        assert not torch.allclose(q[i], q[i + 1]), (
            'query tokens are identical at init; step embeddings must '
            'differentiate them')
    print('init OK (non-zero branches, independent layers, distinct queries)')


# ---------------------------------------------------------------------------
# 11. validation
# ---------------------------------------------------------------------------
def test_validation():
    enc = _build_encoder(2)
    obs = _fake_obs(2, 1, seed=13)
    for key in ('view_slot_00_image', 'view_slot_01_cam', 'view_mask',
                'eef_hist_base'):
        bad = dict(obs)
        bad.pop(key)
        try:
            enc.forward_full(bad)
            raise AssertionError(f'missing {key} should raise')
        except KeyError:
            pass
    bad = copy.deepcopy(obs)
    bad['eef_hist_base'] = bad['eef_hist_base'][:, :3]
    try:
        enc.forward_full(bad)
        raise AssertionError('wrong history length should raise')
    except ValueError:
        pass
    bad = copy.deepcopy(obs)
    bad['eef_hist_base'][0, 0, 0] = float('nan')
    try:
        enc.forward_full(bad)
        raise AssertionError('non-finite history should raise')
    except ValueError:
        pass
    bad = copy.deepcopy(obs)
    bad['view_slot_00_cam'][0, 8] = 80.0
    try:
        enc.forward_full(bad)
        raise AssertionError('camera h/w mismatch should raise')
    except RuntimeError:
        pass

    # constructor gates
    for kw in (dict(out_token_dim=64), dict(patch_size=5),
               dict(crop_shape=(76, 76)), dict(d_model=255),
               dict(eef_hist_steps=3)):
        try:
            _build_encoder(1, **kw)
            raise AssertionError(f'invalid constructor args should raise: {kw}')
        except (ValueError, NotImplementedError):
            pass
    print('validation OK (missing/malformed keys and bad args fail loudly)')


# ---------------------------------------------------------------------------
# 12. configs
# ---------------------------------------------------------------------------
def test_config():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_enc_lq_latent')
    assert cfg.task.name == 'enc_lq_image_abs_multiview_az75'
    assert 'DiffusionUnetImagePolicy' in cfg.policy._target_
    assert 'LQProbe' in cfg._target_
    assert sorted(cfg.task.dataset.view_pool) == list(range(1, 12))
    assert tuple(cfg.task.dataset.view_count_range) == (1, 11)
    assert cfg.task.env_runner.m3_slots == 11
    assert 'cam_key_image_runner' in cfg.task.env_runner._target_
    assert bool(cfg.task.dataset.emit_eef_hist_base) is True
    assert bool(cfg.task.env_runner.serve_eef_hist_base) is True
    assert cfg.task.dataset.eef_hist_window == 'episode'
    assert cfg.task.dataset.eef_hist_steps == 4
    assert cfg.latent_probe.n_states == 128
    assert cfg.latent_probe.az0_view == 6
    obs_cfg = cfg.policy.obs_encoder
    assert obs_cfg._target_.endswith(
        'lvsm_eef_query_obs_encoder.LvsmEefQueryObsEncoder')
    assert obs_cfg.d_model == 256 and obs_cfg.n_heads == 8
    assert obs_cfg.n_layers == 4 and obs_cfg.ffn_dim == 1024
    assert obs_cfg.out_token_dim == 128
    assert obs_cfg.patch_size == 8
    assert tuple(obs_cfg.crop_shape) == (72, 72)
    assert bool(obs_cfg.random_crop) is True
    assert bool(obs_cfg.imagenet_norm) is True
    assert float(obs_cfg.ray_scale) == 1.0
    assert obs_cfg.eef_hist_key == 'eef_hist_base'
    assert obs_cfg.eef_hist_steps == 4
    assert bool(obs_cfg.history_query) is True
    assert bool(obs_cfg.visual_reads_history) is True
    assert 'rgb_model' not in obs_cfg
    enc = hydra.utils.instantiate(obs_cfg)
    assert tuple(enc.output_shape()) == (521,)
    assert enc.n_slots == 11 and enc.token_grid == (GRID, GRID)
    assert enc.cam_keys == [f'view_slot_{k:02d}_cam' for k in range(11)]
    policy = hydra.utils.instantiate(cfg.policy)
    # the policy does not keep global_cond_dim as an attribute; it is
    # obs_feature_dim * n_obs_steps = 521 * 2 = 1042
    assert policy.obs_feature_dim == 521
    assert policy.obs_feature_dim * policy.n_obs_steps == 1042
    print('config OK (LQ workspace composes; 521 per frame, 1042 global_cond)')


def test_probe_extras():
    """The LQ probe subclass persists query tokens and post-transformer
    visual means through the base workspace's _probe_extras seam."""
    from diffusion_policy.workspace.train_diffusion_unet_image_workspace_lqprobe import (
        TrainDiffusionUnetImageWorkspaceLQProbe)

    S, K = 4, 3

    def out(k):
        return {'query_tokens': torch.randn(S, HIST_STEPS, D_MODEL),
                'z_visual_out': torch.randn(S, k, D_MODEL),
                'z_views': torch.randn(S, k, D_MODEL),
                'z_global': torch.randn(S, CONDITION_WIDTH),
                'view_active': torch.ones(S, k, dtype=torch.bool)}

    obj = TrainDiffusionUnetImageWorkspaceLQProbe.__new__(
        TrainDiffusionUnetImageWorkspaceLQProbe)
    arrays, stats = obj._probe_extras(out(K), out(1), {})
    assert set(arrays) == {'qt', 'qtn1', 'zvout', 'zvout_n1'}
    assert arrays['qt'].shape == (S, HIST_STEPS, D_MODEL)
    assert arrays['qtn1'].shape == (S, HIST_STEPS, D_MODEL)
    assert arrays['zvout'].shape == (S, K, D_MODEL)
    assert arrays['zvout_n1'].shape == (S, 1, D_MODEL)
    for k, v in stats.items():
        assert math.isfinite(v), (k, v)
    print('LQ probe extras OK (qt/qtn1/zvout/zvout_n1 saved, finite stats)')


def test():
    test_architecture()
    test_tokenizer()
    test_query_features()
    test_layout_and_contract()
    test_dead_slots_inert()
    test_permutation()
    test_output_shape_and_n()
    test_gradients()
    test_controls()
    test_init()
    test_validation()
    test_probe_extras()
    test_config()
    print('ALL LVSM EEF-QUERY ENCODER CHECKS PASSED')


if __name__ == '__main__':
    test()
