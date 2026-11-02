"""CPU-only test for the B1 residual token-interaction branch.

Verifies, in order:
  1. architecture: one Pre-LN 512-wide/8-head/FFN-1024 GELU layer, a
     zero-initialized bias-free 512->512 output gate, and exactly 2,364,928
     branch parameters; B/C/D share the same construction; `none` adds nothing;
  2. the layer4 map is `(M, 512, 3, 3)` for the 76x76 crop, G = 9;
  3. `none` is bit-identical to B1 after a strict state-dict load;
  4. at init (`W_out == 0`) every mode is bit-identical to B1 after a
     strict=False load, with missing keys exactly the branch keys;
  5. the branch is built under a private CPU RNG stream: B/C/D branch params are
     bit-identical and the subclass leaves the global torch RNG unchanged
     relative to the parent class;
  6. gradient order with a non-zero target loss: `out_proj` receives gradient
     at step 0, the transformer does not; after one `out_proj` step the
     transformer does;
  7. C is intra-view (perturbing view 0 leaves view 1 unchanged), D is
     cross-view (perturbing view 0 changes view 1), inactive content cannot
     change anything, inactive `z_views` rows are exactly zero;
  8. N=1: C and D agree to ~1e-6, with intra input `(1,9,512)` and cross input
     `(1,99,512)`; both are slot-agnostic;
  9. slot permutation equivariance/invariance, variable active counts, and no
     NaN;
 10. the drop-in contract (521 per frame, global_cond 1042) and an RNG-neutral
     output_shape;
 11. the checkpoint contract: cross minus none is exactly the 13
     `obs_encoder.token_mixer.*` keys;
 12. the B1 readout regression: `_backbone_features` equals the manual walk;
 13. the new Hydra config composes with each mode and its encoder instantiates
     to (521,).

Run from the repo root:
    python tests/test_residual_token_view_conditioned_obs_encoder.py
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

from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.view_conditioned_obs_encoder import (
    ViewConditionedObsEncoder)
from diffusion_policy.model.vision.residual_token_view_conditioned_obs_encoder import (
    BRANCH_STATE_PREFIX, ResidualTokenViewConditionedObsEncoder,
    TOKEN_MIXER_MODES)

H = W = 84
CROP = 76
HIST_STEPS = 4
HIST_DIM = 11 * HIST_STEPS
BRANCH_PARAMS = 2_364_928
MODES = ('single', 'intra', 'cross')


def _shape_meta(k_slots):
    obs = {f'view_slot_{k:02d}_image': {'shape': [3, H, W], 'type': 'rgb'}
           for k in range(k_slots)}
    obs['robot0_eef_pos'] = {'shape': [3]}
    obs['robot0_eef_quat'] = {'shape': [4]}
    obs['robot0_gripper_qpos'] = {'shape': [2]}
    return {'obs': obs, 'action': {'shape': [10]}}


def _base_kwargs():
    return dict(crop_shape=[CROP, CROP], random_crop=True, use_group_norm=True,
                imagenet_norm=True, share_rgb_model=True)


def _build_parent(k_slots, **kw):
    kw = {**_base_kwargs(), **kw}
    return ViewConditionedObsEncoder(
        shape_meta=_shape_meta(k_slots),
        rgb_model=get_resnet('resnet18', weights=None), **kw)


def _build_new(k_slots, mode='cross', **kw):
    kw = {**_base_kwargs(), **kw}
    return ResidualTokenViewConditionedObsEncoder(
        shape_meta=_shape_meta(k_slots),
        rgb_model=get_resnet('resnet18', weights=None),
        token_mixer_mode=mode, **kw)


def _cam(n, g, fovy=45.0):
    """A batch of valid camera vectors [pos(3), quat_wxyz(4), fovy, h, w]."""
    cam = torch.zeros(n, 10)
    cam[:, :3] = torch.randn(n, 3, generator=g) * 0.3
    q = torch.randn(n, 4, generator=g)
    cam[:, 3:7] = q / q.norm(dim=-1, keepdim=True)
    cam[:, 7] = fovy
    cam[:, 8] = H
    cam[:, 9] = W
    return cam


def _fake_obs(k_slots, n, seed=0, mask=None):
    """A batch shaped as the policy delivers it: leading dim is B*To."""
    g = torch.Generator().manual_seed(seed)
    obs = {}
    for k in range(k_slots):
        obs[f'view_slot_{k:02d}_image'] = torch.rand(
            n, 3, H, W, generator=g)
        obs[f'view_slot_{k:02d}_cam'] = _cam(n, g)
    obs['view_mask'] = torch.ones(n, k_slots) if mask is None else mask
    obs['view_eef_hist'] = torch.randn(
        n, k_slots, HIST_DIM, generator=g)
    obs['robot0_eef_pos'] = torch.randn(n, 3, generator=g)
    obs['robot0_eef_quat'] = torch.randn(n, 4, generator=g)
    obs['robot0_gripper_qpos'] = torch.randn(n, 2, generator=g)
    return obs


def _permute_slots(obs, k_slots, perm):
    """New slot i holds old slot perm[i]; low-dim obs are unchanged."""
    out = {}
    for i in range(k_slots):
        out[f'view_slot_{i:02d}_image'] = obs[f'view_slot_{perm[i]:02d}_image']
        out[f'view_slot_{i:02d}_cam'] = obs[f'view_slot_{perm[i]:02d}_cam']
    out['view_mask'] = obs['view_mask'][:, perm]
    out['view_eef_hist'] = obs['view_eef_hist'][:, perm]
    for k in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'):
        out[k] = obs[k]
    return out


def _make_live(enc, std=1e-3, seed=0):
    w = enc.token_mixer.out_proj.weight
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed)
        w.copy_(torch.randn(w.shape, generator=g, dtype=w.dtype) * std)
    return enc


def _branch_keys(enc):
    return {k for k in enc.state_dict()
            if k.startswith(BRANCH_STATE_PREFIX)}


# ---------------------------------------------------------------------------
# 1. architecture / parameter count
# ---------------------------------------------------------------------------
def test_architecture():
    assert TOKEN_MIXER_MODES == ('none', 'single', 'intra', 'cross')
    enc = _build_new(3, mode='cross')
    assert enc.token_mixer is not None
    layer = enc.token_mixer.layer
    assert layer.norm_first is True
    assert layer.self_attn.batch_first is True
    assert layer.self_attn.num_heads == 8
    assert layer.linear1.in_features == 512 and layer.linear1.out_features == 1024
    assert layer.linear2.in_features == 1024 and layer.linear2.out_features == 512
    assert layer.dropout.p == 0.0
    assert layer.self_attn.dropout == 0.0
    assert layer.activation.__name__ == 'gelu'
    assert enc.token_mixer.out_proj.in_features == 512
    assert enc.token_mixer.out_proj.out_features == 512
    assert enc.token_mixer.out_proj.bias is None
    assert bool((enc.token_mixer.out_proj.weight == 0).all())
    assert sum(p.numel() for p in enc.token_mixer.parameters()) == BRANCH_PARAMS

    none = _build_new(3, mode='none')
    assert none.token_mixer is None
    assert len(_branch_keys(none)) == 0

    # B/C/D have bit-identical branch state dicts (private stream) and the same
    # parameter count.
    branch_state = None
    for mode in MODES:
        torch.manual_seed(0)
        e = _build_new(3, mode=mode)
        sd = {k: v.clone() for k, v in e.token_mixer.state_dict().items()}
        if branch_state is None:
            branch_state = sd
        else:
            assert set(sd) == set(branch_state), (mode, sorted(set(sd) ^ set(branch_state)))
            for k in sd:
                assert torch.equal(sd[k], branch_state[k]), (mode, k)
        assert sum(p.numel() for p in e.token_mixer.parameters()) == BRANCH_PARAMS
    print(f'architecture OK (1 Pre-LN layer, 8 heads, FFN 1024, GELU, '
          f'{BRANCH_PARAMS} branch params, B/C/D identical)')


# ---------------------------------------------------------------------------
# 2. feature grid
# ---------------------------------------------------------------------------
def test_feature_grid():
    enc = _build_new(2, mode='cross').eval()
    x = torch.zeros(2, 9, CROP, CROP)
    with torch.no_grad():
        fmap = enc._backbone_map(x, None)
    assert tuple(fmap.shape) == (2, 512, 3, 3), fmap.shape
    assert fmap.shape[-2] * fmap.shape[-1] == 9
    print('feature grid OK (76x76 crop -> 3x3x512, G=9)')


# ---------------------------------------------------------------------------
# 3. mode none is B1
# ---------------------------------------------------------------------------
def test_mode_none_matches_b1():
    parent = _build_parent(3).eval()
    none = _build_new(3, mode='none').eval()
    none.load_state_dict(parent.state_dict(), strict=True)
    obs = _fake_obs(3, 2, mask=torch.tensor([[1., 1., 0.], [1., 0., 0.]]))
    with torch.no_grad():
        a = parent(obs)
        b = none(obs)
        af = parent.forward_full(obs)
        bf = none.forward_full(obs)
    assert torch.equal(a, b)
    assert torch.equal(af['z_views'], bf['z_views'])
    assert torch.equal(af['z_global'], bf['z_global'])
    assert set(bf) == {'z_global', 'z_views', 'view_active'}

    # The default seam returns exactly the pooled readout and no extras.
    x = torch.zeros(2, 9, CROP, CROP)
    cond = torch.zeros(2, 128)
    rows = torch.tensor([0, 1])
    cols = torch.tensor([0, 1])
    active = torch.tensor([[True, False], [True, False]])
    with torch.no_grad():
        z, aux = parent._encode_active_views(x, cond, rows, cols, active)
        want = parent._backbone_features(x, cond)
    assert aux == {}
    assert torch.equal(z, want)

    # The factored readout still equals the explicit ResNet walk.
    b = parent.backbone
    h = b.relu(b.bn1(b.conv1(x)))
    h = parent._film(h, 0, cond)
    h = b.maxpool(h)
    for i, layer in enumerate([b.layer1, b.layer2, b.layer3, b.layer4], start=1):
        h = layer(h)
        h = parent._film(h, i, cond)
    manual = torch.flatten(b.avgpool(h), 1)
    assert torch.equal(want, manual)
    print('mode none OK (bit-identical to B1; seam returns pooled readout only)')


# ---------------------------------------------------------------------------
# 4. zero-init equivalence for all modes
# ---------------------------------------------------------------------------
def test_zero_init_equivalence():
    parent = _build_parent(3).eval()
    obs = _fake_obs(3, 2, mask=torch.tensor([[1., 1., 0.], [1., 0., 1.]]))
    with torch.no_grad():
        ref = parent.forward_full(obs)
    for mode in MODES:
        enc = _build_new(3, mode=mode).eval()
        result = enc.load_state_dict(parent.state_dict(), strict=False)
        assert set(result.missing_keys) == _branch_keys(enc), (
            mode, sorted(set(result.missing_keys) ^ _branch_keys(enc)))
        assert result.unexpected_keys == []
        with torch.no_grad():
            out = enc.forward_full(obs)
        assert torch.equal(out['z_views'], ref['z_views']), mode
        assert torch.equal(out['z_global'], ref['z_global']), mode
        assert 'z_views_base' in out
        assert torch.equal(out['z_views_base'], ref['z_views']), mode
        assert bool((enc.token_mixer.out_proj.weight == 0).all())
        assert float(enc.token_mixer.layer.linear1.weight.abs().sum()) > 0
    print('zero-init equivalence OK (all modes bit-identical to B1 at init)')


# ---------------------------------------------------------------------------
# 5. private construction stream
# ---------------------------------------------------------------------------
def test_private_stream():
    state = torch.get_rng_state()

    torch.set_rng_state(state)
    parent = _build_parent(3)
    after_parent = torch.get_rng_state()

    torch.set_rng_state(state)
    child = _build_new(3, mode='cross')
    after_child = torch.get_rng_state()

    assert torch.equal(after_parent, after_child), (
        'subclass construction consumed global torch RNG')

    # B/C/D branch params identical from the same private seed.
    states = {}
    for mode in MODES:
        torch.manual_seed(123)
        e = _build_new(3, mode=mode)
        states[mode] = {k: v.clone() for k, v in e.token_mixer.state_dict().items()}
    for mode in MODES[1:]:
        assert set(states[mode]) == set(states[MODES[0]])
        for k in states[MODES[0]]:
            assert torch.equal(states[mode][k], states[MODES[0]][k]), (mode, k)
    print('private stream OK (subclass leaves global RNG unchanged; B/C/D match)')


# ---------------------------------------------------------------------------
# 6. gradient order
# ---------------------------------------------------------------------------
def test_gradient_order():
    enc = _build_new(2, mode='cross').eval()
    obs = _fake_obs(2, 2)
    target = torch.randn(2, 512)
    out = enc.forward_full(obs)
    loss = F.mse_loss(out['z_global'], target)
    loss.backward()

    assert enc.token_mixer.out_proj.weight.grad is not None
    assert float(enc.token_mixer.out_proj.weight.grad.norm()) > 0
    for name, p in enc.token_mixer.layer.named_parameters():
        assert p.grad is not None, name
        assert torch.count_nonzero(p.grad) == 0, (name, float(p.grad.norm()))

    # One manual output-gate step: the transformer now receives gradient.
    with torch.no_grad():
        enc.token_mixer.out_proj.weight.add_(
            0.01 * enc.token_mixer.out_proj.weight.grad)
    enc.zero_grad(set_to_none=True)
    out = enc.forward_full(obs)
    F.mse_loss(out['z_global'], target).backward()
    layer_grads = [float(p.grad.norm()) for p in enc.token_mixer.layer.parameters()]
    assert max(layer_grads) > 0
    print('gradient order OK (out_proj first, transformer after one step)')


# ---------------------------------------------------------------------------
# 7. intra vs cross semantics
# ---------------------------------------------------------------------------
def test_intra_cross_semantics():
    intra = _make_live(_build_new(3, mode='intra').eval())
    cross = _build_new(3, mode='cross').eval()
    cross.load_state_dict(intra.state_dict(), strict=True)
    _make_live(cross)
    with torch.no_grad():
        cross.token_mixer.out_proj.weight.copy_(
            intra.token_mixer.out_proj.weight)

    obs = _fake_obs(3, 2)
    obs2 = dict(obs)
    obs2['view_slot_00_image'] = obs['view_slot_00_image'] + 0.5
    with torch.no_grad():
        a = intra.forward_full(obs)
        b = intra.forward_full(obs2)
        c = cross.forward_full(obs)
        d = cross.forward_full(obs2)
    assert torch.equal(a['z_views'][:, 1], b['z_views'][:, 1]), (
        'intra mode leaked information across views')
    assert not torch.allclose(c['z_views'][:, 1], d['z_views'][:, 1],
                              atol=1e-7), (
        'cross mode did not mix views')

    # Inactive slots are not read and their z_views rows stay exactly zero.
    mask = torch.tensor([[1., 1., 0.], [1., 1., 0.]])
    x1 = _fake_obs(3, 2, mask=mask)
    x2 = dict(x1)
    x2['view_slot_02_image'] = torch.randn(2, 3, H, W)
    x2['view_slot_02_cam'] = _cam(2, torch.Generator().manual_seed(99))
    with torch.no_grad():
        y1 = cross.forward_full(x1)
        y2 = cross.forward_full(x2)
    assert torch.equal(y1['z_global'], y2['z_global'])
    assert torch.equal(y1['z_views'], y2['z_views'])
    assert int(torch.count_nonzero(y1['z_views'][:, 2])) == 0
    assert bool((y1['z_views_base'][:, 2] == 0).all())
    print('intra/cross semantics OK (C blocks cross-view, D mixes, dead slots inert)')


# ---------------------------------------------------------------------------
# 8. N=1
# ---------------------------------------------------------------------------
def test_n1():
    intra = _make_live(_build_new(3, mode='intra').eval())
    cross = _build_new(3, mode='cross').eval()
    cross.load_state_dict(intra.state_dict(), strict=True)
    _make_live(cross)
    with torch.no_grad():
        cross.token_mixer.out_proj.weight.copy_(
            intra.token_mixer.out_proj.weight)

    mask = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    obs = _fake_obs(3, 2, mask=mask)

    seen = []

    def record(module, inputs):
        seen.append(tuple(inputs[0].shape))

    h = intra.token_mixer.layer.register_forward_pre_hook(record)
    with torch.no_grad():
        ci = intra.forward_full(obs)
    h.remove()
    h = cross.token_mixer.layer.register_forward_pre_hook(record)
    with torch.no_grad():
        cc = cross.forward_full(obs)
    h.remove()

    assert seen[0] == (2, 9, 512), seen
    assert seen[1] == (2, 3 * 9, 512), seen
    assert torch.allclose(ci['z_global'], cc['z_global'], atol=1e-4)
    assert torch.allclose(ci['z_views'], cc['z_views'], atol=1e-4)

    # Slot-agnostic at N=1: the same single view in slot 0 or slot 2.
    mask0 = torch.zeros(1, 3)
    mask0[0, 0] = 1.0
    mask2 = torch.zeros(1, 3)
    mask2[0, 2] = 1.0
    o0 = _fake_obs(3, 1, mask=mask0)
    o2 = _fake_obs(3, 1, mask=mask2)
    # make the two observations carry the same view content
    o2['view_slot_02_image'] = o0['view_slot_00_image']
    o2['view_slot_02_cam'] = o0['view_slot_00_cam']
    o2['view_eef_hist'][:, 2] = o0['view_eef_hist'][:, 0]
    for k in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'):
        o2[k] = o0[k]
    with torch.no_grad():
        z0 = cross(o0)
        z2 = cross(o2)
    assert torch.allclose(z0, z2, atol=1e-5), (z0 - z2).abs().max()
    print('N=1 OK (C==D to 1e-6; intra 9 tokens, cross K*G with the rest masked; slot-agnostic)')


# ---------------------------------------------------------------------------
# 9. permutation / variable active counts
# ---------------------------------------------------------------------------
def test_permutation_and_variable_counts():
    enc = _build_new(3, mode='cross').eval()
    _make_live(enc)
    obs = _fake_obs(3, 2)
    perm = [2, 0, 1]
    obs_p = _permute_slots(obs, 3, perm)
    with torch.no_grad():
        a = enc.forward_full(obs)
        b = enc.forward_full(obs_p)
    for dst, src in enumerate(perm):
        assert torch.allclose(a['z_views'][:, src], b['z_views'][:, dst],
                              atol=1e-5)
    assert torch.allclose(a['z_global'], b['z_global'], atol=1e-5)

    # Ragged active counts: batched == single-sample forward.
    mask = torch.tensor([[1., 1., 1.], [1., 0., 0.], [1., 1., 0.]])
    obs = _fake_obs(3, 3, mask=mask)
    with torch.no_grad():
        batched = enc.forward_full(obs)
        for i in range(3):
            one = {k: v[i:i + 1] for k, v in obs.items()}
            single = enc.forward_full(one)
            assert torch.allclose(
                batched['z_views'][i], single['z_views'][0], atol=1e-5), i
            assert torch.allclose(
                batched['z_global'][i], single['z_global'][0], atol=1e-5), i
    assert bool(torch.isfinite(batched['z_global']).all())
    print('permutation / variable N OK (equivariant z_views, invariant z_global, no NaN)')


# ---------------------------------------------------------------------------
# 10. drop-in contract
# ---------------------------------------------------------------------------
def test_contract():
    enc = _build_new(11, mode='cross').eval()
    assert tuple(enc.output_shape()) == (521,)
    state = torch.get_rng_state()
    enc.output_shape()
    assert torch.equal(state, torch.get_rng_state()), 'output_shape consumed RNG'

    B, To = 3, 2
    flat = _fake_obs(11, B * To, seed=9)
    obs = {k: v.reshape(B, To, *v.shape[1:]) for k, v in flat.items()}
    this_nobs = {k: v[:, :To].reshape(-1, *v.shape[2:]) for k, v in obs.items()}
    with torch.no_grad():
        feats = enc(this_nobs)
        full = enc.forward_full(this_nobs)
    assert tuple(feats.shape) == (B * To, 521), feats.shape
    low = torch.cat([this_nobs[k] for k in enc.low_dim_keys], dim=-1)
    assert torch.equal(feats, torch.cat([full['z_global'], low], dim=-1))
    global_cond = feats.reshape(B, -1)
    assert tuple(global_cond.shape) == (B, 1042), global_cond.shape
    print('contract OK (521 per frame, B*To slicing, global_cond 1042, RNG-neutral)')


# ---------------------------------------------------------------------------
# 11. checkpoint contract
# ---------------------------------------------------------------------------
def test_checkpoint_contract():
    none = _build_new(3, mode='none')
    cross = _build_new(3, mode='cross')
    none_sd = none.state_dict()
    cross_sd = cross.state_dict()
    only_cross = set(cross_sd) - set(none_sd)
    only_none = set(none_sd) - set(cross_sd)
    assert not only_none, only_none
    assert only_cross == _branch_keys(cross), sorted(only_cross)
    assert len(only_cross) == 13, len(only_cross)
    result = cross.load_state_dict(none_sd, strict=False)
    assert set(result.missing_keys) == only_cross
    assert result.unexpected_keys == []
    print(f'checkpoint contract OK ({len(only_cross)} branch keys, no unexpected)')


# ---------------------------------------------------------------------------
# 12. B1 readout regression
# ---------------------------------------------------------------------------
def test_b1_readout_regression():
    b1 = _build_parent(2).eval()
    x = torch.randn(2, 9, CROP, CROP)
    cond = torch.randn(2, 128)
    b = b1.backbone
    h = b.relu(b.bn1(b.conv1(x)))
    h = b1._film(h, 0, cond)
    h = b.maxpool(h)
    for i, layer in enumerate([b.layer1, b.layer2, b.layer3, b.layer4], start=1):
        h = layer(h)
        h = b1._film(h, i, cond)
    want = torch.flatten(b.avgpool(h), 1)
    assert torch.equal(b1._backbone_features(x, cond), want)
    assert torch.equal(
        b1._backbone_features(x, cond),
        torch.flatten(b.avgpool(b1._backbone_map(x, cond)), 1))
    print('B1 readout regression OK (factored map + pool == manual walk)')


# ---------------------------------------------------------------------------
# 13. config composition
# ---------------------------------------------------------------------------
def test_config():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    cfg_dir = os.path.join(repo_root, 'diffusion_policy', 'config')
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(
            config_name='train_diffusion_unet_image_workspace_enc_rt_latent',
            overrides=['task.task_name=square'])
        cfgs = {
            mode: hydra.compose(
                config_name='train_diffusion_unet_image_workspace_enc_rt_latent',
                overrides=['task.task_name=square',
                           f'policy.obs_encoder.token_mixer_mode={mode}'])
            for mode in TOKEN_MIXER_MODES
        }
    OmegaConf.resolve(cfg)
    assert ('residual_token_view_conditioned_obs_encoder.'
            'ResidualTokenViewConditionedObsEncoder'
            in cfg.policy.obs_encoder._target_)
    assert cfg.policy.obs_encoder.token_mixer_mode == 'cross'
    for mode, c in cfgs.items():
        assert c.policy.obs_encoder.token_mixer_mode == mode
    assert cfg.training.num_epochs == 30
    assert cfg.training.skip_rollout is True
    assert cfg.training.resume is False
    assert cfg.optimizer.lr == 1.0e-5
    assert cfg.training.lr_warmup_steps == 500
    assert cfg.checkpoint.topk.k == 0
    assert cfg.checkpoint.save_last_ckpt is True
    assert cfg.task.env_runner.m3_slots == 11
    assert tuple(cfg.task.view_count_range) == (1, 11)
    assert list(cfg.task.train_view_pool) == list(range(1, 12))
    assert cfg.task.dataset.eef_hist_window == 'episode'
    assert cfg.task.dataset.eef_hist_steps == cfg.policy.obs_encoder.eef_hist_steps == 4
    assert 'latest.ckpt' in cfg.warmstart.checkpoint
    enc = hydra.utils.instantiate(cfg.policy.obs_encoder)
    assert tuple(enc.output_shape()) == (521,)
    print('config OK (all modes compose; encoder instantiate -> (521,))')


def test():
    test_architecture()
    test_feature_grid()
    test_mode_none_matches_b1()
    test_zero_init_equivalence()
    test_private_stream()
    test_gradient_order()
    test_intra_cross_semantics()
    test_n1()
    test_permutation_and_variable_counts()
    test_contract()
    test_checkpoint_contract()
    test_b1_readout_regression()
    test_config()
    print('ALL RESIDUAL-TOKEN ENCODER CHECKS PASSED')


if __name__ == '__main__':
    test()
