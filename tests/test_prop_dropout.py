"""Proprioception dropout: what it pins, and why each check can fail.

The intervention behind the `[1,3]` balance test (PROGRESS.md, *The second
failure mode*) is small enough to look obviously correct -- which is exactly the
kind of change this repository has been burned by (NOTES.md, "a check that
cannot fail proves nothing"). Five properties, each with the failure it catches:

1. `proprio_dropout=0.0` is bit-identical to the parent -- loss, every gradient,
   AND the global RNG state afterwards -- so every committed M3 number is
   untouched and the p=0 path draws nothing extra.
2. At p>0 exactly the three proprio keys change, all three drop the SAME
   samples, the image key is untouched, and the caller's batch is not mutated.
3. The dropped value is the normalizer's own image of 0.0: the neutral value *in
   the space the trunk consumes*. The mutation check shows a raw-space zeroing
   lands far from it, so the assertion has power.
4. `predict_action` is bit-identical with the dropout live in train mode. The
   diagnostics (`screen_conditioning.py`, `probe_relpose.py`) read the policy
   through that path, so the intervention cannot corrupt the measurement it
   exists to precede.
5. `eval()` switches it off -- pinned even though this workspace never evals
   `self.model` (it evals the EMA copy), because that is what a future caller
   would assume.
Plus: the suffix tuple is asserted EQUAL to screen_conditioning.py's, so the
answer to "which keys are proprioception" cannot drift between the two files.
"""
import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(this_dir)
os.chdir(repo_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, this_dir)

import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.policy.diffusion_unet_image_policy_propdrop import (
    DiffusionUnetImagePolicyPropDrop, PROPRIO_SUFFIX)
from screen_conditioning import PROPRIO_SUFFIX as SCREEN_PROPRIO_SUFFIX

H, W = 32, 32
T, B = 16, 8
PROP_DIMS = {'robot0_eef_pos': 3, 'robot0_eef_quat': 4, 'robot0_gripper_qpos': 2}
# the dropper writes the normalizer's image of 0.0, so the round trip bottoms
# out at float32 rounding (~1e-7 relative), not at 0 exactly
TOL = 1e-6


def _shape_meta():
    obs = {'agentview_image': {'shape': [3, H, W], 'type': 'rgb'}}
    for k, d in PROP_DIMS.items():
        obs[k] = {'shape': [d]}
    return {'obs': obs, 'action': {'shape': [10]}}


def _make_normalizer(shape_meta, rng):
    """Fitted on uniform(0,1), so each key's neutral value is ~0.5 -- far from
    the [0, 0.1] band `_batch` draws, which is what makes "was this sample
    dropped?" decidable without reading the mask."""
    nz = LinearNormalizer()
    nz['agentview_image'] = get_image_range_normalizer()
    for k, d in PROP_DIMS.items():
        nz[k] = SingleFieldLinearNormalizer.create_fit(
            torch.from_numpy(rng.uniform(0, 1, (512, d)).astype('float32')),
            mode='limits')
    nz['action'] = SingleFieldLinearNormalizer.create_fit(
        torch.from_numpy(rng.uniform(-1, 1, (512, 10)).astype('float32')),
        mode='limits')
    return nz


def _batch(rng, proprio_lo=0.0, proprio_hi=0.1, n=B):
    obs = {'agentview_image': torch.from_numpy(
        rng.uniform(0, 1, (n, T, 3, H, W)).astype('float32'))}
    for k, d in PROP_DIMS.items():
        obs[k] = torch.from_numpy(
            rng.uniform(proprio_lo, proprio_hi, (n, T, d)).astype('float32'))
    return {'obs': obs,
            'action': torch.from_numpy(
                rng.uniform(-1, 1, (n, T, 10)).astype('float32'))}


def _make_encoder(shape_meta):
    # `share_rgb_model=False` is not a preference: the stock encoder's
    # BatchNorm->GroupNorm replacement (multi_image_obs_encoder.py:61-69) is only
    # reachable inside the `not share_rgb_model` branch, so with sharing on, the
    # constructor's batch-1 `output_shape()` probe dies in BatchNorm. The dropout
    # under test is encoder-agnostic; this one is chosen so the policy builds.
    return MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet('resnet18', weights=None),
        crop_shape=None, random_crop=False, use_group_norm=True,
        share_rgb_model=False, imagenet_norm=True)


def _make_policy(cls, shape_meta, encoder, **over):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    sched = DDPMScheduler(
        num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
        beta_schedule='squaredcos_cap_v2', variance_type='fixed_small',
        clip_sample=True, prediction_type='epsilon')
    kwargs = dict(
        shape_meta=shape_meta, noise_scheduler=sched, obs_encoder=encoder,
        horizon=T, n_action_steps=8, n_obs_steps=2, num_inference_steps=100,
        obs_as_global_cond=True, diffusion_step_embed_dim=16,
        down_dims=(32, 64), kernel_size=5, n_groups=8,
        cond_predict_scale=True)
    kwargs.update(over)
    return cls(**kwargs)


def _pair(shape_meta, norm, p):
    """A parent and a propdrop policy sharing ONE encoder and ONE normalizer,
    with identical weights -- so every difference below is the dropout."""
    enc = _make_encoder(shape_meta)
    base = _make_policy(DiffusionUnetImagePolicy, shape_meta, enc)
    pd = _make_policy(DiffusionUnetImagePolicyPropDrop, shape_meta, enc,
                      proprio_dropout=p)
    for pol in (base, pd):
        pol.set_normalizer(norm)
    pd.load_state_dict(base.state_dict())      # no new params -> shapes match
    return base, pd


# ---------------------------------------------------------------------------
# 1. p=0 is the parent, exactly
# ---------------------------------------------------------------------------

def test_zero_is_the_parent():
    rng = np.random.default_rng(0)
    sm = _shape_meta()
    norm = _make_normalizer(sm, rng)
    base, pd = _pair(sm, norm, 0.0)
    base.train(), pd.train()
    batch = _batch(rng)

    torch.manual_seed(0)
    l_base = base.compute_loss(batch)
    rng_base = torch.get_rng_state()
    torch.manual_seed(0)
    l_pd = pd.compute_loss(batch)
    assert torch.equal(l_base, l_pd), (float(l_base), float(l_pd))
    # nothing extra was drawn on the p=0 path
    assert torch.equal(rng_base, torch.get_rng_state()), \
        'p=0 consumed RNG, so it cannot be the parent path'

    base.zero_grad(set_to_none=True)
    pd.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    base.compute_loss(batch).backward()
    torch.manual_seed(0)
    pd.compute_loss(batch).backward()
    n_grad = 0
    for (n1, p1), (n2, p2) in zip(base.named_parameters(), pd.named_parameters()):
        assert n1 == n2
        if p1.grad is None:
            assert p2.grad is None, n1
            continue
        assert torch.equal(p1.grad, p2.grad), n1
        n_grad += 1
    assert n_grad > 50, n_grad
    print(f'  p=0 is bit-identical to the parent: loss, all {n_grad} gradients, '
          f'and the RNG state after the call')


# ---------------------------------------------------------------------------
# 2/3. what p>0 does, and that the test could tell a wrong version apart
# ---------------------------------------------------------------------------

def test_dropout_semantics():
    rng = np.random.default_rng(1)
    sm = _shape_meta()
    norm = _make_normalizer(sm, rng)
    _, pd = _pair(sm, norm, 0.5)
    pd.train()
    batch = _batch(rng, proprio_lo=0.0, proprio_hi=0.1)
    untouched = {k: v.clone() for k, v in batch['obs'].items()}

    dropped = pd.drop_proprioception(batch)
    assert torch.equal(dropped['obs']['agentview_image'],
                       batch['obs']['agentview_image']), 'the image moved'
    for k, copy in untouched.items():
        assert torch.equal(batch['obs'][k], copy), \
            f'{k} was mutated in the caller batch'

    nn_orig = norm.normalize(batch['obs'])
    nn_drop = norm.normalize(dropped['obs'])
    masks = {}
    worst = 0.0
    for k in PROP_DIMS:
        # the neutral value is ~0.5, the data is in [0, 0.1]: a changed value is
        # a dropped sample, decidable without reading the mask
        m = (dropped['obs'][k] - batch['obs'][k]).abs().sum(dim=(1, 2)) > 0
        masks[k] = m
        assert torch.equal(nn_drop[k][~m], nn_orig[k][~m]), \
            f'{k}: kept samples were altered'
        worst = max(worst, float(nn_drop[k][m].abs().max()))
        assert float(nn_drop[k][m].abs().max()) <= TOL, \
            f'{k}: dropped samples are not at the neutral value'
    assert masks['robot0_eef_pos'].equal(masks['robot0_eef_quat']) and \
        masks['robot0_eef_pos'].equal(masks['robot0_gripper_qpos']), \
        'the three proprio keys dropped different samples'
    frac = float(masks['robot0_eef_pos'].float().mean())
    assert 0.25 <= frac <= 0.75, frac          # B=8, p=0.5 -- loose but real
    print(f'  p=0.5: the 3 proprio keys drop together and land at the neutral '
          f'value (max |normalized| {worst:.2e}); image and caller batch untouched')

    # the realized fraction, at a batch size where the binomial is tight
    big = _batch(rng, n=256)
    fracs = []
    for _ in range(4):
        pd.drop_proprioception(big)
        fracs.append(pd.last_prop_drop_frac)
    mean_frac = sum(fracs) / len(fracs)
    assert 0.45 <= mean_frac <= 0.55, fracs
    print(f'  realized drop fraction over 4 draws of n=256: '
          f'{["%.3f" % f for f in fracs]}')

    # MUTATION POWER: the raw-space variant -- zero the raw values instead of
    # finding the raw value that normalizes to 0 -- is a DIFFERENT intervention,
    # and this test's assertion has the margin to see it
    for k in PROP_DIMS:
        raw_zeroed = torch.zeros_like(batch['obs'][k])
        got = norm[k].normalize(raw_zeroed)        # per-key: a bare tensor
        assert float(got.abs().min()) > 0.3, (k, float(got.abs().min()))
    print('  raw-space zeroing would land ~0.9 from the neutral value, so the '
          'assertion above can fail')


# ---------------------------------------------------------------------------
# 4/5. the two paths that must NOT change
# ---------------------------------------------------------------------------

def test_inference_and_eval_are_untouched():
    rng = np.random.default_rng(2)
    sm = _shape_meta()
    norm = _make_normalizer(sm, rng)
    base, pd = _pair(sm, norm, 0.5)
    batch = _batch(rng)
    base.train(), pd.train()

    # predict_action, with the dropout live in train mode
    torch.manual_seed(0)
    a_base = base.predict_action(batch['obs'])['action']
    torch.manual_seed(0)
    a_pd = pd.predict_action(batch['obs'])['action']
    assert torch.equal(a_base, a_pd), \
        'the dropout leaked into the inference path the diagnostics use'
    print('  predict_action is bit-identical with dropout live (p=0.5, train '
          'mode) -- screen_conditioning.py and probe_relpose.py read this path')

    # eval() switches it off
    pd.eval()
    torch.manual_seed(0)
    l_eval = pd.compute_loss(batch)
    torch.manual_seed(0)
    l_base = base.compute_loss(batch)
    assert torch.equal(l_eval, l_base), (float(l_eval), float(l_base))
    assert pd.last_prop_drop_frac == 0.0
    print('  eval() switches it off: the loss is the parent\'s again')


# ---------------------------------------------------------------------------
# the suffix tuple, and the loud constructor
# ---------------------------------------------------------------------------

def test_suffixes_and_construction():
    assert PROPRIO_SUFFIX == SCREEN_PROPRIO_SUFFIX, \
        (PROPRIO_SUFFIX, SCREEN_PROPRIO_SUFFIX)
    print(f'  PROPRIO_SUFFIX == screen_conditioning.py\'s {PROPRIO_SUFFIX}')

    rng = np.random.default_rng(3)
    sm = _shape_meta()
    norm = _make_normalizer(sm, rng)
    enc = _make_encoder(sm)
    pd = _make_policy(DiffusionUnetImagePolicyPropDrop, sm, enc, proprio_dropout=0.5)
    # scope: the intervention is by SUFFIX, so a task config that grew another
    # low-dim key would leave it alone rather than silently dropping it
    assert pd.prop_keys == ['robot0_eef_pos', 'robot0_eef_quat',
                            'robot0_gripper_qpos'], pd.prop_keys
    print(f'  scoped by suffix to {pd.prop_keys}; any other low-dim key is '
          f'left alone')

    # a proprio key that is *renamed* (so the suffix no longer matches) leaves
    # only 2 matching keys, and the constructor refuses rather than dropping
    # whatever it happened to find
    renamed = {'obs': dict(sm['obs']), 'action': sm['action']}
    del renamed['obs']['robot0_gripper_qpos']
    renamed['obs']['robot0_gripper_width'] = {'shape': [2]}
    for bad, why in (
            ({'obs': {k: v for k, v in sm['obs'].items()
                      if k != 'robot0_eef_quat'}, 'action': sm['action']},
             'a missing proprio key'),
            (renamed, 'a renamed proprio key')):
        try:
            _make_policy(DiffusionUnetImagePolicyPropDrop, bad, enc,
                         proprio_dropout=0.5)
        except ValueError as e:
            print(f'  {why} is refused: {str(e)[:58]}...')
        else:
            raise AssertionError(f'{why} was accepted')
    try:
        _make_policy(DiffusionUnetImagePolicyPropDrop, sm, enc, proprio_dropout=1.5)
    except ValueError:
        print('  p=1.5 is refused')
    else:
        raise AssertionError('p=1.5 was accepted')


def test():
    test_zero_is_the_parent()
    test_dropout_semantics()
    test_inference_and_eval_are_untouched()
    test_suffixes_and_construction()
    print('ALL PROP-DROPOUT CHECKS PASSED')


if __name__ == '__main__':
    test()
