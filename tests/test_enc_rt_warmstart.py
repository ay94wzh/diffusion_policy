"""CPU-only test for the B1 warm-start loader and the pre-loop continuity hook.

No dataset, no checkpoint, no simulator. Covers:
  1. a baseline state dict loads into a model with an extra branch, with the
     missing set exactly the branch keys and no unexpected keys;
  2. an unexpected key fails loudly;
  3. a missing non-branch key fails loudly;
  4. model and EMA branch values are identical after the load, and every
     shared tensor equals the checkpoint;
  5. a mode-none model (no branch) has an empty missing set;
  6. the pre-loop hook restores `ema.optimization_step` from the checkpoint's
     `global_step`, so the first EMA update is not the step-0 `decay=0` copy;
  7. the hook resets torch/numpy/Python RNG to the same state on every call.

Run from the repo root:
    python tests/test_enc_rt_warmstart.py
"""
import copy
import os
import random
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(this_dir)
os.chdir(repo_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, this_dir)

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.workspace.train_diffusion_unet_image_workspace_enc_rt import (
    TrainDiffusionUnetImageWorkspaceEncRT, load_warmstart_state_dicts)


class _ToyPolicy(nn.Module):
    """Minimal stand-in with the real `obs_encoder.token_mixer.*` key layout."""

    def __init__(self, branch=True):
        super().__init__()
        self.obs_encoder = nn.Module()
        self.obs_encoder.base = nn.Linear(4, 4)
        if branch:
            self.obs_encoder.token_mixer = nn.Linear(4, 4)


def _payload(model, ema_model=None, extra=None, drop=None):
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if drop is not None:
        sd.pop(drop)
    if extra is not None:
        sd.update(extra)
    return {'model': sd, 'ema_model': copy.deepcopy(sd)}


def _expect_raises(fn, needle):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the test is about the message
        text = str(exc)
        assert needle in text, (needle, text)
        return
    raise AssertionError(f'expected an exception containing {needle!r}')


def test_load_ok():
    base = _ToyPolicy(branch=False)
    sd = _payload(base)
    model = _ToyPolicy(branch=True)
    ema = copy.deepcopy(model)
    report = load_warmstart_state_dicts(model, ema, sd)
    assert report['missing'] == ['obs_encoder.token_mixer.bias',
                                 'obs_encoder.token_mixer.weight']
    assert report['unexpected'] == []
    assert report['shared_model_tensors'] == 2
    assert report['shared_ema_tensors'] == 2
    for key in report['missing']:
        assert torch.equal(model.state_dict()[key], ema.state_dict()[key])
    assert torch.equal(model.obs_encoder.base.weight,
                       base.obs_encoder.base.weight)
    print('load OK (branch-only missing, no unexpected, model/EMA match)')


def test_unexpected_key():
    base = _ToyPolicy(branch=False)
    sd = _payload(base)
    sd['model']['obs_encoder.surprise.weight'] = torch.zeros(1)
    model = _ToyPolicy(branch=True)
    _expect_raises(
        lambda: load_warmstart_state_dicts(model, None, sd),
        'unexpected keys')
    print('unexpected key OK (fails loudly)')


def test_missing_non_branch():
    base = _ToyPolicy(branch=False)
    sd = _payload(base, drop='obs_encoder.base.weight')
    model = _ToyPolicy(branch=True)
    _expect_raises(
        lambda: load_warmstart_state_dicts(model, None, sd),
        'missing non-branch keys')
    print('missing non-branch OK (fails loudly)')


def test_mode_none():
    base = _ToyPolicy(branch=False)
    sd = _payload(base)
    model = _ToyPolicy(branch=False)
    report = load_warmstart_state_dicts(model, None, sd)
    assert report['missing'] == []
    assert report['unexpected'] == []
    print('mode none OK (empty missing set)')


def test_pre_loop_hook():
    # Build a tiny workspace shell; the hook only needs cfg + the saved step.
    ws = object.__new__(TrainDiffusionUnetImageWorkspaceEncRT)
    ws.cfg = OmegaConf.create(
        {'training': {'seed': 42, 'device': 'cpu'}})
    ws._warmstart_global_step = 12345

    model = _ToyPolicy(branch=False)
    ema = EMAModel(copy.deepcopy(model))
    assert ema.optimization_step == 0
    ws._reset_and_log_rng(ema=ema)
    assert ema.optimization_step == 12345, ema.optimization_step

    state1 = (torch.get_rng_state().clone(),
              np.random.get_state()[1].copy(),
              random.getstate())
    ws._reset_and_log_rng(ema=ema)
    state2 = (torch.get_rng_state(), np.random.get_state()[1], random.getstate())
    assert torch.equal(state1[0], state2[0])
    assert np.array_equal(state1[1], state2[1])
    assert state1[2] == state2[2]

    # With the restored step the first EMA update has late-training decay and
    # does not copy the new model over the loaded EMA.
    before = ema.averaged_model.obs_encoder.base.weight.detach().clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.step(model)
    assert ema.decay > 0.9, ema.decay
    after = ema.averaged_model.obs_encoder.base.weight.detach().clone()
    assert not torch.equal(after, model.obs_encoder.base.weight)
    assert torch.allclose(after, before * ema.decay + model.obs_encoder.base.weight * (1 - ema.decay))
    print(f'pre-loop hook OK (EMA step 12345, first decay {ema.decay:.6f}, RNG stable)')


def test():
    test_load_ok()
    test_unexpected_key()
    test_missing_non_branch()
    test_mode_none()
    test_pre_loop_hook()
    print('ALL ENC-RT WARM-START CHECKS PASSED')


if __name__ == '__main__':
    test()
