"""Warm-start continuation for the B1 residual-token screen (A/B/C/D).

This workspace is the latent-probe workspace plus one explicit job: copy the
B1 model **and** B1 EMA weights into a fresh run, then continue for a fixed
short budget with a fresh optimizer/scheduler. It is the training half of
ENCODER_B1_RESIDUAL_TOKEN_PLAN.md section 5.1.

Why the load is strict=False
----------------------------
The new encoder adds `obs_encoder.token_mixer.*` parameters, so the B1 state
dict is missing exactly those keys. Everything else must load exactly: the
helper uses the `_IncompatibleKeys` returned by `load_state_dict` (the
normalizer's `_load_from_state_dict` swallows its own keys, so a naive key-set
difference is wrong), asserts no unexpected keys, asserts the missing set is
exactly the branch keys, and compares every shared tensor against the
checkpoint. Model and EMA branch values must be identical (they are deepcopies
of the same fresh encoder).

EMA continuation
----------------
`EMAModel` starts at `optimization_step=0`, whose first update has `decay=0`
and would copy the B1 *model* over the loaded B1 *EMA*. The pre-loop hook
restores `ema.optimization_step` from the checkpoint's `global_step`, so the
first update uses the late-training decay and preserves the loaded EMA.

RNG continuity
--------------
The hook also resets torch CPU/CUDA, numpy and Python `random` after
env_runner/wandb construction and before the first DataLoader iterator. The
new encoder builds its branch under a private CPU stream, so A (no branch) and
B/C/D leave the global stream in the same state even before this reset. The
printed digest is the launch gate; the epoch-0 average `train_loss` matching
across cells is the stronger proof that every data/crop/noise draw was paired.

This class only loads weights in `run()`. `eval_novel_view.py` constructs the
class from a checkpoint config and calls `load_payload` (never `run()`), so
evaluation is unaffected.
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import hashlib
import os
import pickle
import random

import dill
import numpy as np
import torch

from diffusion_policy.model.vision.residual_token_view_conditioned_obs_encoder import (
    BRANCH_STATE_PREFIX)
from diffusion_policy.workspace.train_diffusion_unet_image_workspace_latentprobe import (
    TrainDiffusionUnetImageWorkspaceLatentProbe)

# The policy model's state dict prefixes the encoder's own keys.
WARMSTART_BRANCH_PREFIX = 'obs_encoder.' + BRANCH_STATE_PREFIX


def _sha256_hex(data):
    return hashlib.sha256(data).hexdigest()[:16]


def load_warmstart_state_dicts(
        model, ema_model, state_dicts,
        allowed_prefixes=(WARMSTART_BRANCH_PREFIX,)):
    """Load a B1 payload into `model`/`ema_model`, allowing only new-branch keys.

    Returns a report dict. Raises on any unexpected key, any missing key outside
    `allowed_prefixes`, a missing set that differs from the model's actual
    branch keys, a model/EMA branch mismatch, or a shared tensor that did not
    load exactly.
    """
    if 'model' not in state_dicts:
        raise KeyError("warm-start payload has no 'state_dicts/model'")
    model_sd = model.state_dict()
    branch_keys = {
        k for k in model_sd
        if any(k.startswith(p) for p in allowed_prefixes)
    }
    model_result = model.load_state_dict(state_dicts['model'], strict=False)
    if model_result.unexpected_keys:
        raise RuntimeError(
            f'warm-start model has unexpected keys: '
            f'{sorted(model_result.unexpected_keys)[:10]}')
    missing = set(model_result.missing_keys)
    outside = sorted(k for k in missing
                     if not any(k.startswith(p) for p in allowed_prefixes))
    if outside:
        raise RuntimeError(
            f'warm-start model is missing non-branch keys: {outside[:10]}')
    if missing != branch_keys:
        raise RuntimeError(
            'warm-start missing keys are not exactly the model branch keys: '
            f'missing_only={sorted(missing - branch_keys)[:10]} '
            f'branch_only={sorted(branch_keys - missing)[:10]}')

    # Every shared tensor must have loaded exactly.
    model_loaded = model.state_dict()
    checked = 0
    for key, value in state_dicts['model'].items():
        if key in model_loaded and key not in branch_keys:
            if not torch.equal(model_loaded[key], value):
                raise RuntimeError(
                    f'warm-start model tensor did not load exactly: {key}')
            checked += 1

    ema_checked = 0
    if ema_model is not None:
        if 'ema_model' not in state_dicts:
            raise KeyError("warm-start payload has no 'state_dicts/ema_model'")
        ema_result = ema_model.load_state_dict(
            state_dicts['ema_model'], strict=False)
        if ema_result.unexpected_keys:
            raise RuntimeError(
                f'warm-start EMA has unexpected keys: '
                f'{sorted(ema_result.unexpected_keys)[:10]}')
        ema_missing = set(ema_result.missing_keys)
        if ema_missing != branch_keys:
            raise RuntimeError(
                'warm-start EMA missing keys are not the branch keys: '
                f'{sorted(ema_missing ^ branch_keys)[:10]}')
        ema_loaded = ema_model.state_dict()
        for key in sorted(branch_keys):
            if not torch.equal(model_loaded[key], ema_loaded[key]):
                raise RuntimeError(
                    f'warm-start model/EMA branch values differ: {key}')
        for key, value in state_dicts['ema_model'].items():
            if key in ema_loaded and key not in branch_keys:
                if not torch.equal(ema_loaded[key], value):
                    raise RuntimeError(
                        f'warm-start EMA tensor did not load exactly: {key}')
                ema_checked += 1

    return dict(
        missing=sorted(missing),
        unexpected=[],
        branch_keys=sorted(branch_keys),
        shared_model_tensors=checked,
        shared_ema_tensors=ema_checked,
    )


class TrainDiffusionUnetImageWorkspaceEncRT(
        TrainDiffusionUnetImageWorkspaceLatentProbe):
    """Latent-probe workspace + B1 warm start + RNG/EMA continuity gate."""

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        # Loaded by run(), never by __init__: eval constructs this class from a
        # checkpoint cfg and must not touch the warm-start path.
        self._warmstart_global_step = None
        self._warmstart_report = None

    # ------------------------------------------------------------- validation
    def _check_warmstart_cfg(self, payload_cfg):
        """Fail loudly if the B1 checkpoint was trained on a different setup."""
        checks = list()

        def cmp(name, old, new):
            if old != new:
                checks.append(f'{name}: checkpoint={old!r} new={new!r}')

        pe = payload_cfg.policy.obs_encoder
        ne = self.cfg.policy.obs_encoder
        for key in ('fused_dim', 'n_heads', 'cond_dim', 'use_plucker',
                    'use_eef_hist', 'eef_hist_steps', 'random_crop',
                    'use_group_norm', 'imagenet_norm', 'share_rgb_model'):
            cmp(f'policy.obs_encoder.{key}', pe.get(key, None),
                ne.get(key, None))

        def crop(value):
            return None if value is None else tuple(int(x) for x in value)

        cmp('policy.obs_encoder.crop_shape',
            crop(pe.get('crop_shape', None)),
            crop(ne.get('crop_shape', None)))

        pd = payload_cfg.task.dataset
        nd = self.cfg.task.dataset
        for key in ('dataset_path', 'eef_hist_window', 'eef_hist_steps'):
            cmp(f'task.dataset.{key}', pd.get(key, None), nd.get(key, None))
        cmp('task.dataset.view_pool',
            [int(x) for x in pd.view_pool],
            [int(x) for x in nd.view_pool])
        cmp('task.dataset.view_count_range',
            tuple(int(x) for x in pd.view_count_range),
            tuple(int(x) for x in nd.view_count_range))

        def rgb_slots(shape_meta):
            return sum(1 for attr in shape_meta['obs'].values()
                       if attr.get('type', 'low_dim') == 'rgb')

        cmp('task.shape_meta.rgb_slots',
            rgb_slots(payload_cfg.task.shape_meta),
            rgb_slots(self.cfg.task.shape_meta))
        if checks:
            raise RuntimeError(
                'warm-start checkpoint is not structurally compatible with this '
                'run:\n  ' + '\n  '.join(checks))

    # ------------------------------------------------------------------ load
    def _load_warmstart(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'warm-start checkpoint not found: {path}')
        payload = torch.load(path, pickle_module=dill, map_location='cpu')
        try:
            self._check_warmstart_cfg(payload['cfg'])
            state_dicts = payload['state_dicts']
            report = load_warmstart_state_dicts(
                self.model, self.ema_model, state_dicts)
            global_step = payload['pickles'].get('global_step', None)
            if isinstance(global_step, (bytes, bytearray)):
                global_step = dill.loads(global_step)
            if global_step is None:
                raise RuntimeError(
                    'warm-start checkpoint has no global_step pickle; cannot '
                    'restore the EMA continuation step')
            self._warmstart_global_step = int(global_step)
            self._warmstart_report = report
        finally:
            # Drop the ~4.6 GB payload before super().run() builds the dataset.
            del payload
        print(
            f'WARMSTART_OK ckpt={path} '
            f'missing={len(report["missing"])} unexpected=0 '
            f'global_step={self._warmstart_global_step} '
            f'shared_model={report["shared_model_tensors"]} '
            f'shared_ema={report["shared_ema_tensors"]}',
            flush=True)

    # ------------------------------------------------------------- pre-loop
    def _reset_and_log_rng(self, ema=None):
        """Restore EMA continuation and reset every RNG stream; print digests."""
        if ema is not None and self._warmstart_global_step is not None:
            ema.optimization_step = int(self._warmstart_global_step)
            print(f'WARMSTART_EMA_STEP {ema.optimization_step}', flush=True)

        seed = int(self.cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cuda_hash = '-'
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            device = torch.device(self.cfg.training.device)
            state = torch.cuda.get_rng_state(device)
            cuda_hash = _sha256_hex(state.cpu().numpy().tobytes())

        cpu_hash = _sha256_hex(torch.get_rng_state().numpy().tobytes())
        np_state = np.random.get_state()
        np_hash = _sha256_hex(
            np.asarray(np_state[1], dtype=np.uint32).tobytes()
            + str(np_state[2]).encode())
        py_hash = _sha256_hex(pickle.dumps(random.getstate()))
        print(
            f'WARMSTART_RNG seed={seed} cpu={cpu_hash} cuda={cuda_hash} '
            f'numpy={np_hash} py={py_hash}',
            flush=True)

    # ------------------------------------------------------------------ probe
    def _probe_extras(self, o_full, o_n1, cache):
        arrays, stats = super()._probe_extras(o_full, o_n1, cache)
        if 'z_views_base' not in o_full:
            return arrays, stats

        def rel_stats(out):
            active = out['view_active']
            base = out['z_views_base']
            delta = (out['z_views'] - base).norm(dim=-1)
            denom = base.norm(dim=-1).clamp_min(1e-12)
            rel = delta / denom
            return (float(rel[active].mean()),
                    float(rel[active].max()),
                    float(delta[active].mean()),
                    float(base.norm(dim=-1)[active].mean()))

        rel_mean, rel_max, delta_mean, base_norm = rel_stats(o_full)
        stats.update({
            'rt_delta_rel_mean': rel_mean,
            'rt_delta_rel_max': rel_max,
            'rt_delta_norm_mean': delta_mean,
            'rt_z_base_norm_mean': base_norm,
        })
        if 'z_views_base' in o_n1:
            stats['rt_delta_rel_n1'] = rel_stats(o_n1)[0]
        return arrays, stats

    # -------------------------------------------------------------------- run
    def run(self):
        if bool(self.cfg.training.get('resume', False)):
            raise RuntimeError(
                'warm-start runs must set training.resume=false; refusing to '
                'silently resume a same-directory checkpoint')
        warm = self.cfg.get('warmstart', None)
        if warm is None or not warm.get('checkpoint', None):
            raise RuntimeError('missing warmstart.checkpoint in the config')
        self._load_warmstart(str(warm.checkpoint))

        # Called by the base workspace after env_runner/wandb init and before
        # the epoch loop.
        self._pre_train_loop_hook = self._reset_and_log_rng
        super().run()

        # The mid-loop latest checkpoints are epoch 0/10/20; overwrite latest
        # at the fixed epoch-30 endpoint. use_thread=True keeps the saved
        # tensors on CPU (/tmp of the threaded path), which eval expects.
        if getattr(self, '_saving_thread', None) is not None:
            self._saving_thread.join()
        path = self.save_checkpoint(tag='latest', use_thread=True)
        if getattr(self, '_saving_thread', None) is not None:
            self._saving_thread.join()
        print(
            f'FINAL_CHECKPOINT path={path} epoch={self.epoch} '
            f'global_step={self.global_step}',
            flush=True)
