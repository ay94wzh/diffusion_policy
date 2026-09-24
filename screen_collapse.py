"""Screen a trained encoder for REPRESENTATION COLLAPSE, and optionally control it.

Why this exists
---------------
Every cell in this project that fails, fails the same way: its encoder outputs a
near-constant vector, so the policy has nothing to condition on and acts open-loop.
That is why the failures are at the TRAINED pose rather than only off-axis, and it is
what separates the tasks L1 solves from the tasks it destroys (PROGRESS.md, *Collapse is
the unifying failure mode*).

Anchors on square, relative spread = std across states (max over dims) / mean norm:

    random init (architecture baseline)   1.3e-02
    L1 lift        (works)                2.5e-02
    m3off [1,7]    (works)                3.3e-02
    L1 square      (fails)                1.5e-04
    L1 can         (fails)                3.1e-05
    m3v12 [1,2]    (floor)                5.4e-07

So the ordering is the signal: ~1e-02 is healthy, ~1e-04 and below is degenerate. Read it
against a `--random-init` run of the SAME checkpoint, never against the absolute number --
the architecture baseline is what distinguishes "training collapsed this" from "this
architecture always looks like this", and without it the number means nothing.

This works on any image encoder, M3's view-conditioned one included, which is why it is a
separate script rather than a mode of `probe_relpose.py` (that one needs per-slot cam keys).

    python screen_collapse.py -c <run>/checkpoints/latest.ckpt --random-init
"""
import os
import sys
import json
import argparse
import tempfile

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import hydra
import dill
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply

# train.py registers this; a config loaded outside it needs it too
OmegaConf.register_new_resolver('eval', eval, replace=True)

FEATURE_DIM = 512          # the encoder's contribution, before any low-dim concat


def load_policy(checkpoint, device):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=tempfile.mkdtemp())
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    return policy, cfg


def measure(policy, cfg, device, n_states, random_init):
    """Relative spread of the encoder's features across states.

    `std` is taken across STATES and maxed over dims, then divided by the mean feature
    norm -- dividing by the norm is what makes the number comparable between encoders
    that happen to produce different magnitudes, which is the whole point of a cross-cell
    screen. Only the first `FEATURE_DIM` columns are used, so an encoder that concatenates
    low-dim observations (M3's `forward` does) is measured on its features alone.
    """
    encoder = policy.obs_encoder
    saved = None
    if random_init:
        saved = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
        encoder.apply(lambda m: m.reset_parameters()
                      if hasattr(m, 'reset_parameters') else None)
    try:
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        Z = []
        for i in range(min(n_states, len(dataset))):
            obs = dict_apply(dataset[i]['obs'],
                             lambda x: x.unsqueeze(0).to(device))
            nobs = policy.normalizer.normalize(obs)
            this = dict_apply(nobs, lambda x: x[:, :2].reshape(-1, *x.shape[2:]))
            with torch.no_grad():
                out = encoder(this)
            Z.append(out.reshape(-1, out.shape[-1])[0, :FEATURE_DIM].float().cpu())
        Z = torch.stack(Z)
        return dict(n_states=int(Z.shape[0]),
                    feature_dim=int(Z.shape[1]),
                    mean_norm=float(Z.norm(dim=-1).mean()),
                    max_spread=float(Z.std(dim=0).max()),
                    relative_spread=float(Z.std(dim=0).max() / Z.norm(dim=-1).mean()),
                    random_init=bool(random_init))
    finally:
        if saved is not None:
            encoder.load_state_dict(saved)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('-o', '--output', default=None, help='also write the JSON here')
    ap.add_argument('-d', '--device', default='cuda:0')
    ap.add_argument('--n-states', type=int, default=16,
                    help='consecutive dataset indices; a screen, not a measurement')
    ap.add_argument('--random-init', action='store_true',
                    help='discard the weights and re-measure -- THE control: without it '
                         'the number cannot distinguish collapse from architecture')
    ap.add_argument('--num-threads', type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    policy, cfg = load_policy(args.checkpoint, device)
    use_plucker = getattr(cfg.policy.obs_encoder, 'use_plucker', None)
    print(f'checkpoint: {args.checkpoint}')
    print(f'  encoder={cfg.policy.obs_encoder._target_.split(".")[-1]} '
          f'use_plucker={use_plucker}')

    out = measure(policy, cfg, device, args.n_states, args.random_init)
    out['checkpoint'] = args.checkpoint
    print(f"  relative spread = {out['relative_spread']:.3e}  "
          f"(mean |z| {out['mean_norm']:.4f}, spread {out['max_spread']:.3e}, "
          f"n={out['n_states']})")
    if args.random_init:
        print('  ^ RANDOM INIT -- this is the architecture baseline to compare against')
    else:
        print('  compare against a --random-init run of this same checkpoint: ~1e-02 is '
              'healthy, ~1e-04 and below is collapsed')
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
