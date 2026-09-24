"""Does the POLICY actually respond to its observation, and to which part of it?

Why this exists
---------------
A representation-only screen cannot see the failure that matters most. `screen_collapse.py`
measures the encoder; `probe_relpose.py` measures what is decodable from it; neither says
whether the *action head* uses any of it. This does, by holding the diffusion sampling noise
fixed and varying one input path at a time -- so a difference in the output is attributable to
the input, not to the sampler.

`global_cond` is ``concat([z_global, low-dim])``, and the low-dim keys are 9 dims of
proprioception (``robot0_eef_pos/quat/gripper_qpos``) that vary across samples too. Measured
naively, sensitivity is dominated by those, which is exactly how the first version of this
measurement misled: it reported the *collapsed* cell as the MOST observation-sensitive, because
its constant image was accompanied by normally-varying proprioception. The two paths must be
separated.

Measured on square (8 observations, fixed noise, one path varied):

    cell                          image-only   proprio-only
    m3off [1,7]   works             0.0068        0.0386
    m3v13 [1,3]   floor             0.0067        0.0674
    m3v12 [1,2]   floor, collapsed  0.0000        0.0686

`0.0000` is the interesting one: m3v12's image path is *severed* -- changing the entire image
leaves the action bit-identical, which is what an encoder collapsed to a constant must produce.
That is the end-to-end, behavioural confirmation of the collapse finding.

Usage:
    python screen_conditioning.py -c <run>/checkpoints/latest.ckpt
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
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver('eval', eval, replace=True)

# the obs keys that carry PROPRIOCEPTION, i.e. not the image path
PROPRIO_SUFFIX = ('eef_pos', 'eef_quat', 'gripper_qpos')


def measure(checkpoint, device, n_obs, seed):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=tempfile.mkdtemp())
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model
    policy.to(device)
    policy.eval()
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    def to_dev(x):
        return x.unsqueeze(0).to(device)

    base = dict_apply(dataset[0]['obs'], to_dev)

    def is_proprio(k):
        return any(k.endswith(s) for s in PROPRIO_SUFFIX)

    def act(obs):
        # the SAME noise every call, so the only thing that can move the output is obs
        torch.manual_seed(seed)
        with torch.no_grad():
            return policy.predict_action(obs)['action'].float().cpu()

    def spread(vary_proprio):
        out = []
        for i in range(n_obs):
            obs = dict_apply(dataset[i]['obs'], to_dev)
            merged = {k: (obs[k] if is_proprio(k) == vary_proprio else base[k])
                      for k in base}
            out.append(act(merged))
        A = torch.stack(out)
        return float(A.std(dim=0).mean() / A.abs().mean()), A

    img, _ = spread(False)
    pro, A = spread(True)
    return dict(image_only=img, proprio_only=pro,
                action_scale=float(A.abs().mean()), n_obs=int(n_obs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('-d', '--device', default='cuda:0')
    ap.add_argument('--n-obs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num-threads', type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    out = measure(args.checkpoint, torch.device(args.device), args.n_obs, args.seed)
    out['checkpoint'] = args.checkpoint
    print(f"  image-only {out['image_only']:.4f}   proprio-only {out['proprio_only']:.4f}   "
          f"(n={out['n_obs']})")
    print('  image-only 0.0000 means the image path is SEVERED: the policy cannot see the '
          'scene at all')
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
