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

Measured on square at **n=64**, fixed sampling noise, one path varied at a time. The old
n=8 anchors are **superseded and not comparable** -- the arm samples `dataset[0..n-1]`, so the
reading scales with n (proprio moved 5.2x from n=8 to n=64):

    cell                          image-only        proprio-only
    m3off [1,7]   works           0.0121 - 0.0276      0.3467
    m3v13 [1,3]   floor           0.0087 - 0.0216      0.3500
    m3v12 [1,2]   floor, collapsed 2.4e-05 - 5.0e-05   0.3470

Two things to read off that. `m3v12`'s image path is **severed** -- changing the entire image
barely moves the action, which is what an encoder collapsed to a constant must produce; that is
the end-to-end behavioural confirmation of the collapse finding, and it survives at n=64. And
`proprio_only` is **equal across all three cells** (1.008x m3v13/m3off), which is what retired
the n=8 "balance" reading of 1.75x.

`proprio_only` is the arm to trust: it is draw-independent, because the proprio keys are the
same low-dim rows whichever views are live, and it held to 0.2-0.4% across repeated runs.
`image_only` is the fragile one -- it is measured over a view ensemble that `dataset[i]` redraws
with `np.random`, so it moved 2.3-2.5x run-to-run until `--seed` was made to cover numpy too.
Pass `--view-count-range` to match the draw **across cells** as well; without it each cell draws
from its own checkpoint cfg and a cross-cell comparison confounds the cell with the draw.

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
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
# the same range-override helper `screen_collapse.py` uses; shared so the two screens
# cannot drift on what "matched draw" means
from probe_relpose import _apply_range

OmegaConf.register_new_resolver('eval', eval, replace=True)

# the obs keys that carry PROPRIOCEPTION, i.e. not the image path
PROPRIO_SUFFIX = ('eef_pos', 'eef_quat', 'gripper_qpos')


def measure(checkpoint, device, n_obs, seed, view_count_range=None):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=tempfile.mkdtemp())
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model
    policy.to(device)
    policy.eval()
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    # What the checkpoint's own cfg draws, recorded BEFORE any override so the artifact
    # shows whether a comparison was matched.
    cell_range = getattr(dataset, 'view_count_range', None)
    cell_range = tuple(int(x) for x in cell_range) if cell_range is not None else None
    # None is a no-op that just returns `cell_range`; an override re-implements the guard.
    effective_range = _apply_range(dataset, view_count_range)

    # Seed numpy, not just torch. `dataset[i]` draws that sample's view subset with
    # np.random (`multiview_image_dataset._m3_slots`), and torch's seeding does NOT cover
    # numpy -- so without this every invocation draws a DIFFERENT view ensemble, and two
    # cells are never measured on the same one. That is the confound that made the n=64
    # gate unreadable: image_only swung 2.3-2.5x run-to-run while proprio_only (which the
    # view draw cannot touch) held to 0.2%. Seeding here, after the range is fixed so the
    # draw sequence is identical for the same (seed, effective_range), makes matched
    # comparisons actually matched rather than assumed. See NOTES.md.
    np.random.seed(seed)

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
                action_scale=float(A.abs().mean()), n_obs=int(n_obs),
                cell_view_count_range=(list(cell_range) if cell_range else None),
                effective_range=(list(effective_range) if effective_range else None))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('-d', '--device', default='cuda:0')
    ap.add_argument('--n-obs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--view-count-range', default=None,
                    help='override the view draw range for EVERY cell, as "lo,hi". '
                         'Matched-across-cells is the point: without it each cell draws '
                         'from its own checkpoint cfg range, so a comparison across cells '
                         'confounds the cell with the draw (screen_collapse.py has the '
                         'same flag for the same reason).')
    ap.add_argument('--num-threads', type=int, default=4)
    args = ap.parse_args()

    vcr = (tuple(int(x) for x in args.view_count_range.split(','))
           if args.view_count_range else None)

    torch.set_num_threads(args.num_threads)
    out = measure(args.checkpoint, torch.device(args.device), args.n_obs, args.seed, vcr)
    out['checkpoint'] = args.checkpoint
    # `:.6g`, NOT `:.4f`. A severed path reads ~2.6e-05, and `:.4f` prints that as `0.0000`
    # -- which was then written up as "bit-identical". The format width turned a measurable
    # number into a claim the data did not support; do not reintroduce it.
    print(f"  image-only {out['image_only']:.6g}   proprio-only {out['proprio_only']:.6g}   "
          f"(n={out['n_obs']})")
    print(f"  draw: cell {out['cell_view_count_range']} -> effective {out['effective_range']}"
          f"  seed {args.seed}")
    print('  image-only at ~1e-05 or below means the image path is SEVERED: the policy '
          'cannot see the scene at all')
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
