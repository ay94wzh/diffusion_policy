"""The z-round's semantic readout — PLAN_Z idea 1's own instrument.

`L_align` (the shared-semantics alignment loss) is the only mechanism cell A/C
add, and its degenerate minimum is P_s ≡ const: a head that maps every view to
one point scores a PERFECT loss while putting zero pressure on the encoder
(risk R1 in `data/z_run.sh`). This tool measures, on the run's own fixed 128-
state probe set, what the trained head actually does — so that signature (and
its absence) is readable from the artifacts rather than inferred.

Reads (no simulator, no GPU; runs on the laptop):
  * `-n` the run's `latent_snapshots/` dir (fp16 npz; the LAST epoch is read)
    or a single `epoch_XXXX.npz`. The probe encodes every state at a FULL
    draw (all K slots live), and slot k -> pool view is fixed per state
    (`meta.json` carries `pool_ring`; slot order == ring order at full draw).
  * `-c` the run's checkpoint, for the EMA semantic head's weights (EMA is
    what the probes and the sweeps use) and its `semantic_hidden_dim` /
    `semantic_init_seed`.

Reports, over the (S, K, D) z_v at the last epoch, with s = normalize(P_s(z_v)):
  L_align      the training term's value on this set: per-row mean over live
               pairs of ||s_i − s_j||², then mean over rows (all C(K,2) pairs,
               all rows — a full draw has no dead slots);
  cos_within   mean pairwise cosine of s across views of the SAME state; a
               view-invariant shared-semantics space sends this to 1;
  cos_across   mean cosine of s at the SAME slot across different states —
               the global-collapse control: a constant head sends BOTH to 1,
               while a healthy view-invariant head keeps cos_across low.
and the same three numbers under a RE-INITIALIZED head (`semantic_init_seed`)
— the control that separates "the trained head learned the invariance" from
"any head this size would score this on this data".

Registered reading (data/z_run.sh, risk R1): head collapse = L_align ≪ the
re-init control's AND cos_within ≈ 1 AND cos_across ≈ 1. A trained head that
beats its re-init control on cos_within while keeping cos_across low did the
designed job at the representation level; whether it moved the encoder is the
latent digest's question, not this tool's.

fp16 storage floor: 4.9e-4 relative — differences below that are noise.

Usage:
    python analyze_semantic_head.py -c <run>/checkpoints/latest.ckpt \
        -n <run>/latent_snapshots -o data/analysis_z_a/semantic_readout.json
"""
import os
import sys
import json
import glob
import tempfile

# bootstrap: make repo root importable regardless of cwd
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import click
import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F

from diffusion_policy.model.vision.shared_semantic_head import (
    SharedSemanticHead)


def load_policy_cfg(checkpoint):
    """The probe_relpose.py idiom: rebuild the workspace from the payload."""
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    tmp = tempfile.mkdtemp(prefix='semantic_readout_')
    workspace = cls(cfg, output_dir=tmp)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    return policy, cfg


def latest_npz(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, 'epoch_*.npz')))
        if not files:
            raise FileNotFoundError(f'no epoch_*.npz under {path}')
        return files[-1]
    return path


def pairwise_stats(s):
    """``(S, K, D)`` unit vectors -> (L_align, cos_within, cos_across).

    L_align uses the policy's own reduction: per-row mean over the K(K-1)/2
    pairs, then mean over rows (every row here is a full draw).
    """
    S, K, D = s.shape
    sim = torch.einsum('sid,sjd->sij', s, s)                 # (S, K, K)
    iu = torch.triu_indices(K, K, offset=1)
    pair_cos = sim[:, iu[0], iu[1]]                          # (S, P)
    l_align = (2.0 - 2.0 * pair_cos).mean().item()           # ||s_i - s_j||^2
    cos_within = pair_cos.mean().item()
    # same slot, different states -- the global-collapse control
    off = ~torch.eye(S, dtype=torch.bool)
    vals = []
    for k in range(K):
        m = torch.einsum('sd,td->st', s[:, k], s[:, k])
        vals.append(m[off].mean())
    cos_across = torch.stack(vals).mean().item()
    return l_align, cos_within, cos_across


def collapse_signature(trained, control):
    """The registered R1 predicate: trained = (L, cos_within, cos_across),
    control = the re-initialized head's tuple. Collapse = the trained head
    scores LESS than half the control's L_align while cos_within AND
    cos_across are both ~1 (a constant map on this set)."""
    l_t, cw_t, ca_t = trained
    l_c, _, _ = control
    return bool(l_t < 0.5 * l_c and cw_t > 0.99 and ca_t > 0.99)


@click.command()
@click.option('-c', '--checkpoint', required=True, help='run checkpoint (latest.ckpt)')
@click.option('-n', '--npz', required=True, help='latent_snapshots/ dir or an epoch_XXXX.npz')
@click.option('-o', '--output', required=True, help='output JSON path')
def main(checkpoint, npz, output):
    policy, cfg = load_policy_cfg(checkpoint)
    head = getattr(policy, 'semantic_head', None)
    if head is None:
        raise SystemExit(
            f'{checkpoint} is not a z-round policy (no semantic_head)')
    path = latest_npz(npz)
    z = np.load(path, allow_pickle=False)
    zv = torch.from_numpy(z['zv'].astype(np.float32))        # (S, K, D) fp16 upcast
    S, K, D = zv.shape

    def stats_with(mod):
        with torch.no_grad():
            s = F.normalize(mod(zv.reshape(-1, D)), dim=-1, eps=1e-6)
            s = s.reshape(S, K, D)
            return pairwise_stats(s)

    l_t, cos_w_t, cos_a_t = stats_with(head)
    hidden = head.net[0].out_features
    init_seed = getattr(cfg.policy, 'semantic_init_seed', 42)
    # the control head, built under the same private stream the policy used
    cpu_state = torch.get_rng_state()
    try:
        gen = torch.Generator().manual_seed(int(init_seed))
        torch.set_rng_state(gen.get_state())
        control = SharedSemanticHead(in_dim=D, hidden_dim=hidden)
    finally:
        torch.set_rng_state(cpu_state)
    l_c, cos_w_c, cos_a_c = stats_with(control)

    out = {
        'schema': 1,
        'tool': 'analyze_semantic_head.py',
        'checkpoint': checkpoint,
        'npz': path,
        'n_states': S, 'n_slots': K, 'dim': D,
        'semantic_hidden_dim': hidden,
        'semantic_init_seed': int(init_seed),
        'trained': {'L_align': l_t, 'cos_within': cos_w_t, 'cos_across': cos_a_t},
        'reinit_control': {'L_align': l_c, 'cos_within': cos_w_c,
                           'cos_across': cos_a_c},
        'collapse_signature': collapse_signature((l_t, cos_w_t, cos_a_t),
                                                 (l_c, cos_w_c, cos_a_c)),
        'fp16_storage_floor_rel': 4.9e-4,
    }
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as f:
        json.dump(out, f, indent=1)

    print(f'READOUT semantic {path} (S={S} K={K} D={D})')
    print(f'  trained  : L_align {l_t:.4f}  cos_within {cos_w_t:.4f}  '
          f'cos_across {cos_a_t:.4f}')
    print(f'  reinit   : L_align {l_c:.4f}  cos_within {cos_w_c:.4f}  '
          f'cos_across {cos_a_c:.4f}')
    if out['collapse_signature']:
        print('  COLLAPSE SIGNATURE: trained head ≈ constant (both cosines '
              '≈ 1 with L_align far below the re-init control) -- the '
              'pressure was consumed by the head, not the encoder '
              '(data/z_run.sh risk R1). Report it; do not round it away.')
    else:
        print('  no collapse signature on this set')


if __name__ == '__main__':
    main()
