"""Per-epoch latent snapshots for M3/M4 training (the "clear" run).

The base workspace's epoch loop has one guarded seam that calls
``self._latent_probe_hook(policy)`` once per epoch; this subclass installs that
hook (see ``train_diffusion_unet_image_workspace.py``, the ``latent_hook``
block). Base-class runs never set the attribute and are bit-identical -- the
same pattern as the existing ``aux_loss`` seam.

What the hook does
------------------
Encodes a FIXED probe set (fixed dataset indices, fixed per-state view draws)
with the EMA weights in eval mode, at two matched conditions:

- ``full``   -- every pool view active (the draw range is forced to [K, K], so
                each probe sample is a permutation of the whole pool);
- ``n1_az0`` -- a single live slot holding the az_0 view, everything else zero
                and masked -- exactly the N=1 inference condition that the
                rollouts and ``eval_novel_view.py`` serve.

It returns small scalar summaries into the per-epoch step log (norms, the
``screen_collapse`` relative spread, participation ratio, a fusion-free
view-vs-state pair ratio -- the same statistic ``probe_relpose`` reports) and
optionally writes raw fp16 ``z_v``/``z_g`` tensors to
``<output_dir>/latent_snapshots/`` (remote-only; never committed).

Why it cannot disturb training
------------------------------
- numpy / torch-CPU / CUDA RNG states are saved and restored around EVERY call
  (the one-time cache build included): reseeding in the main process would
  change the DataLoader's per-epoch base_seed and therefore the training draws.
- The probe inputs are decoded ONCE into a small CPU cache (~11 MB at the
  defaults), so the per-epoch cost is one encoder pass (~1-2 s), not K image
  decodes per state.
- It builds its own dataset instance and never touches the training loop's
  datasets, optimizer, or model parameters (encoding is under torch.no_grad()).

Config block (see ``train_diffusion_unet_image_workspace_m4latent.yaml``):

    latent_probe:
      n_states: 128        # probe states, fixed across epochs
      seed: 12345          # fixes the state choice and per-state view draws
      az0_view: 6          # ring index of the az_0 view (the N=1 condition)
      save_raw: true
      out_dir: latent_snapshots
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import json
import os

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace)
from diffusion_policy.common.pytorch_util import dict_apply


def _rel_dist(a, b):
    """Mean relative L2 between two row-sets -- ``probe_relpose._rel_dist``'s
    definition, so the snapshot's pair ratio is the same statistic the probe
    reports (scale-free, comparable across checkpoints)."""
    return float(((a - b).norm(dim=-1) /
                  (0.5 * (a.norm(dim=-1) + b.norm(dim=-1))).clamp_min(1e-6)).mean())


def _rel_spread(z):
    """screen_collapse's relative spread: std across rows (max over dims) over
    the mean row norm. ~1e-02 is healthy, ~1e-04 and below degenerate; read it
    against the same checkpoint's ``--random-init`` screen, not in absolute."""
    return float(z.std(dim=0).max() / z.norm(dim=-1).mean().clamp_min(1e-12))


def _participation_ratio(z):
    """(sum lambda)^2 / sum lambda^2 over the covariance spectrum of (N, D)."""
    x = z.float() - z.float().mean(dim=0, keepdim=True)
    lam = torch.linalg.svdvals(x).pow(2)
    return float(lam.sum().pow(2) / lam.pow(2).sum().clamp_min(1e-30))


class TrainDiffusionUnetImageWorkspaceLatentProbe(TrainDiffusionUnetImageWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        if cfg.get('latent_probe', None) is None:
            raise ValueError(
                'the latent-probe workspace needs a `latent_probe:` block in the '
                'config (n_states, seed, az0_view, save_raw, out_dir)')
        self._latent_probe_cfg = copy.deepcopy(cfg.latent_probe)
        # the base workspace's seam calls self._latent_probe_hook(policy)
        self._latent_probe_hook = self._run_latent_probe
        self._probe_cache = None

    # ------------------------------------------------------------------ cache
    def _build_probe_cache(self):
        """Decode the fixed probe set ONCE into CPU tensors.

        The draw range is forced to a full draw ([K, K]) so every probe sample
        contains the entire pool (in some slot order). Each state's draw is
        seeded individually, so the same states, the same draws and therefore
        the same slot->view mapping are reproduced every epoch -- which is what
        makes the per-epoch snapshots comparable.
        """
        pcfg = self._latent_probe_cfg
        n_states = int(pcfg.n_states)
        seed = int(pcfg.seed)
        az0_view = int(pcfg.az0_view)

        # own dataset instance from the run's own config (interpolations
        # resolved against the workspace root, like the workspace does)
        ds_conf = OmegaConf.create(
            OmegaConf.to_container(self.cfg.task.dataset, resolve=True))
        n_pool = len(ds_conf.view_pool)
        ds_conf.view_count_range = [n_pool, n_pool]
        dataset = hydra.utils.instantiate(ds_conf)

        pool = [int(v) for v in dataset.view_pool]
        pool_sorted = sorted(pool)
        if az0_view not in pool:
            raise ValueError(f'az0_view {az0_view} not in view_pool {pool}')
        az0_pos = pool_sorted.index(az0_view)
        K = int(dataset.n_slots)
        if len(pool_sorted) != K:
            raise ValueError(
                f'the probe needs slots == pool size for a complete canonical '
                f'ordering; got K={K}, pool={pool}')

        rng = np.random.RandomState(seed)
        n = int(min(n_states, len(dataset)))
        state_idx = np.sort(rng.choice(len(dataset), size=n, replace=False))

        rgb_keys = list(dataset.rgb_keys)
        cam_keys = list(dataset.cam_keys)
        images, cams, hists, masks = [], [], [], []
        views = np.zeros((n, K), dtype=np.int64)
        az0_slots = np.zeros(n, dtype=np.int64)
        for i in range(n):
            # fixed per-state draw: identical every epoch and independent of the
            # global numpy stream (which the RNG save/restore preserves anyway)
            np.random.seed(seed + i)
            obs = dataset[int(state_idx[i])]['obs']
            images.append(torch.stack([obs[k][0] for k in rgb_keys]))
            cams.append(torch.stack([obs[k][0] for k in cam_keys]))
            hists.append(obs[dataset.eef_hist_key][0])
            masks.append(obs[dataset.view_mask_key][0])
            for slot in range(K):
                hit = np.nonzero(
                    np.all(dataset.cam_table == cams[-1][slot].numpy(), axis=1))[0]
                if len(hit) != 1:
                    raise RuntimeError(
                        f'probe state {i} slot {slot}: cam vector matches '
                        f'{len(hit)} pool views; the slot->view map must be exact')
                views[i, slot] = int(hit[0])
            present = set(int(v) for v in views[i])
            if present != set(pool):
                raise RuntimeError(
                    f'probe state {i}: full draw missing pool views '
                    f'{sorted(set(pool) - present)}')
            az0_slots[i] = int(np.nonzero(views[i] == az0_view)[0][0])

        cache = dict(
            images=torch.stack(images),            # (S, K, 3, H, W)
            cams=torch.stack(cams),                # (S, K, 10)
            hists=torch.stack(hists),              # (S, K, H*11)
            masks=torch.stack(masks),              # (S, K)
            views=torch.from_numpy(views),         # (S, K) ring index per slot
            az0_slots=torch.from_numpy(az0_slots), # (S,)
            state_idx=torch.from_numpy(state_idx),
            rgb_keys=rgb_keys,
            cam_keys=cam_keys,
            mask_key=dataset.view_mask_key,
            hist_key=dataset.eef_hist_key,
            pool_ring=pool_sorted,
            az0_ring=az0_view,
            az0_pos=az0_pos,
            K=K,
            n_states=n,
            seed=seed,
        )
        if not bool((cache['masks'] > 0.5).all()):
            raise RuntimeError('probe masks are not all-active at a full draw')
        self._probe_cache = cache
        return cache

    # ------------------------------------------------------------------- hook
    def _run_latent_probe(self, policy):
        """One epoch's snapshot; returns scalar summaries for the step log."""
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
        try:
            if self._probe_cache is None:
                self._build_probe_cache()
            return self._encode_and_save(policy)
        finally:
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    @torch.no_grad()
    def _encode_and_save(self, policy):
        cache = self._probe_cache
        pcfg = self._latent_probe_cfg
        device = next(policy.parameters()).device
        enc = policy.obs_encoder
        S, K = cache['n_states'], cache['K']

        # canonical order on CPU: canonical position j holds ring view
        # pool_ring[j], for images, camera vectors and EE history alike
        views = cache['views']
        ar = torch.arange(S)
        img_c = torch.empty_like(cache['images'])
        cam_c = torch.empty_like(cache['cams'])
        hist_c = torch.empty_like(cache['hists'])
        for j, v in enumerate(cache['pool_ring']):
            sel = (views == int(v)).float().argmax(dim=1)     # (S,)
            img_c[:, j] = cache['images'][ar, sel]
            cam_c[:, j] = cache['cams'][ar, sel]
            hist_c[:, j] = cache['hists'][ar, sel]

        def _obs(images, cams, hists, mask):
            d = dict()
            for k, key in enumerate(cache['rgb_keys']):
                d[key] = images[:, k]
            for k, key in enumerate(cache['cam_keys']):
                d[key] = cams[:, k]
            d[cache['mask_key']] = mask
            d[cache['hist_key']] = hists
            return d

        obs_full = _obs(img_c, cam_c, hist_c, torch.ones(S, K))

        # N=1 az_0: one live slot, everything else zero + masked (the eval
        # convention). The encoder is slot-agnostic at N=1 (the fusion softmax
        # is over a single unmasked key), so canonical position is fine here.
        z_img = torch.zeros_like(img_c)
        z_cam = torch.zeros_like(cam_c)
        z_hist = torch.zeros_like(hist_c)
        z_img[:, cache['az0_pos']] = img_c[:, cache['az0_pos']]
        z_cam[:, cache['az0_pos']] = cam_c[:, cache['az0_pos']]
        z_hist[:, cache['az0_pos']] = hist_c[:, cache['az0_pos']]
        mask_n1 = torch.zeros(S, K)
        mask_n1[:, cache['az0_pos']] = 1.0
        obs_n1 = _obs(z_img, z_cam, z_hist, mask_n1)

        def _encode(ob):
            ob = dict_apply(ob, lambda x: x.to(device))
            nob = policy.normalizer.normalize(ob)
            return enc.forward_full(nob)

        o_full = _encode(obs_full)
        o_n1 = _encode(obs_n1)
        zv = o_full['z_views'].float()        # (S, K, D) canonical
        zg = o_full['z_global'].float()       # (S, D)
        zg_n1 = o_n1['z_global'].float()      # (S, D)

        # ---- summaries (torch on device; small) --------------------------
        pair = [_rel_dist(zv[:, i], zv[:, j])
                for i in range(K) for j in range(i + 1, K)]
        iu = torch.triu_indices(S, S, offset=1)
        cross = [_rel_dist(zv[:, v][iu[0]], zv[:, v][iu[1]]) for v in range(K)]
        per_view = [_rel_spread(zv[:, v]) for v in range(K)]
        stats = {
            'zv_norm_mean': float(zv.norm(dim=-1).mean()),
            'zv_rel_spread_mean': float(np.mean(per_view)),
            'zv_rel_spread_min': float(np.min(per_view)),
            'zg_norm_mean': float(zg.norm(dim=-1).mean()),
            'zg_rel_spread': _rel_spread(zg),
            'zg_n1_norm_mean': float(zg_n1.norm(dim=-1).mean()),
            'zg_n1_rel_spread': _rel_spread(zg_n1),
            'zv_pair_ratio': float(np.mean(pair) / max(float(np.mean(cross)), 1e-12)),
            'zv_pr': _participation_ratio(zv.reshape(-1, zv.shape[-1])),
            'zg_pr': _participation_ratio(zg),
        }

        # ---- raw snapshots ------------------------------------------------
        if bool(pcfg.save_raw):
            out_dir = os.path.join(self.output_dir, str(pcfg.out_dir))
            os.makedirs(out_dir, exist_ok=True)
            meta_path = os.path.join(out_dir, 'meta.json')
            if not os.path.exists(meta_path):
                with open(meta_path, 'w') as f:
                    json.dump(dict(
                        tool='train_diffusion_unet_image_workspace_latentprobe',
                        n_states=S, K=K, fused_dim=int(zv.shape[-1]),
                        seed=cache['seed'], pool_ring=cache['pool_ring'],
                        az0_ring=cache['az0_ring'], az0_pos=int(cache['az0_pos']),
                        conditions=['full', 'n1_az0'],
                        arrays=dict(
                            zv='(S,K,D) fp16; canonical slot j == ring pool_ring[j]',
                            zg='(S,D) fp16 (full draw)',
                            zg_n1='(S,D) fp16 (single az_0 slot)',
                            state_idx='(S,) int64 dataset indices'),
                        normalizer='policy.normalizer (identity for cam/mask/hist)',
                        note='EMA weights, eval mode; draws fixed per state by seed',
                    ), f, indent=2)
            epoch = int(self.epoch)
            np.savez(os.path.join(out_dir, f'epoch_{epoch:04d}.npz'),
                     zv=zv.half().cpu().numpy(),
                     zg=zg.half().cpu().numpy(),
                     zg_n1=zg_n1.half().cpu().numpy(),
                     state_idx=cache['state_idx'].numpy())
            with open(os.path.join(out_dir, 'snapshots.jsonl'), 'a') as f:
                f.write(json.dumps(dict(epoch=epoch, **stats)) + '\n')

        return {f'latent_probe/{k}': v for k, v in stats.items()}
