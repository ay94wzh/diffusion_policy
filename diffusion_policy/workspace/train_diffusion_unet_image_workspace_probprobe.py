"""The probabilistic arm's per-epoch probe (PLAN_G.md stage 1, cells B/C/D).

`TrainDiffusionUnetImageWorkspaceLatentProbe` plus the ``_probe_extras``
override: the same fixed probe set, the same two conditions, the same scalar
summaries -- extended with the per-view Gaussian state and a handful of
scalars that make "is the predicted variance learning anything?" checkable
*during* the run, before any eval:

arrays (fp16, into the same ``latent_snapshots/epoch_XXXX.npz``):

* ``mu``  -- per-view Gaussian mean, canonical slot order (same layout as ``zv``);
* ``lv``  -- the clamped log-variance, same layout;
* ``lv_n1`` -- the N=1 az_0 condition's log-variance (the inference setting);
* ``lvg``  -- the PoE-combined fused log-variance at the full draw (``poe`` only).

scalars (into ``logs.json.txt`` under ``latent_probe/``):

* ``mu_resid_rel`` -- mean over live (frame, view) pairs of ``||mu - z||/||z||``:
  exactly 0 at init (the residual head is zero-initialized), so a positive
  value early is the head working; a *stuck* 0 is the head dead.
* ``lv_mean``, ``lv_std`` -- level and dispersion of the predicted log-variance.
  In cell D the logvar branch has no gradient path at all, so ``lv_std == 0``
  exactly is the expected (and asserted) reading -- never compare D's
  "uncertainty" numbers as if they meant anything.
* ``lv_view_spread`` -- per state, the spread of the per-view scalar logvar
  across the live views, averaged: "do views disagree about their own
  uncertainty" -- the quantity PoE is supposed to exploit.
* ``lv_at_min_frac`` / ``lv_at_max_frac`` -- fraction of live entries sitting on
  the clamp bounds. Rising fractions mean the bound is binding, i.e. the
  precision ratio is railed and PoE degrades toward winner-take-all. Registered
  reading: past 0.9 at e200, report the bound as binding; the bounds are never
  changed mid-campaign.
* ``lvg_mean`` and ``lvg_n1_vs_lv`` (``poe`` only) -- the fused log-variance's
  level, and the N=1 degeneracy measured on the real model (the fused logvar
  at a single live slot must equal that slot's own logvar; ~1e-7 in fp32/EMA).

Storage: mu/lv/lv_n1 are (S,K,512) fp16 ~ 1.44 MB each, lvg 0.13 MB, so about
+4.5 MB per epoch on top of the existing snapshots. Set
``latent_probe.save_gaussian: false`` to drop back to the parent's artifacts
byte-for-byte; the arrays also simply are not written when ``save_raw: false``
(the parent's npz block), while the scalars are still logged.
"""
import torch

from diffusion_policy.workspace.train_diffusion_unet_image_workspace_latentprobe import (
    TrainDiffusionUnetImageWorkspaceLatentProbe)


class TrainDiffusionUnetImageWorkspaceProbProbe(
        TrainDiffusionUnetImageWorkspaceLatentProbe):

    EXTRA_ARRAY_DOCS = dict(
        mu='(S,K,D) fp16; per-view Gaussian mean, canonical slot j == ring pool_ring[j]',
        lv='(S,K,D) fp16; clamped log-variance, same canonical layout',
        lv_n1='(S,K,D) fp16; the N=1 az_0 condition (one live slot; dead rows zero)',
        lvg='(S,D) fp16; PoE-combined fused log-variance at the full draw (poe mode only)')

    def __init__(self, cfg, output_dir=None):
        # validate before the base init (which starts wandb etc.): a
        # save_gaussian flag on a non-probabilistic encoder is a config error,
        # not a reason to still bring up a run
        lp = cfg.get('latent_probe', None)
        self._save_gaussian = (bool(lp.get('save_gaussian', False))
                               if lp is not None else False)
        if self._save_gaussian:
            enc_cfg = cfg.policy.obs_encoder
            if 'fusion_mode' not in enc_cfg:
                raise ValueError(
                    'latent_probe.save_gaussian needs the probabilistic encoder '
                    '(policy.obs_encoder.fusion_mode); the configured encoder '
                    'is not ProbViewConditionedObsEncoder')
            # read the bounds from the same config the head is built from, so a
            # clamp-fraction reading cannot silently use different numbers
            self._logvar_min = float(enc_cfg.logvar_min)
            self._logvar_max = float(enc_cfg.logvar_max)
        super().__init__(cfg, output_dir=output_dir)

    @torch.no_grad()
    def _probe_extras(self, o_full, o_n1, cache):
        if not self._save_gaussian:
            return {}, {}
        for o, cond in ((o_full, 'full draw'), (o_n1, 'N=1 az0')):
            if 'mu_views' not in o or 'logvar_views' not in o:
                raise ValueError(
                    f'{cond}: encoder output has no mu_views/logvar_views -- '
                    f'save_gaussian needs ProbViewConditionedObsEncoder; got '
                    f'keys {sorted(o)}')

        z = o_full['z_views'].float()                 # (S, K, D) raw z_v
        mu = o_full['mu_views'].float()
        lv = o_full['logvar_views'].float()
        live = o_full['view_active'].unsqueeze(-1)    # (S, K, 1) bool

        resid = ((mu - z).norm(dim=-1)
                 / z.norm(dim=-1).clamp_min(1e-6))             # (S, K)
        lv_flat = lv[live.expand_as(lv)]
        lo, hi = self._logvar_min, self._logvar_max
        # a full draw is asserted all-active by the cache build, so the
        # per-view spread below uses every slot
        per_view_scalar = lv.mean(dim=-1)                       # (S, K)
        stats = dict(
            mu_resid_rel=float(resid[live.squeeze(-1)].mean()),
            lv_mean=float(lv_flat.mean()),
            lv_std=float(lv_flat.std(unbiased=False)),
            lv_view_spread=float(per_view_scalar.std(dim=1, unbiased=False).mean()),
            lv_at_min_frac=float((lv_flat <= lo + 1e-6).float().mean()),
            lv_at_max_frac=float((lv_flat >= hi - 1e-6).float().mean()),
        )

        arrays = dict(
            mu=mu.half().cpu().numpy(),
            lv=lv.half().cpu().numpy(),
            lv_n1=o_n1['logvar_views'].half().cpu().numpy())

        lvg = o_full['z_global_logvar']
        if lvg is not None:                                   # 'poe' only
            lvg = lvg.float()
            lv_n1 = o_n1['logvar_views'].float()
            stats['lvg_mean'] = float(lvg.mean())
            # N=1 degeneracy on the real model: the fused logvar at the single
            # live slot must equal that slot's own logvar (~1 ulp)
            stats['lvg_n1_vs_lv'] = float(
                (o_n1['z_global_logvar'].float()
                 - lv_n1[:, int(cache['az0_pos'])]).abs().max())
            arrays['lvg'] = lvg.half().cpu().numpy()

        for k, v in stats.items():
            if not torch.isfinite(torch.tensor(v)):
                raise ValueError(f'non-finite probe extra {k}={v}')
        return arrays, stats
