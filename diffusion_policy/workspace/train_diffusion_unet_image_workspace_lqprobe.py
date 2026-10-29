"""LQ probe: the standard latent probe plus the query-token diagnostics.

``TrainDiffusionUnetImageWorkspaceLatentProbe`` reads only ``z_views`` and
``z_global``. LQ's encoder also returns the final query-position outputs
(``query_tokens``, before the 256 -> 128 output projection) and the
post-transformer visual means (``z_visual_out``); this subclass persists them
through the existing ``_probe_extras`` seam, following
``train_diffusion_unet_image_workspace_probprobe.py``.

The base arrays are unchanged, so LQ snapshots are read with
``analyze_latent_series.py`` exactly like the other cells (with ``zv_dim``
recorded alongside ``fused_dim`` because LQ's ``z_views`` is 256-wide while
``z_global`` is 512-wide). The visualizer is deliberately not updated yet.

Semantics: ``query_tokens`` and ``z_visual_out`` are context-dependent -- they
depend on the sampled view set and on the history -- unlike ``z_views``, which
is the mean of the per-view tokens BEFORE the joint transformer. Do not compare
their norms to a per-view readout from an older cell.
"""
import torch

from diffusion_policy.workspace.train_diffusion_unet_image_workspace_latentprobe import (
    TrainDiffusionUnetImageWorkspaceLatentProbe)


class TrainDiffusionUnetImageWorkspaceLQProbe(
        TrainDiffusionUnetImageWorkspaceLatentProbe):

    EXTRA_ARRAY_DOCS = dict(
        qt='(S,H,D) fp16; final query-position outputs before out_proj',
        qtn1='(S,H,D) fp16; the N=1 az_0 condition',
        zvout='(S,K,D) fp16; per-view mean of POST-transformer visual outputs',
        zvout_n1='(S,K,D) fp16; the N=1 az_0 condition (dead rows zero)')

    def __init__(self, cfg, output_dir=None):
        target = cfg.policy.obs_encoder.get('_target_', '')
        if not target.endswith('LvsmEefQueryObsEncoder'):
            raise ValueError(
                'the LQ probe needs the LQ encoder '
                '(policy.obs_encoder._target_ ending in LvsmEefQueryObsEncoder); '
                f'got {target!r}')
        super().__init__(cfg, output_dir=output_dir)

    @torch.no_grad()
    def _probe_extras(self, o_full, o_n1, cache):
        for o, cond in ((o_full, 'full draw'), (o_n1, 'N=1 az0')):
            for key in ('query_tokens', 'z_visual_out'):
                if key not in o:
                    raise ValueError(
                        f'{cond}: encoder output has no {key!r} -- the LQ '
                        'probe needs LvsmEefQueryObsEncoder; got keys '
                        f'{sorted(o)}')
        qt = o_full['query_tokens'].float()          # (S, H, D)
        qtn1 = o_n1['query_tokens'].float()
        zvout = o_full['z_visual_out'].float()       # (S, K, D)
        zvout_n1 = o_n1['z_visual_out'].float()

        stats = dict(
            qt_norm_mean=float(qt.norm(dim=-1).mean()),
            qt_n1_norm_mean=float(qtn1.norm(dim=-1).mean()),
            qt_n1_vs_qt_rel=float(
                (qtn1 - qt).norm(dim=-1).mean()
                / qt.norm(dim=-1).mean().clamp_min(1e-6)),
            zvout_norm_mean=float(zvout.norm(dim=-1).mean()),
        )
        for k, v in stats.items():
            if not torch.isfinite(torch.tensor(v)):
                raise ValueError(f'non-finite probe extra {k}={v}')

        arrays = dict(
            qt=qt.half().cpu().numpy(),
            qtn1=qtn1.half().cpu().numpy(),
            zvout=zvout.half().cpu().numpy(),
            zvout_n1=zvout_n1.half().cpu().numpy())
        return arrays, stats
