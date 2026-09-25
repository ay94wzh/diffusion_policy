"""Proprioception dropout on the denoising network's conditioning.

Why this exists
---------------
`[1,3]`'s floor is the project's live question (PROGRESS.md, *The second failure
mode*): the cell's representation beats the working cell's on four
separately-measured stages and it still scores 0.080. The one mechanism anyone
has proposed is **balance** -- its image->action sensitivity is identical to the
working cell's (0.0067 vs 0.0068), and the only measured difference is that it
leans ~1.75x harder on proprioception (0.0674 vs 0.0386), which cannot see where
the nut is. That reading is n=8, one seed. This policy makes the mechanism
falsifiable by intervention: drop proprioception for a random half of the
training samples and see whether the floor lifts.

Which path -- there are two, and only one is touched
---------------------------------------------------
  * ``robot0_eef_pos`` / ``robot0_eef_quat`` / ``robot0_gripper_qpos`` (9 dims)
    pass through the encoder untouched and are concatenated into the denoising
    network's ``global_cond = concat([z_global, low-dim])``.  **This one.**
  * ``view_eef_hist`` -> AdaGN/FiLM inside the encoder (the "trajectory
    conditioning", ``use_eef_hist``).  Untouched -- and inert in every ladder
    cell, which all run ``use_eef_hist=false``.

This is also the path ``screen_conditioning.py`` measures: its proprio arm varies
only keys matching its ``PROPRIO_SUFFIX`` and holds ``view_eef_hist`` fixed. The
suffix tuple is duplicated here rather than imported (that module is a script
that pulls in hydra/dill); ``tests/test_prop_dropout.py`` asserts the two copies
agree, so they cannot drift apart silently.

Why ``compute_loss`` and not the dataset
----------------------------------------
Diagnostics instantiate ``cfg.task.dataset`` straight from the checkpoint
(``screen_conditioning.py:63``, ``probe_relpose.py``), so a dataset-level dropout
would corrupt the very measurements it is meant to precede -- and would be baked
into the saved cfg. In ``compute_loss`` it is training-only by construction:
rollouts, eval and every probe go through ``predict_action``.

Why the neutral value comes from ``unnormalize``
------------------------------------------------
The trunk consumes *normalized* values, and a dropped sample should present the
neutral one -- normalized 0.0 -- so the raw value that maps there is recovered
with the normalizer's own inverse. Zeroing the RAW values instead would hand the
trunk whatever the normalizer maps 0 to, which is somewhere else entirely; the
test mutation-checks that distinction.

Properties, all pinned by tests/test_prop_dropout.py
----------------------------------------------------
  * ``proprio_dropout=0.0`` delegates to ``super().compute_loss``, so M3's loss
    and gradients are bit-identical **by construction** -- this class copies no
    upstream body, so there is nothing to drift.
  * One Bernoulli draw per **sample**, shared across the three keys and all
    observation steps: a dropped sample is one whose proprioception is missing
    for the whole window, which is the deployment failure this regularises
    against. The three keys are one modality and drop together.
  * ``self.training`` gates it, with a caveat worth knowing: this workspace never
    puts ``self.model`` into eval mode (it evals the EMA copy at
    ``train_diffusion_unet_image_workspace.py:221-223``), so the dropout **is**
    live during validation and ``val_loss`` is the loss under dropout. That makes
    it a third val_loss, incomparable to M3's and M4's -- compare it only at
    matched epochs, as PROGRESS.md's *Noise and resolution* already requires.
  * The input batch is never mutated: a copy is made, so the caller (and any
    logging) still sees the real observations.
  * **A `p>0` run is NOT RNG-locked against `m3v13`.** Drawing the mask consumes
    one `torch.rand` per step *before* the parent draws its noise and timesteps,
    so the two runs' noise streams differ from the first step on. Unlike M4's
    aux pair -- which is RNG-locked, and says so -- this comparison is
    run-to-run, at the same n=1-per-cell resolution as every other cross-cell
    number here. The p=0 case *is* exact: it delegates before any draw.
"""
from typing import Dict, List

import torch

from diffusion_policy.policy.diffusion_unet_image_policy import (
    DiffusionUnetImagePolicy)

# the 3 low-dim keys carrying proprioception; MUST equal screen_conditioning.py's
PROPRIO_SUFFIX = ('eef_pos', 'eef_quat', 'gripper_qpos')


class DiffusionUnetImagePolicyPropDrop(DiffusionUnetImagePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler,
            obs_encoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            proprio_dropout: float = 0.0,
            **kwargs):
        """
        NOTE: `proprio_dropout` is a NAMED argument and must stay that way. The
        parent stores `self.kwargs = kwargs` and `conditional_sample` forwards
        `**self.kwargs` into `scheduler.step(...)`, so a stray key would raise
        TypeError at ROLLOUT time -- hours into a run, not at construction.
        (Same trap M4 documents in diffusion_unet_image_policy_aux.py.)
        """
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            **kwargs)

        if not 0.0 <= proprio_dropout <= 1.0:
            raise ValueError(f'proprio_dropout must be in [0, 1], got {proprio_dropout}')
        self.proprio_dropout = float(proprio_dropout)
        self.last_prop_drop_frac = 0.0

        # loud, not silent: the keys are taken from shape_meta, so a mismatch
        # between this module and the task config is a construction-time error
        obs_keys = list(shape_meta['obs'].keys())
        self.prop_keys: List[str] = sorted(
            k for k in obs_keys if k.endswith(PROPRIO_SUFFIX))
        if len(self.prop_keys) != len(PROPRIO_SUFFIX):
            raise ValueError(
                f'expected {len(PROPRIO_SUFFIX)} proprio keys ending in '
                f'{PROPRIO_SUFFIX}, found {self.prop_keys} among {obs_keys}')

        if self.proprio_dropout > 0.0:
            # the launch gate: `grep propdrop <run>.log` proves the override took
            print(f'[propdrop] ACTIVE p={self.proprio_dropout} on {self.prop_keys} '
                  f'(denoising-network global_cond only; the trajectory '
                  f'conditioning is untouched)')

    # ========= training  ============
    def compute_loss(self, batch):
        if not self.proprio_dropout > 0.0 or not self.training:
            # p=0 is the unchanged M3 path *by construction* -- nothing copied,
            # so there is no body that could drift from upstream
            return super().compute_loss(batch)
        return super().compute_loss(self.drop_proprioception(batch))

    def drop_proprioception(self, batch: Dict) -> Dict:
        """`batch` with the proprio keys set to their neutral (normalized-0)
        value for a random `proprio_dropout` fraction of samples.

        Exposed separately so tests and diagnostics can call it without going
        through the loss, and so the realized fraction is observable.
        """
        obs = batch['obs']
        x0 = obs[self.prop_keys[0]]
        B = x0.shape[0]
        keep_shape = (B,) + (1,) * (x0.dim() - 1)
        drop = torch.rand(B, device=x0.device) < self.proprio_dropout
        self.last_prop_drop_frac = float(drop.float().mean())

        new_obs = dict(obs)
        for key in self.prop_keys:
            x = obs[key]
            # the raw value the normalizer maps to exactly 0 -- the trunk's
            # neutral input for this channel
            neutral = self.normalizer[key].unnormalize(torch.zeros_like(x))
            new_obs[key] = torch.where(drop.view(keep_shape), neutral, x)

        out = dict(batch)
        out['obs'] = new_obs
        return out
