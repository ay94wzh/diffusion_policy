"""PLAN_Z's shared-semantics projection head ``P_s`` (the `z_v` alignment arm).

One **shared** MLP maps each live view's latent ``z_v`` to a projection in which
the *shared semantics* of a scene are supposed to live: views of the same
physical state (all live views of one sample) are pulled together there by the
alignment loss, while the raw ``z_v`` is left free to keep its view-specific
information. ("Align the shared semantics, do not force `z_v` itself to agree.")

Shared, not per-slot, for the same reason as the other per-view heads: the
dataset fills slots from a per-sample random draw of the view pool, so a
per-slot head could key off slot identity (which the architecture deliberately
randomizes away) and could not be trained at all when a slot is inactive.

**Default (random) initialization, deliberately NOT zero-init** — the opposite
choice from `PerViewAuxActionHead` and `PerViewGaussianHead`. Zero-init works
there because their outputs are unnormalized (the aux head regresses a nonzero
target; the Gaussian head's mean branch is a residual around ``z``). Here the
alignment loss L2-normalizes the output first, so a zero-init head would emit
zero vectors whose normal form is zero: ``L_align ≡ 0`` AND the gradient into
the head is exactly zero (the upstream factor is ``2(s_i − s_j) = 0``), so the
head could never leave zero and the term would be dead for the whole run.
Random init keeps both the loss and its gradient live from step 1.

No dropout / no stochastic layers anywhere: the head must consume no RNG at
forward time, or the align cell would stop being RNG-locked to its control.
Construction consumes no RNG either — the policy builds the head under a
private-stream save/reseed/restore (the `prob_view_conditioned_obs_encoder`
idiom). The module registers no buffers, on purpose (the pairwise index set in
the policy is computed per call), so cross-class `state_dict()` comparisons in
the tests can stay strict.
"""
import torch
import torch.nn as nn


class SharedSemanticHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1:
            raise ValueError(
                f'bad dims: in={in_dim} hidden={hidden_dim}')
        self.in_dim = int(in_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.in_dim),
        )
        # NOTE: no zero-init, no buffer registration — see the module docstring.

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(M, in_dim)`` -> ``(M, in_dim)``, one row per (frame, view) pair."""
        if z.dim() != 2 or z.shape[-1] != self.in_dim:
            raise ValueError(
                f'expected (M, {self.in_dim}) per-view latents, got '
                f'{tuple(z.shape)}')
        return self.net(z)
