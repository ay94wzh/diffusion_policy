"""M4's per-view auxiliary action head (PROPOSAL.md section 2.4).

One **shared** MLP maps a per-view latent ``z_v`` to that view's camera-frame
action chunk. This is the supervision the proposal calls "what forces ``z_v`` to
be geometrically meaningful rather than merely view-tagged": to predict where to
act *in camera k's frame* from a single view, ``z_v`` has to encode where the
scene is relative to that camera.

Shared, not per-slot: the dataset fills slots from a per-sample random draw of
the view pool, so a per-slot head could key off slot identity (which the
architecture deliberately randomizes away) and could not be trained at all when
a slot is inactive.

The output layer is zero-initialized, matching this repo's pattern for added
branches (the FiLM heads, the zeroed Pluecker channels): the aux term starts at
exactly zero and ramps in, so an aux-on run begins as M3. Note this does NOT
delay the shaping by more than one step -- the head's own weights receive
gradient immediately, so the trunk starts seeing the aux gradient from step 1.

No dropout / no stochastic layers anywhere: the head must consume no RNG, or
the aux-on and aux-off arms of the ablation would stop being RNG-locked and the
pair would no longer differ by exactly one loss term.
"""
import torch
import torch.nn as nn


class PerViewAuxActionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        if in_dim < 1 or out_dim < 1 or hidden_dim < 1:
            raise ValueError(
                f'bad dims: in={in_dim} out={out_dim} hidden={hidden_dim}')
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(M, in_dim)`` -> ``(M, out_dim)``, one row per (frame, view) pair."""
        if z.dim() != 2 or z.shape[-1] != self.in_dim:
            raise ValueError(
                f'expected (M, {self.in_dim}) per-view latents, got '
                f'{tuple(z.shape)}')
        return self.net(z)
