"""z_g pose-set head: fused latent -> the pooled camera-pose set of the live
views (PLAN candidate 2; the z_g-supervision line in `PLAN.md`).

One small MLP maps the FUSED latent (`z_global`, one row per (frame) obs step)
to a fixed ``(V, 9)`` pose set: row i == pool view i (**pool ORDER**, not ring
index), each row ``[pos(3) | rot6d(6)]`` of that view's camera. The loss scores
ALL rows against the zero-filled target -- dead views have ZERO targets the
head must match -- so the head is supervised to report WHICH pool views are
live as well as where their cameras are. That is the information the fused
latent does not currently carry (the fusion is ~N-invariant: `cka(n1, full)`
0.998 on the committed parents) and the ONLY sample-dependent content of the
target (the pose table itself is static), which is why a constant head can
only reach the per-row-constant floor, never zero.

Why a new head rather than `PerViewAuxActionHead`
-------------------------------------------------
Same architecture (Linear -> SiLU -> Linear, zero-init output), different
semantics: `PerViewAuxActionHead` reads per-view `z_v` rows and returns a flat
action chunk; this one reads fused `z_global` rows and returns a structured
pose set. Kept separate so neither file's docstring lies about what it does;
the zero-init and no-RNG properties are copied deliberately.

The output layer is zero-initialized, matching this repo's pattern for added
branches (the FiLM heads, the zeroed Pluecker channels, M4's head): the aux
term starts at exactly zero and ramps in. That is also what makes the driver's
pre-launch weight calibration dataset-only -- a zero head predicts zero, so the
batch-0 aux value is exactly the mean target-square over all rows.

No dropout / no stochastic layers anywhere, and no RNG at forward time: the
head must never consume RNG, or train-time and eval-time paths would drift.
"""
import torch
import torch.nn as nn


class ViewPoseSetHead(nn.Module):
    def __init__(self, in_dim: int, n_views: int, pose_dim: int = 9,
                 hidden_dim: int = 256):
        super().__init__()
        if in_dim < 1 or n_views < 1 or pose_dim < 1 or hidden_dim < 1:
            raise ValueError(
                f'bad dims: in={in_dim} views={n_views} pose={pose_dim} '
                f'hidden={hidden_dim}')
        self.in_dim = int(in_dim)
        self.n_views = int(n_views)
        self.pose_dim = int(pose_dim)
        self.out_dim = self.n_views * self.pose_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(M, in_dim)`` -> ``(M, n_views, pose_dim)``, one row per frame.

        The head owns the set structure (it returns the reshaped tensor), so
        the policy cannot mis-reshape a flat ``(M, V*P)`` output.
        """
        if z.dim() != 2 or z.shape[-1] != self.in_dim:
            raise ValueError(
                f'expected (M, {self.in_dim}) fused latents, got '
                f'{tuple(z.shape)}')
        return self.net(z).reshape(-1, self.n_views, self.pose_dim)
