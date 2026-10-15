"""Per-view Gaussian state estimates and the Product-of-Experts fusion (PLAN_G.md stage 1).

One **shared** head maps a per-view latent ``z_v`` to a diagonal Gaussian
``q_v = N(mu_v, diag(exp(logvar_v)))`` over the latent state used for control:

* the **mean** branch is residual and zero-initialized, ``mu_v = z_v + f_mu(z_v)``,
  so at init ``mu_v`` is *bitwise* ``z_v`` -- the probabilistic cells start as an
  exact detour of the deterministic baseline, not as a re-randomized model;
* the **log-variance** branch is zero-initialized and clamped,
  ``logvar_v = clamp(f_sigma(z_v), logvar_min, logvar_max)``, so at init every
  view claims ``sigma^2 = 1`` (uniform precision) and the PoE fusion starts as a
  plain average of the means. ``0`` is strictly inside the clamp by construction,
  so the init is never a bound.

``poe_fuse`` multiplies the live views' Gaussians:
``q_g ∝ Prod_v q_v  =>  precision p_v = exp(-logvar_v)``,
``mu_g = sum p_v mu_v / sum p_v``, ``logvar_g = -log sum p_v``.

One gauge fact that falls out of the normalization and matters for reading the
experiment: only precision *ratios* enter ``mu_g`` -- a uniform additive shift
of every live slot's ``logvar`` (e.g. the logvar head's bias direction) leaves the
fused mean bit-for-bit unchanged and moves only ``logvar_g``. The absolute
uncertainty level is carried by the fused logvar, not by the mean; the test
suite pins this by perturbing the logvar branch heterogeneously.

Two things this module is explicit about:

1. **PoE is a fusion choice, not the true posterior.** If the per-view estimates
   were both posteriors of one prior, the product would need dividing by that
   prior (PLAN_G.md section 2.3), and views of the same scene are strongly
   correlated, so a plain product can double-count evidence and end up
   overconfident. That is a *question for the experiment*, not an assumption of
   this code -- the uniform-mean fusion (``uniform_fuse``) is the control arm,
   and the per-epoch log-variance instrument is what makes overconfidence
   visible.
2. **The mask is applied inside the weight normalizer**, not by the caller.
   Dead slots are zero vectors carrying ``logvar = 0`` at init, i.e. ``p = 1`` --
   a caller that forgets the mask silently adds one certain phantom expert per
   dead slot and shrinks ``mu_g`` toward zero. Masking here makes that failure
   unrepresentable.

Degeneration and numerics (used as test pins):

* N=1: the weights ``w_v / sum(w)`` come out as exactly ``1.0`` (a float
  divided by itself), so ``mu_g`` is *bitwise* the single live mean; the
  combined ``logvar_g = -log(exp(-logvar))`` round-trips to ~1 ulp, not exactly
  (``log``/``exp`` are not inverses in IEEE754).
* With ``logvar in [-6, 2]`` (the default clamp): ``p in [0.135, 403]``,
  ``sum p <= 11 * 403`` over the 11 slots, so ``logvar_g in [-8.4, 2]`` --
  everything far inside fp32. The clamp *is* the overflow guarantee.
* ``mu_g`` is a convex combination of the live means, so its norm can never
  exceed the largest live ``|mu_v|`` for any precision values.
* Computed in fp32 and cast on return: ``exp(-x)`` overflows in fp16 above
  ``x ~ -11``, so this must not run under autocast in half precision.

No dropout / no stochastic layers: the head must consume no RNG at forward time
(the draw-level lockstep between the fusion cells and the deterministic baseline
rests on it), and it is shared across slots for the same reason as
``PerViewAuxActionHead`` -- slots are filled from a per-sample random draw, so a
slot-specific head could key off slot identity.
"""
import torch
import torch.nn as nn


def masked_weighted_mean(tokens, weight, active):
    """``(N,K,D),(N,K,D),(N,K)`` -> ``((N,D) mean, (N,D) denominator)``.

    ``mean[n] = sum_v active[n,v] w[n,v,:] tokens[n,v,:] / sum_v active[n,v] w[n,v,:]``
    per latent dimension.

    The mask is applied HERE (dead-slot weights are exactly zeroed, not merely
    ignored downstream) and the normalizer is formed by that same masked weight,
    so every live slot's contribution is ``w / sum w``: at N=1 live slot that is
    ``p / p == 1.0`` exactly, and the returned mean is bitwise the live token.
    Any row with a zero total weight (all slots dead, or a caller passing zero
    weights on live slots -- a possible fp underflow) raises rather than divide
    to NaN.
    """
    if tokens.dim() != 3 or weight.shape != tokens.shape:
        raise ValueError(
            f'expected tokens and weight both (N, K, D), got '
            f'{tuple(tokens.shape)} and {tuple(weight.shape)}')
    if active.dim() != 2 or active.shape != tokens.shape[:2]:
        raise ValueError(
            f'expected active (N, K) matching tokens, got '
            f'{tuple(active.shape)} vs {tuple(tokens.shape[:2])}')
    if active.dtype != torch.bool:
        raise ValueError(f'active must be bool, got {active.dtype}')
    w = torch.where(active.unsqueeze(-1),
                    weight.float(),
                    torch.zeros((), dtype=torch.float32, device=weight.device))
    denom = w.sum(dim=1, keepdim=True)                       # (N, 1, D)
    if not bool((denom > 0).all()):
        raise ValueError(
            'a row has zero total weight over its live slots -- every sample '
            'needs at least one live slot with positive weight')
    mean = (w / denom * tokens.float()).sum(dim=1)           # (N, D)
    return mean, denom.squeeze(1)


def poe_fuse(mu, logvar, active):
    """Product of Experts over the live slots -> ``(mu_g (N,D), logvar_g (N,D))``.

    ``q_g ∝ Prod_v N(mu_v, diag(exp(logvar_v)))``: precision ``p_v = exp(-logvar_v)``
    and ``logvar_g = -log sum_v p_v``. Expects ``logvar`` already clamped (the
    head's contract); the fp32 conversion happens here.
    """
    p = torch.exp(-logvar.float())
    mu_g, denom = masked_weighted_mean(mu, p, active)
    logvar_g = -torch.log(denom)
    return mu_g.to(mu.dtype), logvar_g.to(mu.dtype)


def uniform_fuse(mu, active):
    """The control fusion: an unweighted mean over the live slots -> ``(N,D)``.

    Exactly ``poe_fuse`` at ``logvar == 0`` -- ``exp(-0.0)`` is ``1.0``, the same
    float the mask contributes, so the two constructions are bitwise identical at
    the head's init. That identity is a test pin: it is what makes "C starts as
    D" a checkable claim rather than a comment.
    """
    w = torch.ones_like(mu, dtype=torch.float32)
    mu_g, _ = masked_weighted_mean(mu, w, active)
    return mu_g.to(mu.dtype)


class PerViewGaussianHead(nn.Module):
    """``(M, D) -> ((M, D) mu, (M, D) logvar)``, one row per (frame, view) pair.

    House pattern (`per_view_aux_head.py`): ``Linear -> SiLU -> Linear`` with the
    output layer zero-initialized. Both last layers are zeroed, and the mean
    branch adds its input back (residual), so at init ``mu == z`` and
    ``logvar == 0`` *bitwise* -- see the module docstring for why that matters.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 logvar_min: float = -6.0, logvar_max: float = 2.0):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1:
            raise ValueError(f'bad dims: in={in_dim} hidden={hidden_dim}')
        if not (logvar_min < 0.0 < logvar_max):
            raise ValueError(
                f'logvar bounds must straddle 0 (so the zero-init logvar '
                f'branch starts unclamped), got [{logvar_min}, {logvar_max}]')
        if not (float('-inf') < logvar_min and logvar_max < float('inf')):
            raise ValueError(
                f'logvar bounds must be finite, got [{logvar_min}, {logvar_max}]')
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.mu_net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.in_dim),
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.in_dim),
        )
        for net in (self.mu_net, self.logvar_net):
            last = net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor):
        """``(M, in_dim)`` per-view latents -> ``(mu, logvar)``, both ``(M, in_dim)``."""
        if z.dim() != 2 or z.shape[-1] != self.in_dim:
            raise ValueError(
                f'expected (M, {self.in_dim}) per-view latents, got '
                f'{tuple(z.shape)}')
        mu = z + self.mu_net(z)
        logvar = self.logvar_net(z).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar
