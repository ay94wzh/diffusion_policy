"""PLAN_Z's z-supervision round: two cross-view losses on the m4base recipe.

Idea 1 -- `L_align` on ``z_v``: a shared projection head ``P_s`` maps each live
view's latent to a *shared-semantics* space; views of the same physical state
(all live views of one sample) are pulled together there by the mean pairwise
squared distance between L2-normalized projections. Only ``P_s``'s output is
constrained -- ``z_v`` itself stays free, which is the point ("do not force the
whole ``z_v`` to agree; keep the view-specific information").

Idea 2 -- `L_cons` on ``z_g``: an independent SECOND view draw (top-level
``batch['obs2']``, emitted by the dataset under a per-sample private stream)
produces a second fused latent ``z_g2``; the SAME noisy action, timestep and
noise go through the UNet under both conditionings, and the two noise
predictions are pulled together (the actions are in the base frame, so the
targets agree by construction).

One class, two weights -- the cells are CLI overrides:
    A zalign: (align_weight=W_A, cons_weight=0,   emit_second_draw=False)
    B zcons : (0,               W_C, emit_second_draw=True)
    C zboth : (W_A,             W_C, emit_second_draw=True)
and the same `(W_A, W_C)` are used in all cells, so C decomposes as A + B.
The weights are calibrated pre-launch by the family convention
``W = 0.33 * diff0 / loss0`` (one batch-0 forward), registered in the driver.

RNG contract (claimed and CPU-pinned in tests/test_z_supervision.py):
  * construction: every parent parameter/head is built exactly as the aux policy
    builds it; the semantic head is built under a private-stream
    save/reseed/restore and the global stream is restored bit-exactly, so
    constructing this policy consumes nothing extra.
  * step, cell A: `L_align` consumes no RNG and adds no forward -- every draw
    (crops, noise, timesteps, dataset draws) is bit-identical to the aux
    control; only the parameter trajectory differs.
  * step, cells B/C: the second encoder pass runs under a torch RNG
    save/restore, so the parent's draws keep their stream positions too; the
    second draw's own subset rides the dataset's per-sample private stream.
  * the enc2 crops are a REUSE of draw-1's saved state (not fresh entropy) --
    deliberate: the cons residual should reflect the view set, not crop noise.

Loss-log convention (workspace seam): ``aux_loss`` stays the UNWEIGHTED aux
term (comparable to m4base's log), and the objective decomposes as
``train_loss = diff_loss + aux_loss_weight*aux_loss
              [+ align_weight*last_align_loss][+ cons_weight*last_cons_loss]``.

NOT claimed: identical parameter count / optimizer parameter list (+ ~262 k for
the shared semantic head, ~0.09% of the ~290 M policy), and anything about the
gradient path of `L_cons` beyond its registered form -- it is symmetric (no
stop-gradient), so it flows into BOTH passes and, through both conditionings,
into the shared encoder.
"""
from typing import Dict

import torch
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.policy.diffusion_unet_image_policy_aux import (
    DiffusionUnetImagePolicyAux)
from diffusion_policy.model.vision.shared_semantic_head import (
    SharedSemanticHead)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.multiview_image_dataset import (
    AUX_ACTION_KEY, SECOND_OBS_KEY)


def masked_pair_mean(pair_loss: torch.Tensor, pair_rows: torch.Tensor,
                     n_rows: int) -> torch.Tensor:
    """``(P,)`` per-pair losses + ``(P,)`` row indices -> scalar.

    Mean over each row's pairs, then mean over the rows that HAVE at least one
    pair. `masked_view_mean` (the aux loss's reduction) cannot be reused here:
    a row with fewer than two live views contributes no pair and would divide
    0 by 0 -> NaN, poisoning the scalar for the whole batch. Rows with no pair
    are excluded from the mean rather than zero-filled -- zero-filling would
    dilute the term by the share of N=1 draws (a uniform 1/11 at [1, 11]).
    """
    if pair_loss.numel() == 0:
        raise ValueError('masked_pair_mean needs at least one pair')
    per_row = pair_loss.new_zeros(n_rows).index_add_(0, pair_rows, pair_loss)
    counts = pair_loss.new_zeros(n_rows).index_add_(
        0, pair_rows, pair_loss.new_ones(pair_loss.shape[0]))
    live = counts > 0
    return (per_row[live] / counts[live]).mean()


class DiffusionUnetImagePolicyZSup(DiffusionUnetImagePolicyAux):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler,
            obs_encoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            aux_loss_weight: float = 0.0,
            aux_n_steps: int = 8,
            aux_hidden_dim: int = 256,
            align_weight: float = 0.0,
            cons_weight: float = 0.0,
            semantic_hidden_dim: int = 256,
            semantic_init_seed: int = 42,
            **kwargs):
        """
        NOTE: every new parameter is a NAMED argument above and must stay that
        way -- the ancestor's warning, inherited: `conditional_sample` forwards
        `self.kwargs` into `scheduler.step(...)`, so a stray key would raise
        TypeError at ROLLOUT time, hours into a run.
        """
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            aux_loss_weight=aux_loss_weight,
            aux_n_steps=aux_n_steps,
            aux_hidden_dim=aux_hidden_dim,
            **kwargs)
        self.align_weight = float(align_weight)
        self.cons_weight = float(cons_weight)

        # Built unconditionally -- including at weight 0 -- so all three cells
        # hold identical parameters at construction and the weight-0 arm is a
        # bit-identical control. Private stream: the parent's construction above
        # consumed exactly the stream the deterministic baseline would have
        # consumed, and this head consumes none of it -- restore bit-exactly.
        # (Do NOT keep the Generator around; and the head is deliberately NOT
        # zero-init -- see shared_semantic_head.py's docstring.)
        cpu_state = torch.get_rng_state()
        try:
            generator = torch.Generator().manual_seed(int(semantic_init_seed))
            torch.set_rng_state(generator.get_state())
            self.semantic_head = SharedSemanticHead(
                in_dim=self.obs_encoder.fused_dim,
                hidden_dim=semantic_hidden_dim)
        finally:
            torch.set_rng_state(cpu_state)
        self.last_align_loss = 0.0
        self.last_cons_loss = 0.0

    # ========= training  ============
    def compute_loss(self, batch):
        if not (self.align_weight > 0.0 or self.cons_weight > 0.0):
            # the unchanged aux path, deliberately -- for this family that IS
            # the m4base control (not the base policy: the encoder path and the
            # base-frame aux term are the recipe the round is added onto). Same
            # draws, no z-supervision graph.
            loss = super().compute_loss(batch)
            self.last_align_loss = 0.0
            self.last_cons_loss = 0.0
            return loss
        return self._compute_loss_with_zsup(batch)

    def align_loss(self, z_views: torch.Tensor,
                   active: torch.Tensor) -> torch.Tensor:
        """PLAN_Z idea 1: shared-semantics agreement over live view pairs.

        ``z_views`` ``(M, K, D)`` raw per-view latents (one row per
        (frame, view) pair, as `forward_full` emits), ``active`` ``(M, K)``
        bool. A row is one (sample, obs step): all of its live views are
        renders of the SAME physical state, which is what makes the pairwise
        target well-posed. Rows with fewer than two live views are excluded
        (see `masked_pair_mean`); an all-N=1 batch returns a zero that still
        carries the graph (backward works; the head's grads are zeros, not
        None), so a lone-N=1 batch cannot crash a run.

        ``sum`` over D, not mean: PLAN_Z's literal squared L2 norm; the loss
        weight absorbs the D = 512 factor. ``eps=1e-6`` on the normalization
        bounds the gradient for very small head outputs.
        """
        M, K, D = z_views.shape
        s = F.normalize(
            self.semantic_head(z_views.reshape(-1, D)), dim=-1, eps=1e-6
            ).reshape(M, K, D)
        iu = torch.triu_indices(K, K, offset=1, device=z_views.device)
        live = active[:, iu[0]] & active[:, iu[1]]           # (M, P)
        pair_rows, pair_idx = live.nonzero(as_tuple=True)
        if pair_rows.numel() == 0:
            return s.sum() * 0.0
        d = s[pair_rows, iu[0][pair_idx]] - s[pair_rows, iu[1][pair_idx]]
        return masked_pair_mean(d.pow(2).sum(dim=-1), pair_rows, M)

    def cons_loss(self, pred: torch.Tensor, pred2: torch.Tensor,
                  loss_mask: torch.Tensor) -> torch.Tensor:
        """PLAN_Z idea 2: two view draws, one noise prediction.

        Mirrors the parent's diffusion reduction exactly (same three lines,
        same `loss_mask`), so the cons term and the diff term are on the same
        scale and the family's 0.33 share means what it says. In the committed
        config `loss_mask` is all-True (`LowdimMaskGenerator` with
        `action_visible=False` leaves nothing to inpaint); the three lines are
        kept anyway so a future inpainting port stays correct.
        """
        loss = F.mse_loss(pred2, pred, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()

    def _second_global_cond(self, batch, batch_size, this_nobs):
        """Conditioning for the second view draw -- consumes no RNG.

        The second encoder pass would otherwise shift the parent's noise and
        timestep draws (crop indices are drawn from the global torch stream),
        so it runs under a CPU+CUDA RNG save/restore, the latentprobe idiom.
        Consequence, stated on purpose: the second draw's crop windows are a
        REUSE of the saved state rather than fresh entropy -- deliberate, so
        the cons residual reflects the view subset, not crop noise.

        The low-dim concat reuses draw 1's `this_nobs`: `forward_full` reads
        only the rgb/cam/mask/eef-hist keys (all view-dependent and all in
        `obs2`), while the low-dim keys are view-independent and come from the
        policy -- so the two fused latents differ only by the view subset.
        """
        if SECOND_OBS_KEY not in batch:
            raise ValueError(
                f'cons_weight={self.cons_weight} needs batch[{SECOND_OBS_KEY!r}]: '
                f'set task.dataset.emit_second_draw=true (only then does the '
                f'dataset emit the second view draw)')
        nobs2 = self.normalizer.normalize(batch[SECOND_OBS_KEY])
        this_nobs2 = dict_apply(nobs2,
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        cpu_state = torch.get_rng_state()
        cuda_states = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
        try:
            enc2 = self.obs_encoder.forward_full(this_nobs2)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        nobs_features2 = torch.cat(
            [enc2['z_global']]
            + [this_nobs[k] for k in self.obs_encoder.low_dim_keys], dim=-1)
        return nobs_features2.reshape(batch_size, -1)

    def _compute_loss_with_zsup(self, batch):
        """The aux policy's `_compute_loss_with_aux` body, plus the two terms.

        Copied rather than refactored, the established pattern here -- and the
        copy IS the pin: at (0, 0) `compute_loss` delegates to the parent and
        tests/test_z_supervision.py asserts the two agree bit-for-bit, loss and
        all shared gradients included; at positive weights the two additions
        are the only differences. The parent's draw ORDER (noise, then
        timesteps) is untouched -- the second encoder pass runs under an RNG
        save/restore, so its position is stream-neutral by construction.
        """
        if not self.obs_as_global_cond:
            raise NotImplementedError(
                'zsup implements the obs_as_global_cond path only; the '
                'inpainting path would need its own wiring')

        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        trajectory = nactions
        cond_data = trajectory

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs,
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        enc = self.obs_encoder.forward_full(this_nobs)
        nobs_features = torch.cat(
            [enc['z_global']]
            + [this_nobs[k] for k in self.obs_encoder.low_dim_keys], dim=-1)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)

        # ---- PLAN_Z idea 2: the second draw's conditioning -------------------
        # Placed before the parent's two adjacent draws so the copy reads like
        # the parent with one block inserted; the save/restore inside
        # `_second_global_cond` makes the position stream-neutral.
        global_cond2 = None
        if self.cons_weight > 0.0:
            global_cond2 = self._second_global_cond(
                batch, batch_size, this_nobs)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps,
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        diff_loss = loss.mean()

        # ---- PLAN_Z idea 2: same noisy action, timestep and noise ------------
        if self.cons_weight > 0.0:
            # SAME noisy_trajectory, SAME timesteps -- only the conditioning
            # differs (the second view draw). Symmetric, no stop-gradient,
            # exactly PLAN_Z's form.
            pred2 = self.model(noisy_trajectory, timesteps,
                local_cond=local_cond, global_cond=global_cond2)
            cons_loss = self.cons_loss(pred, pred2, loss_mask)
            self.last_cons_loss = float(cons_loss.detach())
        else:
            cons_loss = None
            self.last_cons_loss = 0.0

        # ---- PLAN_Z idea 1: shared-semantics alignment on draw 1's z_v -------
        # Draw 1 only, so cells A and C carry an identical align term (one
        # variable at a time); consumes no RNG, adds no forward.
        if self.align_weight > 0.0:
            align_loss = self.align_loss(enc['z_views'], enc['view_active'])
            self.last_align_loss = float(align_loss.detach())
        else:
            align_loss = None
            self.last_align_loss = 0.0

        # ---- M4: per-view auxiliary base-frame action prediction -------------
        if self.aux_loss_weight > 0.0:
            aux_loss = self.aux_loss(enc, batch, batch_size)
            self.last_aux_loss = float(aux_loss.detach())
        else:
            aux_loss = None
            self.last_aux_loss = 0.0
        self.last_diff_loss = float(diff_loss.detach())

        total = diff_loss
        if aux_loss is not None:
            total = total + self.aux_loss_weight * aux_loss
        if align_loss is not None:
            total = total + self.align_weight * align_loss
        if cons_loss is not None:
            total = total + self.cons_weight * cons_loss
        return total
