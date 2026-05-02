"""Tests for compute_aux_nll_loss(...).

Cases:
 1. Mixed mask: aux_loss equals the closed-form masked-mean NLL over valid
    samples.
 2. All-False mask: aux_loss == 0 with no NaN; diag values are zero.
 3. Episode-step composition: when episode_step < episode_start_mask_steps,
    samples are masked even if tri_valid=True.
 4. NLL clamp: extreme inputs do not produce per-dim values above the clamp.
"""
import traceback

import torch

from mappo_with_aux import compute_aux_nll_loss


def _expected_nll(target, mu, log_var):
    inv_var = (-log_var).exp()
    return 0.5 * ((target - mu).pow(2) * inv_var + log_var).sum(dim=-1)


def run_masking_tests(results, device, verbose=False):
    # ----------------------------------------------------------------
    # 1. Mixed mask matches manual closed-form
    # ----------------------------------------------------------------
    try:
        torch.manual_seed(0)
        N = 8
        mu = torch.randn(N, 3, device=device)
        log_var = torch.full((N, 3), -1.0, device=device)
        target = mu + 0.3 * torch.randn(N, 3, device=device)
        valid = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.bool, device=device)
        episode_step = torch.full((N,), 100, dtype=torch.int64, device=device)  # all past warm-up

        loss, diag = compute_aux_nll_loss(
            mu, log_var, target, valid, episode_step,
            episode_start_mask_steps=0, nll_clamp=1e9,
        )

        nll_per = _expected_nll(target, mu, log_var)
        expected = (nll_per * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        assert torch.allclose(loss, expected, atol=1e-6), \
            f"loss={loss.item()}, expected={expected.item()}"
        # Diagnostics on this batch
        assert torch.isfinite(diag["predicted_rmse"])
        assert torch.isfinite(diag["calibration_2sigma_coverage"])
        assert torch.allclose(diag["effective_sample_fraction"], valid.float().mean(), atol=1e-6)
        results.add_pass("Mixed mask matches masked-mean NLL")
    except Exception:
        results.add_fail("Mixed mask matches masked-mean NLL", traceback.format_exc())

    # ----------------------------------------------------------------
    # 2. All-False mask returns 0; no NaN; diag values are zero
    # ----------------------------------------------------------------
    try:
        N = 5
        mu = torch.randn(N, 3, device=device, requires_grad=True)
        log_var = torch.zeros(N, 3, device=device, requires_grad=True)
        target = torch.randn(N, 3, device=device)
        valid = torch.zeros(N, dtype=torch.bool, device=device)
        episode_step = torch.full((N,), 100, dtype=torch.int64, device=device)

        loss, diag = compute_aux_nll_loss(
            mu, log_var, target, valid, episode_step,
            episode_start_mask_steps=0, nll_clamp=1e9,
        )
        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        assert loss.item() == 0.0, f"loss should be 0, got {loss.item()}"
        # Diagnostics default to zero on empty mask
        assert diag["predicted_rmse"].item() == 0.0
        assert diag["calibration_2sigma_coverage"].item() == 0.0
        assert diag["effective_sample_fraction"].item() == 0.0
        # Backward should produce zero grad on inputs since loss is structurally zero.
        loss.backward()
        assert mu.grad is None or mu.grad.abs().sum().item() == 0.0
        assert log_var.grad is None or log_var.grad.abs().sum().item() == 0.0
        results.add_pass("All-False mask: aux_loss == 0, no NaN, no grad leak")
    except Exception:
        results.add_fail("All-False mask: aux_loss == 0, no NaN, no grad leak",
                         traceback.format_exc())

    # ----------------------------------------------------------------
    # 3. Episode-step mask composes with tri_valid
    # ----------------------------------------------------------------
    try:
        N = 6
        mu = torch.zeros(N, 3, device=device)
        log_var = torch.zeros(N, 3, device=device)
        target = torch.ones(N, 3, device=device)  # nonzero error -> nonzero NLL where mask=True
        valid = torch.ones(N, dtype=torch.bool, device=device)  # all valid by triangulation
        # First 4 samples are pre-warm-up (episode_step < 8), last 2 are post.
        episode_step = torch.tensor([0, 1, 4, 7, 8, 20], dtype=torch.int64, device=device)
        episode_start = 8

        loss, diag = compute_aux_nll_loss(
            mu, log_var, target, valid, episode_step,
            episode_start_mask_steps=episode_start, nll_clamp=1e9,
        )

        # Manual: only samples with episode_step >= 8 contribute.
        passing = episode_step >= episode_start
        nll_per = _expected_nll(target, mu, log_var)  # all equal: 0.5 * 3 = 1.5
        expected = (nll_per * passing.float()).sum() / passing.float().sum().clamp_min(1.0)
        assert torch.allclose(loss, expected, atol=1e-6), \
            f"episode_step composition failed: loss={loss.item()}, expected={expected.item()}"
        assert torch.allclose(
            diag["effective_sample_fraction"],
            passing.float().mean(), atol=1e-6,
        )
        results.add_pass("Episode-step mask composes with tri_valid")
    except Exception:
        results.add_fail("Episode-step mask composes with tri_valid",
                         traceback.format_exc())

    # ----------------------------------------------------------------
    # 4. NLL clamp caps absurd per-dim values
    # ----------------------------------------------------------------
    try:
        N = 2
        # Force extreme NLL: tiny σ (large -log_var > 0) and large (target - mu).
        # log_var = -10 -> 1/σ² = e^10 ≈ 22000. (target - mu)² = 100 -> per-dim NLL ≈ 1.1e6.
        mu = torch.zeros(N, 3, device=device)
        log_var = torch.full((N, 3), -10.0, device=device)
        target = torch.full((N, 3), 10.0, device=device)
        valid = torch.ones(N, dtype=torch.bool, device=device)
        episode_step = torch.full((N,), 100, dtype=torch.int64, device=device)

        clamp = 1000.0
        loss, _ = compute_aux_nll_loss(
            mu, log_var, target, valid, episode_step,
            episode_start_mask_steps=0, nll_clamp=clamp,
        )
        # Per-sample NLL is sum over 3 dims, each clamped to ≤ clamp -> overall sum ≤ 3*clamp.
        assert loss.item() <= 3.0 * clamp + 1e-6, \
            f"NLL clamp violated: loss={loss.item()}, ceiling={3.0*clamp}"
        assert torch.isfinite(loss), f"loss is not finite under clamp: {loss}"
        results.add_pass("NLL per-dim clamp caps extreme values")
    except Exception:
        results.add_fail("NLL per-dim clamp caps extreme values", traceback.format_exc())

    # ----------------------------------------------------------------
    # 5. Tri_valid as float (size-1 last dim, mirrors memory storage)
    # ----------------------------------------------------------------
    try:
        N = 4
        mu = torch.randn(N, 3, device=device)
        log_var = torch.zeros(N, 3, device=device)
        target = mu.clone()
        valid_float = torch.tensor([[1.0], [0.0], [1.0], [0.0]], device=device)
        episode_step = torch.full((N,), 100, dtype=torch.int64, device=device)

        loss, diag = compute_aux_nll_loss(
            mu, log_var, target, valid_float, episode_step,
            episode_start_mask_steps=0, nll_clamp=1e9,
        )
        # Perfect prediction -> per-dim NLL = 0.5 * (0 + 0) = 0 -> sum = 0 -> mean = 0.
        assert loss.item() == 0.0, f"perfect prediction should give zero loss, got {loss.item()}"
        assert torch.allclose(diag["effective_sample_fraction"],
                              torch.tensor(0.5, device=device), atol=1e-6)
        results.add_pass("tri_valid accepts (N, 1) float storage layout")
    except Exception:
        results.add_fail("tri_valid accepts (N, 1) float storage layout",
                         traceback.format_exc())
