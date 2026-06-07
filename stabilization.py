"""
stabilization.py — Algorithmic stabilization mechanisms (§3.3, §4).

Three components that modify the standard GRPO loss:
  1. K3 KL Estimator     — Unbiased low-variance KL divergence (Schulman)
  2. PSPO                 — Probability Smoothing Policy Optimization
  3. Entropy Filter       — Threshold-based advantage masking

TRL 1.0.0 already includes K3 KL by default, but we provide the
reference implementation and the building blocks for PSPO + entropy gating.
"""

import torch
import math


# ═══════════════════════════════════════════════════════════════
# K3 Unbiased KL Estimator (§4)
# ═══════════════════════════════════════════════════════════════

def k3_kl_divergence(log_ratio: torch.Tensor) -> torch.Tensor:
    """
    Schulman K3 unbiased KL estimator:  r - log(r) - 1
    where r = π_θ / π_ref (the density ratio).

    TRL 1.0.0 uses: exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1
    which is equivalent to: π_ref/π_θ - log(π_ref/π_θ) - 1
    Both forms are valid K3 estimators.
    """
    # r = π_θ / π_ref = exp(log_π_θ - log_π_ref)
    r = torch.exp(log_ratio)
    return r - log_ratio - 1


# ═══════════════════════════════════════════════════════════════
# PSPO — Probability Smoothing Policy Optimization (§3.3, §4)
# ═══════════════════════════════════════════════════════════════

def pspo_smooth_logprobs(
    logprobs_current: torch.Tensor,
    logprobs_reference: torch.Tensor,
    delta: float = 0.1,
) -> torch.Tensor:
    """
    Smooth current policy probabilities toward the reference policy.

    r_i^PSPO = ((1-δ) * π_θ + δ * π_ref) / π_old

    In log space:
      π_pspo = log((1-δ)*exp(log_π_θ) + δ*exp(log_π_ref))

    Args:
        logprobs_current: log π_θ(o | q)        shape (B, T)
        logprobs_reference: log π_ref(o | q)    shape (B, T)
        delta: smoothing coefficient δ ∈ [0, 1]

    Returns:
        Smoothed log-probabilities, same shape.
    """
    probs_current = torch.exp(logprobs_current)
    probs_reference = torch.exp(logprobs_reference)
    probs_smoothed = (1 - delta) * probs_current + delta * probs_reference
    return torch.log(probs_smoothed.clamp(min=1e-12))


# ═══════════════════════════════════════════════════════════════
# Entropy-Filtered Advantage Masking (§3.3, §4)
# ═══════════════════════════════════════════════════════════════

def entropy_filter_mask(
    entropies: torch.Tensor,
    completion_mask: torch.Tensor,
    threshold: float = 0.693,  # ln(2)
) -> torch.Tensor:
    """
    Build a mask that zeros out advantages for sequences exceeding
    the entropy threshold (§3.3: if H > ln(2), zero the advantage).

    Args:
        entropies: per-token entropy, shape (B, T)
        completion_mask: valid completion tokens, shape (B, T)
        threshold: entropy threshold (default ln(2) ≈ 0.693)

    Returns:
        Mask of shape (B, 1) or (B, T) — 0 for high-entropy sequences.
    """
    # Compute mean entropy per sequence (over valid tokens)
    masked_entropies = entropies * completion_mask
    seq_lengths = completion_mask.sum(dim=-1).clamp(min=1)
    mean_entropy = masked_entropies.sum(dim=-1) / seq_lengths  # (B,)

    # Sequences with mean entropy > threshold get zeroed
    keep_mask = (mean_entropy <= threshold).float()  # (B,)
    return keep_mask


# ═══════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════

def compute_per_token_entropy(logprobs: torch.Tensor) -> torch.Tensor:
    """
    Approximate per-token entropy from log probabilities.
    H ≈ -mean(logprob) for a single sample.

    When true distribution is unknown, the negative log-probability
    under the current policy serves as a sample-based entropy estimate.
    """
    return -logprobs
