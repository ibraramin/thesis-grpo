"""
rewards.py — Reward functions for GRPO training.

Implements all reward signals defined in the methodology (§3.4, §7.4):
  - Binary correctness reward (exact numerical match)
  - Format adherence reward (<think> + <answer> tags)
  - Graduated correctness reward (Cohort C: tiered partial credit)
  - Dynamic weight factory (Cohort D: decaying format gate)
  - Cohort-aware composite reward builder

TRL GRPOTrainer expects reward functions with signature:
    fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]

All functions operate on batches (lists) for compatibility.
"""

import re
import math


# ── Core Parsing ───────────────────────────────────────────────

def _extract_answer(completion: str) -> float | None:
    """Extract numeric answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>", completion)
    if not match:
        return None
    return float(match.group(1))


def _has_tags(completion: str) -> bool:
    """Check if completion has both <think> and <answer> tags."""
    has_think = re.search(r"<think>.*?</think>", completion, re.DOTALL) is not None
    has_answer = re.search(r"<answer>.*?</answer>", completion, re.DOTALL) is not None
    return has_think and has_answer


# ── Binary Correctness Reward (§7.4) ──────────────────────────

def correctness_reward(prompts: list[str], completions: list[str],
                       answer: list[str], **kwargs) -> list[float]:
    """
    Binary reward: 1.0 if parsed <answer> matches ground truth exactly,
    0.0 otherwise.  (§3.4: r_correct ∈ {0, 1})
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        predicted = _extract_answer(completion)
        if predicted is None:
            rewards.append(0.0)
            continue
        try:
            target = float(gt)
        except (ValueError, TypeError):
            rewards.append(0.0)
            continue
        rewards.append(1.0 if abs(predicted - target) < 1e-5 else 0.0)
    return rewards


# ── Format Adherence Reward (§7.4) ─────────────────────────────

def format_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Binary reward: 1.0 if completion has both <think>...</think>
    and <answer>...</answer> tags, 0.0 otherwise.
    """
    return [1.0 if _has_tags(c) else 0.0 for c in completions]


# ── Graduated Correctness Reward (Cohort C, §3.4) ─────────────

def graduated_correctness_reward(prompts: list[str], completions: list[str],
                                 answer: list[str], **kwargs) -> list[float]:
    """
    Continuous reward for Cohort C (§3.4 table):
      +1.0  — exact match
      +0.5  — within 10% of ground truth
      +0.2  — within 20% of ground truth
      -0.2  — logical failure (answer present but >20% off)
       0.0  — no answer found
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        predicted = _extract_answer(completion)
        if predicted is None:
            rewards.append(0.0)
            continue
        try:
            target = float(gt)
        except (ValueError, TypeError):
            rewards.append(0.0)
            continue

        if target == 0:
            # Avoid division by zero — fall back to exact match
            rewards.append(1.0 if abs(predicted) < 1e-5 else -0.2)
            continue

        relative_error = abs(predicted - target) / abs(target)

        if abs(predicted - target) < 1e-5:
            rewards.append(1.0)
        elif relative_error <= 0.10:
            rewards.append(0.5)
        elif relative_error <= 0.20:
            rewards.append(0.2)
        else:
            rewards.append(-0.2)
    return rewards


# ── Dynamic Difficulty Scalarization (§3.3) ───────────────────

def scale_by_difficulty(rewards: list[float], prompts: list[str],
                        group_size: int, alpha: float = 0.5) -> list[float]:
    """
    Per-question λ_correct scaling based on reward-gap variance (§3.3, DW-GRPO).

    λ_correct(q) = 1 + α · Var({R(q, o_i)}_i=1^G)

    Where Var is the variance of rewards across the G completions for each
    prompt. High variance (mixed right/wrong) indicates a valuable difficulty
    gradient — scale reward UP. Low variance (all right or all wrong) provides
    no useful signal — leave unchanged.

    This replaces the static λ_correct with dynamic, per-question values.

    Args:
        rewards: flat list of B·G rewards, grouped by prompt
        prompts: list of B prompts (used for grouping)
        group_size: G — number of completions per prompt
        alpha: scaling coefficient (default 0.5 per §3.3)

    Returns:
        Scaled rewards, same length as input.
    """
    import numpy as np
    B = len(prompts)
    scaled = list(rewards)  # copy
    for i in range(B):
        start = i * group_size
        end = start + group_size
        group_rewards = rewards[start:end]
        var = float(np.var(group_rewards)) if len(group_rewards) > 1 else 0.0
        # Scale: λ_correct(q) = 1 + α · Var({R(q, o_i)})
        difficulty_weight = 1.0 + alpha * var
        for j in range(start, end):
            scaled[j] *= difficulty_weight
    return scaled


# ── Dynamic Reward Gating (Cohort D, DW-GRPO, §3.3) ───────────

class DynamicRewardGater:
    """
    Manages the time-varying format weight for Cohort D (§3.4):

      λ_format(t) = 1.0 * e^(-γ * t)

    where t = training step and γ = decay coefficient.
    Also tracks entropy for entropy-filtered gating (§3.3).
    """

    def __init__(self, gamma: float = 0.01, entropy_threshold: float = 0.693):
        self.gamma = gamma
        self.entropy_threshold = entropy_threshold  # ln(2)
        self.current_step = 0

    def get_format_weight(self) -> float:
        """Return the current λ_format based on exponential decay."""
        return 1.0 * math.exp(-self.gamma * self.current_step)

    def step(self) -> None:
        """Advance the training step counter."""
        self.current_step += 1

    def should_throttle(self, mean_entropy: float) -> bool:
        """Check if entropy exceeds threshold for advantage masking (§3.3)."""
        return mean_entropy > self.entropy_threshold


# ── Cohort Reward Factory (§3.4) ───────────────────────────────

def make_cohort_reward_fns(cohort_cfg: dict, gater: DynamicRewardGater | None = None):
    """
    Build the list of reward functions for a given cohort.

    Returns (reward_funcs, reward_weights) where:
      - reward_funcs is a list of callables for TRL's GRPOTrainer
      - reward_weights is passed via reward_processing_classes or applied externally

    Cohort configurations:
      baseline: {correctness: 1.0, format: 0.0, graduated: false, dynamic: false}
      A:        {correctness: 0.5, format: 1.5, graduated: false, dynamic: false}
      B:        {correctness: 1.5, format: 0.2, graduated: false, dynamic: false}
      C:        {correctness: 1.0, format: 0.2, graduated: true,  dynamic: false}
      D:        {correctness: 1.0, format: 1.0, graduated: false, dynamic: true}
    """
    l_correct = cohort_cfg.get("correctness", 1.0)
    l_format = cohort_cfg.get("format", 0.0)
    graduated = cohort_cfg.get("graduated", False)
    dynamic = cohort_cfg.get("dynamic", False)

    reward_funcs = []

    # Correctness reward
    if l_correct > 0.0:
        if graduated:
            reward_funcs.append(graduated_correctness_reward)
        else:
            reward_funcs.append(correctness_reward)

    # Format reward
    if dynamic and gater is not None:
        # For Cohort D, the format weight decays over time.
        # TRL expects static reward functions, so we handle the decay
        # by wrapping the format_reward to multiply by the dynamic weight.
        def dynamic_format_reward(prompts, completions, **kwargs):
            w = gater.get_format_weight()
            return [w * (1.0 if _has_tags(c) else 0.0) for c in completions]
        reward_funcs.append(dynamic_format_reward)
    elif l_format > 0.0:
        reward_funcs.append(format_reward)

    return reward_funcs


# ── TRL Integration ────────────────────────────────────────────

def make_combined_reward_fn(cohort_cfg: dict, gater: DynamicRewardGater | None = None,
                            difficulty_scale: bool = False, num_generations: int = 16,
                            difficulty_alpha: float = 0.5):
    """
    Build a single combined reward function for TRL's GRPOTrainer.

    Although TRL supports multiple reward_funcs, some configurations
    need weighted combination. This returns a single callable that
    computes the total reward per completion.

    R_total(q, o_i) = λ_correct * r_correct + λ_format * r_format

    When difficulty_scale=True, applies Dynamic Difficulty Scalarization (§3.3):
      λ_correct(q) = 1 + α · Var({R(q, o_i)}_i=1^G)
    """
    l_correct = cohort_cfg.get("correctness", 1.0)
    l_format = cohort_cfg.get("format", 0.0)
    graduated = cohort_cfg.get("graduated", False)
    dynamic = cohort_cfg.get("dynamic", False)

    def combined_reward(prompts: list[str], completions: list[str],
                        answer: list[str], **kwargs) -> list[float]:
        rewards = []

        # Correctness component
        for completion, gt in zip(completions, answer):
            total = 0.0

            if l_correct > 0.0:
                predicted = _extract_answer(completion)
                if predicted is not None:
                    try:
                        target = float(gt)
                    except (ValueError, TypeError):
                        target = None

                    if target is not None:
                        if graduated:
                            if target == 0:
                                r = 1.0 if abs(predicted) < 1e-5 else -0.2
                            else:
                                rel_err = abs(predicted - target) / abs(target)
                                if abs(predicted - target) < 1e-5:
                                    r = 1.0
                                elif rel_err <= 0.10:
                                    r = 0.5
                                elif rel_err <= 0.20:
                                    r = 0.2
                                else:
                                    r = -0.2
                        else:
                            r = 1.0 if abs(predicted - target) < 1e-5 else 0.0
                        total += l_correct * r

            # Format component
            if l_format > 0.0:
                if dynamic and gater is not None:
                    w = gater.get_format_weight()
                else:
                    w = l_format
                total += w * (1.0 if _has_tags(completion) else 0.0)

            rewards.append(total)

        # ── Dynamic Difficulty Scalarization (§3.3) ──────────
        if difficulty_scale:
            rewards = scale_by_difficulty(rewards, prompts, num_generations, difficulty_alpha)

        return rewards

    return combined_reward
