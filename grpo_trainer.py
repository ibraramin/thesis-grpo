"""
grpo_trainer.py — Custom GRPOTrainer with stabilization mechanisms (§3.3, §5).

Subclasses TRL's GRPOTrainer to inject:
  - PSPO probability smoothing (§3.3)
  - Entropy-filtered advantage masking with hard ln(2) threshold (§3.3)
  - K3 Schulman KL estimator (already default in TRL 1.0.0, §4)

The _compute_loss override mirrors TRL 1.0.0's default GRPO loss with
three modifications clearly annotated [PSPO], [ENTROPY], [K3].
"""

import torch
from trl import GRPOTrainer, GRPOConfig
from stabilization import pspo_smooth_logprobs, entropy_filter_mask


class StabilizedGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer subclass that implements the full methodology from §3.3-§4:

      - PSPO: r_i^PSPO = ((1-δ)·π_θ + δ·π_ref) / π_old
      - Entropy filter: zero advantage when mean completion H > ln(2)
      - K3 KL: r - log(r) - 1 (native in TRL 1.0.0)
    """

    def __init__(
        self,
        pspo_delta: float = 0.1,
        pspo_enabled: bool = True,
        entropy_threshold: float = 0.693,
        entropy_filter_enabled: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._pspo_delta = pspo_delta
        self._pspo_enabled = pspo_enabled
        self._entropy_threshold = entropy_threshold
        self._entropy_filter_enabled = entropy_filter_enabled

    # ── Override: _compute_loss ─────────────────────────────────
    # Same logic as TRL 1.0.0 GRPOTrainer._compute_loss with
    # [PSPO] and [ENTROPY] modifications injected.

    def _compute_loss(self, model, inputs):
        # ── Standard: get per-token logprobs and entropies ─────
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            pixel_position_ids=inputs.get("pixel_position_ids"),
        )

        # ── [PSPO] Probability Smoothing (§3.3, §4) ───────────
        if self._pspo_enabled:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_logps = pspo_smooth_logprobs(
                per_token_logps, ref_per_token_logps, delta=self._pspo_delta
            )

        # ── [ENTROPY] Hard-threshold advantage filter (§3.3) ──
        if self._entropy_filter_enabled:
            entropy_keep = entropy_filter_mask(
                entropies, mask, threshold=self._entropy_threshold
            )
        else:
            entropy_keep = None

        # ── Standard: compute advantages ───────────────────────
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        # ── Standard: log-ratio ────────────────────────────────
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        )
        log_ratio = per_token_logps - old_per_token_logps

        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)

        # ── [K3] KL Divergence (native in TRL 1.0.0, §4) ──────
        per_token_kl = None
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # ── Standard: PPO-style clipped loss (GRPO loss_type) ──
        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        # ── [ENTROPY] Apply entropy mask ───────────────────────
        if entropy_keep is not None:
            # entropy_keep is (B,); broadcast to (B, T) or (B, 1)
            if per_token_loss.dim() > 1 and per_token_loss.size(1) > 1:
                entropy_keep = entropy_keep.unsqueeze(1)
            per_token_loss = per_token_loss * entropy_keep

        # ── Standard: KL penalty ───────────────────────────────
        if self.beta != 0.0 and per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # ── Standard: loss aggregation ─────────────────────────
        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        else:
            loss = per_token_loss.mean()

        # ── Standard: metrics ──────────────────────────────────
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * mask).sum() / completion_token_count

        if self.beta != 0.0 and per_token_kl is not None:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        return loss
