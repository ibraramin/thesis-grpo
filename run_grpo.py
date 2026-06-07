"""
run_grpo.py — Single-cohort, single-seed GRPO training run.

Loads the SFT checkpoint, applies LoRA, and runs GRPO with
cohort-specific reward functions and stabilization mechanisms.

Auto-resumes from the latest checkpoint if training was interrupted.
Writes a .resume_marker file during training; deletes it on clean completion.

Usage:
    python run_grpo.py --cohort baseline --seed 42
    python run_grpo.py --cohort D --seed 1 --dry-run
    python run_grpo.py --cohort baseline --seed 0 --test-run
"""

import argparse
import yaml
import os
import sys
import glob
import random
import numpy as np
import torch
from datetime import datetime

from data import load_grpo_dataset, get_tokenizer, tokenize_grpo, filter_non_all_zero
from rewards import make_combined_reward_fn, DynamicRewardGater
from grpo_trainer import StabilizedGRPOTrainer

OUTPUT_DIR = "outputs"
RESUME_MARKER = ".resume_marker"
FILTERED_DATASET_DIR = "outputs/filtered_grpo"


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint subdirectory under output_dir.
    
    HuggingFace Trainer names checkpoints as 'checkpoint-N' where N is the
    global training step. Returns the path with the largest N, or None.
    Also checks if the output_dir itself contains adapter files (post-training save).
    """
    adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        return output_dir

    checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    if not checkpoints:
        return None
    return checkpoints[-1]


def is_training_incomplete(output_dir: str) -> bool:
    """Check if a previous training run was interrupted."""
    return os.path.exists(os.path.join(output_dir, RESUME_MARKER))


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training Run")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cohort", required=True,
                        help="Cohort name: baseline, A, B, C, D")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-checkpoint", default="outputs/sft_checkpoint/merged",
                        help="Path to merged SFT checkpoint")
    parser.add_argument("--output", default=None,
                        help="Output dir (default: outputs/{cohort}/{seed}/)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-run", action="store_true",
                        help="Small-scale run for format validation")
    parser.add_argument("--filter-dataset", action="store_true",
                        help="Run all-zero filter on dataset before training (expensive, once)")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _log_reward_smoke_test(reward_fn, cohort_cfg: dict, report_path: str | None, cohort_name: str):
    """Run reward function on sample completions and log to report."""
    if not report_path:
        return
    sample_prompts = ["<|im_start|>user\nWhat is 2+2?<|im_end|>"]
    sample_completions = [
        "<think>2 plus 2 equals 4</think>\n<answer>4</answer>",
        "just the number 4",
        "<think>let me think...</think>\n<answer>5</answer>",
    ]
    sample_answers = ["4"]
    try:
        # reward_fn signature: (prompts, completions, answer=..., **kwargs)
        r = reward_fn(sample_prompts, sample_completions, answer=sample_answers)
        with open(report_path, "a") as rf:
            rf.write(f"\n─── STAGE 2: GRPO ({cohort_name}) ───\n\n")
            rf.write(f"  λ_correct={cohort_cfg['correctness']}, "
                     f"λ_format={cohort_cfg['format']}, "
                     f"graduated={cohort_cfg['graduated']}, "
                     f"dynamic={cohort_cfg.get('dynamic', False)}\n\n")
            rf.write("  REWARD SMOKE TEST:\n")
            for i, (c, s) in enumerate(zip(sample_completions, r)):
                has_tags = ("<think>" in c and "<answer>" in c)
                rf.write(f"  [{i}] reward={s:.2f}  format_tags={'✓' if has_tags else '✗'}\n")
                rf.write(f"      completion: {c[:120]}...\n")
            rf.write(f"  ✓ Reward function smoke test PASSED\n\n")
        print(f"[GRPO] Reward smoke test logged to {report_path}")
    except Exception as e:
        with open(report_path, "a") as rf:
            rf.write(f"  ✗ Reward smoke test FAILED: {e}\n\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    grpo_cfg = cfg["training"]["grpo"]
    stab_cfg = cfg["training"]["stabilization"]
    model_cfg = cfg["model"]
    cohorts_cfg = cfg["cohorts"]

    if args.cohort not in cohorts_cfg:
        print(f"ERROR: Unknown cohort '{args.cohort}'. Choices: {list(cohorts_cfg.keys())}")
        sys.exit(1)

    cohort = cohorts_cfg[args.cohort]
    set_seed(args.seed)

    output_dir = args.output or os.path.join(OUTPUT_DIR, args.cohort, f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    # ── Test-run overrides ──────────────────────────────────────
    test_run = args.test_run
    report_path = None
    if test_run:
        tr = cfg.get("test_run", {})
        grpo_cfg = dict(grpo_cfg)  # shallow copy
        grpo_cfg["max_samples"] = tr.get("grpo_max_samples", 10)
        grpo_cfg["epochs"] = tr.get("epochs", 1)
        grpo_cfg["num_generations"] = tr.get("num_generations", 4)
        grpo_cfg["max_completion_length"] = tr.get("max_completion_length", 128)
        grpo_cfg["max_prompt_length"] = tr.get("max_prompt_length", grpo_cfg["max_prompt_length"])
        grpo_cfg["batch_size"] = tr.get("batch_size", 1)
        grpo_cfg["gradient_accumulation_steps"] = tr.get("gradient_accumulation_steps", 1)
        output_dir = tr.get("output_dir", "outputs/test_run") + f"/{args.cohort}/seed_{args.seed}"
        os.makedirs(output_dir, exist_ok=True)
        report_path = tr.get("report_file", "outputs/test_run/report.txt")

    print(f"[GRPO] Cohort: {args.cohort} | Seed: {args.seed}")
    print(f"[GRPO] Config: λ_correct={cohort['correctness']}, λ_format={cohort['format']}, "
          f"graduated={cohort['graduated']}, dynamic={cohort['dynamic']}")

    # ── Dynamic reward gater (Cohort D) ────────────────────────
    gater = None
    if cohort["dynamic"]:
        gamma = cohort.get("gamma", 0.01)
        gater = DynamicRewardGater(gamma=gamma, entropy_threshold=stab_cfg["entropy_threshold"])
        print(f"[GRPO] Dynamic gater: γ={gamma}")

    # ── Reward function ────────────────────────────────────────
    reward_fn = make_combined_reward_fn(
        cohort,
        gater,
        difficulty_scale=grpo_cfg.get("difficulty_scale", False),
        num_generations=grpo_cfg["num_generations"],
        difficulty_alpha=grpo_cfg.get("difficulty_alpha", 0.5),
    )
    print("[GRPO] Reward function built")

    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer = get_tokenizer(model_cfg["name"])
    print(f"[GRPO] Tokenizer loaded: {model_cfg['name']}")

    if args.dry_run:
        print(f"[GRPO] Dry-run: would train with G={grpo_cfg['num_generations']}, "
              f"epochs={grpo_cfg['epochs']}, lr={grpo_cfg['learning_rate']}")
        print(f"[GRPO] Output: {output_dir}")
        _log_reward_smoke_test(reward_fn, cohort, report_path, args.cohort)
        return

    # ── Dataset ────────────────────────────────────────────────
    print(f"[GRPO] Loading dataset: {grpo_cfg['dataset']} (max {grpo_cfg['max_samples']})")
    dataset = load_grpo_dataset(cfg, max_samples=grpo_cfg["max_samples"])
    print(f"[GRPO] Dataset size: {len(dataset)}")

    # ── All-Zero Filter (§3.1) ──────────────────────────────────
    # Runs once: excludes samples the SFT model cannot solve at all.
    # Must run BEFORE tokenization (filter expects prompt/answer columns).
    # Cached to outputs/filtered_grpo/ for subsequent runs.
    if args.filter_dataset:
        filtered_path = os.path.join(FILTERED_DATASET_DIR, "filtered_dataset")
        if os.path.exists(filtered_path):
            print(f"[GRPO] Filtered dataset already exists at {filtered_path}")
            from datasets import load_from_disk
            dataset = load_from_disk(filtered_path)
            print(f"[GRPO] Filtered dataset size: {len(dataset)}")
        else:
            print("[GRPO] Running all-zero filter (this is expensive, ~10-30 min)...")
            sft_path = args.sft_checkpoint
            print(f"[GRPO] Loading SFT base model for filtering: {sft_path}")
            from transformers import AutoModelForCausalLM
            filter_model = AutoModelForCausalLM.from_pretrained(
                sft_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            dataset = filter_non_all_zero(
                dataset, filter_model, tokenizer,
                group_size=grpo_cfg["num_generations"],
                max_completion=grpo_cfg["max_completion_length"],
            )
            print(f"[GRPO] Filtered dataset size: {len(dataset)} (removed all-zero trajectories)")
            os.makedirs(FILTERED_DATASET_DIR, exist_ok=True)
            dataset.save_to_disk(filtered_path)
            print(f"[GRPO] Saved filtered dataset to {filtered_path}")
            del filter_model
            torch.cuda.empty_cache()
    elif os.path.exists(os.path.join(FILTERED_DATASET_DIR, "filtered_dataset")):
        # Auto-use cached filtered dataset if available
        from datasets import load_from_disk
        print(f"[GRPO] Using cached filtered dataset from {FILTERED_DATASET_DIR}")
        dataset = load_from_disk(os.path.join(FILTERED_DATASET_DIR, "filtered_dataset"))
        print(f"[GRPO] Filtered dataset size: {len(dataset)}")

    dataset = tokenize_grpo(dataset, tokenizer, max_prompt_length=grpo_cfg["max_prompt_length"])
    print("[GRPO] Dataset tokenized")

    # ── Test-run: log sample prompts & reward smoke test ────────
    if test_run:
        _log_reward_smoke_test(reward_fn, cohort, report_path, args.cohort)
        with open(report_path, "a") as rf:
            rf.write(f"  GRPO Dataset: {grpo_cfg['dataset']}\n")
            rf.write(f"  Samples loaded: {len(dataset)}\n")
            rf.write(f"  G (num_generations): {grpo_cfg['num_generations']}\n")
            rf.write(f"  Max completion: {grpo_cfg['max_completion_length']}\n\n")
            rf.write("  SAMPLE PROMPTS (first 3):\n")
            for i in range(min(3, len(dataset))):
                prompt = dataset[i].get("prompt", dataset[i].get("text", ""))
                answer = dataset[i].get("answer", "N/A")
                truncated = prompt[:400] + ("..." if len(str(prompt)) > 400 else "")
                rf.write(f"  [{i}] answer={answer}\n{truncated}\n\n")

    # ── Model ──────────────────────────────────────────────────
    # Load the merged SFT checkpoint. Prefer Unsloth if available,
    # fall back to standard transformers + bitsandbytes.
    sft_path = args.sft_checkpoint
    if test_run and not os.path.exists(sft_path):
        print(f"[GRPO] Test-run: SFT checkpoint not found at {sft_path}")
        print("[GRPO] Running reward smoke test only (skipping model load).")
        with open(report_path, "a") as rf:
            rf.write("  (Model load skipped: SFT checkpoint not available)\n")
            rf.write(f"  ✓ GRPO pipeline validated (config + data + rewards)\n\n")
        return
    if not os.path.exists(sft_path) and not test_run:
        print(f"[GRPO] ERROR: SFT checkpoint not found at {sft_path}")
        print("[GRPO] Run 'python run_sft.py' first.")
        sys.exit(1)

    print(f"[GRPO] Loading SFT checkpoint: {sft_path}")

    USE_UNSLOTH = False
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.from_pretrained  # trigger lazy import
        USE_UNSLOTH = True
    except Exception:
        print("[GRPO] Unsloth unavailable/incompatible — using standard transformers + bitsandbytes")

    if USE_UNSLOTH:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=sft_path,
            max_seq_length=grpo_cfg["max_prompt_length"] + grpo_cfg["max_completion_length"],
            load_in_4bit=False,
            fast_inference=True,
        )
        if grpo_cfg.get("apply_lora", True):
            model = FastLanguageModel.get_peft_model(
                model,
                r=model_cfg["lora_r"],
                target_modules=model_cfg["lora_targets"],
                lora_alpha=model_cfg["lora_alpha"],
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
    else:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_cfg["load_in_4bit"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            sft_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        peft_config = LoraConfig(
            r=model_cfg["lora_r"],
            lora_alpha=model_cfg["lora_alpha"],
            target_modules=model_cfg["lora_targets"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # ── GRPO Configuration (§3.2, §7.3) ────────────────────────
    from trl import GRPOConfig

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=grpo_cfg["epochs"],
        per_device_train_batch_size=1,  # GRPO generates G completions per prompt
        gradient_accumulation_steps=1,
        learning_rate=grpo_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=grpo_cfg["warmup_ratio"],
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        max_prompt_length=grpo_cfg["max_prompt_length"],
        max_completion_length=grpo_cfg["max_completion_length"],
        num_generations=grpo_cfg["num_generations"],
        beta=grpo_cfg["beta"],
        loss_type="grpo",
        use_vllm=not test_run,          # Disable vLLM for test-run (too heavy)
        vllm_gpu_memory_utilization=0.3,
        report_to="none",
        seed=args.seed,
    )

    # ── Trainer ────────────────────────────────────────────────
    print(f"[GRPO] Initializing StabilizedGRPOTrainer")
    print(f"[GRPO]   PSPO: {stab_cfg['pspo_enabled']} (δ={stab_cfg['pspo_delta']})")
    print(f"[GRPO]   Entropy filter: {stab_cfg['entropy_filter_enabled']} "
          f"(threshold={stab_cfg['entropy_threshold']})")

    trainer = StabilizedGRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        pspo_enabled=stab_cfg["pspo_enabled"],
        pspo_delta=stab_cfg["pspo_delta"],
        entropy_filter_enabled=stab_cfg["entropy_filter_enabled"],
        entropy_threshold=stab_cfg["entropy_threshold"],
        dynamic_gater=gater,
    )

    # ── Train with auto-resume ─────────────────────────────────
    resume_ckpt = find_latest_checkpoint(output_dir)
    if resume_ckpt and resume_ckpt != output_dir:
        print(f"[GRPO] Resuming from checkpoint: {resume_ckpt}")
    elif resume_ckpt == output_dir:
        print(f"[GRPO] Adapter already saved at {output_dir} — skipping training")
        if test_run:
            with open(report_path, "a") as rf:
                rf.write("  (Already complete — skipping)\n")
                rf.write(f"  ✓ GRPO PASSED (cached)\n\n")
        return

    # Write resume marker
    marker_path = os.path.join(output_dir, RESUME_MARKER)
    with open(marker_path, "w") as f:
        f.write(datetime.now().isoformat())
        f.write(f"\ncohort={args.cohort}\nseed={args.seed}\n")

    print("[GRPO] Starting training...")
    trainer.train(resume_from_checkpoint=resume_ckpt or False)

    # Delete resume marker on clean completion
    if os.path.exists(marker_path):
        os.remove(marker_path)

    # ── Save ───────────────────────────────────────────────────
    print(f"[GRPO] Saving checkpoint to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if test_run:
        with open(report_path, "a") as rf:
            rf.write(f"  Training completed: {grpo_cfg['epochs']} epoch(s)\n")
            rf.write(f"  Checkpoint saved to: {output_dir}\n")
            rf.write("  ✓ GRPO PASSED\n\n")

    print(f"[GRPO] Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
