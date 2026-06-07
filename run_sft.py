"""
run_sft.py — Stage 1 Supervised Fine-Tuning.

Trains the Qwen2.5-1.5B base model on OpenR1-Math-220k to bootstrap
reasoning structure and teach <think>/<answer> formatting before GRPO.

Auto-resumes from the latest checkpoint if training was interrupted.
Writes a .resume_marker file during training; deletes it on clean completion.

Usage:
    python run_sft.py                         # uses config.yaml defaults
    python run_sft.py --config config.yaml    # explicit config path
    python run_sft.py --test-run              # small-scale format validation
"""

import argparse
import yaml
import os
import sys
import glob
import torch
from datetime import datetime

from data import load_sft_dataset, get_tokenizer, tokenize_sft

OUTPUT_DIR = "outputs/sft_checkpoint"
RESUME_MARKER = ".resume_marker"


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint subdirectory under output_dir.
    
    HuggingFace Trainer names checkpoints as 'checkpoint-N' where N is the
    global training step. Returns the path with the largest N, or None.
    Also checks if the output_dir itself contains adapter files (post-training save).
    """
    # Check for post-training save (adapter directly in output_dir)
    adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        return output_dir
    
    checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    if not checkpoints:
        return None
    return checkpoints[-1]  # Highest step number


def is_training_incomplete(output_dir: str) -> bool:
    """Check if a previous training run was interrupted."""
    return os.path.exists(os.path.join(output_dir, RESUME_MARKER))


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training Stage")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Checkpoint output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training")
    parser.add_argument("--test-run", action="store_true",
                        help="Small-scale run: 10 samples, 1 step, for format validation")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    sft_cfg = cfg["training"]["sft"]
    model_cfg = cfg["model"]

    # ── Test-run overrides ──────────────────────────────────────
    test_run = args.test_run
    if test_run:
        tr = cfg.get("test_run", {})
        sft_cfg = dict(sft_cfg)  # shallow copy for overrides
        sft_cfg["max_samples"] = tr.get("sft_max_samples", 10)
        sft_cfg["epochs"] = tr.get("epochs", 1)
        sft_cfg["batch_size"] = tr.get("batch_size", 1)
        sft_cfg["gradient_accumulation_steps"] = tr.get("gradient_accumulation_steps", 1)
        sft_cfg["max_seq_length"] = tr.get("max_seq_length", sft_cfg["max_seq_length"])
        report_path = tr.get("report_file", "outputs/test_run/report.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as rf:
            rf.write("TEST RUN REPORT\n")
            rf.write("=" * 60 + "\n\n")
            rf.write("─── STAGE 1: SFT ───\n\n")

    print(f"[SFT] Loading tokenizer: {model_cfg['name']}")
    tokenizer = get_tokenizer(model_cfg["name"])

    if args.dry_run:
        print("[SFT] Dry-run: skipping dataset download (slow on this connection).")
        print(f"[SFT] Config ok, tokenizer loaded, all imports valid.")
        print(f"[SFT] Would train on: {sft_cfg['dataset']} ({sft_cfg['max_samples']} samples)")
        if test_run:
            with open(report_path, "a") as rf:
                rf.write("[DRY-RUN] Would train with above config\n")
                rf.write("✓ SFT dry-run PASSED\n\n")
        return

    print(f"[SFT] Loading dataset: {sft_cfg['dataset']} (max {sft_cfg['max_samples']} samples)")
    dataset = load_sft_dataset(cfg, max_samples=sft_cfg["max_samples"])
    print(f"[SFT] Dataset size: {len(dataset)}")

    # ── Test-run: log sample inputs ─────────────────────────────
    if test_run:
        with open(report_path, "a") as rf:
            rf.write(f"  Dataset: {sft_cfg['dataset']}\n")
            rf.write(f"  Samples loaded: {len(dataset)}\n")
            rf.write(f"  Max sequence length: {sft_cfg['max_seq_length']}\n\n")
            rf.write("  SAMPLE TRAINING TEXTS (first 3):\n")
            for i in range(min(3, len(dataset))):
                text = dataset[i]["text"]
                truncated = text[:500] + ("..." if len(text) > 500 else "")
                rf.write(f"  [{i}]\n{truncated}\n\n")
            rf.write(f"  ✓ SFT dataset loaded ({len(dataset)} samples)\n\n")

    print("[SFT] Tokenizing...")
    dataset = tokenize_sft(dataset, tokenizer, max_seq_length=sft_cfg["max_seq_length"])

    # ── Model Loading (requires Unsloth) ────────────────────────
    print(f"[SFT] Loading model: {model_cfg['name']} (4-bit NF4)")

    if torch.cuda.is_available():
        print(f"[SFT] GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB)")
    else:
        print("[SFT] WARNING: No GPU detected. Training will be extremely slow.")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("[SFT] ERROR: unsloth not installed. Install with: pip install unsloth")
        print("[SFT] On the Docker image (Dockerfile) this is pre-installed.")
        if test_run:
            with open(report_path, "a") as rf:
                rf.write("✗ SFT FAILED: unsloth not installed\n")
        sys.exit(1)

    # In test-run mode, skip actual model loading if too slow:
    # just validate the pipeline with dry-run
    if test_run and not torch.cuda.is_available():
        print("[SFT] Test-run on CPU: skipping model load (format validation only)")
        with open(report_path, "a") as rf:
            rf.write("  (Model load skipped — no GPU available in test-run)\n")
            rf.write("  ✓ SFT pipeline validated (config + data + tokenization)\n\n")
        return

    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=sft_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        fast_inference=False,  # Training mode
    )

    # Apply LoRA adapters (§3.2)
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_cfg["lora_r"],
        target_modules=model_cfg["lora_targets"],
        lora_alpha=model_cfg["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized GCO
        random_state=42,
    )
    model.print_trainable_parameters()

    # ── SFT Trainer (§7.2) ─────────────────────────────────────
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments

    if sft_cfg.get("training_args_override"):
        training_args = TrainingArguments(**sft_cfg["training_args_override"])
    else:
        # Compute save_steps: save ~10 times total across all steps
        batch_size = sft_cfg["batch_size"]
        grad_accum = sft_cfg["gradient_accumulation_steps"]
        total_steps_per_epoch = max(1, len(dataset) // (batch_size * grad_accum))
        save_steps = max(10, total_steps_per_epoch * sft_cfg["epochs"] // 10)
        training_args = SFTConfig(
            output_dir=args.output,
            num_train_epochs=sft_cfg["epochs"],
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=sft_cfg["learning_rate"],
            lr_scheduler_type="cosine",
            warmup_ratio=sft_cfg["warmup_ratio"],
            logging_steps=10,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=1,            # Keep only last checkpoint (32GB storage)
            fp16=not model_cfg["load_in_4bit"],
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            max_seq_length=sft_cfg["max_seq_length"],
            packing=False,
            report_to="none",
            load_best_model_at_end=False,
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ── Auto-resume from latest checkpoint ─────────────────────
    resume_ckpt = find_latest_checkpoint(args.output)
    if resume_ckpt and resume_ckpt != args.output:
        # Found intermediate checkpoint — resume training
        print(f"[SFT] Resuming from checkpoint: {resume_ckpt}")
    elif resume_ckpt == args.output:
        # Final adapter already exists — training is complete
        print(f"[SFT] Adapter already saved at {args.output} — skipping training")
        return

    # Write resume marker
    marker_path = os.path.join(args.output, RESUME_MARKER)
    with open(marker_path, "w") as f:
        f.write(datetime.now().isoformat())
        f.write(f"\nconfig={args.config}\n")

    print("[SFT] Starting training...")
    trainer.train(resume_from_checkpoint=resume_ckpt or False)
    
    # Delete resume marker on clean completion
    if os.path.exists(marker_path):
        os.remove(marker_path)

    # ── Save checkpoint (§7.2) ───────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    print(f"[SFT] Saving LoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Save merged model for vLLM fast inference during GRPO
    # Test-run: skip merging (slow, not needed for format validation)
    if test_run:
        print("[SFT] Test-run: skipping model merge")
        with open(report_path, "a") as rf:
            rf.write(f"  Training completed: {sft_cfg['epochs']} epoch(s)\n")
            rf.write(f"  LoRA saved to: {args.output}\n")
            rf.write("  ✓ SFT PASSED\n\n")
    else:
        merged_path = os.path.join(args.output, "merged")
        print(f"[SFT] Merging and saving full model to {merged_path}")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

    print("[SFT] Done.")


if __name__ == "__main__":
    main()
