"""
filter_tune.py — Standalone filter probe tuner.

Tests the all-zero filter on a small sample set (50-100 problems) 
with different group sizes (G=1, G=2, G=4) and reports retention rates.
Saves per-problem results, questions, and answers to disk for manual verification.

Usage:
    python filter_tune.py                           # default 50 samples, G=[1,2,4]
    python filter_tune.py --samples 100             # 100 samples
    python filter_tune.py --group-sizes 2 4         # test only G=2 and G=4
    python filter_tune.py --max-completion 1024     # custom completion length
"""

import argparse
import yaml
import os
import csv
import torch
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "outputs/filter_tune"


def parse_args():
    parser = argparse.ArgumentParser(description="Filter Probe Tuner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--samples", type=int, default=50, help="Number of problems to probe")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 2, 4],
                        help="G values to test (e.g. --group-sizes 2 4)")
    parser.add_argument("--max-completion", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_answer(completion: str, ground_truth: str) -> bool:
    """Check if a completion's <answer> tag matches the ground truth."""
    import re
    match = re.search(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>", completion)
    if not match:
        return False
    try:
        predicted = float(match.group(1))
        target = float(ground_truth)
        return abs(predicted - target) < 1e-5
    except (ValueError, TypeError):
        return False


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    grpo_cfg = cfg["training"]["grpo"]

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("All-Zero Filter Tuner")
    print("=" * 60)
    print(f"Samples: {args.samples}")
    print(f"G values: {args.group_sizes}")
    print(f"Max completion: {args.max_completion}")
    print()

    # ── Load tokenizer ──────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Load SFT model ──────────────────────────────────────────
    sft_path = "outputs/sft_checkpoint/merged"
    if not os.path.exists(sft_path):
        sft_path = "outputs/sft_checkpoint"
    print(f"Loading SFT model from: {sft_path}")

    from transformers import AutoModelForCausalLM
    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    sft_model.eval()
    device = next(sft_model.parameters()).device
    print(f"Model loaded on: {device}")

    # ── Load dataset samples ────────────────────────────────────
    print(f"Loading {args.samples} samples from {grpo_cfg['dataset']}")
    ds = load_dataset(grpo_cfg["dataset"], split="train", streaming=True, token=True)
    samples = []
    for i, row in enumerate(ds):
        if i >= args.samples:
            break
        samples.append({"problem": row["problem"], "answer": str(row["answer"])})
    print(f"Loaded {len(samples)} samples")

    # ── Save questions and answers ──────────────────────────────
    q_path = os.path.join(args.output, "questions.json")
    a_path = os.path.join(args.output, "answers.json")
    import json
    with open(q_path, "w") as f:
        json.dump({i: s["problem"] for i, s in enumerate(samples)}, f, indent=2)
    with open(a_path, "w") as f:
        json.dump({i: s["answer"] for i, s in enumerate(samples)}, f, indent=2)
    print(f"Questions saved to {q_path}")
    print(f"Answers saved to {a_path}")

    if args.dry_run:
        print("[DRY-RUN] Would test filter with above config")
        return

    # ── Chat template ───────────────────────────────────────────
    def format_prompt(problem: str) -> str:
        from data import format_grpo_prompt
        return format_grpo_prompt(problem)

    # ── Run filter for each G value ─────────────────────────────
    summary_lines = [f"Filter Tune Results — {len(samples)} samples, max_completion={args.max_completion}\n"]
    summary_lines.append("=" * 60 + "\n")

    for g in args.group_sizes:
        print(f"\n{'─' * 40}")
        print(f"Testing G={g}...")
        results = []
        correct_count = 0

        for batch_start in tqdm(range(0, len(samples), args.batch_size),
                                desc=f"G={g}"):
            batch = samples[batch_start:batch_start + args.batch_size]
            prompts = [format_prompt(s["problem"]) for s in batch]
            answers = [s["answer"] for s in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=True, max_length=256).to(device)

            with torch.no_grad():
                outputs = sft_model.generate(
                    **inputs,
                    max_new_tokens=args.max_completion,
                    do_sample=(g > 1),
                    temperature=0.7 if g > 1 else 1.0,
                    top_p=0.9,
                    num_return_sequences=g,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            completions = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            for j in range(len(batch)):
                batch_completions = completions[j * g:(j + 1) * g]
                any_correct = any(
                    check_answer(comp, answers[j]) for comp in batch_completions
                )
                idx = batch_start + j
                results.append({
                    "idx": idx,
                    "any_correct": any_correct,
                    "num_correct": sum(1 for c in batch_completions if check_answer(c, answers[j])),
                    "total_attempts": g,
                })
                if any_correct:
                    correct_count += 1

        # ── Save per-problem results ─────────────────────────────
        csv_path = os.path.join(args.output, f"g{g}_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "any_correct", "num_correct", "total_attempts"])
            writer.writeheader()
            writer.writerows(results)

        retention = correct_count / len(samples) * 100
        line = f"G={g}: {correct_count}/{len(samples)} retained ({retention:.1f}%)"
        print(f"  {line}")
        print(f"  Results: {csv_path}")
        summary_lines.append(line + "\n")

    # ── Save summary ────────────────────────────────────────────
    summary_path = os.path.join(args.output, "summary.txt")
    with open(summary_path, "w") as f:
        f.writelines(summary_lines)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
