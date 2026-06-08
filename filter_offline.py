"""
filter_offline.py — Offline distillation filter for GRPO dataset.

Runs the SFT model against 30K Big-Math-RL-Verified prompts via vLLM,
scores each completion with _check_answer(), and saves prompts where
at least 1 of G completions is correct.

This guarantees the GRPO dataset contains only "solvable" prompts,
preventing format-reward hacking when correctness signal is absent.

Usage:
    python filter_offline.py                    # uses config.yaml defaults
    python filter_offline.py --prompts 20000    # custom prompt count
    python filter_offline.py --g 8              # custom generations/prompt

Output: outputs/filtered_grpo/filtered_dataset.jsonl
"""

import argparse
import json
import os
import sys

import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

from data import format_grpo_prompt, _check_answer, DATA_DIR

GRPO_LOCAL = os.path.join(DATA_DIR, "openmath_instruct_2_grpo.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Dataset Filter")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sft-checkpoint", default="outputs/sft_checkpoint/merged")
    parser.add_argument("--prompts", type=int, default=30000)
    parser.add_argument("--g", type=int, default=4, help="Generations per prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", default="outputs/filtered_grpo/filtered_dataset.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dataset_name = cfg["training"]["grpo"]["dataset"]
    sft_path = args.sft_checkpoint

    if not os.path.exists(sft_path):
        print(f"ERROR: SFT checkpoint not found at {sft_path}")
        print("Run SFT first: python run_sft.py")
        sys.exit(1)

    # ── Load model via vLLM ─────────────────────────────────────
    print(f"Loading model from: {sft_path}")
    llm = LLM(
        model=sft_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.7,        # sampling for diversity across G completions
        max_tokens=args.max_tokens,
        n=args.g,
    )

    # ── Load dataset ────────────────────────────────────────────
    print(f"Loading dataset: {dataset_name} (max {args.prompts} prompts)")
    # Prefer local JSONL (offline), fall back to HF streaming
    if os.path.exists(GRPO_LOCAL):
        print(f"  Using local file: {GRPO_LOCAL}")
        rows = []
        with open(GRPO_LOCAL) as f:
            for i, line in enumerate(f):
                if i >= args.prompts:
                    break
                row = json.loads(line)
                problem = row.get("problem", "")
                answer = row.get("answer", "")
                if problem and answer:
                    rows.append({"problem": problem, "answer": answer})
    else:
        from datasets import load_dataset
        print(f"  Streaming from HF: {dataset_name}")
        ds = load_dataset(dataset_name, split="train", streaming=True, token=True)
        rows = []
        for i, row in enumerate(ds):
            if i >= args.prompts:
                break
            problem = row.get("problem", "")
            answer = row.get("answer") or row.get("expected_answer", "")
            if problem and answer:
                rows.append({"problem": problem, "answer": str(answer)})
    print(f"  Loaded {len(rows)} prompts")

    kept = []
    total = 0
    correct_prompts = 0

    for entry in tqdm(rows, desc="Filtering", unit="prompts"):
        total += 1
        prompt_str = format_grpo_prompt(entry["problem"])

        # Generate G completions
        outputs = llm.generate([prompt_str], sampling_params)
        completions = [o.text for o in outputs[0].outputs]

        # Check if any completion matches the ground truth
        any_correct = any(_check_answer(comp, entry["answer"]) for comp in completions)

        if any_correct:
            kept.append({"prompt": prompt_str, "answer": entry["answer"]})
            correct_prompts += 1

    # ── Save ────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        for entry in kept:
            f.write(json.dumps(entry) + "\n")

    retention = (len(kept) / total * 100) if total > 0 else 0
    print(f"\nFiltered: {len(kept)} / {total} prompts kept ({retention:.1f}%)")
    print(f"Saved to: {args.output}")

    if retention < 1:
        print("WARNING: Retention < 1% — SFT model may be too weak for this dataset.")
        print("Consider: more SFT epochs, larger SFT dataset, or a simpler math dataset.")


if __name__ == "__main__":
    main()
