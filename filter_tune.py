"""
filter_tune.py — vLLM-powered filter probe tuner.

Tests the all-zero filter on a small sample set (50-100 problems)
with different group sizes (G=1, G=2, G=4) using vLLM's PagedAttention
for fast batch generation. Reports retention rates.

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
import json
import re
from datasets import load_dataset

OUTPUT_DIR = "outputs/filter_tune"


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Filter Probe Tuner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--samples", type=int, default=50, help="Number of problems to probe")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 2, 4],
                        help="G values to test (e.g. --group-sizes 2 4)")
    parser.add_argument("--max-completion", type=int, default=1024)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Save raw completions for first N samples to debug_completions.json")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _normalize_str(s: str) -> str:
    """Collapse whitespace and strip for string comparison."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def check_answer(completion: str, ground_truth: str) -> bool:
    """Check if a completion's <answer> tag matches the ground truth.

    Handles: pure numeric, LaTeX (via sympy), equations, and multi-value
    answers joined by \"or\".  Sympy is optional; falls back to string
    comparison when not available.
    """
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if not match:
        return False

    predicted = match.group(1).strip()
    target = ground_truth.strip()

    # 1. Pure numeric comparison
    try:
        pred_num = float(predicted.replace(",", ""))
        tgt_num = float(target.replace(",", ""))
        if tgt_num == 0:
            return abs(pred_num) < 1e-5
        return abs(pred_num - tgt_num) / max(abs(tgt_num), 1e-6) < 1e-4
    except (ValueError, TypeError):
        pass

    # 2. Sympy symbolic comparison (LaTeX / equations)
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, N
        a_expr = parse_latex(predicted.replace(r"\text{ or }", ""))
        b_expr = parse_latex(target.replace(r"\text{ or }", ""))
        diff = simplify(a_expr - b_expr)
        if diff == 0:
            return True
        if diff.is_number:
            return abs(float(N(diff))) < 1e-4
    except Exception:
        pass

    # 3. Normalized string comparison
    if _normalize_str(predicted) == _normalize_str(target):
        return True

    # 4. \"or\"-separated answers — match any alternative
    alt_sep = r"(?:\s+or\s+|\\text\{\s*or\s*\})"
    if re.search(alt_sep, target):
        alternatives = re.split(alt_sep, target)
        for alt in alternatives:
            alt = alt.strip()
            if check_answer(f"<answer>{alt}</answer>", alt):
                if check_answer(completion, alt):
                    return True
            elif _normalize_str(predicted) == _normalize_str(alt):
                return True

    return False


def main():
    args = parse_args()
    cfg = load_config(args.config)
    grpo_cfg = cfg["training"]["grpo"]

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("vLLM All-Zero Filter Tuner")
    print("=" * 60)
    print(f"Samples: {args.samples}")
    print(f"G values: {args.group_sizes}")
    print(f"Max completion: {args.max_completion}")
    print()

    # ── Resolve SFT checkpoint ──────────────────────────────────
    sft_path = "outputs/sft_checkpoint/merged"
    if not os.path.exists(sft_path):
        sft_path = "outputs/sft_checkpoint"
    print(f"SFT checkpoint: {sft_path}")

    # ── Load dataset samples ────────────────────────────────────
    print(f"Loading {args.samples} samples from {grpo_cfg['dataset']}")
    ds = load_dataset(grpo_cfg["dataset"], split="train", streaming=True, token=True)
    samples = []
    for i, row in enumerate(ds):
        if i >= args.samples:
            break
        samples.append({"problem": row["problem"], "answer": str(row["answer"])})
    print(f"Loaded {len(samples)} samples")

    # ── Format prompts with chat template ───────────────────────
    from data import format_grpo_prompt
    prompts = [format_grpo_prompt(s["problem"]) for s in samples]
    answers = [s["answer"] for s in samples]

    # ── Save questions and answers ──────────────────────────────
    q_path = os.path.join(args.output, "questions.json")
    a_path = os.path.join(args.output, "answers.json")
    with open(q_path, "w") as f:
        json.dump({i: s["problem"] for i, s in enumerate(samples)}, f, indent=2)
    with open(a_path, "w") as f:
        json.dump({i: s["answer"] for i, s in enumerate(samples)}, f, indent=2)
    print(f"Questions saved to {q_path}")
    print(f"Answers saved to {a_path}")

    if args.dry_run:
        print("[DRY-RUN] Would test filter with above config")
        return

    # ── Load vLLM engine once ───────────────────────────────────
    print(f"\nLoading SFT model into vLLM: {sft_path}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=sft_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    print(f"vLLM engine ready")

    # ── Run filter for each G value ─────────────────────────────
    summary_lines = [
        f"Filter Tune Results — {len(samples)} samples, max_completion={args.max_completion}\n",
        "=" * 60 + "\n",
    ]

    for g in args.group_sizes:
        print(f"\n{'─' * 40}")
        print(f"Testing G={g} (n={g} completions per prompt)...")

        sampling_params = SamplingParams(
            n=g,
            max_tokens=args.max_completion,
            temperature=0.7 if g > 1 else 0.0,
            top_p=0.9,
        )

        # Generate all completions in one vLLM call
        outputs = llm.generate(prompts, sampling_params)

        results = []
        correct_count = 0
        debug_entries = {} if args.debug else None

        for i, output in enumerate(outputs):
            completions = [c.text for c in output.outputs]
            any_correct = any(check_answer(c, answers[i]) for c in completions)
            num_correct = sum(1 for c in completions if check_answer(c, answers[i]))

            results.append({
                "idx": i,
                "any_correct": any_correct,
                "num_correct": num_correct,
                "total_attempts": g,
            })
            if any_correct:
                correct_count += 1

            if debug_entries is not None:
                debug_entries[str(i)] = {
                    "problem": samples[i]["problem"][:200],
                    "ground_truth": answers[i],
                    "completions": completions,
                    "any_correct": any_correct,
                    "num_correct": num_correct,
                }

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

        # ── Save debug completions ───────────────────────────────
        if debug_entries is not None:
            debug_path = os.path.join(args.output, f"debug_g{g}_completions.json")
            with open(debug_path, "w") as f:
                json.dump(debug_entries, f, indent=2)
            print(f"  Debug: {debug_path}")

    # ── Save summary ────────────────────────────────────────────
    summary_path = os.path.join(args.output, "summary.txt")
    with open(summary_path, "w") as f:
        f.writelines(summary_lines)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
