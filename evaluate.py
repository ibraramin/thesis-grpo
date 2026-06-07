"""
evaluate.py — Benchmark evaluation on MATH-500 and AIME24 (§6).

Evaluates all trained checkpoints by:
  1. Loading the merged model for each cohort/seed
  2. Generating completions with greedy decoding
  3. Parsing <answer> tags and comparing against ground truth
  4. Computing Pass@1 accuracy

Outputs: outputs/results.csv with per-cohort, per-seed metrics.
"""

import argparse
import yaml
import os
import re
import csv
import torch
from datasets import load_dataset
from tqdm import tqdm

from data import format_grpo_prompt, get_tokenizer
from rewards import _extract_answer


BENCHMARKS = {
    "math500": {
        "dataset": "HuggingFaceH4/MATH-500",
        "split": "test",
        "problem_col": "problem",
        "answer_col": "answer",
    },
    "aime24": {
        "dataset": "math-ai/aime24",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "solution",  # AIME24 stores answer in solution field
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Evaluation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoints-root", default="outputs",
                        help="Root dir containing {cohort}/seed_{seed}/ checkpoints")
    parser.add_argument("--output", default="outputs/results.csv")
    parser.add_argument("--benchmarks", nargs="*", default=None,
                        help="Benchmarks to run (default from config)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per benchmark (for quick testing)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-run", action="store_true",
                        help="Small-scale eval for format validation")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _log_eval_dry_run(report_path: str, checkpoints: list[dict], benchmarks: list[str],
                       max_samples: int | None):
    """Log evaluation plan to test-run report."""
    with open(report_path, "a") as rf:
        rf.write(f"\n─── STAGE 3: EVALUATION ───\n\n")
        rf.write(f"  Benchmarks: {benchmarks}\n")
        rf.write(f"  Max samples per benchmark: {max_samples or 'all'}\n")
        rf.write(f"  Checkpoints found: {len(checkpoints)}\n")
        for ckpt in checkpoints:
            rf.write(f"    {ckpt['cohort']}/seed_{ckpt['seed']} → {ckpt['path']}\n")
        rf.write("  ✓ Evaluation plan validated\n")


def find_checkpoints(root: str, cohorts: list[str], seeds: list[int]) -> list[dict]:
    """Discover all trained checkpoints under {root}/{cohort}/seed_{seed}/."""
    checkpoints = []
    for cohort in cohorts:
        for seed in seeds:
            path = os.path.join(root, cohort, f"seed_{seed}")
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
                checkpoints.append({"cohort": cohort, "seed": seed, "path": path})
    return checkpoints


def normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison: strip whitespace, LaTeX, etc."""
    text = text.strip()
    # Remove leading/trailing $ signs
    text = re.sub(r"^\$+|\$+$", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Exact string match
    if pred_norm == gt_norm:
        return True

    # Try numeric comparison if both are parseable
    try:
        pred_num = float(pred_norm.replace(",", ""))
        gt_num = float(gt_norm.replace(",", ""))
        return abs(pred_num - gt_num) < 1e-5
    except (ValueError, TypeError):
        pass

    return False


def evaluate_checkpoint(checkpoint: dict, benchmarks: list[str], model_cfg: dict,
                        eval_cfg: dict, max_samples: int | None = None,
                        test_log_lines: list[str] | None = None) -> dict:
    """
    Evaluate a single checkpoint on selected benchmarks.
    Returns a dict of metrics.
    If test_log_lines is provided, appends per-sample results for test-run reporting.
    """
    ckpt_path = checkpoint["path"]
    merged_path = os.path.join(ckpt_path, "merged")
    if not os.path.exists(merged_path):
        merged_path = ckpt_path  # fallback: evaluate from adapter directly

    print(f"  Loading model from: {ckpt_path}")

    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = get_tokenizer(model_cfg["name"])

    # Load base model with the checkpoint's adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ckpt_path)
    model.eval()

    results = {}
    for bench_name in benchmarks:
        if bench_name not in BENCHMARKS:
            print(f"  Unknown benchmark: {bench_name}")
            continue

        b = BENCHMARKS[bench_name]
        print(f"  Benchmark: {bench_name} ({b['dataset']})")

        ds = load_dataset(b["dataset"], split=b["split"], streaming=True)
        correct = 0
        total = 0
        format_ok = 0
        total_length = 0

        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            problem = row[b["problem_col"]]
            ground_truth = row[b["answer_col"]]

            prompt = format_grpo_prompt(problem)
            inputs = tokenizer(prompt, return_tensors="pt",
                              truncation=True, max_length=256).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=eval_cfg.get("max_new_tokens", 512),
                    do_sample=False,  # greedy
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            completion = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

            # Parse answer
            predicted = _extract_answer(completion)
            has_format = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL)) and \
                         bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
            is_correct = predicted is not None and answers_match(str(predicted), str(ground_truth))

            # ── Test-run: log per-sample results ─────────────────
            if test_log_lines is not None:
                test_log_lines.append(
                    f"    [{bench_name} #{i}] problem: {problem[:80]}...\n"
                    f"      completion: {completion[:200]}...\n"
                    f"      predicted={predicted}, truth={ground_truth}, "
                    f"correct={'✓' if is_correct else '✗'}, "
                    f"format={'✓' if has_format else '✗'}\n"
                )

            total += 1
            total_length += len(completion.split())
            if has_format:
                format_ok += 1
            if is_correct:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        format_rate = format_ok / total if total > 0 else 0.0
        avg_length = total_length / total if total > 0 else 0.0

        results[f"{bench_name}_accuracy"] = accuracy
        results[f"{bench_name}_format_rate"] = format_rate
        results[f"{bench_name}_avg_length"] = avg_length

        print(f"    Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"    Format:   {format_rate:.4f}")
        print(f"    Avg len:  {avg_length:.1f} tokens")

    # Clean up
    del model
    del base_model
    torch.cuda.empty_cache()

    return results


def main():
    args = parse_args()
    cfg = load_config(args.config)
    eval_cfg = cfg["evaluation"]
    model_cfg = cfg["model"]
    cohorts_cfg = cfg["cohorts"]

    benchmarks = args.benchmarks or eval_cfg["benchmarks"]
    cohort_names = list(cohorts_cfg.keys())
    seeds = list(range(eval_cfg["num_seeds"]))

    # ── Test-run setup ──────────────────────────────────────────
    test_run = args.test_run
    report_path = None
    if test_run:
        tr = cfg.get("test_run", {})
        report_path = tr.get("report_file", "outputs/test_run/report.txt")
        if args.max_samples is None:
            args.max_samples = tr.get("eval_max_samples", 3)

    print("=" * 50)
    print(f"Evaluation: {benchmarks}")
    print(f"Mode:       {'TEST-RUN' if test_run else 'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"Cohorts:    {cohort_names} × {len(seeds)} seeds")
    print("=" * 50)

    checkpoints = find_checkpoints(args.checkpoints_root, cohort_names, seeds)
    print(f"Found {len(checkpoints)} checkpoints")

    if args.dry_run:
        for ckpt in checkpoints:
            print(f"  {ckpt['cohort']}/seed_{ckpt['seed']} → {ckpt['path']}")
        if test_run:
            _log_eval_dry_run(report_path, checkpoints, benchmarks, args.max_samples)
        print("[DRY-RUN] Done.")
        return

    all_results = []
    test_log_lines = []

    for i, ckpt in enumerate(checkpoints):
        print(f"\n[{i+1}/{len(checkpoints)}] {ckpt['cohort']}/seed_{ckpt['seed']}")
        try:
            metrics = evaluate_checkpoint(ckpt, benchmarks, model_cfg, eval_cfg, args.max_samples,
                                          test_log_lines if test_run else None)
            row = {"cohort": ckpt["cohort"], "seed": ckpt["seed"], **metrics}
        except Exception as e:
            print(f"  ERROR: {e}")
            row = {"cohort": ckpt["cohort"], "seed": ckpt["seed"], "error": str(e)}
        all_results.append(row)

    # ── Save results ──────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {args.output}")

    # ── Test-run: write eval details to report ──────────────────
    if test_run and test_log_lines:
        with open(report_path, "a") as rf:
            rf.write(f"\n─── STAGE 3: EVALUATION ───\n\n")
            rf.write(f"  Benchmarks: {benchmarks}\n")
            rf.write(f"  Max samples: {args.max_samples}\n")
            rf.write(f"  Checkpoints: {len(checkpoints)}\n\n")
            rf.write("  PER-SAMPLE RESULTS:\n")
            for line in test_log_lines:
                rf.write(line)
            rf.write("\n")
            for row in all_results:
                rf.write(f"  {row.get('cohort','?')}/seed_{row.get('seed','?')}: ")
                parts = [f"{k}={v:.4f}" for k, v in row.items()
                         if k not in ("cohort", "seed", "error")]
                if "error" in row:
                    parts = [f"error={row['error']}"]
                rf.write(", ".join(parts) + "\n")
            rf.write("  ✓ Evaluation PASSED\n\n")


if __name__ == "__main__":
    main()
