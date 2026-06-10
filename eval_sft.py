"""
eval_sft.py — Benchmark the SFT merged model (no LoRA) as baseline control.

Uses vLLM for fast batch evaluation on GSM8k + SVAMP. Outputs a results.csv-compatible
row so analyze.py can compare SFT against GRPO cohorts directly.

Usage:
    python eval_sft.py --sft-checkpoint outputs/sft_checkpoint/merged
"""
import argparse
import csv
import json
import os
import re
import time
import torch

from data import format_grpo_prompt, get_tokenizer
from rewards import _extract_answer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

BENCHMARKS = {
    "gsm8k": {
        "dataset": "gsm8k", "split": "test",
        "problem_col": "problem", "answer_col": "answer",
        "local": os.path.join(DATA_DIR, "gsm8k_eval.jsonl"),
    },
    "svamp": {
        "dataset": "ChilleD/SVAMP", "split": "test",
        "problem_col": "problem", "answer_col": "answer",
        "local": os.path.join(DATA_DIR, "svamp_eval.jsonl"),
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SFT model as baseline")
    p.add_argument("--sft-checkpoint", default="outputs/sft_checkpoint/merged")
    p.add_argument("--output", default="outputs/sft_baseline.csv")
    return p.parse_args()


def load_local_jsonl(path: str, max_samples: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def answers_match(predicted: str, ground_truth: str) -> bool:
    pred_norm = predicted.strip()
    gt_norm = ground_truth.strip()
    if pred_norm == gt_norm:
        return True
    try:
        return abs(float(pred_norm.replace(",", "")) - float(gt_norm.replace(",", ""))) < 1e-5
    except (ValueError, TypeError):
        return False


def eval_benchmark(vllm_model_path: str, bench_name: str, max_tokens: int = 256) -> dict:
    from vllm import LLM, SamplingParams

    b = BENCHMARKS[bench_name]
    local = b.get("local")
    if local and os.path.exists(local):
        rows = load_local_jsonl(local)
        print(f"  Loaded {len(rows)} samples from {local}")
    else:
        from datasets import load_dataset
        print(f"  Streaming {bench_name} from HF...")
        ds = load_dataset(b["dataset"], split=b["split"], streaming=True, token=True)
        rows = list(ds)

    prompts = [format_grpo_prompt(r[b["problem_col"]]) for r in rows]
    answers = [str(r[b["answer_col"]]) for r in rows]

    print(f"  vLLM batch-generation of {len(prompts)} prompts...")
    llm = LLM(model=vllm_model_path, trust_remote_code=True, gpu_memory_utilization=0.5)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)

    correct, total, format_ok, total_len = 0, 0, 0, 0
    for out, ans in zip(outputs, answers):
        comp = out.outputs[0].text
        pred = _extract_answer(comp)
        has_fmt = bool(re.search(r"<think>.*?</think>", comp, re.DOTALL)) and \
                  bool(re.search(r"<answer>.*?</answer>", comp, re.DOTALL))
        ok = pred is not None and answers_match(str(pred), ans)
        total += 1
        total_len += len(comp.split())
        if has_fmt: format_ok += 1
        if ok:       correct += 1

    del llm; torch.cuda.empty_cache()
    acc = correct / total if total else 0
    fmt = format_ok / total if total else 0
    avg_len = total_len / total if total else 0
    print(f"    Accuracy: {acc:.4f} ({correct}/{total})  Format: {fmt:.4f}  "
          f"Avg len: {avg_len:.0f}  ({time.time()-t0:.0f}s)")
    return {
        f"{bench_name}_accuracy": acc,
        f"{bench_name}_format_rate": fmt,
        f"{bench_name}_avg_length": avg_len,
    }


def main():
    args = parse_args()
    sft_merged = os.path.join(args.sft_checkpoint, "merged") \
        if not args.sft_checkpoint.endswith("merged") else args.sft_checkpoint

    if not os.path.isdir(sft_merged):
        print(f"ERROR: SFT merged checkpoint not found at {sft_merged}")
        return

    results = {"cohort": "SFT", "seed": "—"}
    for bench in ["gsm8k", "svamp"]:
        print(f"\n  Benchmark: {bench}")
        results.update(eval_benchmark(sft_merged, bench))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results.keys()))
        w.writeheader()
        w.writerow(results)
    print(f"\nSFT baseline saved to {args.output}")
    for k, v in results.items():
        if k not in ("cohort", "seed"):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
