"""
filter_benchmark.py — Compare base vs SFT model on Big-Math solve rate.

Runs the same 500 Big-Math-RL-Verified prompts through both:
  - Base model: Qwen/Qwen2.5-1.5B (no fine-tuning)
  - SFT model: outputs/sft_checkpoint/merged

For each model, generates G=4 completions per prompt (T=0, 512 tokens)
and reports how many prompts yield at least one correct answer.

This measures whether SFT actually increases math capability.

Usage:
    python filter_benchmark.py
    python filter_benchmark.py --prompts 1000 --g 8
"""

import argparse
import yaml
import shutil
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from data import format_grpo_prompt, _check_answer


def parse_args():
    parser = argparse.ArgumentParser(description="Filter Benchmark: base vs SFT")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sft-checkpoint", default="outputs/sft_checkpoint/merged")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompts", type=int, default=500)
    parser.add_argument("--g", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def vllm_stop(llm):
    """Attempt to clean up vLLM GPU memory."""
    try:
        import gc, torch
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def benchmark_model(model_path: str, model_label: str, prompts_data: list[dict],
                    args) -> tuple[set[int], list[dict]]:
    """Run vLLM inference on prompts, return indices of solved prompts + per-prompt stats."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_label}")
    print(f"Model path:  {model_path}")
    print(f"Prompts:     {len(prompts_data)}")
    print(f"G:           {args.g}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"{'='*60}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        n=args.g,
    )

    solved_indices = set()
    per_prompt = []

    for i, entry in enumerate(tqdm(prompts_data, desc=model_label, unit="prompt")):
        prompt_str = entry["prompt"]
        answer = entry["answer"]

        outputs = llm.generate([prompt_str], sampling_params)
        completions = [o.text for o in outputs[0].outputs]
        results = [_check_answer(comp, answer) for comp in completions]
        any_correct = any(results)
        num_correct = sum(results)

        if any_correct:
            solved_indices.add(i)

        per_prompt.append({
            "idx": i,
            "problem": entry["problem"][:100],
            "answer": answer,
            "num_correct": num_correct,
            "any_correct": any_correct,
        })

    vllm_stop(llm)
    return solved_indices, per_prompt


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["training"]["grpo"]["dataset"]

    # ── Load sample prompts (same set for both models) ─────────
    print(f"Loading {args.prompts} prompts from {dataset_name}")
    ds = load_dataset(dataset_name, split="train", streaming=True, token=True)

    prompts_data = []
    for row in ds:
        if len(prompts_data) >= args.prompts:
            break
        problem = row.get("problem", "")
        answer = row.get("answer", "")
        if problem and answer:
            prompt_str = format_grpo_prompt(problem)
            prompts_data.append({
                "prompt": prompt_str,
                "answer": answer,
                "problem": problem,
            })

    print(f"Loaded {len(prompts_data)} prompts\n")

    # ── Benchmark base model ───────────────────────────────────
    base_solved, base_details = benchmark_model(
        args.base_model, "BASE", prompts_data, args
    )

    # ── Benchmark SFT model ────────────────────────────────────
    import os
    if not os.path.exists(args.sft_checkpoint):
        print(f"\nERROR: SFT checkpoint not found at {args.sft_checkpoint}")
        print("Run SFT first: python run_sft.py")
        return

    sft_solved, sft_details = benchmark_model(
        args.sft_checkpoint, "SFT ", prompts_data, args
    )

    # ── Analysis ───────────────────────────────────────────────
    total = len(prompts_data)
    base_n = len(base_solved)
    sft_n = len(sft_solved)
    shared = len(base_solved & sft_solved)
    base_only = len(base_solved - sft_solved)
    sft_only = len(sft_solved - base_solved)
    neither = total - len(base_solved | sft_solved)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Correct':>8} {'Rate':>8}")
    print(f"{'-'*30}")
    print(f"{'BASE':<12} {base_n:>8} {base_n/total*100:>7.1f}%")
    print(f"{'SFT':<12} {sft_n:>8} {sft_n/total*100:>7.1f}%")
    print(f"{'Δ':<12} {sft_n-base_n:>+8} {(sft_n-base_n)/total*100:>+7.1f}pp")
    print(f"\nOverlap analysis:")
    print(f"  Both correct:     {shared:>4} ({shared/total*100:5.1f}%)")
    print(f"  BASE only:        {base_only:>4} ({base_only/total*100:5.1f}%)")
    print(f"  SFT only:         {sft_only:>4} ({sft_only/total*100:5.1f}%)")
    print(f"  Neither correct:  {neither:>4} ({neither/total*100:5.1f}%)")

    # ── SFT-only problems (new capabilities gained) ────────────
    if sft_only > 0:
        print(f"\nSFT-GAINED problems ({sft_only} new):")
        for i in sorted(list(sft_solved - base_solved))[:5]:
            entry = prompts_data[i]
            print(f"  [{i}] {entry['problem'][:80]}...")
            print(f"       answer: {entry['answer']}")

    # ── Conclusion ─────────────────────────────────────────────
    delta = sft_n - base_n
    if delta > 0:
        print(f"\n✓ SFT IMPROVEMENT: +{delta} prompts ({delta/total*100:.1f}pp)")
        print(f"  SFT boosted solve rate from {base_n/total*100:.1f}% → {sft_n/total*100:.1f}%")
    elif delta == 0:
        print(f"\n⚠  SFT had NO EFFECT on solve rate. Both models at {base_n/total*100:.1f}%")
        print(f"  Consider: more SFT data, more epochs, or a larger base model.")
    else:
        print(f"\n✗ SFT DEGRADED: {delta} prompts ({-delta/total*100:.1f}pp)")
        print(f"  Something went wrong — check SFT training.")


if __name__ == "__main__":
    main()
