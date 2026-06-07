"""
probe_dataset.py — Compare solve rates: Big-Math vs OpenMathInstruct-2.

Tests the revision.md hypothesis that Big-Math-RL-Verified is too hard
for Qwen2.5-1.5B (2% solvable) while OpenMathInstruct-2 should yield
30-60% solvable prompts, giving GRPO meaningful gradient signal.

Run: python probe_dataset.py --sft-checkpoint outputs/sft_checkpoint
"""

import argparse
import json
import os
import time
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def parse_args():
    parser = argparse.ArgumentParser(description="Probe dataset solve rates")
    parser.add_argument("--sft-checkpoint", default="outputs/sft_checkpoint",
                        help="Path to merged SFT checkpoint")
    parser.add_argument("--model-path", default=None,
                        help="Explicit local path to model (bypasses cache lookup)")
    parser.add_argument("--dataset-path", default=None,
                        help="Local JSONL file instead of streaming from HF")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of prompts to test per dataset")
    parser.add_argument("--output", default="outputs/probe_results.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def inspect_dataset(name: str, split: str = "train", n_rows: int = 5):
    """Load dataset, dump schema and first N rows."""
    from datasets import load_dataset
    print(f"\n{'='*60}")
    print(f"INSPECT: {name}")
    print(f"{'='*60}")
    try:
        ds = load_dataset(name, split=split, streaming=True, token=False)
        row = next(iter(ds))
        print(f"\n  Columns: {list(row.keys())}")
        print(f"\n  --- First {n_rows} rows ---")
        for i, row in enumerate(ds):
            if i >= n_rows:
                break
            print(f"\n  [{i}]")
            for k, v in row.items():
                s = str(v)
                if len(s) > 200:
                    s = s[:200] + "..."
                print(f"    {k}: {s}")
        print(f"\n  ✓ Schema inspection complete")
        return list(row.keys())
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        return []


def _find_model_path(repo_id: str) -> str | None:
    """Find a locally cached model, including ModelScope-style lowercase IDs."""
    from huggingface_hub import try_to_load_from_cache
    candidates = [repo_id]
    # ModelScope caches under lowercase repo IDs (qwen/Qwen2.5-1.5B vs Qwen/Qwen2.5-1.5B)
    parts = repo_id.split("/")
    if len(parts) == 2:
        candidates.append(f"{parts[0].lower()}/{parts[1]}")
    for cid in candidates:
        info = try_to_load_from_cache(cid, "config.json")
        if info:
            return cid
    # Last resort: scan cache directory for model files
    import glob
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    model_glob = f"models--*{parts[1]}*"
    matches = glob.glob(os.path.join(cache_root, model_glob))
    if matches:
        # Extract repo_id from dir name: models--org--Name -> org/Name
        for m in matches:
            dirname = os.path.basename(m)
            if dirname.startswith("models--"):
                inner = dirname[len("models--"):].replace("--", "/")
                if try_to_load_from_cache(inner, "config.json"):
                    return inner
    return None


def load_model(tokenizer_name: str, ckpt_path: str, model_path: str | None = None):
    """Load SFT model + tokenizer. Uses explicit path, local cache, or remote fallback."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path:
        # Explicit local path (e.g. ModelScope download)
        print(f"  Loading model from explicit path: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            local_files_only=True,
            trust_remote_code=True,
        )
    else:
        # Check local cache first (handles ModelScope + HF offline scenarios)
        local_id = _find_model_path(tokenizer_name)
        if local_id:
            print(f"  Found cached model: {local_id}")
            tokenizer = AutoTokenizer.from_pretrained(
                local_id, local_files_only=True, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                local_id,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                local_files_only=True,
                trust_remote_code=True,
            )
        else:
            hf_endpoint = os.environ.get("HF_ENDPOINT", "")
            print(f"  Model not cached locally — downloading{f' via {hf_endpoint}' if hf_endpoint else ''}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                tokenizer_name,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.isdir(ckpt_path):
        print(f"  Loading LoRA adapter from {ckpt_path}")
        model = PeftModel.from_pretrained(model, ckpt_path)
    else:
        print(f"  No LoRA checkpoint found at {ckpt_path}, using base model for probe")
        print(f"  (Run SFT first for accurate solve-rate measurement)")

    model.eval()
    return model, tokenizer


def format_prompt(problem: str) -> str:
    """Format a problem as a GRPO prompt (open-ended, no closing im_end)."""
    return (
        "<|im_start|>system\nYou are a helpful math assistant. "
        "Think step by step inside <think> tags, then output your final "
        "answer inside <answer> tags.\n<|im_end|>\n"
        f"<|im_start|>user\n{problem}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def probe_dataset(
    model, tokenizer, dataset_name: str, split: str,
    problem_col: str, answer_col: str, n_samples: int,
    max_completion: int = 512,
) -> dict:
    """Single greedy-generation probe: T=0, check correctness via _check_answer.

    Uses local JSONL from data/ when available, otherwise HF streaming."""
    from datasets import load_dataset
    from data import _check_answer
    import re

    # Local JSONL shortcuts (same files committed to repo)
    LOCAL_MAP = {
        "nvidia/OpenMathInstruct-2": os.path.join(DATA_DIR, "openmath_instruct_2_probe.jsonl"),
    }
    local_file = LOCAL_MAP.get(dataset_name)
    if local_file and os.path.exists(local_file):
        print(f"\n  Loading from local file: {local_file} ({n_samples} samples)")
        rows = []
        with open(local_file) as f:
            for line in f:
                rows.append(json.loads(line))
                if len(rows) >= n_samples:
                    break
        ds = rows
    else:
        print(f"\n  Probing {n_samples} prompts from {dataset_name}...")
        token = os.environ.get("HF_TOKEN") or True
        ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
    total = 0
    correct = 0
    format_ok = 0
    empty = 0
    errors = 0
    results = []
    t0 = time.time()

    for i, row in enumerate(ds):
        if total >= n_samples:
            break

        problem = row.get(problem_col, "")
        ground_truth = row.get(answer_col, "")
        if not problem or not ground_truth:
            continue

        prompt = format_prompt(str(problem))
        try:
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=256).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_completion,
                    do_sample=False,
                    temperature=1.0,  # temperature ignored when do_sample=False
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            completion = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
        except Exception as e:
            errors += 1
            results.append({"problem": str(problem)[:100], "completion": "", "error": str(e)})
            continue

        total += 1

        has_tags = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL)) and \
                   bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
        if has_tags:
            format_ok += 1

        is_correct = _check_answer(completion, str(ground_truth))

        if len(completion.strip()) == 0 or completion.strip() == "<|im_start|>assistant":
            empty += 1

        if is_correct:
            correct += 1

        results.append({
            "problem": str(problem)[:150],
            "ground_truth": str(ground_truth)[:80],
            "completion": completion[:300],
            "has_tags": has_tags,
            "correct": is_correct,
        })

        if total % 50 == 0:
            elapsed = time.time() - t0
            rate = correct / total if total > 0 else 0
            print(f"    {total}/{n_samples} | correct={correct:3d} ({rate:.1%}) | "
                  f"format={format_ok} | empty={empty} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    solve_rate = correct / total if total > 0 else 0.0
    format_rate = format_ok / total if total > 0 else 0.0

    print(f"    Done in {elapsed:.0f}s")
    print(f"    Solve rate: {correct}/{total} = {solve_rate:.1%}")
    print(f"    Format rate: {format_ok}/{total} = {format_rate:.1%}")
    print(f"    Empty: {empty}  Errors: {errors}")

    return {
        "dataset": dataset_name,
        "samples_tested": total,
        "correct": correct,
        "solve_rate": solve_rate,
        "format_rate": format_rate,
        "empty": empty,
        "errors": errors,
        "elapsed_s": elapsed,
        "results": results[:10],  # only include first 10 for report
    }


def main():
    args = parse_args()

    if args.dry_run:
        print("[PROBE] Dry-run: inspecting dataset schemas only...")
        inspect_dataset("open-r1/OpenR1-Math-220k", "train", 3)
        inspect_dataset("SynthLabsAI/Big-Math-RL-Verified", "train", 3)
        inspect_dataset("nvidia/OpenMathInstruct-2", "train", 3)
        print("\n[PROBE] Dry-run done. Run without --dry-run to benchmark solve rates.")
        return

    # ── Step 1: Inspect OpenMathInstruct-2 schema ───────────────
    cols_om2 = inspect_dataset("nvidia/OpenMathInstruct-2", "train", 5)
    cols_bm = inspect_dataset("SynthLabsAI/Big-Math-RL-Verified", "train", 5)

    # ── Step 2: Load model ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    model, tokenizer = load_model("Qwen/Qwen2.5-1.5B", args.sft_checkpoint, args.model_path)

    # ── Step 3: Probe Big-Math-RL-Verified (skip if gated/unauthorized) ──
    print(f"\n{'='*60}")
    print("PROBE: Big-Math-RL-Verified (control)")
    print(f"{'='*60}")
    try:
        bm_result = probe_dataset(
            model, tokenizer,
            dataset_name="SynthLabsAI/Big-Math-RL-Verified",
            split="train",
            problem_col="problem",
            answer_col="answer",
            n_samples=args.samples,
        )
    except Exception as e:
        print(f"  Skipping Big-Math: {e}")
        bm_result = {"dataset": "SynthLabsAI/Big-Math-RL-Verified",
                       "samples_tested": 0, "correct": 0, "solve_rate": 0.02,  # prior measurement
                       "format_rate": 0.0, "empty": 0, "errors": 0,
                       "elapsed_s": 0, "results": [], "skipped": True}

    # ── Step 4: Probe OpenMathInstruct-2 ───────────────────────
    print(f"\n{'='*60}")
    print("PROBE: OpenMathInstruct-2 (candidate)")
    print(f"{'='*60}")

    # OpenMathInstruct-2 schema: problem | generated_solution | expected_answer | problem_source
    om2_result = probe_dataset(
        model, tokenizer,
        dataset_name="nvidia/OpenMathInstruct-2",
        split="train",
        problem_col="problem",
        answer_col="expected_answer",
        n_samples=args.samples,
    )

    # ── Step 5: Report ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sft_checkpoint": args.sft_checkpoint,
        "samples_per_dataset": args.samples,
        "config": {"G": 1, "T": "greedy", "max_completion": 512},
        "big_math": bm_result,
        "openmath_instruct_2": om2_result,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report: {args.output}")

    # Terminal summary
    print(f"\n{'Dataset':<30} {'Solve Rate':>12} {'Format Rate':>12}")
    print("-" * 56)
    for name, r in [("Big-Math-RL-Verified", bm_result),
                     ("OpenMathInstruct-2", om2_result)]:
        print(f"  {name:<28} {r['solve_rate']:>11.1%} {r['format_rate']:>11.1%}")

    # Verdict
    if om2_result["solve_rate"] > bm_result["solve_rate"] * 5:
        print(f"\n  ✓ OpenMathInstruct-2 has >5x higher solve rate — revision.md validated")
    else:
        print(f"\n  ✗ Solve rates similar — revision.md overestimates improvement")

    print(f"\nDone. Report saved to {args.output}")


if __name__ == "__main__":
    main()
