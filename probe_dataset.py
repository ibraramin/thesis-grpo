"""
probe_dataset.py — Dataset solve-rate probe with EOS-looping diagnostics.

Tests the revision.md hypothesis that Big-Math-RL-Verified is too hard
for Qwen2.5-1.5B (2% solvable) while OpenMathInstruct-2 should yield
30-60% solvable prompts, giving GRPO meaningful gradient signal.

Now with EOS-looping detection: repetitive tag generation (</think>,
</blockquote>, etc.) that masks true solve rates.

Run: python probe_dataset.py --sft-checkpoint outputs/sft_checkpoint --samples 500
"""

import argparse
import json
import os
import time
import torch

from data import format_grpo_prompt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

LOOP_TAGS = ["</think>", "</blockquote>", "</answer>", "\n\n"]


def _detect_eos_loop(completion: str):
    """Detect EOS-looping: repetitive tag sequences at end of completion.

    Returns (is_looping: bool, loop_tag: str | None, loop_count: int).
    """
    tail = completion[-400:] if len(completion) > 400 else completion
    for tag in LOOP_TAGS:
        count = 0
        pos = len(tail)
        tag_len = len(tag)
        while pos >= tag_len and tail[pos - tag_len:pos] == tag:
            count += 1
            pos -= tag_len
        if count >= 5:
            return True, tag, count
    return False, None, 0


def _total_tokens(completion: str) -> int:
    return len(completion.split())


def _completion_stats(completion: str, ground_truth: str):
    """Compute per-completion diagnostics: correctness, format, looping, length."""
    import re
    from data import _check_answer

    has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
    has_tags = has_think and has_answer
    is_correct = _check_answer(completion, str(ground_truth))
    is_empty = len(completion.strip()) == 0 or completion.strip() == "<|im_start|>assistant"
    is_looping, loop_tag, loop_n = _detect_eos_loop(completion)

    return {
        "has_tags": has_tags,
        "has_think": has_think,
        "has_answer": has_answer,
        "correct": is_correct,
        "empty": is_empty,
        "eos_loop": is_looping,
        "loop_tag": loop_tag,
        "loop_count": loop_n,
        "length_tokens": _total_tokens(completion),
        "length_chars": len(completion),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Probe dataset solve rates + EOS-loop diagnostics")
    parser.add_argument("--sft-checkpoint", default="outputs/sft_checkpoint",
                        help="Path to merged SFT checkpoint")
    parser.add_argument("--model-path", default=None,
                        help="Explicit local path to model (bypasses cache lookup)")
    parser.add_argument("--dataset-path", default=None,
                        help="Local JSONL file instead of streaming from HF")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of prompts to test per dataset")
    parser.add_argument("--max-completion", type=int, default=512,
                        help="Max new tokens per generation (default: 512)")
    parser.add_argument("--output", default="outputs/probe_results.json")
    parser.add_argument("--debug-file", default=None,
                        help="Dump all EOS-looped completions to this JSON file")
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


def probe_dataset(
    model, tokenizer, dataset_name: str, split: str,
    problem_col: str, answer_col: str, n_samples: int,
    max_completion: int = 512,
    debug_loop_file: str | None = None,
) -> dict:
    """Single greedy-generation probe: T=0, check correctness + EOS-looping.

    Uses local JSONL from data/ when available, otherwise HF streaming."""
    from datasets import load_dataset
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
    looped = 0
    results = []
    loop_debug_entries = [] if debug_loop_file else None
    t0 = time.time()

    for i, row in enumerate(ds):
        if total >= n_samples:
            break

        problem = row.get(problem_col, "")
        ground_truth = row.get(answer_col, "")
        if not problem or not ground_truth:
            continue

        prompt = format_grpo_prompt(str(problem))
        try:
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=256).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_completion,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            completion = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
        except Exception as e:
            errors += 1
            results.append({"problem": str(problem)[:100], "error": str(e)})
            continue

        total += 1
        stats = _completion_stats(completion, ground_truth)

        if stats["correct"]:
            correct += 1
        if stats["has_tags"]:
            format_ok += 1
        if stats["empty"]:
            empty += 1
        if stats["eos_loop"]:
            looped += 1

        entry = {
            "problem": str(problem)[:150],
            "ground_truth": str(ground_truth)[:80],
            "completion": completion,  # full, not truncated
            **stats,
        }
        results.append(entry)

        if loop_debug_entries is not None and stats["eos_loop"]:
            loop_debug_entries.append(entry)

        if total % 10 == 0 or total == 1:
            print(f"  [{total}/{n_samples}] acc={correct/total:.1%} "
                  f"fmt={format_ok/total:.1%} loop={looped/total:.1%}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if total % 50 == 0:
            elapsed = time.time() - t0
            print(f"    {total}/{n_samples} | correct={correct:3d} ({correct/total:.1%}) | "
                  f"format={format_ok} | looped={looped} | empty={empty} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    solve_rate = correct / total if total > 0 else 0.0
    format_rate = format_ok / total if total > 0 else 0.0
    loop_rate = looped / total if total > 0 else 0.0

    print(f"    Done in {elapsed:.0f}s")
    print(f"    Solve rate:  {correct}/{total} = {solve_rate:.1%}")
    print(f"    Format rate: {format_ok}/{total} = {format_rate:.1%}")
    print(f"    EOS-loop:    {looped}/{total} = {loop_rate:.1%}")
    print(f"    Empty: {empty}  Errors: {errors}")

    result = {
        "dataset": dataset_name,
        "samples_tested": total,
        "correct": correct,
        "solve_rate": solve_rate,
        "format_rate": format_rate,
        "loop_rate": loop_rate,
        "looped_count": looped,
        "empty": empty,
        "errors": errors,
        "elapsed_s": elapsed,
        "results": results,
    }

    if loop_debug_entries is not None and len(loop_debug_entries) > 0:
        os.makedirs(os.path.dirname(debug_loop_file), exist_ok=True)
        with open(debug_loop_file, "w") as f:
            json.dump({
                "description": "Completions with EOS-looping (repetitive tag generation)",
                "total_looped": len(loop_debug_entries),
                "loop_rate": loop_rate,
                "entries": loop_debug_entries,
            }, f, indent=2)
        print(f"    Looped completions dumped to: {debug_loop_file}")

    return result


def probe_dataset_vllm(
    merged_model_path: str, dataset_name: str, split: str,
    problem_col: str, answer_col: str, n_samples: int,
    max_completion: int = 256,
    debug_loop_file: str | None = None,
) -> dict:
    """vLLM-based probe: fast batch greedy generation + EOS-looping detection."""
    from vllm import LLM, SamplingParams
    import re, json

    print(f"\n  Loading vLLM model from {merged_model_path} ...")
    llm = LLM(model=merged_model_path, trust_remote_code=True, gpu_memory_utilization=0.85)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_completion)

    # Load data
    local_file = os.path.join(DATA_DIR, "openmath_instruct_2_probe.jsonl")
    if os.path.exists(local_file):
        print(f"  Loading from local: {local_file}")
        rows = []
        with open(local_file) as f:
            for i, line in enumerate(f):
                if i >= n_samples:
                    break
                rows.append(json.loads(line))
    else:
        from datasets import load_dataset
        token = os.environ.get("HF_TOKEN") or True
        ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
        rows = []
        for i, row in enumerate(ds):
            if i >= n_samples:
                break
            rows.append(row)

    prompts = []
    answers = []
    for row in rows:
        p = row.get(problem_col, "")
        a = row.get(answer_col, "")
        if not p or not a:
            continue
        prompts.append(format_grpo_prompt(str(p)))
        answers.append(str(a))

    print(f"  Generating {len(prompts)} prompts via vLLM (max_tokens={max_completion}) ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({len(prompts)/elapsed:.0f} samples/s)")

    correct = 0
    format_ok = 0
    empty = 0
    errors = 0
    looped = 0
    results = []
    loop_debug_entries = [] if debug_loop_file else None

    for i, (prompt, answer, output) in enumerate(zip(prompts, answers, outputs)):
        completion = output.outputs[0].text
        stats = _completion_stats(completion, answer)

        if stats["correct"]:
            correct += 1
        if stats["has_tags"]:
            format_ok += 1
        if stats["empty"]:
            empty += 1
        if stats["eos_loop"]:
            looped += 1

        entry = {
            "problem": prompt[:150],
            "ground_truth": answer[:80],
            "completion": completion,  # full
            **stats,
        }
        results.append(entry)

        if loop_debug_entries is not None and stats["eos_loop"]:
            loop_debug_entries.append(entry)

    total = len(prompts)
    solve_rate = correct / total if total > 0 else 0
    format_rate = format_ok / total if total > 0 else 0
    loop_rate = looped / total if total > 0 else 0

    print(f"    Solve rate:  {correct}/{total} = {solve_rate:.1%}")
    print(f"    Format rate: {format_ok}/{total} = {format_rate:.1%}")
    print(f"    EOS-loop:    {looped}/{total} = {loop_rate:.1%}")

    r = {
        "dataset": dataset_name,
        "samples_tested": total,
        "correct": correct,
        "solve_rate": solve_rate,
        "format_rate": format_rate,
        "loop_rate": loop_rate,
        "looped_count": looped,
        "empty": empty,
        "errors": errors,
        "elapsed_s": elapsed,
        "results": results,
    }

    if loop_debug_entries is not None and len(loop_debug_entries) > 0:
        os.makedirs(os.path.dirname(debug_loop_file), exist_ok=True)
        with open(debug_loop_file, "w") as f:
            json.dump({
                "description": "Completions with EOS-looping (repetitive tag generation)",
                "total_looped": len(loop_debug_entries),
                "loop_rate": loop_rate,
                "entries": loop_debug_entries,
            }, f, indent=2)
        print(f"    Looped completions dumped to: {debug_loop_file}")

    return r


def main():
    args = parse_args()

    if args.dry_run:
        print("[PROBE] Dry-run: inspecting dataset schemas only...")
        inspect_dataset("open-r1/OpenR1-Math-220k", "train", 3)
        inspect_dataset("nvidia/OpenMathInstruct-2", "train", 3)
        print("\n[PROBE] Dry-run done. Run without --dry-run to benchmark solve rates.")
        return

    debug_loop = args.debug_file or args.output.replace(".json", "_loops.json")

    # ── Try vLLM first (requires merged model) ──────────────────
    merged_path = os.path.join(args.sft_checkpoint, "merged")
    use_vllm = False
    try:
        from vllm import LLM
        if os.path.isdir(merged_path) and os.path.exists(os.path.join(merged_path, "config.json")):
            use_vllm = True
    except ImportError:
        pass

    if use_vllm:
        print(f"\n{'='*60}")
        print("PROBE (vLLM): OpenMathInstruct-2")
        print(f"{'='*60}")
        om2_result = probe_dataset_vllm(
            merged_model_path=merged_path,
            dataset_name="nvidia/OpenMathInstruct-2",
            split="train",
            problem_col="problem",
            answer_col="expected_answer",
            n_samples=args.samples,
            max_completion=args.max_completion,
            debug_loop_file=debug_loop,
        )
    else:
        print(f"\n{'='*60}")
        print("LOADING MODEL (HF)")
        print(f"{'='*60}")
        model, tokenizer = load_model("Qwen/Qwen2.5-1.5B", args.sft_checkpoint, args.model_path)

        print(f"\n{'='*60}")
        print("PROBE (HF): OpenMathInstruct-2")
        print(f"{'='*60}")
        om2_result = probe_dataset(
            model, tokenizer,
            dataset_name="nvidia/OpenMathInstruct-2",
            split="train",
            problem_col="problem",
            answer_col="expected_answer",
            n_samples=args.samples,
            max_completion=args.max_completion,
            debug_loop_file=debug_loop,
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sft_checkpoint": args.sft_checkpoint,
        "samples_per_dataset": args.samples,
        "config": {
            "G": 1, "T": "greedy", "max_completion": args.max_completion,
            "engine": "vLLM" if use_vllm else "HF",
        },
        "openmath_instruct_2": om2_result,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report: {args.output}")

    # Terminal summary
    r = om2_result
    print(f"\n{'Dataset':<30} {'Solve':>10} {'Format':>10} {'EOS-Loop':>10}")
    print("-" * 62)
    print(f"  {'OpenMathInstruct-2':<28} {r['solve_rate']:>9.1%} "
          f"{r['format_rate']:>9.1%} {r.get('loop_rate', 0):>9.1%}")
    print(f"\n  {r['samples_tested']} samples | {r['correct']} correct | "
          f"{r.get('looped_count', 0)} looped | {r['elapsed_s']:.0f}s")

    if r.get("loop_rate", 0) > 0.05:
        print(f"\n  ⚠  EOS-looping detected in {r['loop_rate']:.1%} of completions!")
        print(f"  This may suppress true solve rate. Check {debug_loop} for examples.")

    print(f"\nDone. Report saved to {args.output}")


if __name__ == "__main__":
    main()
