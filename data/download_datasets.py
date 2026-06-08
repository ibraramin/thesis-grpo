"""
download_datasets.py — Pull production datasets locally for air-gapped deployment.

Downloads each dataset to data/*.jsonl (or .jsonl.gz for large files).
Outputs:
  data/openr1_math_220k_sft.jsonl.gz   — SFT samples (gzipped, long completions)
  data/openmath_instruct_2_grpo.jsonl   — GRPO samples
  data/math500_eval.jsonl               — MATH-500 eval
  data/aime24_eval.jsonl                — AIME24 eval
"""

import json
import os
import gzip
import argparse
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__) or ".", "")


def download_openr1_sft(n: int = 5000):
    """OpenR1-Math-220k SFT.  Saves gzipped (completions are large)."""
    print(f"Downloading open-r1/OpenR1-Math-220k (default, train) — {n} samples...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train",
                      streaming=True, token=True)
    out = os.path.join(DATA_DIR, "openr1_math_220k_sft.jsonl.gz")
    written = 0
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= n:
                break
            generation = None
            if row.get("generations") and row.get("correctness_math_verify"):
                for gen, ok in zip(row["generations"], row["correctness_math_verify"]):
                    if ok:
                        generation = gen
                        break
            if generation is None and row.get("generations"):
                generation = row["generations"][0]
            if not generation or not row.get("problem"):
                continue
            f.write(json.dumps({"problem": row["problem"], "completion": generation}) + "\n")
            written += 1
    print(f"  Saved {written} samples → {out} ({os.path.getsize(out)/1024:.0f} KB)")


def download_openmath_grpo(n: int = 50000):
    """OpenMathInstruct-2 GRPO dataset.  Saves problem + expected_answer."""
    print(f"Downloading nvidia/OpenMathInstruct-2 (train) — {n} samples...")
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train",
                      streaming=True, token=True)
    out = os.path.join(DATA_DIR, "openmath_instruct_2_grpo.jsonl")
    written = 0
    with open(out, "w") as f:
        for i, row in enumerate(ds):
            if i >= n:
                break
            problem = row.get("problem", "")
            answer = row.get("expected_answer", "")
            if not problem or not answer:
                continue
            f.write(json.dumps({"problem": problem, "answer": str(answer)}) + "\n")
            written += 1
    print(f"  Saved {written} samples → {out} ({os.path.getsize(out)/1024:.0f} KB)")


def download_math500(n: int = 500):
    """MATH-500 evaluation dataset."""
    print(f"Downloading HuggingFaceH4/MATH-500 (test) — {n} samples...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test",
                      streaming=True, token=True)
    out = os.path.join(DATA_DIR, "math500_eval.jsonl")
    written = 0
    with open(out, "w") as f:
        for i, row in enumerate(ds):
            if i >= n:
                break
            f.write(json.dumps({"problem": row["problem"], "answer": row["answer"]}) + "\n")
            written += 1
    print(f"  Saved {written} samples → {out} ({os.path.getsize(out)/1024:.0f} KB)")


def download_aime24(n: int = 30):
    """AIME24 evaluation dataset."""
    print(f"Downloading math-ai/aime24 — {n} samples...")
    ds = load_dataset("math-ai/aime24", split="test", streaming=True, token=True)
    out = os.path.join(DATA_DIR, "aime24_eval.jsonl")
    written = 0
    with open(out, "w") as f:
        for i, row in enumerate(ds):
            if i >= n:
                break
            f.write(json.dumps({"problem": row["problem"], "answer": row["solution"]}) + "\n")
            written += 1
    print(f"  Saved {written} samples → {out} ({os.path.getsize(out)/1024:.0f} KB)")


def download_gsm8k(n: int = 1319):
    """GSM8k evaluation dataset — grade-school math word problems."""
    print(f"Downloading gsm8k (test) — {n} samples...")
    ds = load_dataset("gsm8k", "main", split="test", streaming=True, token=True)
    out = os.path.join(DATA_DIR, "gsm8k_eval.jsonl")
    written = 0
    with open(out, "w") as f:
        for i, row in enumerate(ds):
            if i >= n:
                break
            # GSM8k answer format: "#### <number>", extract just the number
            answer = row.get("answer", "")
            if "####" in answer:
                answer = answer.split("####")[-1].strip()
            f.write(json.dumps({"problem": row["question"], "answer": str(answer)}) + "\n")
            written += 1
    print(f"  Saved {written} samples → {out} ({os.path.getsize(out)/1024:.0f} KB)")


def download_svamp(n: int = 1000):
    """SVAMP — simple math word problems with variations."""
    print(f"Downloading SVAMP (test) — {n} samples...")
    ds = load_dataset("ChilleD/SVAMP", split="test", streaming=True, token=True)
    out = os.path.join(DATA_DIR, "svamp_eval.jsonl")
    written = 0
    with open(out, "w") as f:
        for i, row in enumerate(ds):
            if i >= n:
                break
            problem = f"{row.get('Body', '')} {row.get('Question', '')}"
            answer = str(row.get("Answer", ""))
            f.write(json.dumps({"problem": problem, "answer": answer}) + "\n")
            written += 1
    print(f"  Saved {written} samples → {out} ({os.path.getsize(out)/1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Download all production datasets")
    parser.add_argument("--sft-samples", type=int, default=5000)
    parser.add_argument("--grpo-samples", type=int, default=50000)
    parser.add_argument("--math500-samples", type=int, default=500)
    parser.add_argument("--aime24-samples", type=int, default=30)
    parser.add_argument("--gsm8k-samples", type=int, default=1319)
    parser.add_argument("--svamp-samples", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    download_openr1_sft(args.sft_samples)
    download_openmath_grpo(args.grpo_samples)
    download_math500(args.math500_samples)
    download_aime24(args.aime24_samples)
    download_gsm8k(args.gsm8k_samples)
    download_svamp(args.svamp_samples)

    print("\nAll datasets downloaded to data/*.jsonl")
    print("Add these files to git and push.")


if __name__ == "__main__":
    main()
