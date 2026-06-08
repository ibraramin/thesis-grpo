"""
data.py — Dataset loading, preprocessing, and tokenization pipeline.

Supports three dataset stages as defined in the methodology:
  1. Acquisition Set (SFT): open-r1/OpenR1-Math-220k
  2. Consolidation Set (GRPO): nvidia/OpenMathInstruct-2 (or Big-Math-RL-Verified)
  3. All-zero filtering: excludes samples the SFT model cannot solve

All loaders prefer local JSONL/JSONL.GZ files from data/ over HuggingFace
streaming, enabling fully offline (air-gapped) deployment.
"""

import json
import gzip
import os
import re
from datasets import load_dataset, Dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SFT_LOCAL = os.path.join(DATA_DIR, "openr1_math_220k_sft.jsonl.gz")
GRPO_LOCAL = os.path.join(DATA_DIR, "openmath_instruct_2_grpo.jsonl")

# ── Chat template for Qwen2.5 ──────────────────────────────────
QWEN_CHAT_TEMPLATE = """\
<|im_start|>system
You are a helpful mathematical reasoning assistant. Always structure your answer by placing your step-by-step reasoning inside <think></think> tags, and your final answer inside <answer></answer> tags.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
{completion}<|im_end|>"""

QWEN_PROMPT_TEMPLATE = """\
<|im_start|>system
You are a helpful mathematical reasoning assistant. Always structure your answer by placing your step-by-step reasoning inside <think></think> tags, and your final answer inside <answer></answer> tags.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
"""


def format_sft_chat(problem: str, completion: str) -> str:
    """Format a problem+completion pair for SFT (closed assistant turn)."""
    return QWEN_CHAT_TEMPLATE.format(problem=problem, completion=completion)


def format_grpo_prompt(problem: str) -> str:
    """Format a problem as a GRPO prompt (open-ended assistant turn)."""
    return QWEN_PROMPT_TEMPLATE.format(problem=problem)


# ── Few-Shot Prompting (§Evaluation) ────────────────────────────

FEW_SHOT_EXAMPLES = """\
<|im_start|>user
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|im_end|>
<|im_start|>assistant
<think>Janet's ducks lay 16 eggs/day. She eats 3 + uses 4 for muffins = 7 used. Remaining: 16 - 7 = 9 eggs sold. At $2 each: 9 × $2 = $18.</think>
<answer>18</answer><|im_end|>
<|im_start|>user
A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?<|im_end|>
<|im_start|>assistant
<think>Blue fiber: 2 bolts. White fiber: half of 2 = 1 bolt. Total: 2 + 1 = 3 bolts.</think>
<answer>3</answer><|im_end|>
<|im_start|>user
Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?<|im_end|>
<|im_start|>assistant
<think>Total cost: $80,000 + $50,000 = $130,000. Value increase: 150% of $80,000 = $120,000. New value: $80,000 + $120,000 = $200,000. Profit: $200,000 - $130,000 = $70,000.</think>
<answer>70000</answer><|im_end|>
<|im_start|>user
James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?<|im_end|>
<|im_start|>assistant
<think>Per mailing: 3 pages × 2 friends = 6 pages. Per week: 2 mailings × 6 pages = 12 pages. Per year: 52 weeks × 12 pages = 624 pages.</think>
<answer>624</answer><|im_end|>
<|im_start|>user
Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as yellow and purple flowers. How many flowers does Mark have in his garden?<|im_end|>
<|im_start|>assistant
<think>Yellow: 10. Purple: 10 × 1.80 = 18. Green: 25% of (10 + 18) = 0.25 × 28 = 7. Total: 10 + 18 + 7 = 35.</think>
<answer>35</answer><|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
"""


def format_few_shot_prompt(problem: str) -> str:
    """Format a problem with 5 few-shot examples using <think>/<answer> format."""
    system_msg = """<|im_start|>system
You are a helpful mathematical reasoning assistant. Always structure your answer by placing your step-by-step reasoning inside <think></think> tags, and your final answer inside <answer></answer> tags.<|im_end|>
"""
    return system_msg + FEW_SHOT_EXAMPLES.format(problem=problem)


# ── SFT Dataset ────────────────────────────────────────────────

def load_sft_dataset(config: dict, max_samples: int = 5000) -> Dataset:
    """
    Load and preprocess the OpenR1-Math-220k dataset for SFT.

    Prefers the local gzipped JSONL file (data/openr1_math_220k_sft.jsonl.gz)
    for offline deployment. Falls back to HuggingFace streaming.

    Returns a Dataset with a 'text' column suitable for SFTTrainer.
    """
    # ── Try local gzip copy first ────────────────────────────────
    if os.path.exists(SFT_LOCAL):
        samples = []
        with gzip.open(SFT_LOCAL, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                row = json.loads(line)
                problem = row.get("problem", "")
                completion = row.get("completion", "")
                if not problem or not completion:
                    continue
                text = format_sft_chat(problem, completion)
                samples.append({"text": text})
        print(f"[SFT] Loaded {len(samples)} samples from local: {SFT_LOCAL}")
        return Dataset.from_list(samples)

    # ── HuggingFace fallback ─────────────────────────────────────
    ds = load_dataset(
        config["training"]["sft"]["dataset"],
        "default",
        split="train",
        streaming=True,
        token=True,
    )
    samples = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        generation = None
        if row.get("generations") and row.get("correctness_math_verify"):
            for gen, correct in zip(row["generations"], row["correctness_math_verify"]):
                if correct:
                    generation = gen
                    break
        if generation is None and row.get("generations"):
            generation = row["generations"][0]

        problem = row.get("problem", "")
        if not generation or not problem:
            continue

        text = format_sft_chat(problem, generation)
        samples.append({"text": text})

    return Dataset.from_list(samples)


# ── GRPO Dataset ───────────────────────────────────────────────

def load_grpo_dataset(config: dict, max_samples: int = 2500) -> Dataset:
    """
    Load and preprocess the GRPO dataset for GRPO training.

    Priority order:
      1. Offline-filtered JSONL (filter_offline.py output)
      2. Local OpenMathInstruct-2 copy (data/openmath_instruct_2_grpo.jsonl)
      3. HuggingFace streaming (gated Big-Math or OpenMathInstruct-2)

    Returns a Dataset with 'prompt' and 'answer' columns.
    """
    # 1. Offline-filtered dataset
    filtered_path = config.get("filter_offline", {}).get("output",
                       "outputs/filtered_grpo/filtered_dataset.jsonl")
    if os.path.exists(filtered_path):
        return load_filtered_grpo_dataset(filtered_path, max_samples)

    # 2. Local OpenMathInstruct-2 copy
    if os.path.exists(GRPO_LOCAL):
        samples = []
        with open(GRPO_LOCAL) as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                row = json.loads(line)
                problem = row.get("problem", "")
                answer = row.get("answer", "")
                if not problem or not answer:
                    continue
                prompt = format_grpo_prompt(problem)
                samples.append({"prompt": prompt, "answer": answer})
        print(f"[GRPO] Loaded {len(samples)} samples from local: {GRPO_LOCAL}")
        return Dataset.from_list(samples)

    # 3. HuggingFace fallback
    ds = load_dataset(
        config["training"]["grpo"]["dataset"],
        split="train",
        streaming=True,
        token=True,
    )
    samples = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        problem = row.get("problem", "")
        answer = row.get("answer") or row.get("expected_answer", "")
        if not problem or not answer:
            continue

        prompt = format_grpo_prompt(problem)
        samples.append({"prompt": prompt, "answer": answer})

    return Dataset.from_list(samples)


def load_filtered_grpo_dataset(path: str, max_samples: int = 2500) -> Dataset:
    """
    Load a pre-filtered dataset from a JSONL file produced by filter_offline.py.

    Each line: {"prompt": "...", "answer": "..."}
    """
    import json
    import os
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            entry = json.loads(line)
            samples.append({"prompt": entry["prompt"], "answer": entry["answer"]})
    print(f"[GRPO] Loaded {len(samples)} pre-filtered samples from {path}")
    return Dataset.from_list(samples)


# ── All-Zero Filter ────────────────────────────────────────────

def filter_non_all_zero(grpo_dataset: Dataset, sft_model, tokenizer,
                        group_size: int = 2, max_completion: int = 1024,
                        batch_size: int = 4) -> Dataset:
    """
    Filter the GRPO dataset to only include problems where the SFT-primed
    model can generate at least one correct trajectory (non-all-zero reward).

    This prevents reward sparsity — a zero reward vector across all G
    rollouts provides zero gradient signal for GRPO.

    NOTE: This is computationally expensive. Run once after SFT training.
    On the 3090 this is feasible; on smaller GPUs use a smaller probe set.

    Defaults: G=2 (2 samples per problem), 1024 tokens per completion.
    For a 1.5B model this balances coverage vs throughput.
    """
    import torch
    from tqdm import tqdm

    sft_model.eval()
    keep_indices = []
    device = next(sft_model.parameters()).device

    samples = [grpo_dataset[i] for i in range(len(grpo_dataset))]

    for batch_start in tqdm(range(0, len(samples), batch_size), desc="All-zero filter"):
        batch = samples[batch_start:batch_start + batch_size]
        prompts = [s["prompt"] for s in batch]
        answers = [s["answer"] for s in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=256).to(device)

        # Generate G completions per prompt
        with torch.no_grad():
            outputs = sft_model.generate(
                **inputs,
                max_new_tokens=max_completion,
                do_sample=(group_size > 1),     # greedy for G=1, sample for G>1
                temperature=0.7 if group_size > 1 else 1.0,
                top_p=0.9,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Score each completion
        completions = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                             skip_special_tokens=True)

        for j in range(len(batch)):
            batch_completions = completions[j * group_size:(j + 1) * group_size]
            any_correct = any(
                _check_answer(comp, answers[j]) for comp in batch_completions
            )
            if any_correct:
                keep_indices.append(batch_start + j)

    return grpo_dataset.select(keep_indices)


def _normalize_str(s: str) -> str:
    """Collapse whitespace and strip for string comparison."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_answer(raw: str) -> str:
    """Normalize an answer string: strip LaTeX delimiters, normalize escapes."""
    s = raw.strip()
    s = re.sub(r"^\s*\\\(\s*", "", s)
    s = re.sub(r"\s*\\\)\s*$", "", s)
    s = re.sub(r"^\s*\\\[\s*", "", s)
    s = re.sub(r"\s*\\\]\s*$", "", s)
    s = re.sub(r"^\s*\$\$\s*", "", s)
    s = re.sub(r"\s*\$\$\s*$", "", s)
    s = re.sub(r"^\s*\$\s*", "", s)
    s = re.sub(r"\s*\$\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _check_answer(completion: str, ground_truth: str) -> bool:
    """Check if a completion's <answer> tag matches the ground truth.

    Handles: pure numeric, LaTeX (via sympy), equations, and multi-value
    answers joined by \"or\".  Sympy is optional; falls back to string
    comparison when not available.
    """
    match = re.search(r"<answer>(.*?)(?:</answer>|$)", completion, re.DOTALL)
    if not match:
        return False

    predicted = _normalize_answer(match.group(1))
    target = _normalize_answer(ground_truth)

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
            if _check_answer(f"<answer>{alt}</answer>", alt):
                if _check_answer(completion, alt):
                    return True
            elif _normalize_str(predicted) == _normalize_str(alt):
                return True

    return False


# ── Tokenization helpers ───────────────────────────────────────

def get_tokenizer(model_name: str):
    """Load the tokenizer for the given model."""
    from transformers import AutoTokenizer
    kwargs = {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"       # required for decoder-only generation
    return tokenizer


def tokenize_sft(dataset: Dataset, tokenizer, max_seq_length: int = 2048) -> Dataset:
    """Tokenize SFT dataset for training."""
    def _tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


def tokenize_grpo(dataset: Dataset, tokenizer, max_prompt_length: int = 256) -> Dataset:
    """Tokenize GRPO prompts (completions are generated, so only prompts tokenized)."""
    def _tokenize(examples):
        tokenized = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=max_prompt_length,
            padding=False,
        )
        # Keep 'answer' alongside tokenized prompt for reward computation
        tokenized["answer"] = examples["answer"]
        return tokenized

    return dataset.map(_tokenize, batched=True)
