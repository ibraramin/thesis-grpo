"""
data.py — Dataset loading, preprocessing, and tokenization pipeline.

Supports three dataset stages as defined in the methodology:
  1. Acquisition Set (SFT): open-r1/OpenR1-Math-220k
  2. Consolidation Set (GRPO): SynthLabsAI/Big-Math-RL-Verified
  3. All-zero filtering: excludes samples the SFT model cannot solve
"""

import re
from datasets import load_dataset, Dataset

# ── Chat template for Qwen2.5 ──────────────────────────────────
# Forces assistant responses inside <think></think> tags
QWEN_CHAT_TEMPLATE = """\
<|im_start|>system
You are a helpful mathematical reasoning assistant. Always structure your answer by placing your step-by-step reasoning inside <think></think> tags, and your final answer inside <answer></answer> tags.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
{completion}<|im_end|>"""


def format_sft_chat(problem: str, completion: str) -> str:
    """Format a problem+completion pair using the Qwen2.5 chat template."""
    content = QWEN_CHAT_TEMPLATE.format(problem=problem, completion=completion)
    return content


def format_grpo_prompt(problem: str) -> str:
    """Format a problem as a GRPO prompt (no completion)."""
    return QWEN_CHAT_TEMPLATE.format(problem=problem, completion="")


# ── SFT Dataset ────────────────────────────────────────────────

def load_sft_dataset(config: dict, max_samples: int = 5000) -> Dataset:
    """
    Load and preprocess the OpenR1-Math-220k dataset for SFT.

    Uses the 'default' config which contains full traces with <think> tags.
    Returns a Dataset with a 'text' column suitable for SFTTrainer.
    """
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

        # Use the first generation that passed math_verify as the target
        generation = None
        if row.get("generations") and row.get("correctness_math_verify"):
            for gen, correct in zip(row["generations"], row["correctness_math_verify"]):
                if correct:
                    generation = gen
                    break
        # Fallback: use first generation regardless
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
    Load and preprocess the Big-Math-RL-Verified dataset for GRPO.

    Returns a Dataset with 'prompt' and 'answer' columns.
    """
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
        answer = row.get("answer", "")
        if not problem or not answer:
            continue

        prompt = format_grpo_prompt(problem)
        samples.append({"prompt": prompt, "answer": answer})

    return Dataset.from_list(samples)


# ── All-Zero Filter ────────────────────────────────────────────

def filter_non_all_zero(grpo_dataset: Dataset, sft_model, tokenizer,
                        group_size: int = 16, max_completion: int = 1792,
                        batch_size: int = 4) -> Dataset:
    """
    Filter the GRPO dataset to only include problems where the SFT-primed
    model can generate at least one correct trajectory (non-all-zero reward).

    This prevents reward sparsity — a zero reward vector across all G
    rollouts provides zero gradient signal for GRPO.

    NOTE: This is computationally expensive. Run once after SFT training.
    On the 3090 this is feasible; on smaller GPUs use a smaller probe set.
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
                do_sample=True,
                temperature=0.7,
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


def _check_answer(completion: str, ground_truth: str) -> bool:
    """Check if a completion's <answer> tag matches the ground truth."""
    match = re.search(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>", completion)
    if not match:
        return False
    try:
        predicted = float(match.group(1))
        target = float(ground_truth)
        return abs(predicted - target) < 1e-5
    except (ValueError, TypeError):
        return False


# ── Tokenization helpers ───────────────────────────────────────

def get_tokenizer(model_name: str):
    """Load the tokenizer for the given model."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
