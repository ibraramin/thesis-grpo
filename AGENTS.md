# AGENTS.md — Thesis-GRPO

Dynamic Reward-Gating in GRPO for Small Language Models (Qwen2.5-1.5B, 4-bit LoRA).

## Architecture

- **Single config**: `config.yaml` is the source of truth — all scripts read from it.
- **4-stage pipeline** orchestrated by `run_all.py`:
  1. SFT (`run_sft.py`) → quantized LoRA on OpenR1-Math-220k
  2. Offline dataset filter (`filter_offline.py`) → one-time vLLM pass to find solvable prompts
  3. GRPO (`run_grpo.py`, `grpo_trainer.py`) → cohort × seed training with custom `StabilizedGRPOTrainer`
  4. Evaluation (`evaluate.py`) + Analysis (`analyze.py`) → MATH-500 / AIME24, ANOVA, Tukey HSD
- `StabilizedGRPOTrainer` (in `grpo_trainer.py`) subclasses TRL's `GRPOTrainer` to inject PSPO, entropy filtering, and K3 KL into `_compute_loss`. Do not modify the loss logic without understanding the stabilization modules (`stabilization.py`).

## Commands

Always start with a test-run before any full-scale work:

```bash
python run_all.py --test-run --dry-run    # validates config, data, rewards (no GPU needed)
python run_all.py --test-run              # small-scale end-to-end (needs GPU, ~2 min)
```

### Individual scripts

```bash
# SFT (once; 30-60 min on 24GB GPU)
python run_sft.py
python run_sft.py --test-run

# GRPO (per cohort/seed; ~4-6 hr each)
python run_grpo.py --cohort baseline --seed 0
python run_grpo.py --cohort A --seed 0 --test-run

# Full pipeline with subsets
python run_all.py --skip-sft                          # all cohorts, all seeds
python run_all.py --skip-sft --cohorts baseline B      # specific cohorts
python run_all.py --skip-sft --cohorts A --seeds 0 1   # cohort + seed subset
python run_all.py --skip-sft --steps eval,analyze       # only eval + analysis
```

### Auxiliary scripts (manual investigation)

```bash
python filter_tune.py --samples 100 --group-sizes 2 4    # tune all-zero filter params
python filter_benchmark.py --prompts 500                  # base vs SFT solve-rate comparison
python probe_dataset.py --samples 500                     # dataset solve-rate probe
python data/download_datasets.py --sft-samples 5000       # re-download datasets
```

## Framework version constraints

- **`trl>=1.0.0,<1.1.0`** — `GRPOConfig`, `GRPOTrainer`, and K3 KL are native in 1.0.0. Later versions removed `ConstantLengthDataset` and changed trainer APIs. Do not upgrade without verifying.
- **`vllm>=0.10.2,<0.18.0`** — API surface changes rapidly across minor versions. `LLM.generate()` signature and `SamplingParams` are version-sensitive.
- **`unsloth>=2024.11.0`** — optional; code falls back to `transformers + bitsandbytes + peft` when Unsloth is unavailable.
- **`peft>=0.14,<0.15`** + **`transformers>=4.46`** — needed for checkpoint resume to work (avoids `EmbeddingParallel` import error).

## Resume / idempotency

All trainers use a `.resume_marker` file written at training start and deleted on clean completion:

- **Marker present** → incomplete run, training will resume from latest `checkpoint-N` directory.
- **No marker + `adapter_model.safetensors` exists** → clean completion, training is skipped.
- `find_latest_checkpoint()` (in `run_sft.py:30`, `run_grpo.py:35`) scans for `checkpoint-*` dirs sorted by step number.

When you want to force re-run, delete the output directory or the `.resume_marker` file.

## Data: offline-first

All datasets are pre-downloaded to `data/`:
- `openr1_math_220k_sft.jsonl.gz` — SFT (5K samples, 10K after re-download)
- `openmath_instruct_2_grpo.jsonl` — GRPO (50K samples)
- `math500_eval.jsonl` — MATH-500 eval
- `aime24_eval.jsonl` — AIME24 eval
- `gsm8k_eval.jsonl` — GSM8k eval (optional, diagnostic)
- `svamp_eval.jsonl` — SVAMP eval (optional, diagnostic)
- `openmath_instruct_2_probe.jsonl` — dataset probe

Loaders in `data.py` check local files **before** falling back to HuggingFace streaming. For air-gapped deployments, ensure these files are present — no internet needed.

## Model loading quirks

- **SFT**: loads raw base model (`Qwen/Qwen2.5-1.5B`), applies LoRA, trains, then **merges** to `outputs/sft_checkpoint/merged/`. The merged directory is required for vLLM. `run_sft.py` now checks local HF/ModelScope cache first — no re-downloads if model exists.
- **GRPO**: loads the **merged** SFT checkpoint (not the adapter), then applies a fresh LoRA on top. Uses `load_in_4bit=False` with Unsloth or `BitsAndBytesConfig` otherwise.
- **Evaluation**: merges LoRA adapter into base model via `_merge_ckpt_for_vllm()`, then uses vLLM for batch generation. Strict XML parsing (requires `<think>` + `<answer>` tags).
- **Tokenizer**: `padding_side` must be `"left"` for decoder-only generation — `get_tokenizer()` in `data.py` handles this. Set `pad_token = eos_token` if missing. Now auto-sets `fix_mistral_regex=True` to suppress harmless warning.

## Experimental cohorts (in config.yaml)

| Cohort | λ_correct | λ_format | Graduated | Dynamic (DW-GRPO) |
|--------|-----------|----------|-----------|-------------------|
| baseline | 1.0 | 0.0 | No | No |
| A | 0.5 | 1.5 | No | No |
| B | 1.5 | 0.2 | No | No |
| C | 1.0 | 0.2 | Yes (0.2/0.5/1.0) | No |
| D | 1.0 | 1.0×e^(-γt) | No | Yes (γ=0.005) |

**Empirically validated**: baseline = zero reward (mathematically dead). Cohort D > baseline on all benchmarks. γ=0.01 was too fast (decayed before model learned); γ=0.005 is current optimal. Cohort D requires `gater.step()` called every optimizer step — this is handled by `DynamicGaterCallback` registered in `StabilizedGRPOTrainer.__init__` (`grpo_trainer.py:56`).

## Stabilization layers

Defined in `stabilization.py`, applied inside `StabilizedGRPOTrainer._compute_loss()`:

- **PSPO** (δ=0.1): Smooths policy logprobs toward reference — `pspo_smooth_logprobs()`.
- **Entropy filter** (H > ln(2)): Zeroes advantages for high-entropy rollouts — `entropy_filter_mask()`.
- **K3 KL**: Unbiased low-variance estimator — `r - log(r) - 1`, native in TRL 1.0.0.

These are controlled by `config.yaml → training.stabilization.*`.

## vLLM / GPU Configuration

- **`vllm_memory_utilization: 0.65`** — empirically validated sweet spot for RTX 3090 (24GB). 0.75 caused GPU thrashing (9.9s/step → 2.3s/step). 0.85 OOMs. Do not change.
- **`generation_batch_size = G × 2`** — this is essential for speed. Setting to G (4) creates twice as many vLLM calls, doubling per-step time. Setting to G × 4 (16) causes choppy batching. G×2=8 is optimal.
- **GPU fragmentation**: After multiple Ctrl+C restarts, step_time degrades from ~7s to ~12s. Only fix: reboot instance or rent new GPU. Fresh GPU = fast training.

## Evaluation Protocol

- **Benchmarks**: MATH-500 + AIME24 (per methodology §6). GSM8k and SVAMP are available as optional `--benchmarks gsm8k` overrides for supervisor demos.
- **Scoring**: Strict XML only — requires both `<think>` and `<answer>` tags. Answer extracted from `<answer>` tag, compared with 1e-5 tolerance. No lenient fallback, no few-shot prompting.
- **vLLM batch eval**: Uses `gpu_memory_utilization=0.5` (lower than training to avoid OOM alongside other processes).
- **First eval downloads model** (3GB, ~40 min). After caching, subsequent evals are instant. Use `_resolve_model_path()` which checks: HF cache → ModelScope cache → SFT merged checkpoint.

## Known issues

- **EOS-looping**: Qwen2.5-1.5B generates endless repetitive `</think>`/`</blockquote>` tags. Mitigation: `max_completion_length=512` in config. This caps both GRPO training and evaluation. Raise only after verifying EOS behavior is fixed.
- **Baseline (λ_format=0) = zero reward**: Expected behavior. All completions score 0.0 → σ(R)=0 → advantage=0 → no learning. This is the key finding proving SFT + format guidance are mandatory for 1.5B models.
- **Entropy stalls above ln(2)**: Model entropy hovers at 2.4–3.4 across entire training run. Entropy filter is always active. Stronger SFT (10K×2) helps but doesn't fully resolve. This is a known limitation of 1.5B-scale GRPO.
- **`filter_probe.enabled: false`** in config: the all-zero filter is disabled because it's too strict for the 1.5B model. Use `filter_offline.py` instead.
- **OOM on 24GB**: Reduce `max_completion_length` to 256 or `num_generations` to 2. Typical stable config: G=4, vLLM=0.65, prompt=256, completion=512.
- **`packing=False`** in SFT config — must remain false; the dataset uses pre-formatted chat templates.
- **`load_in_4bit` asymmetry in GRPO**: Unsloth path uses `load_in_4bit=False` (full precision) while the BnB fallback uses `load_in_4bit=True`. Results may differ between environments.
- **Checkpoint resume progress bar**: Resets to 0 after resume — cosmetic. Model/dataset/optimizer are correctly restored.
- **Trainer saves 50-step checkpoints, auto-deletes after training**: `save_total_limit=2` keeps at most 2 during training. After training, checkpoint cleanup deletes all remaining checkpoint dirs (final adapter is saved separately).

## Transferring Between Instances

The only irreplaceable files (not in git or downloadable):

```
outputs/sft_checkpoint/merged/           # ~3GB — trained SFT model
outputs/filtered_grpo/filtered_dataset.jsonl  # ~2MB — 1,287 solvable prompts
```

```bash
# From old instance to Google Cloud / local:
tar czf thesis-grpo.tar.gz outputs/sft_checkpoint/merged outputs/filtered_grpo/filtered_dataset.jsonl *.py config.yaml data/

# On new instance:
tar xzf thesis-grpo.tar.gz -C ~/thesis-grpo/
cd ~/thesis-grpo && python data/download_datasets.py --sft-samples 10000
python run_all.py --skip-sft --cohorts D --seeds 0
```

## References

- **`ARCHITECTURE.md`** — full methodology, validation results, parameter justification, 12 bugs fixed, run commands
- **`chats/SESSION-2026-06-09.md`** — complete session state (pipeline progress, findings, config)
- **`docs/Synthesized Methodology.md`** — theoretical paper (G=16, Big-Math, max_completion=1792 — ideal, not current)
- **`docs/revision.md`** — critique recommending OpenMathInstruct-2, 50K samples, SFT cold start
- **`RUNBOOK.md`** — production deployment guide with troubleshooting

## Style notes

- No type checker, no linter, no formatter configured. Do not add `py.typed`, `mypy.ini`, or lint config unless asked.
- Docstrings follow Google-style but are not enforced. When adding a function, a one-line summary is sufficient.
- `import` order: stdlib → third-party → local. Not enforced, just observed.
- Log output uses `[TAG]` prefixes (e.g., `[SFT]`, `[GRPO]`) — match this pattern for any new scripts.
