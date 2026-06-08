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

## Resume / idempotency

All trainers use a `.resume_marker` file written at training start and deleted on clean completion:

- **Marker present** → incomplete run, training will resume from latest `checkpoint-N` directory.
- **No marker + `adapter_model.safetensors` exists** → clean completion, training is skipped.
- `find_latest_checkpoint()` (in `run_sft.py:30`, `run_grpo.py:35`) scans for `checkpoint-*` dirs sorted by step number.

When you want to force re-run, delete the output directory or the `.resume_marker` file.

## Data: offline-first

All datasets are pre-downloaded to `data/`:
- `openr1_math_220k_sft.jsonl.gz` — SFT (5K samples)
- `openmath_instruct_2_grpo.jsonl` — GRPO (50K samples)
- `math500_eval.jsonl` — MATH-500 eval
- `aime24_eval.jsonl` — AIME24 eval
- `openmath_instruct_2_probe.jsonl` — dataset probe

Loaders in `data.py` check local files **before** falling back to HuggingFace streaming. For air-gapped deployments, ensure these files are present — no internet needed.

## Model loading quirks

- **SFT**: loads raw base model (`Qwen/Qwen2.5-1.5B`), applies LoRA, trains, then **merges** to `outputs/sft_checkpoint/merged/`. The merged directory is required for vLLM.
- **GRPO**: loads the **merged** SFT checkpoint (not the adapter), then applies a fresh LoRA on top. Uses `load_in_4bit=False` with Unsloth or `BitsAndBytesConfig` otherwise.
- **Evaluation**: loads base model + loads LoRA adapter via `PeftModel.from_pretrained()` (no merge). Uses `model_cfg["name"]` not the SFT path for the base model.
- **Tokenizer**: `padding_side` must be `"left"` for decoder-only generation — `get_tokenizer()` in `data.py:337` handles this. Set `pad_token = eos_token` if missing.

## Experimental cohorts (in config.yaml)

| Cohort | λ_correct | λ_format | Graduated | Dynamic (DW-GRPO) |
|--------|-----------|----------|-----------|-------------------|
| baseline | 1.0 | 0.0 | No | No |
| A | 0.5 | 1.5 | No | No |
| B | 1.5 | 0.2 | No | No |
| C | 1.0 | 0.2 | Yes (0.2/0.5/1.0) | No |
| D | 1.0 | 1.0×e^(-γt) | No | Yes (γ=0.01) |

Cohort D requires `gater.step()` called every optimizer step — this is handled by `DynamicGaterCallback` registered in `StabilizedGRPOTrainer.__init__` (`grpo_trainer.py:56`). If you fork training logic, ensure the gater still advances.

## Stabilization layers

Defined in `stabilization.py`, applied inside `StabilizedGRPOTrainer._compute_loss()`:

- **PSPO** (δ=0.1): Smooths policy logprobs toward reference — `pspo_smooth_logprobs()`.
- **Entropy filter** (H > ln(2)): Zeroes advantages for high-entropy rollouts — `entropy_filter_mask()`.
- **K3 KL**: Unbiased low-variance estimator — `r - log(r) - 1`, native in TRL 1.0.0. No custom implementation needed.

These are controlled by `config.yaml → training.stabilization.*`.

## Known issues

- **EOS-looping**: Qwen2.5-1.5B over-generates endless repetitions (confirmed in `debug_g1_completions.json` / `debug_g2_completions.json` — hundreds of `</think>`/`</blockquote>` tags). Mitigation: `max_completion_length` capped at 512 in config. Raise back to 1792 only after verifying EOS behavior.
- **`filter_probe.enabled: false`** in config: the all-zero filter is disabled because it's too strict for the 1.5B model. If you enable it, expect <2% retention and 0-sample datasets.
- **OOM on 24GB**: Reduce `max_completion_length` to 256 or `num_generations` to 2.
- **vLLM in test-run**: `use_vllm=False` is hardcoded in `run_grpo.py:370` for test-runs (too heavy). Full runs use vLLM.
- **`generation_batch_size`** must equal `num_generations` in `GRPOConfig` (TRL 1.0.0 requirement).
- **`packing=False`** in SFT config — must remain false; the dataset uses pre-formatted chat templates.
- **`load_in_4bit` asymmetry in GRPO**: Unsloth path uses `load_in_4bit=False` (full precision from merged checkpoint) while the BnB fallback uses `load_in_4bit=True`. This means GRPO runs with different quantization depending on whether Unsloth is available — results may differ between environments.

## Test-run mode

`--test-run` reduces all scale parameters (samples, epochs, seeds) to values in `config.yaml → test_run`. When `run_all.py` receives `--test-run`, it writes a temporary config to `outputs/test_run/_config.yaml` with overrides and passes it to subprocesses. The test-run report is written incrementally to `outputs/test_run/report.txt`.

Use `--test-run --dry-run` first (no GPU, ~5 seconds) to validate config/imports/rewards before using real GPU time.

## Misc directories

- **`docs/`** — `Synthesized Methodology.md` (theoretical paper, G=16, Big-Math dataset, max_completion=1792) and `revision.md` (critique recommending OpenMathInstruct-2, 50K samples, SFT cold start). These describe the *ideal* experiment. The actual config has been scaled down (G=4, max_completion=512) to work around EOS-looping on Qwen2.5-1.5B.
- **`diag/`** — diagnostic artifacts from a previous test-run (config, log, report). Safe to clean up.
- **`filter_tune/`** — cached filter tuning results (CSV, JSON). Generated by `filter_tune.py`.
- **`stuff/`** — miscellaneous archive. Not part of the pipeline.

## Tool interop

- **`filter_tune.py`** imports `format_grpo_prompt` and `_check_answer` from `data.py` — any changes to answer checking logic in `data.py` propagate here.
- **`probe_dataset.py`** imports `format_grpo_prompt` and `_check_answer` from `data.py` — same.
- **`filter_benchmark.py`** and **`filter_offline.py`** import from `data.py` — consistent.
- **`evaluate.py`** uses its own `_extract_answer` import from `rewards.py` and `format_grpo_prompt` from `data.py`. Local JSONL files are normalized to match HF column names on load.

## Style notes

- No type checker, no linter, no formatter configured. Do not add `py.typed`, `mypy.ini`, or lint config unless asked.
- Docstrings follow Google-style but are not enforced. When adding a function, a one-line summary is sufficient.
- `import` order: stdlib → third-party → local. Not enforced, just observed.
- Log output uses `[TAG]` prefixes (e.g., `[SFT]`, `[GRPO]`) — match this pattern for any new scripts.
