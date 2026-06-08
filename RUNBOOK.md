# GRPO Dynamic Reward-Gating — Production Runbook

Qwen2.5-1.5B · GRPO · RTX 3090 / A5000 (24 GB) · 5 experimental cohorts

## 0. Prerequisites

```bash
# System
apt install -y git python3-pip

# Python deps (CUDA 12.x — works for 12.4–12.8)
# China: add -i https://mirrors.aliyun.com/pypi/simple/
pip install torch transformers accelerate peft bitsandbytes datasets trl>=1.0.0 \
    vllm>=0.10.2,<0.18.0 scipy matplotlib seaborn sympy antlr4-python3-runtime
```

## 1. Get the code

```bash
git clone https://github.com/ibraramin/thesis-grpo.git
cd thesis-grpo
```

If GitHub blocked (China GFW): SCP the repo tarball instead.

## 2. Download the model (air-gapped)

```bash
# Option A — ModelScope (fast in China)
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen2.5-1.5B', cache_dir='/root/.cache/modelscope/hub')
"

# Option B — HuggingFace mirror (set before any Python import)
export HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-1.5B')"
```

## 3. Datasets included in repo

All production datasets are pre-downloaded to `data/`:

| File | Contents | Samples |
|------|----------|---------|
| `openr1_math_220k_sft.jsonl.gz` | SFT training | 5,000 |
| `openmath_instruct_2_grpo.jsonl` | GRPO training | 50,000 |
| `math500_eval.jsonl` | MATH-500 evaluation | 500 |
| `aime24_eval.jsonl` | AIME 2024 evaluation | 30 |
| `openmath_instruct_2_probe.jsonl` | Dataset probe | 1,000 |

The code auto-detects these files and uses them (no internet needed).
If you need to re-download or change sizes, run:

```bash
python data/download_datasets.py --sft-samples 5000 --grpo-samples 50000
```

## 4. Quick validation (test-run)

```bash
python run_all.py --test-run --dry-run
# → outputs/test_run/report.txt — validates all stages, reward functions, formats
```

## 5. Full production pipeline

### 5a. Run SFT (once)

```bash
python run_sft.py
# → outputs/sft_checkpoint/  (~30–60 min on 3090/A5000)
```

### 5b. Run GRPO (5 cohorts × 4 seeds)

```bash
# All 20 runs
python run_all.py --skip-sft

# Or pick a subset
python run_all.py --skip-sft --cohorts baseline B --seeds 0 1
```

**Runtime per seed per cohort**: ~4–6 hours (G=4, max_completion=512, 500 samples, 2 epochs).

### 5c. Evaluate + analyze

```bash
python run_all.py --skip-sft --steps eval,analyze
# → outputs/results.csv  +  outputs/analysis/
```

## 6. Individual script reference

| Script | Key flags |
|--------|-----------|
| `run_sft.py` | `--config` `--output` `--test-run` |
| `run_grpo.py` | `--cohort A` `--seed 42` `--sft-checkpoint` `--test-run` |
| `run_all.py` | `--skip-sft` `--cohorts A B` `--seeds 0 1` `--steps sft,grpo` `--test-run` |
| `evaluate.py` | `--max-samples 100` `--benchmarks math500` `--test-run` |
| `analyze.py` | `--results outputs/results.csv` `--test-run` |

## 7. Experimental cohorts

| Cohort | λ_correct | λ_format | Graduated | Dynamic (DW-GRPO) |
|--------|-----------|----------|-----------|-------------------|
| baseline | 1.0 | 0.0 | No | No |
| A | 0.5 | 1.5 | No | No |
| B | 1.5 | 0.2 | No | No |
| C | 1.0 | 0.2 | Yes (0.2/0.5/1.0) | No |
| D | 1.0 | 1.0×e^(-γt) | No | Yes (γ=0.01) |

## 8. Stabilization mechanisms (all in config.yaml)

- **PSPO** (δ=0.1): smooths policy toward reference to reduce variance
- **Entropy filter** (H < ln(2)): masks advantages for high-entropy rollouts
- **K3 KL**: unbiased low-variance KL estimator (built into TRL 1.0.0)

## 9. Output structure

```
outputs/
├── sft_checkpoint/          # SFT LoRA + merged model
├── baseline/seed_0/         # GRPO LoRA per cohort/seed
├── A/seed_0/
├── ...
├── results.csv              # Evaluation metrics
├── analysis/                # ANOVA, Tukey HSD, plots
├── pipeline_log.csv         # Run history
└── test_run/                # Test-run artifacts
```

## 10. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `bitsandbytes libnvJitLink.so` error | Install matching CUDA lib: `pip install nvidia-cuda-nvjitlink-cu12` |
| Model download 500KB/s (China) | Use ModelScope or `HF_ENDPOINT=https://hf-mirror.com` |
| `ConstantLengthDataset` import error | TRL 1.0.0 removed this — code falls back to peft LoRA, no unsloth needed |
| Gated dataset 401 | Set `HF_TOKEN` env var and accept terms on huggingface.co |
| OOM on 24GB | Reduce `max_completion_length` to 256 or `num_generations` to 2 |
| EOS-looping (endless generation) | Lower `max_completion_length` to 512; the 1.5B model over-generates |
