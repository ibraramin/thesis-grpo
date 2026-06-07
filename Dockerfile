FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    unsloth==2024.11.1 \
    "trl>=0.12.0" \
    "vllm>=0.6.0" \
    "bitsandbytes>=0.44.0" \
    "datasets>=2.20.0" \
    "accelerate>=0.34.0" \
    "peft>=0.12.0" \
    pyyaml>=6.0 \
    scipy>=1.10.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    tqdm>=4.66.0 \
    wandb

WORKDIR /workspace
COPY . .

ENTRYPOINT ["python", "run_all.py"]
