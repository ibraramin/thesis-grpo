FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Use Alibaba mirror for China deployments; remove -i flag for non-China builds.
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ \
    "unsloth>=2024.11.0" \
    "trl>=1.0.0,<1.1.0" \
    "vllm>=0.10.2,<0.18.0" \
    "bitsandbytes>=0.44.0" \
    "transformers>=4.44.0" \
    "datasets>=2.20.0" \
    "accelerate>=0.34.0" \
    "peft>=0.12.0" \
    pyyaml>=6.0 \
    scipy>=1.10.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    tqdm>=4.66.0 \
    numpy>=1.26.0

WORKDIR /workspace
COPY . .

ENTRYPOINT ["python", "run_all.py"]
