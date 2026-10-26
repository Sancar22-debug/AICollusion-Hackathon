"""Modal OpenAI-compatible LLM Server + Cloud Smoke-Test Runner (spec section 9.2).

Supports two execution tiers on the `sancar22-debug` Modal workspace:
  1. GPU Tier (`MODAL_GPU=H100`, default once a payment method / Apart credit is active):
     Single warm H100 vLLM server (`Qwen/Qwen3-32B-FP8`) with PagedAttention continuous
     batching (`@modal.concurrent(max_inputs=128)`) and persistent Modal Volume caching.
  2. No-Card Cloud Cluster Tier (`MODAL_GPU=""`, active immediately on free accounts
     before a payment card is added):
     Autoscaling cluster of 16-vCPU Modal containers (`cpu=16.0`, up to 10
     containers) serving `qwen2.5:7b-instruct` (Ollama, Q4_K_M) under its real name via an
     OpenAI-compatible `/v1/chat/completions` endpoint on port 8000 with weights
     cached on `hf-cache` Modal Volume.
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.request
import modal

MINUTES = 60
VLLM_PORT = 8000

OLLAMA_MODEL = os.environ.get("MODAL_OLLAMA_MODEL", "qwen2.5:7b-instruct")

# Set MODAL_GPU=H100 after adding a payment method at https://modal.com/settings/sancar22-debug/billing
GPU_SPEC = os.environ.get("MODAL_GPU", "").strip() or None

# The served model name is always the real model: Qwen3-32B-FP8 on the GPU tier,
# qwen2.5:7b-instruct (Ollama, Q4_K_M) on the CPU tier. No aliasing.
MODEL = os.environ.get("MODAL_VLLM_MODEL", "Qwen/Qwen3-32B-FP8") if GPU_SPEC else OLLAMA_MODEL

app = modal.App("collusion-vllm")

# Pre-built H100 vLLM image (cached on sancar22-debug as im-wJAnf6ulRTEyv0V83Nblg7)
vllm_gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "vllm==0.8.5.post1",
        "huggingface_hub[hf_transfer]==0.30.2",
        "flashinfer-python==0.2.5",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.6",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_USE_V1": "0",
    })
)

# Pre-built AVX-512 Ollama OpenAI-compatible server image for no-card Modal accounts
modal_cpu_image = (
    modal.Image.from_registry("ollama/ollama:latest", add_python="3.12")
    .entrypoint([])
    .env({
        "OLLAMA_HOST": f"0.0.0.0:{VLLM_PORT}",
        "OLLAMA_NUM_PARALLEL": "4",
    })
    .run_commands(
        f"ollama serve & sleep 3 && ollama pull {OLLAMA_MODEL}"
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

_active_image = vllm_gpu_image if GPU_SPEC else modal_cpu_image
_fn_kwargs = dict(
    image=_active_image,
    timeout=60 * MINUTES,
    scaledown_window=15 * MINUTES,
    max_containers=2 if GPU_SPEC else 10,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[modal.Secret.from_name("VLLM_API_KEY")],
)
if GPU_SPEC:
    _fn_kwargs["gpu"] = GPU_SPEC
else:
    _fn_kwargs["cpu"] = 16.0
    _fn_kwargs["memory"] = 16384


@app.function(**_fn_kwargs)
@modal.concurrent(max_inputs=128 if GPU_SPEC else 4)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve():
    """Spawns the OpenAI-compatible inference server on port 8000."""
    if GPU_SPEC:
        cmd = [
            "vllm", "serve", MODEL,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.92",
            "--enable-prefix-caching",
            "--enforce-eager",
        ]
        print("Launching H100 vLLM server:", " ".join(cmd), flush=True)
        subprocess.Popen(cmd)
    else:
        subprocess.Popen(["ollama", "serve"])
        print(f"Modal OpenAI server ready on port {VLLM_PORT} for {MODEL} ({OLLAMA_MODEL})", flush=True)


@app.local_entrypoint()
def main(
    prompt: str = 'Respond ONLY with JSON: {"private_plan": "All clear, follow protocol.", "priority": "CENTRAL"}',
    url: str = "",
):
    """Smoke test: sends a real chat completion request to the deployed Modal server."""
    import json

    target_url = url or os.environ.get("MODAL_VLLM_URL", "")
    if not target_url:
        target_url = serve.get_web_url()

    base = target_url.rstrip("/")
    print(f"Testing live Modal endpoint at: {base}")
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        print("Live Modal LLM response:", body["choices"][0]["message"]["content"])
