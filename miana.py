#!/usr/bin/env python3
import os
import time
import torch
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler

# — Your Telegram Bot Credentials —
BOT_TOKEN = '7009183863:AlTqWFak'
CHAT_ID   = '551'

# — Environment & Performance Flags —
os.environ["OMP_NUM_THREADS"]                  = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"]                  = str(os.cpu_count())
os.environ["SAFETENSORS_FAST_GPU"]             = "1"    # zero-copy safetensors → GPU
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]  = "1"
os.environ["TOKENIZERS_PARALLELISM"]           = "false"

# Enable TF32 and cuDNN autotune for maximal GPU throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
torch.set_default_device("cuda")
torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)  # disable grad for inference

def send_telegram(image, caption: str):
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
        data={"chat_id": CHAT_ID, "caption": caption},
        files={"photo": ("image.png", buf, "image/png")}
    )

def main():
    t0 = time.time()

    # 1) Prewarm CUDA
    _ = torch.randn(1, device="cuda")
    torch.cuda.synchronize()

    # 2) Parallel load tokenizer, text encoder, transformer
    with ThreadPoolExecutor() as ex:
        tok_f = ex.submit(
            PreTrainedTokenizerFast.from_pretrained,
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            num_workers=os.cpu_count(),
            local_files_only=True
        )
        enc_f = ex.submit(
            LlamaForCausalLM.from_pretrained,
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            attn_implementation="eager",  # SDPA flash attention
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        trf_f = ex.submit(
            HiDreamImageTransformer2DModel.from_pretrained,
            "HiDream-ai/HiDream-I1-Fast",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=True
        )

        tokenizer    = tok_f.result()
        text_encoder = enc_f.result().to("cuda")
        transformer  = trf_f.result().to("cuda")

    t_load = time.time()

    # 3) Build pipeline
    scheduler = FlashFlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False
    )
    pipe = HiDreamImagePipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Fast",
        tokenizer_4=tokenizer,
        text_encoder_4=text_encoder,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    ).to("cuda")
    pipe.transformer = transformer
    pipe.enable_vae_tiling()

    # Keep scheduler sigmas on CPU to avoid sync
    pipe.scheduler.sigmas = pipe.scheduler.sigmas.to("cpu")

    t_build = time.time()

    # 4) Optimize memory format for transformer & VAE
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    # 5) Warm-up (1 step) to load kernels & autotune
    _ = pipe(
        "warmup",
        height=1024, width=1024,
        num_inference_steps=1,
        guidance_scale=0.0
    )
    torch.cuda.synchronize()
    t_warm = time.time()

    # 6) Actual inference (16 steps)
    t_inf_start = time.time()
    result = pipe(
        "A cat holding a sign that says 'Hi-Dreams.ai'.",
        height=1024, width=1024,
        num_inference_steps=16,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(0)
    )
    torch.cuda.synchronize()
    t_inf_end = time.time()

    # 7) Timings
    load_time = t_load - t0
    build_time = t_build - t_load
    warm_time = t_warm - t_build
    infer_time = t_inf_end - t_inf_start
    total_time = t_inf_end - t0

    print(f"✅ Load: {load_time:.2f}s | Build: {build_time:.2f}s | Warm: {warm_time:.2f}s | Infer: {infer_time:.2f}s | Total: {total_time:.2f}s")

    # 8) Send result to Telegram
    caption = (
        f"Load: {load_time:.2f}s | Build: {build_time:.2f}s | "
        f"Warm: {warm_time:.2f}s | Infer: {infer_time:.2f}s | Total: {total_time:.2f}s"
    )
    send_telegram(result.images[0], caption)

if __name__ == "__main__":
    main()
