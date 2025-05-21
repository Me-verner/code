#!/usr/bin/env python3
import os
import time
import torch
import requests
from io import BytesIO
from transformers import AutoTokenizer, LlamaForCausalLM
from diffusers import HiDreamImagePipeline, FlowMatchEulerDiscreteScheduler
from pruna_pro import smash, SmashConfig

# — Telegram Bot Credentials —
BOT_TOKEN = '***********'
CHAT_ID   = '*******************'

# — Pruna Pro API token —
os.environ["PRUNA_TOKEN"] = '**************'

def send_telegram(img, caption: str):
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
        data={"chat_id": CHAT_ID, "caption": caption},
        files={"photo": ("image.png", buf, "image/png")}
    )

def main():
    total_start = time.time()

    # 1) Load tokenizer & LLaMA text encoder
    t0 = time.time()
    tokenizer_4 = AutoTokenizer.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        use_fast=True,
        local_files_only=True
    )
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    ).to("cuda", non_blocking=True)
    load_lm_time = time.time() - t0

    # 2) Scheduler & pipeline
    t1 = time.time()
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=6.0,
        use_dynamic_shifting=False
    )
    pipe = HiDreamImagePipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Dev",
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    ).to("cuda", torch.bfloat16)
    load_pipe_time = time.time() - t1

    # 3) Smash with Pruna Pro (experimental override)
    smash_cfg = SmashConfig()
    smash_cfg["cacher"]            = "auto"
    smash_cfg["auto_cache_mode"]   = "taylor"
    smash_cfg["auto_speed_factor"] = 0.4
    smash_cfg["auto_objective"]    = "fidelity"
    t2 = time.time()
    pipe = smash(
        pipe,
        smash_cfg,
        token=os.environ["PRUNA_TOKEN"],
        experimental=True
    )
    smash_time = time.time() - t2

    # 4) Inference
    prompt = 'A cat holding a sign that says "Hi-Dreams.ai".'
    gen = torch.Generator("cuda").manual_seed(0)
    t3 = time.time()
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=5.0,
        num_inference_steps=28,
        generator=gen
    ).images[0]
    infer_time = time.time() - t3

    total_time = time.time() - total_start

    # 5) Send to Telegram
    caption = (
        f"Load LM: {load_lm_time:.2f}s\n"
        f"Load pipeline: {load_pipe_time:.2f}s\n"
        f"Pruna smash: {smash_time:.2f}s\n"
        f"Infer: {infer_time:.2f}s\n"
        f"Total: {total_time:.2f}s"
    )
    send_telegram(image, caption)
    print("✅ Done.")

if __name__ == "__main__":
    main()
