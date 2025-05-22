#!/usr/bin/env python3
import os
import time
import torch
import requests
from io import BytesIO
from transformers import AutoTokenizer, LlamaForCausalLM
from diffusers import HiDreamImagePipeline, FlowMatchEulerDiscreteScheduler
from pruna_pro import smash, SmashConfig

# — Credentials —
os.environ["PRUNA_TOKEN"] = "****************"
BOT_TOKEN = "****************"
CHAT_ID   = "************"

def send_telegram(img, caption: str):
    """
    Sends the PIL image both as a Telegram photo (preview) and
    as a document (file) so you can download the exact PNG.
    """
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    # send as photo
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
        data={"chat_id": CHAT_ID, "caption": caption},
        files={"photo": ("image.png", buf, "image/png")}
    )
    buf.seek(0)
    # send as document
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
        data={"chat_id": CHAT_ID, "caption": caption},
        files={"document": ("image.png", buf, "image/png")}
    )

def main():
    # 1) Load base pipeline
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        use_fast=True,
        local_files_only=True
    )
    text_encoder = LlamaForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    ).to("cuda", non_blocking=True)

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=6.0,
        use_dynamic_shifting=False
    )
    pipe = HiDreamImagePipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Dev",
        tokenizer_4=tokenizer,
        text_encoder_4=text_encoder,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    ).to("cuda", torch.bfloat16)

    prompts = [
        'A cat holding a sign that says "Hi-Dreams.ai".',
        'A cute, smiling, purple, knitted prune.',
        'An astronaut hatching from an egg on the moon'
    ]

    # 2) Run base model
    base_times = []
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=0.0,
            num_inference_steps=28,
            generator=torch.Generator("cuda").manual_seed(0),
        ).images[0]
        dt = time.time() - t0
        base_times.append(dt)
        send_telegram(image, f"[Base {i}] {dt:.2f}s")

    avg_base = sum(base_times) / len(base_times)
    send_telegram(image, f"Average base time: {avg_base:.2f}s")

    # 3) Smash with Pruna Pro (experimental override)
    smash_cfg = SmashConfig()
    smash_cfg["cacher"]            = "auto"
    smash_cfg["auto_cache_mode"]   = "taylor"
    smash_cfg["auto_speed_factor"] = 0.4
    smash_cfg["auto_objective"]    = "fidelity"

    t0 = time.time()
    smashed_pipe = smash(
        pipe,
        smash_cfg,
        token=os.environ["PRUNA_TOKEN"],
        experimental=True
    )
    smash_time = time.time() - t0
    send_telegram(image, f"Pruna smash: {smash_time:.2f}s")

    # 4) Run smashed model
    smash_times = []
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        image = smashed_pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=0.0,
            num_inference_steps=28,
            generator=torch.Generator("cuda").manual_seed(0),
        ).images[0]
        dt = time.time() - t0
        smash_times.append(dt)
        send_telegram(image, f"[Smashed {i}] {dt:.2f}s")

    avg_smashed = sum(smash_times) / len(smash_times)
    send_telegram(image, f"Average smashed time: {avg_smashed:.2f}s")

if __name__ == "__main__":
    main()
