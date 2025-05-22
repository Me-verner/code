#!/usr/bin/env python3
import time
import torch
import requests
import argparse
from io import BytesIO
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler

# — Telegram Bot Credentials —
BOT_TOKEN = '70091'
CHAT_ID   = ''

# Model configurations
MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

def send_telegram(image, caption: str):
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
        data={"chat_id": CHAT_ID, "caption": caption},
        files={"photo": ("image.png", buf, "image/png")}
    )

def load_models(model_type: str):
    cfg = MODEL_CONFIGS[model_type]
    path = cfg["path"]
    scheduler = cfg["scheduler"](num_train_timesteps=1000, shift=cfg["shift"], use_dynamic_shifting=False)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL, use_fast=False)
    text_encoder = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    pipe = HiDreamImagePipeline.from_pretrained(
        path,
        scheduler=scheduler,
        tokenizer_4=tokenizer,
        text_encoder_4=text_encoder,
        torch_dtype=torch.bfloat16
    ).to("cuda", torch.bfloat16)
    pipe.transformer = transformer

    return pipe, cfg

def parse_resolution(res: str):
    if "1024 × 1024" in res: return 1024, 1024
    if "768 × 1360"  in res: return 768, 1360
    if "1360 × 768"  in res: return 1360, 768
    if "880 × 1168"  in res: return 880, 1168
    if "1168 × 880"  in res: return 1168, 880
    if "1248 × 832"  in res: return 1248, 832
    if "832 × 1248"  in res: return 832, 1248
    return 1024, 1024

def generate_image(pipe, cfg, prompt: str, resolution: str, seed: int):
    h, w = parse_resolution(resolution)
    if seed == -1:
        seed = torch.randint(0, 1_000_000, (1,)).item()
    gen = torch.Generator("cuda").manual_seed(seed)
    out = pipe(
        prompt,
        height=h, width=w,
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        num_images_per_prompt=1,
        generator=gen
    ).images[0]
    return out, seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="fast", choices=["dev","full","fast"])
    args = parser.parse_args()

    total_start = time.time()
    print(f"Loading '{args.model_type}' model...")
    t0 = time.time()
    pipe, cfg = load_models(args.model_type)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    prompt    = "A cat holding a sign that says \"Hi-Dreams.ai\"."
    resolution= "1024 × 1024 (Square)"
    seed      = -1

    print("Generating image...")
    t1 = time.time()
    image, seed = generate_image(pipe, cfg, prompt, resolution, seed)
    infer_time = time.time() - t1
    print(f"Inference time: {infer_time:.2f}s")

    total_time = time.time() - total_start
    caption = (
        f"Model load: {load_time:.2f}s\n"
        f"Inference: {infer_time:.2f}s\n"
        f"Total elapsed: {total_time:.2f}s\n"
        f"Seed: {seed}"
    )
    send_telegram(image, caption)
    print(f"Done. Total elapsed: {total_time:.2f}s")
