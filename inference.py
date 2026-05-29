import scipy
import numpy as np
import io
import base64
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, pipeline as hf_pipeline

import torch
if not hasattr(torch, 'xpu'):
    class MockXPU:
        def empty_cache(self): pass
    torch.xpu = MockXPU()

from diffusers import StableDiffusion3Pipeline

import os
HF_TOKEN = os.environ.get("HF_TOKEN")
CAPTION_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"
SD_MODEL      = "stabilityai/stable-diffusion-3.5-large-turbo"
MUSIC_MODEL   = "facebook/musicgen-large"

LITERATURE_PROMPT = (
    "Generate a poem, with 10 lines, "
    "depicting the artwork but also focusing on conveying its emotions."
)
MUSIC_CAPTION_PROMPT = (
    "Generate a prompt for a music generation model to express the emotions of the artwork. "
    "Output only the prompt text, nothing else."
)
IMAGE_CAPTION_PROMPT = (
    "Generate a detailed image generation prompt that describes the visual style, composition, "
    "colors, mood, and subject matter of this artwork so an AI image model can recreate its essence. "
    "Output only the prompt text, nothing else."
)


def load_models():
    print("Loading JoyCaption...")
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL, use_fast=False)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        CAPTION_MODEL, torch_dtype=torch.bfloat16, device_map=0
    )
    llava_model.eval()

    print("Loading Stable Diffusion 3.5...")
    sd_pipe = StableDiffusion3Pipeline.from_pretrained(
        SD_MODEL, torch_dtype=torch.bfloat16, token=HF_TOKEN
    )
    sd_pipe = sd_pipe.to("cuda")

    print("Loading MusicGen...")
    synthesiser = hf_pipeline("text-to-audio", MUSIC_MODEL)

    print("All models loaded.")
    return {
        "processor": processor,
        "llava": llava_model,
        "sd": sd_pipe,
        "music": synthesiser,
    }


def run_joycaption(processor, model, image, prompt):
    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user",   "content": prompt},
    ]
    convo_string = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs, max_new_tokens=1024, do_sample=True,
            suppress_tokens=None, use_cache=True,
            temperature=0.6, top_k=None, top_p=0.9,
        )[0]

    generate_ids = generate_ids[inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()


def run_inference(models, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # JoyCaption — generate all 3 prompts
    literature   = run_joycaption(models["processor"], models["llava"], image, LITERATURE_PROMPT)
    music_prompt = run_joycaption(models["processor"], models["llava"], image, MUSIC_CAPTION_PROMPT)
    image_prompt = run_joycaption(models["processor"], models["llava"], image, IMAGE_CAPTION_PROMPT)

    # Stable Diffusion
    result_image = models["sd"](
        image_prompt, num_inference_steps=4, guidance_scale=0.0
    ).images[0]
    img_buf = io.BytesIO()
    result_image.save(img_buf, format="PNG")
    image_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

    # MusicGen
    music = models["music"](music_prompt, forward_params={"do_sample": True})
    wav_buf = io.BytesIO()
    scipy.io.wavfile.write(wav_buf, rate=music["sampling_rate"], data=music["audio"])
    audio_b64 = base64.b64encode(wav_buf.getvalue()).decode("utf-8")

    return {
        "literature": literature,
        "image_b64": image_b64,       # PNG, base64
        "audio_b64": audio_b64,       # WAV, base64
        "music_prompt": music_prompt,
        "image_prompt": image_prompt,
    }