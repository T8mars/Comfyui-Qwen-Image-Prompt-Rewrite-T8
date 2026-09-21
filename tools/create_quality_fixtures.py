"""Generate controlled local Qwen Image 2.1 references for the quality set."""

import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8189"
COMFY_OUTPUT = Path("E:/comfyui-t8-onekey-5x/ComfyUI/output")
DEST = ROOT / "evaluation/fixtures"
DEST.mkdir(parents=True, exist_ok=True)

CASES = [
    ("person_a", "A studio portrait photograph of one adult woman with short dark curly hair, warm brown skin, round silver glasses, and a vivid red wool coat. She faces the camera with a neutral expression. Plain gray backdrop. No text."),
    ("person_b", "A studio portrait photograph of one adult man with close-cropped black hair, a neat short beard, and a bright blue denim jacket. He faces the camera with a relaxed expression. Plain gray backdrop. No text."),
    ("rainy_street", "An empty cobblestone street at dusk after rain, warm amber shop windows, reflections on the pavement, no people, wide photographic composition, no readable signs."),
    ("blue_vase", "A single tall cobalt blue ceramic vase with a narrow neck and two small handles, on a pale wooden table against a plain cream wall, product photograph, no text."),
]


def request(path, data=None):
    payload = None if data is None else json.dumps(data).encode("utf-8")
    req = Request(BASE + path, data=payload,
                  headers={"Content-Type": "application/json"} if payload else {})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def graph(prompt, prefix, seed):
    return {
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["3", 0], "prompt": prompt,
            "negative_prompt": "", "resolution": 512}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "seed": seed,
            "steps": 20, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["4", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


for index, (name, prompt) in enumerate(CASES):
    target = DEST / (name + ".png")
    if target.exists():
        print("existing", target, flush=True)
        continue
    queued = request("/prompt", {"prompt": graph(prompt, "QwenPE-QualityRef-" + name, 171 + index),
                                 "client_id": "qwen-pe-quality-fixtures"})
    if queued.get("node_errors"):
        raise RuntimeError(json.dumps(queued["node_errors"], ensure_ascii=False))
    prompt_id = queued["prompt_id"]
    print("queued", name, prompt_id, flush=True)
    started = time.monotonic()
    while time.monotonic() - started < 900:
        history = request("/history/" + prompt_id).get(prompt_id)
        if history:
            if not history.get("status", {}).get("completed"):
                raise RuntimeError(json.dumps(history.get("status"), ensure_ascii=False))
            info = history["outputs"]["9"]["images"][0]
            source = COMFY_OUTPUT / info["subfolder"] / info["filename"]
            target.write_bytes(source.read_bytes())
            print("saved", target, flush=True)
            break
        time.sleep(3)
    else:
        raise TimeoutError(name)
