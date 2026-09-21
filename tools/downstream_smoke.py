"""Local Qwen Image 2.1 end-to-end smoke run against the newer ComfyUI server."""

import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8189"
REPORT = ROOT / "runtime" / "downstream-t2i-result.json"


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    with urlopen(Request(BASE + path, data=data, headers=headers), timeout=30) as response:
        return json.load(response)


graph = {
    "1": {"class_type": "QwenPERewriteT8", "inputs": {
        "user_prompt": "A single red apple on a dark wooden table, soft side light.",
        "task": "auto", "aspect_ratio": "1:1", "output_language": "English",
        "transparent_rgba": False,
        "t2i_model": "Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf",
        "edit_model": "Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf",
        "vision_model": "Auto", "model_lifetime": "after_run", "seed": 42,
    }},
    "2": {"class_type": "UNETLoader", "inputs": {
        "unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
    "3": {"class_type": "CLIPLoader", "inputs": {
        "clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
    "4": {"class_type": "VAELoader", "inputs": {
        "vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
    "5": {"class_type": "TextEncodeQwenImage21", "inputs": {
        "clip": ["3", 0], "prompt": ["1", 0], "negative_prompt": "",
        "resolution": 512}},
    "6": {"class_type": "QwenPECanvasT8", "inputs": {
        "pe_result": ["1", 1], "resolution": 512, "follow_input_size": False}},
    "7": {"class_type": "KSampler", "inputs": {
        "model": ["2", 0], "seed": 42, "steps": 8, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple",
        "positive": ["5", 0], "negative": ["5", 1],
        "latent_image": ["6", 2], "denoise": 1.0}},
    "8": {"class_type": "VAEDecode", "inputs": {
        "samples": ["7", 0], "vae": ["4", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {
        "images": ["8", 0], "filename_prefix": "QwenPE21-T2I-Smoke"}},
}


def main():
    queued = request("/prompt", {"prompt": graph, "client_id": "qwen-pe-downstream-smoke"})
    if queued.get("node_errors"):
        raise RuntimeError(json.dumps(queued["node_errors"], ensure_ascii=False))
    prompt_id = queued["prompt_id"]
    print("queued", prompt_id, flush=True)
    started = time.monotonic()
    last_report = 0
    while time.monotonic() - started < 1800:
        history = request("/history/" + prompt_id)
        entry = history.get(prompt_id)
        if entry:
            REPORT.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            status = entry.get("status", {})
            print("finished", json.dumps(status, ensure_ascii=False), flush=True)
            print("outputs", json.dumps(entry.get("outputs", {}), ensure_ascii=False), flush=True)
            return 0 if status.get("completed") else 1
        if time.monotonic() - last_report > 20:
            print("waiting", round(time.monotonic() - started, 1), flush=True)
            last_report = time.monotonic()
        time.sleep(5)
    raise TimeoutError("downstream smoke did not complete within 30 minutes")


if __name__ == "__main__":
    sys.exit(main())
