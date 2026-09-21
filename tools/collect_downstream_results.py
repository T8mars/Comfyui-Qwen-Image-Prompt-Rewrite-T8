"""Save local ComfyUI downstream results with PNG alpha and prompt diagnostics."""

import json
import os
from pathlib import Path
from urllib.request import urlopen

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("QWEN_PE_COMFY_URL", "http://127.0.0.1:8189")
OUTPUT = Path(os.environ.get("QWEN_PE_COMFY_OUTPUT", "E:/comfyui-t8-onekey-5x/ComfyUI/output"))
REPORT = ROOT / "runtime/downstream-acceptance.json"


def fetch(path):
    with urlopen(BASE + path, timeout=30) as response:
        return json.load(response)


def collect():
    history = fetch("/history?max_items=200")
    records = []
    for prompt_id, entry in history.items():
        outputs = entry.get("outputs", {})
        images = outputs.get("9", {}).get("images", [])
        if not images or not images[0]["filename"].startswith("Qwen-PE-2.1-Full-"):
            continue
        diagnostics = outputs.get("10", {}).get("text", [])
        final_prompt = outputs.get("13", {}).get("text", [])
        canvas_source = outputs.get("14", {}).get("text", [])
        record = {
            "prompt_id": prompt_id,
            "status": entry.get("status", {}).get("status_str"),
            "diagnostics": json.loads(diagnostics[0]) if diagnostics else None,
            "final_prompt": final_prompt[0] if final_prompt else None,
            "canvas_source": canvas_source[0] if canvas_source else None,
            "images": [],
        }
        for info in images:
            path = OUTPUT / info["subfolder"] / info["filename"]
            with Image.open(path) as image:
                alpha = image.getchannel("A") if "A" in image.getbands() else None
                histogram = alpha.histogram() if alpha else None
                record["images"].append({
                    "path": str(path),
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                    "alpha_min_max": alpha.getextrema() if alpha else None,
                    "alpha_under_128_fraction": (sum(histogram[:128]) / (image.width * image.height)
                                                 if histogram else None),
                })
        records.append(record)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(records)} records -> {REPORT}")


if __name__ == "__main__":
    collect()
