"""A direct llama.cpp smoke test; UI verification is tracked separately."""

import json
import base64
import io
from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pe_runtime import SERVER, local_models, pick_mmproj, resolve_model


task = sys.argv[1] if len(sys.argv) > 1 else "t2i"
name = sys.argv[2] if len(sys.argv) > 2 else "Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf"
model = resolve_model(name)
vision = pick_mmproj(name, "Auto") if task == "edit" else None
images = []
if vision:
    image = Image.new("RGB", (512, 384), "#70a7d3")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 250, 480, 365), fill="#49723c")
    draw.ellipse((180, 130, 300, 265), fill="#f0d38e")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    images = ["data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")]
try:
    SERVER.start(model, vision, 32768 if vision else 16384, 99)
    answer, info = SERVER.complete(task, "一只穿蓝色雨衣的柯基坐在雨中的红色长椅上。" if task == "t2i" else "把天空改成晚霞，保持其他内容。", images, 42, 900)
    print(json.dumps({"answer": answer, "info": info}, ensure_ascii=False), flush=True)
finally:
    SERVER.stop()
