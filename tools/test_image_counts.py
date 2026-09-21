"""Exercise every supported image count against the real I2I GGUF and mmproj."""

import base64
import io
import json
import math
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pe_runtime import SERVER, DEFAULT_EDIT, pick_mmproj, resolve_model


COLORS = ["#cf3030", "#20834e", "#265cb9", "#e79719", "#8a3b9b",
          "#149eaa", "#c45b80", "#735c36", "#4e5689", "#e6c827"]
def image_url(index):
    image = Image.new("RGB", (384, 384), COLORS[index])
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, 369, 369), outline="white", width=8)
    if index == 9:
        points = []
        for vertex in range(10):
            radius = 65 if vertex % 2 == 0 else 28
            angle = -math.pi / 2 + vertex * math.pi / 5
            points.append((192 + radius * math.cos(angle),
                           265 + radius * math.sin(angle)))
        draw.polygon(points, fill="black")
    output = io.BytesIO()
    image.save(output, "PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


model = resolve_model(DEFAULT_EDIT)
vision = pick_mmproj(DEFAULT_EDIT, "Auto")
images = [image_url(i) for i in range(10)]
counts = [int(value) for value in sys.argv[1:]] if len(sys.argv) > 1 else list(range(1, 11))
report = (Path(__file__).resolve().parents[1] / "runtime" /
          ("image-count-results-validated.jsonl" if len(sys.argv) > 1 else "image-count-results.jsonl"))
report.parent.mkdir(exist_ok=True)
try:
    with report.open("w", encoding="utf-8") as handle:
        for count in counts:
            context = 32768 if count <= 5 else 65536
            started = time.monotonic()
            entry = {"image_count": count, "context": context}
            try:
                SERVER.start(model, vision, context, 99)
                prompt = (f"使用这{count}张图片制作一张宽幅抽象拼贴画。"
                          "按照图片顺序使用每张图的主色；如果最后一张有明显图形，也保留该图形。")
                answer, info = SERVER.complete("edit", prompt, images[:count], 42, 900)
                entry.update(answer=answer, usage=info["usage"], finish_reason=info["finish_reason"])
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["elapsed_seconds"] = round(time.monotonic() - started, 2)
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"image_count": count, "error": entry.get("error"),
                              "elapsed_seconds": entry["elapsed_seconds"],
                              "prompt_tokens": entry.get("usage", {}).get("prompt_tokens")},
                             ensure_ascii=True), flush=True)
finally:
    SERVER.stop()
