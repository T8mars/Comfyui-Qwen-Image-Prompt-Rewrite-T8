"""Prepare and validate the fixed 60-scene Qwen PE quality set."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
FIXTURES = EVAL / "fixtures"
CASES = EVAL / "quality_cases.json"
FIXTURES.mkdir(parents=True, exist_ok=True)

COPIES = {
    "apple": Path("E:/comfyui-t8-onekey-5x/ComfyUI/output/Qwen-PE-2.1-Full-T2I_00001_.png"),
    "butterfly": Path("E:/comfyui-t8-onekey-5x/ComfyUI/output/Qwen-PE-2.1-Full-Transparent-T2I_00001_.png"),
    "orange_scene": ROOT / "workflows/fixtures/qwen_pe_reference_2.png",
}
for name, source in COPIES.items():
    target = FIXTURES / (name + ".png")
    if not target.exists():
        shutil.copyfile(source, target)

colors = ["#cf3030", "#20834e", "#265cb9", "#e79719", "#8a3b9b",
          "#149eaa", "#c45b80", "#735c36", "#4e5689", "#e6c827"]
for index, color in enumerate(colors, start=1):
    target = FIXTURES / f"swatch_{index}.png"
    if target.exists():
        continue
    image = Image.new("RGB", (128, 128), color)
    if index == 10:
        draw = ImageDraw.Draw(image)
        points = []
        for vertex in range(10):
            radius = 36 if vertex % 2 == 0 else 16
            angle = -math.pi / 2 + vertex * math.pi / 5
            points.append((64 + radius * math.cos(angle), 64 + radius * math.sin(angle)))
        draw.polygon(points, fill="black")
    image.save(target)

cases = json.loads(CASES.read_text(encoding="utf-8"))
ids = [case["id"] for case in cases]
assert len(cases) == 60 and len(set(ids)) == 60
assert sum(case["render"] for case in cases) == 24
assert Counter(case["task"] for case in cases) == {"t2i": 24, "edit": 36}
assert Counter(len(case["images"]) for case in cases if case["id"].startswith("E")) == {
    1: 12, 2: 8, 3: 2, 4: 1, 5: 1,
}
assert Counter(len(case["images"]) for case in cases if case["id"].startswith("M")) == {
    6: 2, 7: 2, 8: 2, 9: 2, 10: 4,
}
for case in cases:
    if case["task"] == "edit":
        assert all((FIXTURES / (name + ".png")).is_file() for name in case["images"]), case["id"]
    else:
        assert not case["images"], case["id"]
    assert case["prompt"].strip(), case["id"]

digest = hashlib.sha256(CASES.read_bytes()).hexdigest()
print("cases", len(cases), "render", sum(case["render"] for case in cases),
      "sha256", digest, "fixtures", len(list(FIXTURES.glob("*.png"))))
