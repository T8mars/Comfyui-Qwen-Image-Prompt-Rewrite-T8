"""Exercise the explicit language, canvas, and transparent-output constraints."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pe_runtime import SERVER, DEFAULT_T2I, resolve_model


try:
    SERVER.start(resolve_model(DEFAULT_T2I), None, 16384, 99)
    answer, info = SERVER.complete(
        "t2i", "一只金色蝴蝶，翅膀有细致花纹，居中展示。", [], 42, 900,
        output_language="English", aspect_ratio="21:9", transparent_rgba=True)
    print(json.dumps({"answer": answer, "info": info}, ensure_ascii=True), flush=True)
finally:
    SERVER.stop()
