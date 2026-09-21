"""Verify Chinese output from an English source request."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pe_runtime import SERVER, DEFAULT_T2I, resolve_model


try:
    SERVER.start(resolve_model(DEFAULT_T2I), None, 16384, 99)
    answer, info = SERVER.complete(
        "t2i", "A watercolor painting of a small mountain cottage beside a lake.",
        [], 42, 900, output_language="中文", aspect_ratio="4:3")
    print(json.dumps({"answer": answer, "info": info}, ensure_ascii=True), flush=True)
finally:
    SERVER.stop()
