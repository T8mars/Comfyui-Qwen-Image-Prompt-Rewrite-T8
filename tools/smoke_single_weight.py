"""Experimental I2I checkpoint as text-only T2I rewriter; not the product default."""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pe_runtime import SERVER, DEFAULT_EDIT, resolve_model


cases = [
    "一只穿蓝色雨衣的柯基坐在雨中的红色长椅上。",
    "Design a simple poster with the exact headline 'FRESH BREAD' and a loaf on a wooden board.",
    "16:9 横向画面：一艘小船穿过雾中的山间湖泊。",
]
report = Path(__file__).resolve().parents[1] / "runtime" / "single-weight-probe.jsonl"
try:
    SERVER.start(resolve_model(DEFAULT_EDIT), None, 16384, 99)
    with report.open("w", encoding="utf-8") as stream:
        for index, prompt in enumerate(cases, 1):
            started = time.monotonic()
            entry = {"case": index, "prompt": prompt}
            try:
                answer, info = SERVER.complete("t2i", prompt, [], 42, 900)
                entry.update(answer=answer, usage=info["usage"],
                             format_retries=info["format_retries"])
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["elapsed_seconds"] = round(time.monotonic() - started, 2)
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({k: entry.get(k) for k in
                              ("case", "error", "elapsed_seconds", "format_retries")},
                             ensure_ascii=True), flush=True)
finally:
    SERVER.stop()
