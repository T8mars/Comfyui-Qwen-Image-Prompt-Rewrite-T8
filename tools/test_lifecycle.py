"""Exercise local model reuse, switching, and explicit unload for 20 cycles."""

import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pe_runtime import (DEFAULT_EDIT, DEFAULT_MMPROJ, DEFAULT_T2I,
                        SERVER, resolve_model)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runtime" / "lifecycle-results.jsonl"
PROFILES = [
    ("t2i", DEFAULT_T2I, None, 16384),
    ("edit", DEFAULT_EDIT, DEFAULT_MMPROJ, 32768),
    ("heretic", "pe_t2i_heretic-Q4_K_M.gguf", None, 16384),
    ("t2i-again", DEFAULT_T2I, None, 16384),
]


def gpu_mb():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return int(output.splitlines()[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


with REPORT.open("w", encoding="utf-8") as stream:
    try:
        for cycle in range(1, 21):
            prior = None
            for label, model_name, vision_name, context in PROFILES:
                started = time.monotonic()
                model = resolve_model(model_name)
                vision = resolve_model(vision_name, True) if vision_name else None
                SERVER.start(model, vision, context, 99)
                current = SERVER.process
                assert current is not None and current.poll() is None
                assert prior is None or prior.poll() is not None, "previous model process remained active"
                current_pid = current.pid
                SERVER.start(model, vision, context, 99)
                assert SERVER.process.pid == current_pid, "same profile did not reuse the process"
                record = {"cycle": cycle, "profile": label, "pid": current_pid,
                          "elapsed_seconds": round(time.monotonic() - started, 2),
                          "gpu_mb": gpu_mb()}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(json.dumps(record), flush=True)
                prior = current
            SERVER.stop()
            assert prior.poll() is not None, "explicit unload left a model process running"
            record = {"cycle": cycle, "profile": "unloaded", "gpu_mb": gpu_mb()}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(json.dumps(record), flush=True)
    finally:
        SERVER.stop()
