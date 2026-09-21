"""Temporary direct check of whether the local GGUF can translate its own prose."""

import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pe_runtime import SERVER, DEFAULT_T2I, resolve_model


try:
    SERVER.start(resolve_model(DEFAULT_T2I), None, 16384, 99)
    payload = {
        "model": "qwen-pe",
        "messages": [
            {"role": "system", "content": "Translate the user's text into fluent Simplified Chinese. Preserve all details, exact quoted strings, and <imageN> tags. Return only the Chinese translation, without explanation."},
            {"role": "user", "content": "A watercolor painting of a small mountain cottage beside a lake, under a pale blue morning sky. The cottage has a warm wooden door."},
        ],
        "temperature": 0.1, "top_p": 0.9, "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
    }
    request = Request(f"http://127.0.0.1:{SERVER.port}/v1/chat/completions",
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    print(json.dumps(result["choices"][0], ensure_ascii=True), flush=True)
finally:
    SERVER.stop()
