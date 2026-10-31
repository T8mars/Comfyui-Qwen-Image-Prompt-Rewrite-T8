"""Targeted local-GGUF checks for language fallback and exact image lettering."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pe_runtime import (DEFAULT_T2I, ROOT, LocalServer, quoted_literals, resolve_model,
                        validate_language, validate_references)


class RecordedServer(LocalServer):
    def __init__(self):
        super().__init__()
        self.responses = []

    def _post_completion(self, payload, timeout):
        result = super()._post_completion(payload, timeout)
        self.responses.append({"request": payload, "response": result})
        (ROOT / "runtime" / "translation-literals-raw.json").write_text(
            json.dumps(self.responses, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="GGUF filename; may be repeated")
    parser.add_argument("--rewrite", action="store_true", help="Also run a full English T2I rewrite")
    args = parser.parse_args()
    models = args.model or [DEFAULT_T2I, "pe_i2i_heretic-Q4_K_M.gguf"]
    cases = [
        ("English", '调亮<image1>的画面，保留标题“春日来信”和标签「DON\'T STOP」。'
         '将<image2>放在右侧，保留广告牌“SUMMER SALE”，背景是柔和的蓝色。'),
        ("中文", 'Brighten <image1> while preserving the title "SUMMER SALE" and the label '
         '"DON\'T STOP". Put <image2> on the right and preserve the sign 「春日来信」 '
         'against a soft blue background.'),
    ]
    report = ROOT / "runtime" / "translation-literals-results.json"
    rows = []
    server = RecordedServer()
    try:
        for model in models:
            server.start(resolve_model(model), None, 16384, 99)
            for language, source in cases:
                started = time.monotonic()
                literals = set(quoted_literals(source))
                correction = None
                corrections = []
                for attempt in range(3):
                    try:
                        translated, usage = server._translate_prose(
                            source, language, 42, 180, correction=correction,
                            temperature=0.0 if attempt == 0 else 0.3,
                            protected_literals=literals)
                        answer = {"rewritten_prompt": translated}
                        validate_language(answer, language, literals)
                        validate_references(answer, "edit", 2, literals)
                        assert quoted_literals(translated) == quoted_literals(source)
                        assert "QWEN_PE_LITERAL_" not in translated
                    except (ValueError, AssertionError) as exc:
                        if attempt == 2:
                            raise
                        correction = str(exc)
                        corrections.append(correction)
                    else:
                        break
                row = {"model": model, "language": language, "source": source,
                       "translated": translated, "usage": usage, "retries": attempt,
                       "corrections": corrections,
                       "seconds": round(time.monotonic() - started, 2), "passed": True}
                rows.append(row)
                report.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(row, ensure_ascii=True), flush=True)
            if args.rewrite and model == DEFAULT_T2I:
                answer, info = server.complete(
                    "t2i", '设计一张蓝色海报，主标题写“春日来信”，副标题写“DON\'T STOP”。',
                    [], 42, 180, output_language="English", aspect_ratio="9:16")
                assert "春日来信" in answer["rewritten_prompt"]
                assert "DON'T STOP" in answer["rewritten_prompt"]
                assert "QWEN_PE_LITERAL_" not in answer["rewritten_prompt"]
                rows.append({"model": model, "check": "full_t2i_rewrite",
                             "answer": answer, "info": info, "passed": True})
                report.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(rows[-1], ensure_ascii=True), flush=True)
            server.stop()
    finally:
        server.stop()
    print(f"Saved {len(rows)} passing checks to {report}", flush=True)


if __name__ == "__main__":
    main()
