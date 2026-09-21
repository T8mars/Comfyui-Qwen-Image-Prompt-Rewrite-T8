"""Reproducible paired T2I probe for the two downloaded Q4 models."""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pe_runtime import SERVER, DEFAULT_T2I, resolve_model


CASES = [
    ("simple_zh", "一只穿蓝色雨衣的柯基坐在雨中的红色长椅上。"),
    ("simple_en", "A ceramic blue teapot on a pale wooden table in soft morning light."),
    ("chinese_text", "做一张茶叶海报，标题必须清晰写成“春日新茶”，不要出现其他文字。"),
    ("english_text", "Design a bakery poster with the exact headline 'FRESH BREAD' and no other text."),
    ("count", "五只颜色各不相同的纸鹤排成一行，从左到右依次是红、橙、黄、绿、蓝。"),
    ("space", "A red cup is left of a blue book; a green apple rests on top of the book."),
    ("product", "一瓶透明玻璃香水放在白色大理石台面上，银色瓶盖，柔和逆光，高级产品摄影。"),
    ("portrait", "一位戴圆框眼镜的老年木匠在工作室里雕刻木鸟，真实纪实摄影。"),
    ("wide_ratio", "16:9 横向电影画面：一艘小船穿过雾中的山间湖泊。"),
    ("tall_ratio", "9:16 手机壁纸，深蓝夜空中一棵发光的白色树。"),
    ("minimal", "A minimal black line icon of a bicycle on a plain white background."),
    ("complex", "博物馆儿童活动海报：中央一只戴围巾的恐龙，左上角太阳，右下角三朵花，背景为淡奶油色。"),
]
MODELS = [DEFAULT_T2I, "pe_t2i_heretic-Q4_K_M.gguf"]
SEEDS = [42]
REPORT = Path(__file__).resolve().parents[1] / "runtime" / "t2i-comparison.jsonl"


try:
    with REPORT.open("w", encoding="utf-8") as stream:
        for model_name in MODELS:
            SERVER.start(resolve_model(model_name), None, 16384, 99)
            for case_id, prompt in CASES:
                for seed in SEEDS:
                    started = time.monotonic()
                    record = {"model": model_name, "case": case_id, "prompt": prompt, "seed": seed}
                    try:
                        answer, info = SERVER.complete("t2i", prompt, [], seed, 900)
                        record.update(answer=answer, usage=info["usage"],
                                      format_retries=info["format_retries"])
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    record["elapsed_seconds"] = round(time.monotonic() - started, 2)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(json.dumps({key: record.get(key) for key in
                                      ("model", "case", "error", "elapsed_seconds", "format_retries")},
                                     ensure_ascii=True), flush=True)
            SERVER.stop()
finally:
    SERVER.stop()
