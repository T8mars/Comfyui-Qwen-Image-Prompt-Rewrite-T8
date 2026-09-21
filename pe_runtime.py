import atexit
import base64
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import torch
from torch.nn.functional import interpolate


ROOT = Path(__file__).resolve().parent
DEFAULT_T2I = "Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf"
DEFAULT_EDIT = "Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf"
DEFAULT_MMPROJ = "Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf"
MAX_VISUAL_PIXELS = 1024 * 1024
MAX_VISUAL_SIDE = 4096
_RATIO = re.compile(r"^[1-9]\d{0,2}:[1-9]\d{0,2}$")
_QUOTED_LITERAL = re.compile(
    r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|「[^」\n]*」|『[^』\n]*』|«[^»\n]*»|'
    r"(?<![A-Za-z0-9])'(?:[^'\n]|(?<=[A-Za-z0-9])'(?=[A-Za-z0-9]))*'(?![A-Za-z0-9])"
)


def _protected_quote(match, protected_literals):
    return protected_literals is None or match.group()[1:-1] in protected_literals


def strip_quoted_literals(text, protected_literals=None):
    """Exclude exact image text without treating English apostrophes as quotes."""
    return _QUOTED_LITERAL.sub(
        lambda match: "" if _protected_quote(match, protected_literals) else match.group(), text)


def mask_quoted_literals(text, protected_literals=None):
    """Keep character offsets while excluding exact text requested for the image."""
    return _QUOTED_LITERAL.sub(
        lambda match: " " * len(match.group()) if _protected_quote(match, protected_literals)
        else match.group(), text)


def replace_unquoted(pattern, text, replace, protected_literals=None):
    masked = mask_quoted_literals(text, protected_literals)
    protected = [(item.start(), item.end()) for item in _QUOTED_LITERAL.finditer(text)
                 if _protected_quote(item, protected_literals)]
    parts = []
    cursor = 0
    count = 0
    for match in pattern.finditer(masked):
        if any(start < match.end() and end > match.start() for start, end in protected):
            continue
        replacement = replace(match, masked)
        parts.extend((text[cursor:match.start()], replacement))
        cursor = match.end()
        count += replacement != match.group()
    parts.append(text[cursor:])
    return "".join(parts), count


def quoted_literals(text, protected_literals=None):
    return [match.group()[1:-1] for match in _QUOTED_LITERAL.finditer(text)
            if _protected_quote(match, protected_literals)]


def model_roots():
    roots = [ROOT / "models" / "llm" / "qwenimage-pe"]
    extra = os.environ.get("QWEN_PE_MODEL_DIR")
    if extra:
        roots.insert(0, Path(extra))
    try:
        import folder_paths
        roots.append(Path(folder_paths.models_dir) / "llm" / "qwenimage-pe")
    except ImportError:
        pass
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen and root.is_dir():
            seen.add(resolved)
            yield root


def local_models(vision=False):
    found = {}
    for root in model_roots():
        for path in root.rglob("*.gguf"):
            is_vision = "mmproj" in path.name.lower()
            if is_vision == vision and path.is_file() and not path.name.endswith(".part"):
                found.setdefault(path.relative_to(root).as_posix(), path)
    return dict(sorted(found.items(), key=lambda item: item[0].lower()))


def resolve_model(name, vision=False):
    path = local_models(vision).get(name)
    if path is None:
        raise FileNotFoundError(f"local {'vision ' if vision else ''}GGUF not found: {name}")
    return path


def pick_mmproj(model_name, chosen):
    model_path = resolve_model(model_name)
    base = re.sub(r"\.(?:Q\d[^.]*)\.gguf$", "", Path(model_name).name, flags=re.IGNORECASE)
    if chosen != "Auto":
        path = resolve_model(chosen, True)
        if not path.name.lower().startswith((base + ".mmproj").lower()):
            raise ValueError(f"vision file {path.name} does not match model {model_name}")
        return path
    matches = [path for path in model_path.parent.glob("*.gguf")
               if path.is_file() and path.name.lower().startswith((base + ".mmproj").lower())]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"No unique matching mmproj for {model_name}; select its vision model explicitly")


def prepare_images(images):
    result = []
    dimensions = []
    for index, tensor in enumerate(images, 1):
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] not in (3, 4):
            raise ValueError(f"image_{index} must contain exactly one RGB/RGBA IMAGE, not a batch")
        height, width = int(tensor.shape[1]), int(tensor.shape[2])
        if height < 1 or width < 1:
            raise ValueError(f"image_{index} must have nonzero width and height")
        dimensions.append([width, height])
        source = tensor.detach().permute(0, 3, 1, 2)
        if width * height > MAX_VISUAL_PIXELS or max(width, height) > MAX_VISUAL_SIDE:
            scale = min(math.sqrt(MAX_VISUAL_PIXELS / (width * height)),
                        MAX_VISUAL_SIDE / max(width, height))
            target = (max(1, int(height * scale)), max(1, int(width * scale)))
            source = interpolate(source, size=target, mode="bilinear", align_corners=False,
                                 antialias=source.dtype in (torch.float32, torch.float64))
        scaled = source[0].permute(1, 2, 0).float().nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        pixels = np.clip(scaled.cpu().numpy() * 255.0,
                         0, 255).astype(np.uint8)
        image = Image.fromarray(pixels)
        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(canvas, image).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        result.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    return result, dimensions


def parse_answer(raw, task, image_count):
    decoder = json.JSONDecoder()
    parsed = []
    cursor = 0
    while cursor < len(raw):
        if raw.startswith("<think>", cursor):
            end_thinking = raw.find("</think>", cursor + len("<think>"))
            if end_thinking < 0:
                raise ValueError("model returned an incomplete thinking block")
            cursor = end_thinking + len("</think>")
            continue
        if raw.startswith("</think>", cursor):
            raise ValueError("model returned an unmatched thinking block close")
        if raw[cursor] not in "{[":
            cursor += 1
            continue
        try:
            candidate, end = decoder.raw_decode(raw[cursor:])
        except json.JSONDecodeError as exc:
            raise ValueError("model returned malformed top-level JSON") from exc
        parsed.append(candidate)
        cursor += end
    if not parsed or not isinstance(parsed[-1], dict):
        raise ValueError("model did not return a valid final JSON object")
    answer = parsed[-1]
    required = {"rewritten_prompt", "wh_ratio"} if task == "t2i" else {"rewritten_prompt", "wh_ratio", "ratio_follow"}
    if set(answer) != required:
        raise ValueError(f"wrong answer fields: got {sorted(answer)}, expected {sorted(required)}")
    if not all(isinstance(value, str) for value in answer.values()):
        raise ValueError("answer fields must all be strings")
    if not answer["rewritten_prompt"].strip():
        raise ValueError("rewritten_prompt is empty")
    ratio = answer["wh_ratio"]
    if task == "t2i":
        if not _RATIO.fullmatch(ratio):
            raise ValueError(f"invalid T2I aspect ratio: {ratio!r}")
    else:
        follow = answer["ratio_follow"]
        if bool(ratio) == bool(follow):
            raise ValueError("edit answer must set exactly one of wh_ratio and ratio_follow")
        if ratio and not _RATIO.fullmatch(ratio):
            raise ValueError(f"invalid edit aspect ratio: {ratio!r}")
        if follow and follow not in {f"<image{i}>" for i in range(1, image_count + 1)}:
            raise ValueError(f"ratio_follow references an unavailable image: {follow}")
    return answer


def validate_references(answer, task, image_count, protected_literals=None):
    prose = strip_quoted_literals(answer["rewritten_prompt"], protected_literals)
    references = set(re.findall(r"<image[^>]*>", prose))
    expected = {f"<image{i}>" for i in range(1, image_count + 1)} if image_count >= 2 else set()
    if references != expected:
        raise ValueError(f"image references in rewritten_prompt are {sorted(references)}, expected {sorted(expected)}")


def validate_language(answer, output_language, protected_literals=None):
    if output_language == "auto":
        return
    prose = strip_quoted_literals(answer["rewritten_prompt"], protected_literals)
    contains_han = bool(re.search(r"[\u4e00-\u9fff]", prose))
    if output_language == "English" and contains_han:
        raise ValueError("rewritten descriptive prose is not fully English")
    if output_language == "中文" and not contains_han:
        raise ValueError("rewritten descriptive prose is not Chinese")
    if output_language == "中文":
        prose = re.sub(r"<image\d+>", "", prose)
        prose = re.sub(r"(?<![A-Za-z0-9])[23]D(?![A-Za-z0-9])", "", prose, flags=re.I)
        english_words = [word for word in re.findall(r"[A-Za-z][A-Za-z-]*", prose)
                         if word.lower() not in {"rgba", "alpha"}]
        if english_words:
            raise ValueError(f"Chinese descriptive prose contains English words: {english_words[:8]}")


def transparent_background_clause(sentence):
    return bool(re.search(
        r"透明.{0,8}背景|背景.{0,16}透明|"
        r"(?:不要|不含|没有|无|去掉|移除|删除).{0,12}背景|"
        r"\btransparent.{0,20}background\b|\bbackground.{0,20}transparent\b|"
        r"\b(?:no|without|remove|erase|cut\s*out|isolate).{0,25}background\b|"
        r"\bbackground.{0,20}(?:removed|erased|absent)\b",
        sentence, re.I))


_OPAQUE_BACKDROP = re.compile(
    r"\b(?:(?:pure|plain|solid|opaque|uniform|matte|light|dark|pale)\s+){0,2}"
    r"(?:white|black|gray|grey|beige|brown|blue|red|green|yellow|checkerboard)\s+background\b|"
    r"\bbackground\s+(?:is|appears|looks|of|in)?\s*(?:a\s+)?"
    r"(?:(?:pure|plain|solid|opaque|uniform|matte|light|dark|pale)\s+){0,2}"
    r"(?:white|black|gray|grey|beige|brown|blue|red|green|yellow|checkerboard)\b|"
    r"(?:纯白|白色|黑色|灰色|米色|棋盘格)背景|"
    r"背景(?:是|为|呈)?[^。！？,.]{0,12}(?:纯白|白色|黑色|灰色|米色|棋盘格)", re.I)


def normalize_opaque_background_phrases(prompt, protected_literals=None):
    def replace(match, masked):
        prefix = masked[max(0, match.start() - 18):match.start()]
        if re.search(r"\b(?:no|without|not)\s+(?:a\s+)?$|(?:不要|没有|不含|无)$", prefix, re.I):
            return match.group()
        return "透明背景" if re.search(r"[\u4e00-\u9fff]", match.group()) else "transparent background"

    return replace_unquoted(_OPAQUE_BACKDROP, prompt, replace, protected_literals)


def normalize_transparent_margins(prompt, protected_literals=None):
    pattern = re.compile(r"\b(?:even\s+)?(?:white|opaque)\s+(?:empty\s+)?margins?\b|"
                         r"白色留白|不透明留白|白色空白边缘", re.I)
    return replace_unquoted(pattern, prompt,
                            lambda match, _: "透明留白" if re.search(r"[\u4e00-\u9fff]", match.group())
                            else "transparent margins", protected_literals)


def validate_mode(answer, aspect_ratio, transparent_rgba, protected_literals=None):
    errors = []
    original = answer["rewritten_prompt"]
    prose = mask_quoted_literals(original, protected_literals)
    if not original.strip():
        errors.append("rewritten prompt became empty after transparency cleanup")
    if aspect_ratio != "auto":
        explicit_ratios = set(re.findall(r"\b\d{1,2}:\d{1,2}\b", prose))
        if explicit_ratios - {aspect_ratio}:
            errors.append(f"prompt describes conflicting aspect ratio {sorted(explicit_ratios - {aspect_ratio})}")
        if aspect_ratio != "1:1" and re.search(
                r"\bsquare-format\b|\bsquare\s+(?:image|canvas|frame|composition|layout)\b|"
                r"正方形(?:画幅|图像|画面)|方形画幅", prose, re.I):
            errors.append("prompt describes a square canvas")
    if transparent_rgba:
        for match in _OPAQUE_BACKDROP.finditer(prose):
            prefix = prose[max(0, match.start() - 18):match.start()]
            if not re.search(r"\b(?:no|without|not)\s+(?:a\s+)?$|(?:不要|没有|不含|无)$",
                             prefix, re.I):
                errors.append("prompt describes an opaque background")
                break
        if re.search(r"\b(?:white|opaque)\s+(?:empty\s+)?margins?\b|"
                     r"白色留白|不透明留白|白色空白边缘", prose, re.I):
            errors.append("prompt describes opaque margins")
        for sentence in re.split(r"[.!?。！？]", prose):
            if (re.search(r"\bbackground\b|背景", sentence, re.I)
                    and not transparent_background_clause(sentence)):
                errors.append("prompt describes a non-transparent background")
                break
    if errors:
        raise ValueError("; ".join(errors))


def remove_opaque_background_sentences(prompt, protected_literals=None):
    kept = []
    removed = 0
    for sentence in re.split(r"(?<=[.!?。！？])\s*", prompt):
        descriptive = mask_quoted_literals(sentence, protected_literals)
        has_backdrop = re.search(r"\bbackground\b|背景", descriptive, re.I)
        has_ground_shadow = re.search(r"\bshadow\b.{0,80}\bground\b|阴影.{0,40}地面", descriptive, re.I)
        safe = transparent_background_clause(descriptive) if has_backdrop else False
        if ((has_backdrop and not safe) or (has_ground_shadow and not safe)) and not quoted_literals(sentence, protected_literals):
            removed += 1
        elif sentence.strip():
            kept.append(sentence.strip())
    return " ".join(kept).strip(), removed


class LocalServer:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.profile = None
        self.port = None
        self.log = None

    @staticmethod
    def binary():
        configured = os.environ.get("QWEN_PE_LLAMA_SERVER")
        if configured:
            path = Path(configured)
            if path.is_file():
                return path
            raise FileNotFoundError(path)
        matches = list((ROOT / "runtime").rglob("llama-server.exe"))
        if len(matches) != 1:
            raise FileNotFoundError("llama-server.exe not found; set QWEN_PE_LLAMA_SERVER")
        return matches[0]

    def stop(self):
        with self.lock:
            try:
                if self.process is not None:
                    if self.process.poll() is None:
                        self.process.terminate()
                    try:
                        self.process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=10)
            finally:
                self.process = None
                if self.log is not None:
                    self.log.close()
                    self.log = None
                self.profile = None
                self.port = None

    def start(self, model, mmproj, context, gpu_layers):
        with self.lock:
            def signature(path):
                stat = path.stat()
                return str(path.resolve()), stat.st_size, stat.st_mtime_ns

            profile = (signature(model), signature(mmproj) if mmproj else None, context, gpu_layers)
            if self.process is not None and self.process.poll() is None and self.profile == profile:
                try:
                    with urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                        if response.status == 200:
                            return
                except (URLError, TimeoutError, HTTPError):
                    pass
            self.stop()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            command = [str(self.binary()), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
                       "-c", str(context), "-ngl", str(gpu_layers), "--jinja", "--reasoning-format", "none",
                       "--parallel", "1", "--alias", "qwen-pe"]
            if mmproj:
                command += ["--mmproj", str(mmproj), "--image-min-tokens", "1024"]
            log_path = ROOT / "runtime" / "server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = log_path.open("ab")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT, creationflags=flags)
                self.profile = profile
                self.port = port
                try:
                    import comfy.model_management as memory
                    check_interrupt = memory.throw_exception_if_processing_interrupted
                except ImportError:
                    check_interrupt = None
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if check_interrupt:
                        check_interrupt()
                    if self.process.poll() is not None:
                        raise RuntimeError(f"llama-server exited with {self.process.returncode}; see {log_path}")
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                            if response.status == 200:
                                return
                    except (URLError, TimeoutError, HTTPError):
                        pass
                    time.sleep(0.5)
                raise TimeoutError("llama-server did not become healthy within 180 seconds")
            except BaseException:
                self.stop()
                raise

    def complete(self, task, prompt, images, seed, timeout,
                 output_language="auto", aspect_ratio="auto", transparent_rgba=False):
        with self.lock:
            exact_literals = set(quoted_literals(prompt))
            if task == "edit":
                # Numbered image tags are references even when the user puts
                # quotation marks around them in an editing instruction.
                exact_literals = {value for value in exact_literals
                                  if not re.fullmatch(r"<image\d+>", value)}
            system_name = "system_prompt_t2i.txt" if task == "t2i" else "system_prompt_edit.txt"
            system = (ROOT / "prompts" / system_name).read_text(encoding="utf-8").strip()
            content = [{"type": "image_url", "image_url": {"url": value}} for value in images]
            if len(images) >= 2:
                tags = "、".join(f"<image{i}>" for i in range(1, len(images) + 1))
                system += (f"\n\nRuntime image-reference rule: The rewritten_prompt must explicitly use every "
                           f"image tag from {tags}, each for its matching input image; omit none.")
            elif len(images) == 1:
                system += ("\n\nRuntime image-reference rule: There is one input image. Do not put "
                           "<image1> or any numbered image tag in rewritten_prompt; refer to it naturally.")
            if output_language != "auto":
                language = "English" if output_language == "English" else "Chinese"
                system += (f"\n\nUser-selected language override: Write all descriptive prose of "
                           f"rewritten_prompt in {language}. Preserve exact user-requested text to be "
                           "rendered inside quotation marks, even if it is in another language.")
            if aspect_ratio != "auto":
                system += (f"\n\nUser-selected canvas override: Set wh_ratio to {aspect_ratio}, "
                           "ratio_follow to an empty string, and describe the output composition for this ratio.")
            if transparent_rgba:
                system += ("\n\nUser-selected output mode: Compose for an RGBA image with an alpha "
                           "channel and fully transparent background. Do not add an opaque backdrop.")
            if output_language == "中文":
                presented_prompt = ("请用中文撰写最终 rewritten_prompt 的全部描述性文字。"
                                    "下面是用户原始创作需求；引号内明确指定的画面文字保留原样：\n" + prompt)
            elif output_language == "English":
                presented_prompt = ("Write all descriptive prose in the final rewritten_prompt in English. "
                                    "The following is the user's original creative request; preserve any "
                                    "exact quoted text to appear in the image:\n" + prompt)
            else:
                presented_prompt = prompt
            content.append({"type": "text", "text": presented_prompt})
            payload = {
                "model": "qwen-pe",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5 if task == "t2i" else 0.0,
                "max_tokens": 8192 if task == "t2i" else 12288,
                "seed": seed,
                "chat_template_kwargs": {"enable_thinking": True},
                "stream": False,
            }
            first_error = None
            for attempt in range(2):
                result = self._post_completion(payload, timeout)
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("model response was truncated at max_tokens")
                raw = choice["message"].get("content") or ""
                try:
                    answer = parse_answer(raw, task, len(images))
                    removed_background_sentences = 0
                    if transparent_rgba:
                        answer["rewritten_prompt"], normalized_background_phrases = (
                            normalize_opaque_background_phrases(answer["rewritten_prompt"], exact_literals))
                        answer["rewritten_prompt"], removed_background_sentences = (
                            remove_opaque_background_sentences(answer["rewritten_prompt"], exact_literals))
                        answer["rewritten_prompt"], normalized_margin_phrases = (
                            normalize_transparent_margins(answer["rewritten_prompt"], exact_literals))
                    else:
                        normalized_background_phrases = 0
                        normalized_margin_phrases = 0
                    validate_references(answer, task, len(images), exact_literals)
                    translation_fallback = False
                    translation_usage = None
                    try:
                        validate_language(answer, output_language, exact_literals)
                    except ValueError:
                        if output_language == "auto":
                            raise
                        answer["rewritten_prompt"], translation_usage = self._translate_prose(
                            answer["rewritten_prompt"], output_language, seed, timeout, exact_literals)
                        translation_fallback = True
                        if transparent_rgba:
                            answer["rewritten_prompt"], normalized_more_backdrop = (
                                normalize_opaque_background_phrases(answer["rewritten_prompt"], exact_literals))
                            normalized_background_phrases += normalized_more_backdrop
                            answer["rewritten_prompt"], removed_more = (
                                remove_opaque_background_sentences(answer["rewritten_prompt"], exact_literals))
                            removed_background_sentences += removed_more
                            answer["rewritten_prompt"], normalized_more = (
                                normalize_transparent_margins(answer["rewritten_prompt"], exact_literals))
                            normalized_margin_phrases += normalized_more
                        validate_references(answer, task, len(images), exact_literals)
                        validate_language(answer, output_language, exact_literals)
                    validate_mode(answer, aspect_ratio, transparent_rgba, exact_literals)
                    return answer, {"finish_reason": choice.get("finish_reason"),
                                    "usage": result.get("usage", {}),
                                    "format_retries": attempt, "first_format_error": first_error,
                                    "removed_background_sentences": removed_background_sentences,
                                    "normalized_background_phrases": normalized_background_phrases,
                                    "normalized_margin_phrases": normalized_margin_phrases,
                                    "translation_fallback": translation_fallback,
                                    "translation_usage": translation_usage}
                except ValueError as exc:
                    if attempt:
                        raise ValueError(f"model failed format validation after one retry: {exc}") from exc
                    first_error = str(exc)
                    system += ("\n\nThe previous output failed validation: " + first_error +
                               ". Retry the same user request and satisfy all JSON, language, "
                               "aspect-ratio, and image-reference constraints.")
                    payload["messages"][0]["content"] = system
            raise AssertionError("unreachable")

    def _translate_prose(self, prose, output_language, seed, timeout, exact_literals=None):
        target = "Simplified Chinese" if output_language == "中文" else "English"
        if output_language == "中文":
            request_text = (
                "把以下图像生成提示词逐句忠实翻译成简体中文，只输出译文，不要标题、JSON 或解释。"
                "描述部分禁止任何英文字母；仅 <imageN> 标签、数字、引号内原文、RGBA 和 alpha 可以保留英文。"
                "不得新增或删除视觉事实，不得更改图片编号和引号内需要呈现的文字。\n\n" + prose)
        else:
            request_text = (
                f"Translate the following image-generation prompt faithfully into {target}. "
                "Do not add or remove visual facts. Keep every <imageN> tag and every exact "
                "quoted string unchanged. Return only plain translated prose, with no heading, "
                "JSON, or explanation.\n\n" + prose)
        payload = {
            "model": "qwen-pe",
            "messages": [{"role": "system", "content": "你是严格忠实的图像提示词翻译编辑。只执行翻译。" if output_language == "中文" else
                         "You are a faithful translation editor for image prompts."},
                         {"role": "user", "content": request_text}],
            "temperature": 0.0, "top_p": 0.9, "max_tokens": 8192,
            "seed": seed, "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
        }
        result = self._post_completion(payload, timeout)
        choice = result["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("translation fallback was truncated")
        translated = choice["message"].get("content") or ""
        translated = re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", translated)
        translated = re.sub(r"(?m)^\s*#{1,6}\s*", "", translated).strip()
        for literal in quoted_literals(prose, exact_literals):
            if literal not in translated:
                raise ValueError(f"translation changed exact quoted image text: {literal!r}")
        if not translated:
            raise ValueError("translation fallback returned empty text")
        return translated, result.get("usage", {})

    def _post_completion(self, payload, timeout):
        """Wait for llama.cpp while honoring ComfyUI's interrupt flag."""
        request = Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                          data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        outcome = {}

        def perform():
            try:
                with urlopen(request, timeout=timeout) as response:
                    outcome["result"] = json.load(response)
            except HTTPError as exc:
                detail = exc.read(2000).decode("utf-8", "replace")
                outcome["error"] = RuntimeError(f"llama-server HTTP {exc.code}: {detail}")
            except BaseException as exc:
                outcome["error"] = exc

        try:
            import comfy.model_management as memory
            check_interrupt = memory.throw_exception_if_processing_interrupted
        except ImportError:
            check_interrupt = None
        worker = threading.Thread(target=perform, name="qwen-pe-http", daemon=True)
        worker.start()
        deadline = time.monotonic() + timeout
        try:
            while worker.is_alive():
                worker.join(0.25)
                if check_interrupt:
                    check_interrupt()
                if time.monotonic() > deadline:
                    raise TimeoutError(f"llama-server generation exceeded {timeout} seconds")
        except BaseException:
            self.stop()
            worker.join(5)
            raise
        if "error" in outcome:
            raise outcome["error"]
        return outcome["result"]


SERVER = LocalServer()
atexit.register(SERVER.stop)
