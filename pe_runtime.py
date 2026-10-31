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
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from PIL import Image
import torch
from torch.nn.functional import interpolate


ROOT = Path(__file__).resolve().parent
# Every HTTP request in this module targets our own loopback llama-server.
# Ignore process-wide proxy settings, which may otherwise intercept 127.0.0.1.
LOCAL_OPENER = build_opener(ProxyHandler({}))
DEFAULT_T2I = "Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf"
DEFAULT_EDIT = "Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf"
DEFAULT_MMPROJ = "Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf"
DEFAULT_STARTUP_TIMEOUT = 900
MAX_VISUAL_PIXELS = 1024 * 1024
MAX_VISUAL_SIDE = 4096
_RATIO = re.compile(r"^[1-9]\d{0,2}:[1-9]\d{0,2}$")
_HAN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_OTHER_SCRIPT = re.compile(r"[\u0400-\u052f\u0590-\u06ff\u0900-\u097f\u0e00-\u0e7f\u3040-\u30ff\uac00-\ud7af]")
_QUOTED_LITERAL = re.compile(
    r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|「[^」\n]*」|『[^』\n]*』|«[^»\n]*»|'
    r"(?<![A-Za-z0-9])'(?:[^'\n]|(?<=[A-Za-z0-9])'(?=[A-Za-z0-9]))*'(?![A-Za-z0-9])"
)
_VISIBLE_TEXT_BEFORE = re.compile(
    r"(?:标题|文字|字样|字幕|标语|写着|写有|写成|印着|印有|显示|标注|"
    r"\b(?:title|text|caption|slogan|reads?|says?)\b)"
    r"[^，,。！？.!?；;\n\"“”'‘’「」『』«»]{0,24}$|"
    r"(?:广告牌|招牌|店招|路牌|标牌|告示牌|牌匾|横幅|标签|"
    r"\b(?:billboard|sign|label|banner)\b)"
    r"[^，,。！？.!?；;\n\"“”'‘’「」『』«»]{0,12}$", re.I)
_VISIBLE_TEXT_AFTER = re.compile(
    r"\s*(?:字样|文字|标题|标语|字幕|招牌|广告牌|路牌|标牌|标签|"
    r"\b(?:text|title|caption|slogan|sign|label)\b)", re.I)


def _visible_text_quote(match):
    """Recognize quoted lettering transcribed from an input image."""
    before = match.string[max(0, match.start() - 80):match.start()]
    after = match.string[match.end():match.end() + 18]
    return bool(_VISIBLE_TEXT_BEFORE.search(before) or _VISIBLE_TEXT_AFTER.match(after))


def _protected_quote(match, protected_literals):
    return (protected_literals is None or match.group()[1:-1] in protected_literals
            or _visible_text_quote(match))


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


def _protect_translation_literals(prose, protected_literals):
    """Keep exact lettering and image references out of translation input."""
    # Use a sentinel absent from all source text, including literal contents.
    sentinel = next((char for char in ("⟦", "§", "¦", "¤") if char not in prose), None)
    if sentinel is None:
        codepoint = 0xE000
        while chr(codepoint) in prose:
            codepoint += 1
        sentinel = chr(codepoint)
    prefix = sentinel + "QWEN_PE_LITERAL_"
    literals = {}

    def protect(match):
        if not _protected_quote(match, protected_literals):
            return match.group()
        marker = f"{prefix}{len(literals)}__"
        literals[marker] = match.group()
        return marker

    masked = _QUOTED_LITERAL.sub(protect, prose)

    def protect_reference(match):
        number = re.search(r"\d+", match.group()).group()
        marker = f"{prefix}IMAGE{number}_{len(literals)}__"
        literals[marker] = match.group()
        return marker

    masked = re.sub(r"<image\d+>", protect_reference, masked)
    # Record immutable spans in source order for translation and reconstruction.
    marker_pattern = re.compile(re.escape(prefix) + r"(?:IMAGE\d+_)?\d+__")
    literals = {marker: literals[marker] for marker in marker_pattern.findall(masked)}
    return masked, literals, prefix


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


def file_signature(path):
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def pick_mmproj(model_name, chosen):
    model_path = resolve_model(model_name)
    base = re.sub(r"[.-](?:Q\d[^.]*|IQ\d[^.]*|BF16|F16|F32)$", "",
                  Path(model_name).stem, flags=re.IGNORECASE)
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


def _composite_rgba_white(source):
    """Composite a channel-first RGBA tensor without changing the caller's IMAGE."""
    alpha = source[:, 3:4].float().nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
    rgb = source[:, :3].float().clone().nan_to_num_(nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
    return rgb.sub_(1).mul_(alpha).add_(1)


def _resize_rgba_white(source, target):
    """Composite before filtering, keeping large RGBA copies bounded by strips."""
    height, width = source.shape[-2:]
    target_height, target_width = target
    if (height, width) == target:
        return _composite_rgba_white(source)
    if width / target_width >= height / target_height:
        strip_rows = max(1, MAX_VISUAL_PIXELS // width)
        intermediate = torch.empty((1, 3, height, target_width),
                                   dtype=torch.float32, device=source.device)
        for start in range(0, height, strip_rows):
            end = min(height, start + strip_rows)
            strip = _composite_rgba_white(source[:, :, start:end, :])
            intermediate[:, :, start:end, :] = interpolate(
                strip, size=(end - start, target_width), mode="bilinear",
                align_corners=False, antialias=True)
    else:
        strip_columns = max(1, MAX_VISUAL_PIXELS // height)
        intermediate = torch.empty((1, 3, target_height, width),
                                   dtype=torch.float32, device=source.device)
        for start in range(0, width, strip_columns):
            end = min(width, start + strip_columns)
            strip = _composite_rgba_white(source[:, :, :, start:end])
            intermediate[:, :, :, start:end] = interpolate(
                strip, size=(target_height, end - start), mode="bilinear",
                align_corners=False, antialias=True)
    if intermediate.shape[-2:] == target:
        return intermediate
    return interpolate(intermediate, size=target, mode="bilinear",
                       align_corners=False, antialias=True)


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
        target = (height, width)
        if width * height > MAX_VISUAL_PIXELS or max(width, height) > MAX_VISUAL_SIDE:
            scale = min(math.sqrt(MAX_VISUAL_PIXELS / (width * height)),
                        MAX_VISUAL_SIDE / max(width, height))
            target = (max(1, int(height * scale)), max(1, int(width * scale)))
        if source.shape[1] == 4:
            source = _resize_rgba_white(source, target)
        elif target != (height, width):
            # Sanitize before interpolation: one non-finite RGB source pixel
            # otherwise contaminates multiple output pixels while filtering.
            clean = source.float().nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
            source = interpolate(clean, size=target, mode="bilinear", align_corners=False,
                                 antialias=True)
        scaled = source[0].permute(1, 2, 0).float().nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        pixels = np.clip(scaled.cpu().numpy() * 255.0,
                         0, 255).astype(np.uint8)
        image = Image.fromarray(pixels)
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
    # Match malformed case/spacing variants too, then require exact canonical
    # tags below. Otherwise <IMAGE2> can bypass the image-count contract.
    references = set(re.findall(r"<\s*image[^>]*>", prose, flags=re.IGNORECASE))
    expected = {f"<image{i}>" for i in range(1, image_count + 1)} if image_count >= 2 else set()
    if references != expected and not (image_count == 1 and references == {"<image1>"}):
        allowed = "[] or ['<image1>']" if image_count == 1 else str(sorted(expected))
        detail = (f"image references in rewritten_prompt are {sorted(references)}, "
                  f"expected {allowed} for {task} with {image_count} input image(s)")
        if "<image>" in references:
            detail += "; <image> is an invalid unnumbered placeholder"
        raise ValueError(detail)


def normalize_single_image_references(prompt):
    """Number an unambiguous bare tag; preserve explicit and quoted references."""
    return replace_unquoted(re.compile(r"<image>"), prompt,
                            lambda _match, _masked: "<image1>")


def validate_language(answer, output_language, protected_literals=None):
    if output_language == "auto":
        return
    prose = strip_quoted_literals(answer["rewritten_prompt"], protected_literals)
    if _OTHER_SCRIPT.search(prose):
        raise ValueError("rewritten descriptive prose contains non-target writing scripts")
    contains_han = bool(_HAN.search(prose))
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
_CONTEXT_BACKDROP = re.compile(
    r"\b(?:against|in\s+front\s+of|on)\s+(?:a|an|the)\s+"
    r"(?:(?!with\b|and\b|that\b|which\b|wearing\b)[a-z-]+\s+){1,6}background\b", re.I)
_BACKGROUND_ONLY_SENTENCE = re.compile(
    r"^\s*(?:(?:the|this)\s+)?background\s+(?:is|appears|looks)\s+"
    r"(?:(?:a|an|the)\s+)?"
    r"(?:(?:pure|plain|solid|warm|light|dark|pale|soft|muted|bright|clean|neutral|"
    r"simple|empty|white|black|grey|gray|beige|brown|blue|red|green|yellow|brick|"
    r"stone|wooden|checkerboard)\s+){0,4}"
    r"(?:surface|wall|backdrop|color|colour|gradient|pattern|field|canvas|white|"
    r"black|grey|gray|beige|brown|blue|red|green|yellow)\s*[.!?]?\s*$|"
    r"^\s*(?:画面|图像|图片)?背景(?:是|为|呈)(?:纯白|白色|黑色|灰色|米色|"
    r"棋盘格|灰白棋盘格)\s*[。！？]?\s*$", re.I)


def normalize_opaque_background_phrases(prompt, protected_literals=None):
    def replace(match, masked):
        prefix = masked[max(0, match.start() - 18):match.start()]
        if re.search(r"\b(?:no|without|not)\s+(?:a\s+)?$|(?:不要|没有|不含|无)$", prefix, re.I):
            return match.group()
        return "透明背景" if re.search(r"[\u4e00-\u9fff]", match.group()) else "transparent background"

    prompt, replacements = replace_unquoted(_OPAQUE_BACKDROP, prompt, replace, protected_literals)
    def replace_context(match, _):
        if re.search(r"\btransparent\s+background$", match.group(), re.I):
            return match.group()
        return "against a transparent background"

    prompt, contextual = replace_unquoted(_CONTEXT_BACKDROP, prompt,
                                          replace_context,
                                          protected_literals)
    return prompt, replacements + contextual


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
        for match in re.finditer(r"\bshadow\b.{0,80}\bground\b|阴影.{0,40}地面", prose, re.I):
            prefix = prose[max(0, match.start() - 20):match.start()]
            if not re.search(r"\b(?:no|without|not)\s+$|(?:没有|无|不要)$", prefix, re.I):
                errors.append("prompt describes a ground shadow")
                break
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
    # Keep original slices: rejoining split text would insert spaces into
    # decimals (1.5) and exact image text ("SALE!TODAY"). Quoted punctuation
    # is masked so it cannot become a sentence boundary.
    masked = mask_quoted_literals(prompt)
    boundaries = []
    for match in re.finditer(r"[.!?。！？]", masked):
        end = match.end()
        while end < len(prompt) and prompt[end].isspace():
            end += 1
        if not boundaries or end > boundaries[-1]:
            boundaries.append(end)
    if not boundaries or boundaries[-1] != len(prompt):
        boundaries.append(len(prompt))
    start = 0
    for end in boundaries:
        sentence = prompt[start:end]
        start = end
        descriptive = mask_quoted_literals(sentence, protected_literals)
        has_backdrop = re.search(r"\bbackground\b|背景", descriptive, re.I)
        safe = transparent_background_clause(descriptive) if has_backdrop else False
        if (has_backdrop and not safe and _BACKGROUND_ONLY_SENTENCE.fullmatch(descriptive)
                and not quoted_literals(sentence, protected_literals)):
            removed += 1
        else:
            kept.append(sentence)
    return "".join(kept).strip(), removed


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
            raise FileNotFoundError(
                f"QWEN_PE_LLAMA_SERVER points to a missing executable: {path}. "
                "See https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8/"
                "blob/main/README.md#install-llama-server-windows")
        matches = list((ROOT / "runtime").rglob("llama-server.exe"))
        if not matches:
            raise FileNotFoundError(
                "llama-server.exe is missing. Install the two official llama.cpp Windows ZIPs "
                "into this node's runtime/llama-b11068/ directory, or set "
                "QWEN_PE_LLAMA_SERVER to its full path. Instructions: "
                "https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8/"
                "blob/main/README.md#install-llama-server-windows")
        if len(matches) != 1:
            raise RuntimeError(
                "Multiple llama-server.exe files found under runtime/. Set "
                "QWEN_PE_LLAMA_SERVER to the one you want to use.")
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

    def start(self, model, mmproj, context, gpu_layers,
              startup_timeout=DEFAULT_STARTUP_TIMEOUT):
        if (isinstance(startup_timeout, bool) or not isinstance(startup_timeout, int)
                or not 30 <= startup_timeout <= 7200):
            raise ValueError("startup_timeout must be an integer between 30 and 7200 seconds")
        with self.lock:
            profile = (file_signature(model), file_signature(mmproj) if mmproj else None,
                       context, gpu_layers)
            if self.process is not None and self.process.poll() is None and self.profile == profile:
                try:
                    with LOCAL_OPENER.open(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
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
                started_at = time.monotonic()
                deadline = started_at + startup_timeout
                next_report = started_at + 30
                last_health_error = "no successful health response"
                while time.monotonic() < deadline:
                    if check_interrupt:
                        check_interrupt()
                    if self.process.poll() is not None:
                        raise RuntimeError(f"llama-server exited with {self.process.returncode}; see {log_path}")
                    try:
                        with LOCAL_OPENER.open(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                            if response.status == 200:
                                return
                            last_health_error = f"HTTP {response.status}"
                    except (URLError, TimeoutError, HTTPError) as exc:
                        last_health_error = str(exc)
                    now = time.monotonic()
                    if now >= next_report:
                        print(f"[Qwen PE T8] Loading {model.name}: waiting for llama-server "
                              f"({now - started_at:.0f}/{startup_timeout} seconds); see {log_path}",
                              flush=True)
                        next_report = now + 30
                    time.sleep(min(0.5, max(0, deadline - now)))
                raise TimeoutError(
                    f"llama-server did not become healthy within {startup_timeout} seconds. "
                    f"Last health check: {last_health_error}. See {log_path}. "
                    "If the log still shows model loading, increase the node's startup_timeout; "
                    "if it shows an error, fix that error first.")
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
            if not images:
                reference_rule = ("There are no input images. Do not write <image>, "
                                  "<image1>, or any other image-reference tag in rewritten_prompt; "
                                  "describe only the finished image.")
            elif len(images) >= 2:
                tags = "、".join(f"<image{i}>" for i in range(1, len(images) + 1))
                reference_rule = (f"The rewritten_prompt must explicitly use every image tag "
                                  f"from {tags}, each for its matching input image; omit none. "
                                  "Do not use the unnumbered <image> tag.")
            else:
                reference_rule = ("There is one input image. Refer to it naturally by default. "
                                  "If rewritten_prompt uses an image tag, only <image1> is valid; "
                                  "never use the unnumbered <image> placeholder.")
            system += "\n\nRuntime image-reference rule: " + reference_rule
            if images:
                system += ("\n\nVisible text in reference images: Keep legible titles, signs, "
                           "advertisements, and labels verbatim inside quotation marks. "
                           "Write the surrounding description in the selected output language; "
                           "do not translate the visible lettering.")
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
                "max_tokens": 16256 if task == "t2i" else 24000,
                "seed": seed,
                "chat_template_kwargs": {"enable_thinking": True},
                "stream": False,
            }
            first_error = None
            truncation_retry = False
            for attempt in range(2):
                result = self._post_completion(payload, timeout)
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    usage = result.get("usage") or {}
                    counts = ", ".join(f"{key}={usage[key]}" for key in
                                       ("prompt_tokens", "completion_tokens") if key in usage)
                    context = self.profile[2] if self.profile else None
                    limits = f"max_tokens={payload['max_tokens']}"
                    if context:
                        limits += f", context={context}"
                    if counts:
                        limits += f", {counts}"
                    error = f"model response reached the generation or context limit ({limits})"
                    if attempt:
                        suffix = ("; retry without thinking was also truncated" if truncation_retry
                                  else "; format-retry response was truncated")
                        raise ValueError(error + suffix)
                    first_error = error
                    truncation_retry = True
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
                    payload["messages"][0]["content"] = (
                        system + "\n\nThe previous response was truncated. Return only the final "
                        "JSON object with the required fields; do not include reasoning or commentary.")
                    continue
                raw = choice["message"].get("content") or ""
                try:
                    answer = parse_answer(raw, task, len(images))
                    # Remember lettering recognized in this rewrite even if its
                    # surrounding words change during translation or cleanup.
                    answer_literals = exact_literals | set(quoted_literals(
                        answer["rewritten_prompt"], exact_literals))
                    normalized_single_image_tags = 0
                    if task == "edit" and len(images) == 1:
                        answer["rewritten_prompt"], normalized_single_image_tags = (
                            normalize_single_image_references(answer["rewritten_prompt"]))
                    removed_background_sentences = 0
                    if transparent_rgba:
                        answer["rewritten_prompt"], normalized_background_phrases = (
                            normalize_opaque_background_phrases(answer["rewritten_prompt"], answer_literals))
                        answer["rewritten_prompt"], removed_background_sentences = (
                            remove_opaque_background_sentences(answer["rewritten_prompt"], answer_literals))
                        answer["rewritten_prompt"], normalized_margin_phrases = (
                            normalize_transparent_margins(answer["rewritten_prompt"], answer_literals))
                    else:
                        normalized_background_phrases = 0
                        normalized_margin_phrases = 0
                    validate_references(answer, task, len(images), answer_literals)
                    translation_fallback = False
                    translation_usage = None
                    try:
                        validate_language(answer, output_language, answer_literals)
                    except ValueError:
                        if output_language == "auto":
                            raise
                        translation_fallback = True
                        source_prose = answer["rewritten_prompt"]
                        correction = None
                        for translation_attempt in range(3):
                            try:
                                translated, translation_usage = self._translate_prose(
                                    source_prose, output_language, seed, timeout,
                                    correction=correction,
                                    temperature=0.0 if translation_attempt == 0 else 0.3,
                                    protected_literals=answer_literals)
                                normalized_more_tags = 0
                                normalized_more_backdrop = 0
                                removed_more = 0
                                normalized_more = 0
                                if task == "edit" and len(images) == 1:
                                    translated, normalized_more_tags = normalize_single_image_references(translated)
                                if transparent_rgba:
                                    translated, normalized_more_backdrop = (
                                        normalize_opaque_background_phrases(translated, answer_literals))
                                    translated, removed_more = (
                                        remove_opaque_background_sentences(translated, answer_literals))
                                    translated, normalized_more = (
                                        normalize_transparent_margins(translated, answer_literals))
                                candidate = {**answer, "rewritten_prompt": translated}
                                validate_references(candidate, task, len(images), answer_literals)
                                validate_language(candidate, output_language, answer_literals)
                                validate_mode(candidate, aspect_ratio, transparent_rgba, answer_literals)
                            except ValueError as exc:
                                if translation_attempt == 2:
                                    raise
                                correction = str(exc)
                            else:
                                answer = candidate
                                normalized_single_image_tags += normalized_more_tags
                                normalized_background_phrases += normalized_more_backdrop
                                removed_background_sentences += removed_more
                                normalized_margin_phrases += normalized_more
                                break
                    validate_mode(answer, aspect_ratio, transparent_rgba, answer_literals)
                    return answer, {"finish_reason": choice.get("finish_reason"),
                                    "usage": result.get("usage", {}),
                                    "format_retries": attempt, "first_format_error": first_error,
                                    "truncation_retry": truncation_retry,
                                    "normalized_single_image_tags": normalized_single_image_tags,
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
                               "aspect-ratio, and image-reference constraints. " +
                               reference_rule + " Return only the final JSON object, "
                               "without reasoning or commentary.")
                    payload["messages"][0]["content"] = system
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
            raise AssertionError("unreachable")

    def _translate_prose(self, prose, output_language, seed, timeout,
                         correction=None, temperature=0.0, protected_literals=None):
        masked_prose, literals, prefix = _protect_translation_literals(prose, protected_literals)
        english = output_language == "English"
        example_image = f"{prefix}IMAGE1_0__"
        example_title = f"{prefix}1__"
        example_en = (f"Brighten {example_image} while keeping the title {example_title}. "
                      "Use a red background.")
        example_zh = f"调亮{example_image}，保留标题{example_title}，背景为红色。"
        if english:
            translation_system = (
                "Translate the input prose faithfully into English. Do not expand or rewrite it. "
                "Preserve every protected placeholder in order. Translate ordinary descriptive "
                "phrases inside quotation marks too. Return JSON with only translated_prose.")
            example_source, example_target = example_zh, example_en
        else:
            translation_system = (
                "把输入文字忠实翻译成简体中文，不得改写、扩写或添加视觉事实。"
                "普通描述性引号内的文字也要翻译。保持所有受保护占位符原样和顺序，"
                "占位符、RGBA、alpha、2D、3D 之外的描述不得保留英文单词。"
                "只返回 JSON 对象，唯一字段为 translated_prose。")
            example_source, example_target = example_en, example_zh
        string_schema = {"type": "string", "minLength": 1}
        if literals:
            # The sentinel is absent from the source. Only the fixed markers may
            # contain it, so constrained decoding cannot drop or duplicate them.
            gap = f"[^{re.escape(prefix[0])}]*"
            string_schema["pattern"] = "^" + gap.join(
                [""] + [re.escape(marker) for marker in literals] + [""]) + "$"
        request_text = masked_prose
        if correction:
            request_text = (
                "上次译文未通过校验，请重新忠实翻译原文。错误：" + correction[:500] + "\n\n"
                if not english else
                "The previous translation failed validation. Retranslate the original faithfully. "
                "Error: " + correction[:500] + "\n\n"
            ) + request_text
        payload = {
            "model": "qwen-pe",
            "messages": [{"role": "system", "content": translation_system},
                         {"role": "user", "content": example_source},
                         {"role": "assistant", "content": json.dumps(
                             {"translated_prose": example_target}, ensure_ascii=False)},
                         {"role": "user", "content": request_text}],
            "temperature": temperature, "top_p": 0.9, "max_tokens": 8192,
            "seed": seed, "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
            # Qwen still emits empty thinking delimiters when thinking is off.
            # Parse them separately so they cannot conflict with JSON decoding.
            "reasoning_format": "auto",
            "response_format": {"type": "json_object", "schema": {
                "type": "object", "properties": {"translated_prose": string_schema},
                "required": ["translated_prose"], "additionalProperties": False}},
        }
        result = self._post_completion(payload, timeout)
        choice = result["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("translation fallback was truncated")
        raw = choice["message"].get("content") or ""
        raw = re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", raw).strip()
        if raw.startswith("```json\n") and raw.endswith("```"):
            raw = raw[8:-3].strip()
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("translation fallback did not return a translated_prose JSON object") from exc
        if (not isinstance(response, dict) or set(response) != {"translated_prose"}
                or not isinstance(response["translated_prose"], str)):
            raise ValueError("translation fallback must return exactly one translated_prose string")
        translated = response["translated_prose"].strip()
        if not translated:
            raise ValueError("translation fallback returned empty text")
        if literals:
            if not re.fullmatch(string_schema["pattern"], translated):
                raise ValueError("translation must preserve every protected text marker exactly once and in order")
            # Tolerate a model wrapping a marker in quotes; restore the original
            # complete quoted span instead of producing doubled quotation marks.
            translated = _QUOTED_LITERAL.sub(
                lambda match: match.group()[1:-1].strip()
                if match.group()[1:-1].strip() in literals else match.group(), translated)
            marker_pattern = re.compile(re.escape(prefix) + r"(?:IMAGE\d+_)?\d+__")
            translated = marker_pattern.sub(lambda match: literals[match.group()], translated)
        if prefix[0] in translated:
            raise ValueError("translation returned an invalid protected text marker")
        if quoted_literals(translated, protected_literals) != quoted_literals(prose, protected_literals):
            raise ValueError("translation changed the order or content of exact quoted image text")
        return translated, result.get("usage", {})

    def _post_completion(self, payload, timeout):
        """Wait for llama.cpp while honoring ComfyUI's interrupt flag."""
        request = Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                          data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        outcome = {}

        def perform():
            try:
                with LOCAL_OPENER.open(request, timeout=timeout) as response:
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
