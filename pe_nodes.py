import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch

from .pe_runtime import (DEFAULT_EDIT, DEFAULT_T2I, SERVER, local_models,
                         pick_mmproj, prepare_images, resolve_model)


ASPECT_RATIOS = ["auto", "1:1", "1:2", "2:3", "3:4", "4:5", "16:9",
                 "21:9", "9:21", "5:4", "4:3", "2:1"]


def _resolved_language(selection, rewritten_prompt):
    if selection == "中文":
        return "zh"
    if selection == "English":
        return "en"
    prose = re.sub(r'"[^\"]*"|“[^”]*”|‘[^’]*’', "", rewritten_prompt)
    return "zh" if re.search(r"[\u4e00-\u9fff]", prose) else "en"


def _format_prompt(prompt, transparent_rgba, language):
    pieces = []
    if transparent_rgba:
        pieces.append("这是一张带有透明度的RGBA图像。" if language == "zh"
                      else "This is an RGBA image with transparency. ")
    body = prompt.strip()
    if transparent_rgba and body and body[-1] not in ".!?。！？":
        body += "。" if language == "zh" else "."
    pieces.append(body)
    if transparent_rgba:
        pieces.append("该图像具有alpha通道，背景是透明的。" if language == "zh"
                      else " The image has an alpha channel, and the background is transparent.")
    return "".join(pieces)


def _choices(vision=False):
    names = list(local_models(vision))
    if vision:
        return ["Auto"] + names
    return names or ["(no local GGUF found)"]


class QwenPERewrite:
    @classmethod
    def INPUT_TYPES(cls):
        models = _choices()
        images = {f"image_{i}": ("IMAGE",) for i in range(1, 11)}
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "task": (["auto", "t2i", "edit"], {"default": "auto"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "auto",
                                                 "tooltip": "auto 使用模型建议；指定比例将覆盖模型比例并控制 Canvas。"}),
                "output_language": (["auto", "中文", "English"], {"default": "auto",
                                                                    "tooltip": "控制改写描述的语言；图内原文保留用户指定文字。"}),
                "transparent_rgba": ("BOOLEAN", {"default": False,
                                                    "tooltip": "将 RGBA、alpha 通道和透明背景要求加入最终提示词。"}),
                "t2i_model": (models, {"default": DEFAULT_T2I if DEFAULT_T2I in models else models[0]}),
                "edit_model": (models, {"default": DEFAULT_EDIT if DEFAULT_EDIT in models else models[0]}),
                "vision_model": (_choices(True), {"default": "Auto"}),
                "model_lifetime": (["after_run", "keep_loaded"], {"default": "after_run"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF}),
            },
            "optional": images,
        }

    RETURN_TYPES = ("STRING", "PE_RESULT", "STRING")
    RETURN_NAMES = ("rewritten_prompt", "pe_result", "diagnostics")
    FUNCTION = "rewrite"
    CATEGORY = "Qwen Image 2.1/Prompt Rewrite"

    @classmethod
    def IS_CHANGED(cls, task, t2i_model, edit_model, vision_model, **kwargs):
        fingerprint = hashlib.sha256()
        root = Path(__file__).resolve().parent
        fingerprint.update((root / "pe_nodes.py").read_bytes())
        fingerprint.update((root / "pe_runtime.py").read_bytes())
        for template in ("system_prompt_t2i.txt", "system_prompt_edit.txt"):
            fingerprint.update((root / "prompts" / template).read_bytes())
        effective_task = ("edit" if any(kwargs.get(f"image_{i}") is not None for i in range(1, 11))
                          else "t2i") if task == "auto" else task
        names = [t2i_model] if effective_task == "t2i" else [edit_model]
        for name in names:
            path = resolve_model(name)
            stat = path.stat()
            fingerprint.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        if effective_task == "edit":
            path = pick_mmproj(edit_model, vision_model)
            stat = path.stat()
            fingerprint.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return fingerprint.hexdigest()

    def rewrite(self, user_prompt, task, aspect_ratio, output_language, transparent_rgba,
                t2i_model, edit_model, vision_model,
                model_lifetime, seed, **kwargs):
        if not user_prompt.strip():
            raise ValueError("Enter a text instruction; image-only requests need an explicit editing goal")
        present = sorted((int(key.split("_")[-1]), value) for key, value in kwargs.items() if value is not None)
        indices = [index for index, _ in present]
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError("Connect images consecutively from image_1; gaps would change <imageN> references")
        images = [value for _, value in present]
        if len(images) > 10:
            raise ValueError("At most 10 reference images are supported")
        actual_task = ("edit" if images else "t2i") if task == "auto" else task
        if actual_task == "edit" and not images:
            raise ValueError("edit requires at least one image")
        if actual_task == "t2i" and images:
            raise ValueError("t2i cannot receive images; use edit for image-conditioned generation")
        model_name = t2i_model if actual_task == "t2i" else edit_model
        model = resolve_model(model_name)
        mmproj = pick_mmproj(model_name, vision_model) if actual_task == "edit" else None
        encoded, dimensions = prepare_images(images)
        image_fingerprints = [hashlib.sha256(value.encode("ascii")).hexdigest() for value in encoded]
        context = 16384 if not images else (32768 if len(images) <= 5 else 65536)
        try:
            import comfy.model_management as memory
            memory.free_memory(12 * 1024**3, memory.get_torch_device())
            memory.soft_empty_cache()
        except ImportError:
            pass
        started = time.monotonic()
        with SERVER.lock:
            try:
                SERVER.start(model, mmproj, context, 99)
                answer, info = SERVER.complete(actual_task, user_prompt, encoded, seed, 900,
                                               output_language, aspect_ratio, transparent_rgba)
            finally:
                if model_lifetime == "after_run":
                    SERVER.stop()
        model_wh_ratio = answer["wh_ratio"]
        model_ratio_follow = answer.get("ratio_follow", "")
        if aspect_ratio != "auto":
            answer["wh_ratio"] = aspect_ratio
            answer["ratio_follow"] = ""
        language = _resolved_language(output_language, answer["rewritten_prompt"])
        final_prompt = _format_prompt(answer["rewritten_prompt"], transparent_rgba, language)
        template_name = "system_prompt_t2i.txt" if actual_task == "t2i" else "system_prompt_edit.txt"
        template_hash = hashlib.sha256((Path(__file__).resolve().parent / "prompts" / template_name).read_bytes()).hexdigest()
        result = {
            "task": actual_task,
            "rewritten_prompt": final_prompt,
            "wh_ratio": answer["wh_ratio"],
            "ratio_follow": answer.get("ratio_follow", ""),
            "model_wh_ratio": model_wh_ratio,
            "model_ratio_follow": model_ratio_follow,
            "selected_aspect_ratio": aspect_ratio,
            "output_language": language,
            "transparent_rgba": transparent_rgba,
            "image_dimensions": dimensions,
            "image_fingerprints": image_fingerprints,
            "model": model.name,
            "mmproj": mmproj.name if mmproj else "",
            "system_prompt_sha256": template_hash,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "seed": seed,
            "finish_reason": info["finish_reason"],
            "usage": info["usage"],
            "format_retries": info.get("format_retries", 0),
            "first_format_error": info.get("first_format_error"),
            "removed_background_sentences": info.get("removed_background_sentences", 0),
            "normalized_background_phrases": info.get("normalized_background_phrases", 0),
            "normalized_margin_phrases": info.get("normalized_margin_phrases", 0),
            "translation_fallback": info.get("translation_fallback", False),
            "translation_usage": info.get("translation_usage"),
        }
        diagnostics = json.dumps({key: value for key, value in result.items() if key != "rewritten_prompt"},
                                 ensure_ascii=False)
        return result["rewritten_prompt"], result, diagnostics


class QwenPECanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pe_result": ("PE_RESULT",),
            "resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
            "follow_input_size": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("INT", "INT", "LATENT", "STRING")
    RETURN_NAMES = ("width", "height", "latent", "ratio_source")
    FUNCTION = "canvas"
    CATEGORY = "Qwen Image 2.1/Prompt Rewrite"

    def canvas(self, pe_result, resolution, follow_input_size):
        follow = pe_result["ratio_follow"]
        if follow:
            index = int(follow.removeprefix("<image").removesuffix(">")) - 1
            width, height = pe_result["image_dimensions"][index]
            source = follow
        else:
            left, right = map(int, pe_result["wh_ratio"].split(":"))
            ratio = left / right
            width = math.sqrt(resolution * resolution * ratio)
            height = math.sqrt(resolution * resolution / ratio)
            source = pe_result["wh_ratio"]
        if follow and not follow_input_size:
            ratio = width / height
            width = math.sqrt(resolution * resolution * ratio)
            height = math.sqrt(resolution * resolution / ratio)
        width = max(16, round(width / 16) * 16)
        height = max(16, round(height / 16) * 16)
        import comfy.model_management as memory
        latent = torch.zeros([1, 64, height // 16, width // 16], device=memory.intermediate_device())
        return width, height, {"samples": latent}, source


class QwenPEUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pe_result": ("PE_RESULT",)}}

    RETURN_TYPES = ("PE_RESULT",)
    RETURN_NAMES = ("pe_result",)
    FUNCTION = "unload"
    CATEGORY = "Qwen Image 2.1/Prompt Rewrite"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def unload(self, pe_result):
        SERVER.stop()
        return (pe_result,)


class QwenPEModelList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"refresh": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("local_models",)
    FUNCTION = "list_models"
    CATEGORY = "Qwen Image 2.1/Prompt Rewrite"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def list_models(self, refresh):
        models = local_models()
        vision = local_models(True)
        return (json.dumps({"models": list(models), "vision_models": list(vision)}, ensure_ascii=False, indent=2),)


NODE_CLASS_MAPPINGS = {
    "QwenPERewriteT8": QwenPERewrite,
    "QwenPECanvasT8": QwenPECanvas,
    "QwenPEUnloadT8": QwenPEUnload,
    "QwenPEModelListT8": QwenPEModelList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenPERewriteT8": "Qwen Image 2.1 PE Rewrite T8",
    "QwenPECanvasT8": "Qwen Image 2.1 PE Canvas T8",
    "QwenPEUnloadT8": "Qwen Image 2.1 PE Unload T8",
    "QwenPEModelListT8": "Qwen Image 2.1 PE Local Models T8",
}
