import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen_pe_test_package", ROOT / "__init__.py",
                                              submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from qwen_pe_test_package.pe_nodes import (ASPECT_RATIOS, QwenPECanvas, QwenPERewrite,
                                           _format_prompt, _resolved_language)
from qwen_pe_test_package.pe_runtime import (DEFAULT_MMPROJ, DEFAULT_T2I,
                                              remove_opaque_background_sentences,
                                              normalize_transparent_margins,
                                              normalize_opaque_background_phrases,
                                              validate_language, validate_mode)


class UserOptionTests(unittest.TestCase):
    def test_dropdowns_contain_requested_choices(self):
        required = QwenPERewrite.INPUT_TYPES()["required"]
        self.assertEqual(required["aspect_ratio"][0], ASPECT_RATIOS)
        self.assertEqual(required["output_language"][0], ["auto", "中文", "English"])
        self.assertEqual(required["transparent_rgba"][0], "BOOLEAN")

    def test_transparent_prompt_uses_selected_language_and_ratio(self):
        chinese = _format_prompt("一只蝴蝶。", True, "zh")
        english = _format_prompt("A butterfly.", True, "en")
        self.assertTrue(chinese.startswith("这是一张带有透明度的RGBA图像。"))
        self.assertTrue(chinese.endswith("该图像具有alpha通道，背景是透明的。"))
        self.assertIn("RGBA image with transparency", english)
        self.assertIn("alpha channel, and the background is transparent", english)

    def test_explicit_language_takes_priority(self):
        self.assertEqual(_resolved_language("English", "中文指令"), "en")
        self.assertEqual(_resolved_language("中文", "English instruction"), "zh")
        self.assertEqual(_resolved_language("auto", "A blue coat on a bench."), "en")
        self.assertEqual(_resolved_language("auto", "一只蓝色外套。"), "zh")
        self.assertEqual(_resolved_language("auto", 'A poster with the title “春日新茶”.'), "en")
        self.assertEqual(_resolved_language("auto", "A poster with the title '春日新茶'."), "en")

    def test_chinese_prose_accepts_common_dimension_notation(self):
        validate_language({"rewritten_prompt": "一幅2D平面插画，主体带有3D浮雕效果。"}, "中文")
        with self.assertRaises(ValueError):
            validate_language({"rewritten_prompt": "一幅watercolor插画。"}, "中文")

    def test_transparent_cleanup_rejects_checkerboard_despite_unrelated_negation(self):
        prompt = ("一只金色蝴蝶。画面背景为灰白棋盘格，没有明显的外部投影。"
                  "主体有细致花纹。")
        cleaned, removed = remove_opaque_background_sentences(prompt)
        self.assertEqual(removed, 1)
        self.assertNotIn("棋盘格", cleaned)
        validate_mode({"rewritten_prompt": cleaned}, "4:5", True)
        with self.assertRaises(ValueError):
            validate_mode({"rewritten_prompt": prompt}, "4:5", True)

    def test_transparent_margins_are_not_opaque(self):
        prompt = "Keep the butterfly centered with even white margins on all four sides."
        with self.assertRaises(ValueError):
            validate_mode({"rewritten_prompt": prompt}, "3:4", True)
        normalized, count = normalize_transparent_margins(prompt)
        self.assertEqual(count, 1)
        self.assertIn("transparent margins on all four sides", normalized)
        validate_mode({"rewritten_prompt": normalized}, "3:4", True)

    def test_opaque_background_is_replaced_even_with_unrelated_no_background_clause(self):
        prompt = ("A butterfly on a pure white background. Center it against the pure white "
                  "background with no other objects and no background elements.")
        with self.assertRaises(ValueError):
            validate_mode({"rewritten_prompt": prompt}, "3:4", True)
        normalized, count = normalize_opaque_background_phrases(prompt)
        self.assertEqual(count, 2)
        self.assertNotIn("white background", normalized)
        validate_mode({"rewritten_prompt": normalized}, "3:4", True)

    def test_transparent_cleanup_preserves_exact_quoted_image_text(self):
        prompt = 'A poster reads "white background" in black letters.'
        normalized, count = normalize_opaque_background_phrases(prompt)
        self.assertEqual((normalized, count), (prompt, 0))
        cleaned, removed = remove_opaque_background_sentences(prompt)
        self.assertEqual((cleaned, removed), (prompt, 0))
        validate_mode({"rewritten_prompt": prompt}, "16:9", True)
        ratio_text = 'A poster reads "1:1" in large letters on a transparent background.'
        validate_mode({"rewritten_prompt": ratio_text}, "16:9", True)
        margins, count = normalize_transparent_margins('Print "white margins" on the poster.')
        self.assertEqual((margins, count), ('Print "white margins" on the poster.', 0))
        interleaved = 'Put white "SALE" background lettering on the shirt.'
        normalized, count = normalize_opaque_background_phrases(interleaved)
        self.assertEqual((normalized, count), (interleaved, 0))
        model_quoted_description = '"A butterfly on a white background"'
        with self.assertRaises(ValueError):
            validate_mode({"rewritten_prompt": model_quoted_description}, "4:5", True, set())
        validate_mode({"rewritten_prompt": model_quoted_description}, "4:5", True,
                      {"A butterfly on a white background"})

    def test_canvas_uses_selected_ratio_or_followed_reference(self):
        comfy = types.ModuleType("comfy")
        memory = types.ModuleType("comfy.model_management")
        memory.intermediate_device = lambda: "cpu"
        comfy.model_management = memory
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": memory}):
            canvas = QwenPECanvas()
            width, height, _, source = canvas.canvas(
                {"wh_ratio": "21:9", "ratio_follow": "", "image_dimensions": []}, 1024, True)
            self.assertEqual(source, "21:9")
            self.assertAlmostEqual(width / height, 21 / 9, delta=0.05)
            width, height, _, source = canvas.canvas(
                {"wh_ratio": "", "ratio_follow": "<image2>",
                 "image_dimensions": [[1024, 1024], [512, 384]]}, 1024, True)
            self.assertEqual((width, height, source), (512, 384, "<image2>"))
            width, height, _, source = canvas.canvas(
                {"wh_ratio": "", "ratio_follow": "<image1>",
                 "image_dimensions": [[8192, 8192]]}, 1024, True)
            self.assertEqual((width, height, source), (1024, 1024, "<image1>"))
            width, height, _, _ = canvas.canvas(
                {"wh_ratio": "", "ratio_follow": "<image1>",
                 "image_dimensions": [[8, 8]]}, 1024, True)
            self.assertEqual((width, height), (16, 16))
            with self.assertRaisesRegex(ValueError, "too extreme"):
                canvas.canvas({"wh_ratio": "1:999", "ratio_follow": "",
                               "image_dimensions": []}, 1024, True)

    def test_auto_cache_handles_unavailable_unused_vision_pair(self):
        value = QwenPERewrite.IS_CHANGED(task="auto", t2i_model=DEFAULT_T2I,
                                         edit_model=DEFAULT_T2I, vision_model=DEFAULT_MMPROJ)
        self.assertEqual(len(value), 64)
        linked_value = QwenPERewrite.IS_CHANGED(task="auto", t2i_model=DEFAULT_T2I,
                                                edit_model=DEFAULT_T2I, vision_model=DEFAULT_MMPROJ,
                                                image_1=None)
        self.assertEqual(value, linked_value)

    def test_auto_cache_includes_edit_model_when_linked_image_is_unresolved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            t2i, edit, vision = (root / name for name in ("t2i.gguf", "edit.gguf", "vision.gguf"))
            for path in (t2i, edit, vision):
                path.write_bytes(b"a")
            def resolve(name):
                return {"t2i": t2i, "edit": edit}[name]
            with (patch("qwen_pe_test_package.pe_nodes.resolve_model", side_effect=resolve),
                  patch("qwen_pe_test_package.pe_nodes.pick_mmproj", return_value=vision)):
                inputs = {"task": "auto", "t2i_model": "t2i", "edit_model": "edit",
                          "vision_model": "Auto", "image_1": None}
                first = QwenPERewrite.IS_CHANGED(**inputs)
                edit.write_bytes(b"changed")
                second = QwenPERewrite.IS_CHANGED(**inputs)
                self.assertNotEqual(first, second)
                vision.write_bytes(b"changed")
                third = QwenPERewrite.IS_CHANGED(**inputs)
                self.assertNotEqual(second, third)


if __name__ == "__main__":
    unittest.main()
