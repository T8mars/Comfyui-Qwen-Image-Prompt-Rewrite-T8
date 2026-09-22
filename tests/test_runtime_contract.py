import os
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from pe_runtime import (DEFAULT_EDIT, DEFAULT_T2I, DEFAULT_MMPROJ,
                        LocalServer, file_signature, local_models,
                        normalize_single_image_references, parse_answer, pick_mmproj,
                        prepare_images, quoted_literals,
                        remove_opaque_background_sentences, validate_language, validate_mode,
                        validate_references)


class RuntimeContractTests(unittest.TestCase):
    def test_parse_t2i_json_after_thinking_text(self):
        raw = '<think>the example {"wrong": 1} is invalid</think>\n{"rewritten_prompt":"a blue dog", "wh_ratio":"3:2"}'
        self.assertEqual(parse_answer(raw, "t2i", 0)["wh_ratio"], "3:2")

    def test_parse_rejects_thinking_only_and_nested_wrapper(self):
        draft = '<think>{"rewritten_prompt":"draft blue dog","wh_ratio":"1:1"}</think>'
        with self.assertRaisesRegex(ValueError, "valid final JSON"):
            parse_answer(draft, "t2i", 0)
        wrapped = '{"answer":{"rewritten_prompt":"blue dog","wh_ratio":"1:1"}}'
        with self.assertRaisesRegex(ValueError, "wrong answer fields"):
            parse_answer(wrapped, "t2i", 0)
        malformed_wrapper = '{"answer":{"rewritten_prompt":"blue dog","wh_ratio":"1:1"}'
        with self.assertRaises(ValueError):
            parse_answer(malformed_wrapper, "t2i", 0)
        with self.assertRaisesRegex(ValueError, "incomplete thinking"):
            parse_answer('<think>{"rewritten_prompt":"draft","wh_ratio":"1:1"}', "t2i", 0)
        literal = '{"rewritten_prompt":"A poster says \\"<think>HELLO</think>\\".","wh_ratio":"1:1"}'
        self.assertIn("<think>HELLO</think>", parse_answer(literal, "t2i", 0)["rewritten_prompt"])

    def test_parse_rejects_extra_field_and_bad_image_reference(self):
        with self.assertRaises(ValueError):
            parse_answer('{"rewritten_prompt":"x","wh_ratio":"1:1","extra":"y"}', "t2i", 0)
        with self.assertRaises(ValueError):
            parse_answer('{"rewritten_prompt":"x","wh_ratio":"","ratio_follow":"<image11>"}', "edit", 10)
        with self.assertRaises(ValueError):
            parse_answer('{"rewritten_prompt":"x","wh_ratio":"1:1","ratio_follow":"<image1>"}', "edit", 1)

    def test_reference_contract(self):
        validate_references({"rewritten_prompt": "Use <image1> and <image2>."}, "edit", 2)
        nine_tags = " ".join(f"<image{i}>" for i in range(1, 10))
        validate_references({"rewritten_prompt": nine_tags}, "edit", 9)
        with self.assertRaises(ValueError):
            validate_references({"rewritten_prompt": nine_tags.replace("<image9>", "")}, "edit", 9)
        validate_references({"rewritten_prompt": "Preserve the original image."}, "edit", 1)
        validate_references({"rewritten_prompt": "Preserve <image1>."}, "edit", 1)
        validate_references({"rewritten_prompt": 'Print "<image1>" on the poster.'}, "t2i", 0)
        validate_references({"rewritten_prompt": "Print '<image1>' on the poster."}, "t2i", 0)
        validate_references({"rewritten_prompt": 'Blend "<image1>" with "<image2>".'},
                            "edit", 2, set())
        validate_references({"rewritten_prompt": 'Print "<image1>" on the poster.'},
                            "t2i", 0, {"<image1>"})
        with self.assertRaises(ValueError):
            validate_references({"rewritten_prompt": "Use <image>."}, "edit", 1)
        with self.assertRaises(ValueError):
            validate_references({"rewritten_prompt": "Use <image1>."}, "edit", 2)
        with self.assertRaises(ValueError):
            validate_references({"rewritten_prompt": "Use <image01> and <image2>."}, "edit", 2)

    def test_ratio_and_transparency_conflicts_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_mode({"rewritten_prompt": "A butterfly in a square-format image with a dark background.",
                           "wh_ratio": "1:1", "ratio_follow": ""}, "21:9", True)
        with self.assertRaises(ValueError):
            validate_mode({"rewritten_prompt": "The background is a warm beige surface.",
                           "wh_ratio": "21:9", "ratio_follow": ""}, "21:9", True)
        validate_mode({"rewritten_prompt": "A butterfly isolated on a transparent background.",
                       "wh_ratio": "21:9", "ratio_follow": ""}, "21:9", True)

    def test_transparency_cleanup_removes_opaque_backdrop_only(self):
        prose, count = remove_opaque_background_sentences(
            "A golden butterfly has detailed wings. The background is a warm beige surface. "
            "Keep the butterfly centered. The image has a transparent background.")
        self.assertEqual(count, 1)
        self.assertIn("detailed wings", prose)
        self.assertNotIn("beige", prose)
        self.assertIn("transparent background", prose)

    def test_explicit_chinese_rejects_stray_english_but_preserves_literal_text(self):
        validate_language({"rewritten_prompt": "一张海报，标题写成\"FRESH BREAD\"。"}, "中文")
        validate_language({"rewritten_prompt": "一张海报，标题写成 'FRESH BREAD'。"}, "中文")
        for delimiters in ("「」", "『』", "«»"):
            prompt = f"一张海报，标题写成{delimiters[0]}FRESH BREAD{delimiters[1]}。"
            self.assertEqual(quoted_literals(prompt), ["FRESH BREAD"])
            validate_language({"rewritten_prompt": prompt}, "中文", set(quoted_literals(prompt)))
        validate_language({"rewritten_prompt": "A poster displays '你好' in red type."}, "English")
        validate_language({"rewritten_prompt": "Don't change the title '你好'."}, "English")
        with self.assertRaises(ValueError):
            validate_language({"rewritten_prompt": "一张 watercolor 海报。"}, "中文")
        with self.assertRaisesRegex(ValueError, "non-target writing scripts"):
            validate_language({"rewritten_prompt": "赤い猫の写真。"}, "中文", set())
        with self.assertRaisesRegex(ValueError, "non-target writing scripts"):
            validate_language({"rewritten_prompt": "빨간 고양이 사진."}, "English", set())

    def test_image_preparation_preserves_original_size_and_rejects_batch(self):
        image = torch.zeros((1, 1200, 1600, 3), dtype=torch.float32)
        encoded, dimensions = prepare_images([image])
        self.assertEqual(dimensions, [[1600, 1200]])
        self.assertTrue(encoded[0].startswith("data:image/png;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(encoded[0].split(",", 1)[1]))) as reduced:
            self.assertLessEqual(reduced.width * reduced.height, 1024 * 1024)
        bf16 = torch.zeros((1, 1200, 1200, 3), dtype=torch.bfloat16)
        bf16_encoded, bf16_dimensions = prepare_images([bf16])
        self.assertEqual(bf16_dimensions, [[1200, 1200]])
        self.assertTrue(bf16_encoded[0].startswith("data:image/png;base64,"))
        needle, original = prepare_images([torch.zeros((1, 1, 8192, 3))])
        self.assertEqual(original, [[8192, 1]])
        with Image.open(io.BytesIO(base64.b64decode(needle[0].split(",", 1)[1]))) as reduced:
            self.assertEqual(reduced.size, (4096, 1))
        very_long, _ = prepare_images([torch.zeros((1, 1, 1_200_000, 3))])
        with Image.open(io.BytesIO(base64.b64decode(very_long[0].split(",", 1)[1]))) as reduced:
            self.assertLessEqual(reduced.width * reduced.height, 1024 * 1024)
            self.assertLessEqual(max(reduced.size), 4096)
        with self.assertRaises(ValueError):
            prepare_images([torch.zeros((2, 100, 100, 3))])

    def test_rgba_downscale_does_not_bleed_invisible_rgb_into_visible_pixels(self):
        image = torch.zeros((1, 1, 8192, 4))
        image[0, 0, ::2, 3] = 1.0  # opaque black
        image[0, 0, 1::2, 0] = 1.0  # transparent red, invisible before filtering
        encoded, _ = prepare_images([image])
        with Image.open(io.BytesIO(base64.b64decode(encoded[0].split(",", 1)[1]))) as reduced:
            red, green, blue = reduced.convert("RGB").getpixel((2048, 0))
        self.assertLessEqual(max(red, green, blue) - min(red, green, blue), 1)
        self.assertTrue(110 <= red <= 145)
        vertical, _ = prepare_images([image.permute(0, 2, 1, 3)])
        with Image.open(io.BytesIO(base64.b64decode(vertical[0].split(",", 1)[1]))) as reduced:
            red, green, blue = reduced.convert("RGB").getpixel((0, 2048))
        self.assertLessEqual(max(red, green, blue) - min(red, green, blue), 1)
        tiny = torch.zeros((1, 1, 1, 4))
        tiny[0, 0, 0, 0] = 1.0
        untouched, _ = prepare_images([tiny])
        with Image.open(io.BytesIO(base64.b64decode(untouched[0].split(",", 1)[1]))) as reduced:
            self.assertEqual(reduced.convert("RGB").getpixel((0, 0)), (255, 255, 255))
        tiny[0, 0, 0, 3] = float("nan")
        cleaned, _ = prepare_images([tiny])
        with Image.open(io.BytesIO(base64.b64decode(cleaned[0].split(",", 1)[1]))) as reduced:
            self.assertEqual(reduced.convert("RGB").getpixel((0, 0)), (255, 255, 255))
        tiny[0, 0, 0, :3] = 1.0
        tiny[0, 0, 0, 3] = float("inf")
        cleaned, _ = prepare_images([tiny])
        with Image.open(io.BytesIO(base64.b64decode(cleaned[0].split(",", 1)[1]))) as reduced:
            self.assertEqual(reduced.convert("RGB").getpixel((0, 0)), (255, 255, 255))

    def test_model_discovery_includes_additional_local_gguf(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "another-Q4.gguf").write_bytes(b"stub")
            (folder / "another.mmproj.gguf").write_bytes(b"stub")
            with patch.dict(os.environ, {"QWEN_PE_MODEL_DIR": str(folder)}):
                self.assertIn("another-Q4.gguf", local_models())
                self.assertIn("another.mmproj.gguf", local_models(True))
                self.assertNotIn("another.mmproj.gguf", local_models())

    def test_known_vision_project_cannot_pair_with_t2i_weight(self):
        self.assertEqual(pick_mmproj(DEFAULT_EDIT, "Auto").name, DEFAULT_MMPROJ)
        with self.assertRaises(ValueError):
            pick_mmproj(DEFAULT_T2I, DEFAULT_MMPROJ)

    def test_auto_vision_model_uses_the_model_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = "Example.Q4_K_M.gguf"
            vision = "Example.mmproj-f16.gguf"
            for folder in ("A", "B"):
                (root / folder).mkdir()
                (root / folder / vision).write_bytes(b"vision")
            (root / "A" / name).write_bytes(b"model")
            with patch.dict(os.environ, {"QWEN_PE_MODEL_DIR": str(root)}):
                self.assertEqual(pick_mmproj("A/" + name, "Auto"), root / "A" / vision)

    def test_auto_vision_model_ignores_duplicate_name_in_other_root(self):
        with tempfile.TemporaryDirectory() as temp:
            model_root, override_root = Path(temp) / "model", Path(temp) / "override"
            model_root.mkdir()
            override_root.mkdir()
            model = model_root / "Example.Q4_K_M.gguf"
            vision = "Example.mmproj-f16.gguf"
            model.write_bytes(b"main")
            (model_root / vision).write_bytes(b"correct")
            (override_root / vision).write_bytes(b"other")
            with patch("pe_runtime.model_roots", return_value=iter([override_root, model_root])):
                self.assertEqual(pick_mmproj(model.name, "Auto"), model_root / vision)

    def test_bf16_edit_weight_pairs_with_vision_project(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "Qwen-Image-2.1-PE-I2I.BF16.gguf"
            vision = root / "Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf"
            model.write_bytes(b"model")
            vision.write_bytes(b"vision")
            with patch("pe_runtime.model_roots", side_effect=lambda: iter([root])):
                self.assertEqual(pick_mmproj(model.name, "Auto"), vision)
                self.assertEqual(pick_mmproj(model.name, vision.name), vision)

    def test_file_signature_changes_for_same_size_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            replacement = Path(temp) / "replacement.gguf"
            path.write_bytes(b"old")
            before = file_signature(path)
            original = path.stat()
            replacement.write_bytes(b"new")
            os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
            os.replace(replacement, path)
            self.assertNotEqual(file_signature(path), before)

    def test_server_cleans_up_after_failed_start_and_interrupt(self):
        class FakeProcess:
            def __init__(self, exited):
                self.exited = exited
                self.terminated = False
                self.returncode = 1 if exited else None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = 0

            def wait(self, timeout):
                return self.returncode

        with tempfile.TemporaryDirectory() as temp:
            import pe_runtime
            model = Path(temp) / "model.gguf"
            model.write_bytes(b"stub")
            exited = FakeProcess(True)
            with (patch.object(pe_runtime, "ROOT", Path(temp)),
                  patch.object(LocalServer, "binary", return_value=Path(temp) / "server"),
                  patch.object(pe_runtime.subprocess, "Popen", return_value=exited)):
                server = LocalServer()
                with self.assertRaisesRegex(RuntimeError, "exited"):
                    server.start(model, None, 1024, 0)
                self.assertIsNone(server.process)
                self.assertIsNone(server.profile)
                self.assertIsNone(server.log)

            interrupted = FakeProcess(False)
            comfy = types.ModuleType("comfy")
            memory = types.ModuleType("comfy.model_management")
            def interrupt():
                raise KeyboardInterrupt()
            memory.throw_exception_if_processing_interrupted = interrupt
            comfy.model_management = memory
            with (patch.object(pe_runtime, "ROOT", Path(temp)),
                  patch.object(LocalServer, "binary", return_value=Path(temp) / "server"),
                  patch.object(pe_runtime.subprocess, "Popen", return_value=interrupted),
                  patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": memory})):
                server = LocalServer()
                with self.assertRaises(KeyboardInterrupt):
                    server.start(model, None, 1024, 0)
                self.assertTrue(interrupted.terminated)
                self.assertIsNone(server.process)
                self.assertIsNone(server.log)

    def test_completion_uses_only_user_quoted_text_as_exact_literal(self):
        server = LocalServer()
        def response(answer):
            return {"choices": [{"message": {"content": json.dumps(answer)},
                                 "finish_reason": "stop"}], "usage": {}}
        edit_answer = {"rewritten_prompt": 'Blend "<image1>" with "<image2>".',
                       "wh_ratio": "4:3", "ratio_follow": ""}
        with patch.object(server, "_post_completion", return_value=response(edit_answer)):
            actual, _ = server.complete("edit", 'Blend "<image1>" with "<image2>".',
                                        ["image-a", "image-b"], 42, 1)
        self.assertEqual(actual["rewritten_prompt"], edit_answer["rewritten_prompt"])

        transparent_answer = {"rewritten_prompt": 'A poster reads "white background" in black letters.',
                              "wh_ratio": "4:5"}
        with patch.object(server, "_post_completion", return_value=response(transparent_answer)):
            actual, _ = server.complete("t2i", 'A poster reads "white background".',
                                        [], 42, 1, aspect_ratio="4:5", transparent_rgba=True)
        self.assertIn('"white background"', actual["rewritten_prompt"])

    def test_truncated_completion_retries_without_thinking_then_validates(self):
        server = LocalServer()
        server.profile = ("model", None, 24576, 99)
        requests = []
        def reply(payload, _timeout):
            requests.append((payload["max_tokens"],
                             payload["chat_template_kwargs"]["enable_thinking"],
                             payload["messages"][0]["content"]))
            if len(requests) == 1:
                return {"choices": [{"message": {"content": "unfinished"},
                                     "finish_reason": "length"}],
                        "usage": {"prompt_tokens": 3000, "completion_tokens": 16256}}
            return {"choices": [{"message": {"content": json.dumps({
                "rewritten_prompt": "A blue cat on a table.", "wh_ratio": "1:1"})},
                                 "finish_reason": "stop"}], "usage": {}}
        with patch.object(server, "_post_completion", side_effect=reply):
            answer, info = server.complete("t2i", "A blue cat", [], 42, 1)
        self.assertEqual(answer["rewritten_prompt"], "A blue cat on a table.")
        self.assertEqual([(limit, thinking) for limit, thinking, _ in requests],
                         [(16256, True), (16256, False)])
        self.assertIn("Return only the final JSON object", requests[1][2])
        self.assertTrue(info["truncation_retry"])
        self.assertEqual(info["format_retries"], 1)

        with patch.object(server, "_post_completion", return_value={
                "choices": [{"message": {"content": "unfinished"},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": 3000, "completion_tokens": 16256}}) as completion:
            with self.assertRaisesRegex(ValueError, "retry without thinking was also truncated") as error:
                server.complete("t2i", "A blue cat", [], 42, 1)
        self.assertEqual(completion.call_count, 2)
        self.assertIn("context=24576", str(error.exception))
        self.assertIn("completion_tokens=16256", str(error.exception))

    def test_generic_image_tag_gets_image_count_specific_retry(self):
        cases = (
            ("t2i", [], {"rewritten_prompt": "Show <image> as a flower.", "wh_ratio": "1:1"},
             {"rewritten_prompt": "A red flower fills the frame.", "wh_ratio": "1:1"},
             "There are no input images"),
            ("edit", ["image-a", "image-b"],
             {"rewritten_prompt": "Blend <image> with <image2>.", "wh_ratio": "",
              "ratio_follow": "<image1>"},
             {"rewritten_prompt": "Blend <image1> with <image2>.", "wh_ratio": "",
              "ratio_follow": "<image1>"},
             "Do not use the unnumbered <image> tag"),
        )
        for task, images, invalid, valid, rule in cases:
            with self.subTest(task=task):
                server = LocalServer()
                requests = []
                def reply(payload, _timeout):
                    requests.append((payload["chat_template_kwargs"]["enable_thinking"],
                                     payload["messages"][0]["content"]))
                    answer = invalid if len(requests) == 1 else valid
                    return {"choices": [{"message": {"content": json.dumps(answer)},
                                         "finish_reason": "stop"}], "usage": {}}
                with patch.object(server, "_post_completion", side_effect=reply):
                    answer, info = server.complete(task, "Change the image.", images, 42, 1)
                self.assertEqual(answer, valid)
                self.assertEqual([thinking for thinking, _ in requests], [True, False])
                self.assertIn(rule, requests[0][1])
                self.assertIn(rule, requests[1][1])
                self.assertIn("<image>", info["first_format_error"])
                self.assertEqual(info["format_retries"], 1)

    def test_single_image_bare_tag_is_normalized_without_retry(self):
        english, count = normalize_single_image_references(
            'Recolor <image> and leave the label "<image1>" untouched.')
        self.assertEqual(count, 1)
        self.assertEqual(english,
                         'Recolor <image1> and leave the label "<image1>" untouched.')
        chinese, count = normalize_single_image_references("把<image>中的猫改为蓝色。")
        self.assertEqual((chinese, count), ("把<image1>中的猫改为蓝色。", 1))
        server = LocalServer()
        response = {"choices": [{"message": {"content": json.dumps({
            "rewritten_prompt": "Change <image> to blue.", "wh_ratio": "",
            "ratio_follow": "<image1>"})}, "finish_reason": "stop"}], "usage": {}}
        with patch.object(server, "_post_completion", return_value=response) as completion:
            answer, info = server.complete("edit", "Change the image to blue.",
                                           ["image-data"], 42, 1)
        self.assertEqual(completion.call_count, 1)
        self.assertEqual(answer["rewritten_prompt"], "Change <image1> to blue.")
        self.assertEqual(info["normalized_single_image_tags"], 1)
        self.assertEqual(info["format_retries"], 0)

    def test_translation_requires_exact_text_to_remain_quoted(self):
        server = LocalServer()
        result = {"choices": [{"message": {"content":
                  "HELLO appears elsewhere; the sign says 'GOODBYE'."},
                  "finish_reason": "stop"}], "usage": {}}
        with patch.object(server, "_post_completion", return_value=result):
            with self.assertRaisesRegex(ValueError, "quoted image text"):
                server._translate_prose("A sign says 'HELLO'.", "English", 42, 1)
        result["choices"][0]["message"]["content"] = '"HELLO" appears elsewhere; the sign says "GOODBYE".'
        with patch.object(server, "_post_completion", return_value=result):
            with self.assertRaisesRegex(ValueError, "quoted image text"):
                server._translate_prose("A sign says 'HELLO'.", "English", 42, 1)
        result["choices"][0]["message"]["content"] = 'The sign says "HELLO".'
        with patch.object(server, "_post_completion", return_value=result):
            translated, _ = server._translate_prose("A sign says 'HELLO'.", "English", 42, 1)
        self.assertEqual(translated, 'The sign says "HELLO".')

    def test_kept_server_reloads_replaced_model_file(self):
        class FakeProcess:
            returncode = None
            terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = 0

            def wait(self, timeout):
                return self.returncode

        with tempfile.TemporaryDirectory() as temp:
            import pe_runtime
            model = Path(temp) / "model.gguf"
            model.write_bytes(b"old")
            first, second = FakeProcess(), FakeProcess()
            health = MagicMock()
            health.status = 200
            health.__enter__.return_value = health
            with (patch.object(pe_runtime, "ROOT", Path(temp)),
                  patch.object(LocalServer, "binary", return_value=Path(temp) / "server"),
                  patch.object(pe_runtime.subprocess, "Popen", side_effect=[first, second]) as launch,
                  patch.object(pe_runtime, "urlopen", return_value=health)):
                server = LocalServer()
                server.start(model, None, 1024, 0)
                server.start(model, None, 1024, 0)
                self.assertEqual(launch.call_count, 1)
                old_stat = model.stat()
                replacement = Path(temp) / "replacement.gguf"
                replacement.write_bytes(b"new")
                os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
                os.replace(replacement, model)
                server.start(model, None, 1024, 0)
                self.assertTrue(first.terminated)
                self.assertEqual(launch.call_count, 2)
                server.stop()


if __name__ == "__main__":
    unittest.main()
