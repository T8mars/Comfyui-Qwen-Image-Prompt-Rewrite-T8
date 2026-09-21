import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pe_runtime import (DEFAULT_EDIT, DEFAULT_T2I, DEFAULT_MMPROJ,
                        local_models, parse_answer, pick_mmproj, prepare_images,
                        remove_opaque_background_sentences, validate_language, validate_mode,
                        validate_references)


class RuntimeContractTests(unittest.TestCase):
    def test_parse_t2i_json_after_thinking_text(self):
        raw = '<think>the example {"wrong": 1} is invalid</think>\n{"rewritten_prompt":"a blue dog", "wh_ratio":"3:2"}'
        self.assertEqual(parse_answer(raw, "t2i", 0)["wh_ratio"], "3:2")

    def test_parse_rejects_extra_field_and_bad_image_reference(self):
        with self.assertRaises(ValueError):
            parse_answer('{"rewritten_prompt":"x","wh_ratio":"1:1","extra":"y"}', "t2i", 0)
        with self.assertRaises(ValueError):
            parse_answer('{"rewritten_prompt":"x","wh_ratio":"","ratio_follow":"<image11>"}', "edit", 10)
        with self.assertRaises(ValueError):
            parse_answer('{"rewritten_prompt":"x","wh_ratio":"1:1","ratio_follow":"<image1>"}', "edit", 1)

    def test_reference_contract(self):
        validate_references({"rewritten_prompt": "Use <image1> and <image2>."}, "edit", 2)
        validate_references({"rewritten_prompt": "Preserve the original image."}, "edit", 1)
        with self.assertRaises(ValueError):
            validate_references({"rewritten_prompt": "Use <image1>."}, "edit", 1)
        with self.assertRaises(ValueError):
            validate_references({"rewritten_prompt": "Use <image1>."}, "edit", 2)

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
        with self.assertRaises(ValueError):
            validate_language({"rewritten_prompt": "一张 watercolor 海报。"}, "中文")

    def test_image_preparation_preserves_original_size_and_rejects_batch(self):
        image = torch.zeros((1, 1200, 1600, 3), dtype=torch.float32)
        encoded, dimensions = prepare_images([image])
        self.assertEqual(dimensions, [[1600, 1200]])
        self.assertTrue(encoded[0].startswith("data:image/png;base64,"))
        with self.assertRaises(ValueError):
            prepare_images([torch.zeros((2, 100, 100, 3))])

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


if __name__ == "__main__":
    unittest.main()
