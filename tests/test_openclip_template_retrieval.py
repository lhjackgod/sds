from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offset_structure.template_bank import COMPONENT_NAMES
from offset_structure.template_retrieval import retrieve_structure_template


class TemplateRetrievalTest(unittest.TestCase):
    def spec(self):
        return {
            "upper": {"enabled": True, "type": "hoodie"},
            "lower": {"enabled": True, "type": "jeans"},
        }

    def test_rule_retrieval_still_works(self) -> None:
        weights, debug = retrieve_structure_template(self.spec(), prompt="a loose hoodie and blue jeans", method="rule")
        self.assertEqual(debug["method"], "rule")
        self.assertTrue(debug["selected"])
        self.assertEqual(set(weights), set(COMPONENT_NAMES))
        for value in weights.values():
            self.assertIsInstance(value, float)

    def test_openclip_missing_error_is_clear(self) -> None:
        if importlib.util.find_spec("open_clip") is not None:
            self.skipTest("open_clip is installed; missing-package path is not applicable")
        with self.assertRaisesRegex(RuntimeError, "open_clip_torch is not installed"):
            retrieve_structure_template(self.spec(), prompt="a loose hoodie and blue jeans", method="openclip")

    def test_openclip_retrieval_if_installed(self) -> None:
        if os.environ.get("RUN_OPENCLIP_RETRIEVAL_TEST") != "1":
            self.skipTest("set RUN_OPENCLIP_RETRIEVAL_TEST=1 to run the model-loading OpenCLIP retrieval test")
        if importlib.util.find_spec("open_clip") is None:
            self.skipTest("open_clip is not installed")
        weights, debug = retrieve_structure_template(
            self.spec(),
            prompt="a person wearing a loose white hoodie and dark blue jeans",
            method="openclip",
            device="cuda:0",
            top_k=3,
        )
        self.assertEqual(set(weights), set(COMPONENT_NAMES))
        for value in weights.values():
            self.assertTrue(float("-inf") < float(value) < float("inf"))
        top_upper = [item["template"] for item in debug.get("upper", {}).get("top_k", [])]
        top_lower = [item["template"] for item in debug.get("lower", {}).get("top_k", [])]
        self.assertTrue(any("hoodie" in name for name in top_upper), top_upper)
        self.assertTrue(any("jeans" in name for name in top_lower), top_lower)


if __name__ == "__main__":
    unittest.main()
