from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main_generate_masks import generate_masks
from parse_prompt import parse_prompt
from part_mapping import build_vertex_masks


class PromptParserTest(unittest.TestCase):
    def test_expected_prompt_variants(self) -> None:
        cases = [
            ("red short sleeve T-shirt and blue jeans", "tshirt", "short", "jeans", "long"),
            ("black sleeveless vest and gray shorts", "vest", "none", "shorts", "short"),
            ("green long sleeve sweater and black pants", "sweater", "long", "pants", "long"),
            ("white hoodie and dark jeans", "hoodie", "long", "jeans", "long"),
            ("tight black long sleeve shirt and leggings", "shirt", "long", "leggings", "long"),
        ]
        for prompt, upper_type, sleeve, lower_type, lower_length in cases:
            with self.subTest(prompt=prompt):
                spec = parse_prompt(prompt)
                self.assertEqual(spec["upper"]["type"], upper_type)
                self.assertEqual(spec["upper"]["sleeve"], sleeve)
                self.assertEqual(spec["lower"]["type"], lower_type)
                self.assertEqual(spec["lower"]["length"], lower_length)


class PartMappingTest(unittest.TestCase):
    labels = [
        "head",
        "neck",
        "torso",
        "left_upper_arm",
        "left_forearm",
        "left_hand",
        "pelvis",
        "left_thigh",
        "left_thigh",
        "left_calf",
        "left_foot",
    ]
    vertices = np.asarray([[0, y, 0] for y in [3, 2, 1, 1, 0, 0, 0, 0, -1, -2, -3]])

    def test_short_sleeve_jeans_keep_forearm_hand_and_foot_skin(self) -> None:
        masks = build_vertex_masks(parse_prompt("short sleeve t-shirt and jeans"), self.labels, self.vertices)
        self.assertTrue(masks.upper[2])
        self.assertTrue(masks.upper[3])
        self.assertTrue(masks.skin[4])
        self.assertTrue(masks.skin[5])
        self.assertTrue(masks.lower[9])
        self.assertTrue(masks.skin[10])

    def test_shorts_only_include_upper_half_of_thigh(self) -> None:
        masks = build_vertex_masks(parse_prompt("vest and shorts"), self.labels, self.vertices)
        self.assertTrue(masks.lower[7])
        self.assertFalse(masks.lower[8])
        self.assertTrue(masks.skin[9])


class EndToEndTest(unittest.TestCase):
    def test_cli_pipeline_exports_masks_and_textured_obj(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = root / "template.obj"
            labels = root / "labels.json"
            config = root / "config.yaml"
            out = root / "out"
            mesh.write_text(
                "\n".join(
                    [
                        "v 0 1 0",
                        "v 1 1 0",
                        "v 1 0 0",
                        "v 0 0 0",
                        "vt 0 1",
                        "vt 1 1",
                        "vt 1 0",
                        "vt 0 0",
                        "f 1/1 2/2 3/3",
                        "f 1/1 3/3 4/4",
                    ]
                ),
                encoding="utf-8",
            )
            labels.write_text('["torso", "torso", "left_upper_arm", "left_upper_arm"]', encoding="utf-8")
            config.write_text(
                "body_up_axis: y\npostprocess:\n  min_component_area: 1\n",
                encoding="utf-8",
            )
            args = Namespace(
                prompt="a red short sleeve t-shirt",
                mesh=str(mesh),
                uv=None,
                part_labels=str(labels),
                out=str(out),
                resolution=32,
                config=str(config),
            )
            generate_masks(args)

            self.assertTrue((out / "debug_mesh.obj").exists())
            self.assertTrue((out / "debug_mesh.mtl").exists())
            self.assertTrue((out / "debug_texture.png").exists())
            self.assertTrue((out / "raw_vertex_debug_texture.png").exists())
            self.assertTrue((out / "vertex_region_mesh.ply").exists())
            upper = np.asarray(Image.open(out / "upper_mask.png"))
            self.assertGreater(np.count_nonzero(upper), 0)


if __name__ == "__main__":
    unittest.main()

