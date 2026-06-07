from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_smplx_template import normalize_part_labels, write_obj
from load_smplx import MeshData


class TemplateGeneratorTest(unittest.TestCase):
    def test_segmentation_conversion_and_missing_vertex_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            segmentation = Path(temp) / "segmentation.json"
            segmentation.write_text(
                '{"head": [0], "spine": [1], "leftArm": [2]}',
                encoding="utf-8",
            )
            with warnings.catch_warnings(record=True) as caught:
                labels = normalize_part_labels(segmentation, 4)
            self.assertEqual(labels, ["head", "torso", "left_upper_arm", "head"])
            self.assertEqual(len(caught), 1)

    def test_uv_obj_export_preserves_separate_uv_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            obj = Path(temp) / "template.obj"
            mesh = MeshData(
                vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                faces=np.asarray([[0, 1, 2]], dtype=np.int32),
                uv_coords=np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
                face_uv_indices=np.asarray([[2, 1, 0]], dtype=np.int32),
            )
            write_obj(mesh, obj)
            text = obj.read_text(encoding="utf-8")
            self.assertIn("vt 1.00000000 0.00000000", text)
            self.assertIn("f 1/3 2/2 3/1", text)


if __name__ == "__main__":
    unittest.main()
