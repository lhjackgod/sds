from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from load_smplx import MeshData
from offset_rules import build_offset_rules
from offset_shell import apply_normal_offset, compute_vertex_normals, smooth_offset_scale
from parse_prompt import parse_prompt
from part_mapping import VertexMasks


class OffsetPhase2Test(unittest.TestCase):
    def synthetic_mesh(self) -> MeshData:
        return MeshData(
            vertices=np.asarray(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
            ),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            uv_coords=np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            face_uv_indices=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        )

    def test_offset_rules_keep_skin_zero_and_clamp(self) -> None:
        config = yaml.safe_load(Path("configs/offset_rules.yaml").read_text(encoding="utf-8"))
        mesh = self.synthetic_mesh()
        labels = ["torso", "left_upper_arm", "left_hand", "pelvis"]
        masks = VertexMasks(
            upper=np.asarray([True, True, False, False]),
            lower=np.asarray([False, False, False, True]),
            skin=np.asarray([False, False, True, False]),
        )
        result = build_offset_rules(parse_prompt("short sleeve t-shirt and jeans"), labels, masks, mesh, config, 32)
        self.assertGreater(result.vertex_init[0], 0.0)
        self.assertEqual(result.vertex_init[2], 0.0)
        self.assertEqual(result.vertex_max[2], 0.0)
        self.assertLessEqual(float(result.vertex_init.max()), float(result.vertex_max.max()))

    def test_normal_offset_and_smoothing_respect_fixed_zero(self) -> None:
        vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)
        normals = compute_vertex_normals(vertices, faces)
        scale = np.asarray([0.01, 0.02, 0.0], dtype=np.float32)
        fixed = np.asarray([False, False, True])
        max_scale = np.asarray([0.02, 0.02, 0.0], dtype=np.float32)
        smoothed = smooth_offset_scale(scale, faces, fixed, max_scale, iterations=2)
        self.assertEqual(smoothed[2], 0.0)
        offset = apply_normal_offset(vertices, normals, smoothed)
        self.assertTrue(np.allclose(offset[2], vertices[2]))
        self.assertGreater(offset[0, 2], vertices[0, 2])


if __name__ == "__main__":
    unittest.main()
