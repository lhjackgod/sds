"""Rasterize vertex-level body regions into UV-space images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from part_mapping import VertexMasks


@dataclass
class UVMasks:
    upper: np.ndarray
    lower: np.ndarray
    skin: np.ndarray
    valid: np.ndarray


def _image_triangle(uv_triangle: np.ndarray, resolution: int) -> np.ndarray:
    points = np.empty((3, 2), dtype=np.float32)
    points[:, 0] = uv_triangle[:, 0] * (resolution - 1)
    points[:, 1] = (1.0 - uv_triangle[:, 1]) * (resolution - 1)
    return np.rint(points).astype(np.int32)


def rasterize_vertex_masks_to_uv(
    vertex_masks: VertexMasks,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    face_uv_indices: np.ndarray,
    resolution: int = 1024,
) -> UVMasks:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if faces.shape != face_uv_indices.shape:
        raise ValueError("faces and face_uv_indices must have matching shapes")

    masks = {
        "upper": np.zeros((resolution, resolution), dtype=np.uint8),
        "lower": np.zeros((resolution, resolution), dtype=np.uint8),
        "skin": np.zeros((resolution, resolution), dtype=np.uint8),
    }
    valid = np.zeros((resolution, resolution), dtype=np.uint8)

    for face, face_uv in zip(faces, face_uv_indices):
        triangle = _image_triangle(uv_coords[face_uv], resolution)
        cv2.fillConvexPoly(valid, triangle, 255)
        if np.count_nonzero(vertex_masks.upper[face]) >= 2:
            region = "upper"
        elif np.count_nonzero(vertex_masks.lower[face]) >= 2:
            region = "lower"
        else:
            region = "skin"
        cv2.fillConvexPoly(masks[region], triangle, 255)

    return UVMasks(valid=valid, **masks)

