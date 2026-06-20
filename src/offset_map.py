"""UV offset-map utilities for bounded normal-offset garments."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OffsetUVMaps:
    init: np.ndarray
    max: np.ndarray
    garment_mask: np.ndarray


def _image_triangle(uv_triangle: np.ndarray, resolution: int) -> np.ndarray:
    pts = np.empty((3, 2), dtype=np.float32)
    pts[:, 0] = uv_triangle[:, 0] * (resolution - 1)
    pts[:, 1] = (1.0 - uv_triangle[:, 1]) * (resolution - 1)
    return np.rint(pts).astype(np.int32)


def rasterize_vertex_scalar_to_uv(
    values: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    face_uv_indices: np.ndarray,
    resolution: int,
    mask_values: np.ndarray | None = None,
) -> np.ndarray:
    accum = np.zeros((resolution, resolution), dtype=np.float32)
    weight = np.zeros((resolution, resolution), dtype=np.float32)
    for face, face_uv in zip(faces, face_uv_indices):
        if mask_values is not None and np.count_nonzero(mask_values[face]) < 2:
            continue
        tri = _image_triangle(uv_coords[face_uv], resolution)
        face_value = float(np.mean(values[face]))
        tri_mask = np.zeros_like(accum, dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, tri, 1)
        accum += tri_mask.astype(np.float32) * face_value
        weight += tri_mask.astype(np.float32)
    out = np.zeros_like(accum)
    np.divide(accum, np.maximum(weight, 1.0), out=out, where=weight > 0)
    return out


def rasterize_garment_mask_to_uv(
    garment_vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    face_uv_indices: np.ndarray,
    resolution: int,
) -> np.ndarray:
    mask = np.zeros((resolution, resolution), dtype=np.float32)
    for face, face_uv in zip(faces, face_uv_indices):
        if np.count_nonzero(garment_vertices[face]) >= 2:
            cv2.fillConvexPoly(mask, _image_triangle(uv_coords[face_uv], resolution), 1.0)
    return mask


def build_vertex_uv_coords(num_vertices: int, faces: np.ndarray, uv_coords: np.ndarray, face_uv_indices: np.ndarray) -> np.ndarray:
    sums = np.zeros((num_vertices, 2), dtype=np.float64)
    counts = np.zeros(num_vertices, dtype=np.float64)
    for face, face_uv in zip(faces, face_uv_indices):
        for vertex_index, uv_index in zip(face, face_uv):
            sums[vertex_index] += uv_coords[uv_index]
            counts[vertex_index] += 1.0
    counts[counts == 0] = 1.0
    return (sums / counts[:, None]).astype(np.float32)


def sample_uv_to_vertices(offset_uv: np.ndarray, vertex_uv_coords: np.ndarray) -> np.ndarray:
    h, w = offset_uv.shape
    u = np.clip(vertex_uv_coords[:, 0], 0.0, 1.0) * (w - 1)
    v = (1.0 - np.clip(vertex_uv_coords[:, 1], 0.0, 1.0)) * (h - 1)
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = u - x0
    dy = v - y0
    top = offset_uv[y0, x0] * (1.0 - dx) + offset_uv[y0, x1] * dx
    bottom = offset_uv[y1, x0] * (1.0 - dx) + offset_uv[y1, x1] * dx
    return (top * (1.0 - dy) + bottom * dy).astype(np.float32)


def compose_offset_uv(init_uv: np.ndarray, max_uv: np.ndarray, garment_mask_uv: np.ndarray, delta_low: np.ndarray | None = None) -> np.ndarray:
    offset = init_uv.astype(np.float32).copy()
    if delta_low is not None:
        up = cv2.resize(delta_low.astype(np.float32), (init_uv.shape[1], init_uv.shape[0]), interpolation=cv2.INTER_CUBIC)
        offset += up
    offset = np.clip(offset, 0.0, max_uv.astype(np.float32))
    offset *= garment_mask_uv.astype(np.float32)
    return offset


def save_offset_uv_png(offset_uv: np.ndarray, path: str, max_value: float | None = None) -> None:
    denom = float(max_value if max_value is not None else max(offset_uv.max(), 1e-8))
    image = np.clip(offset_uv / denom, 0.0, 1.0)
    cv2.imwrite(path, (image * 255.0).astype(np.uint8))
