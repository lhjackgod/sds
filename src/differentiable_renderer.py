"""Minimal orthographic debug renderer.

This module intentionally provides a lightweight CPU renderer for Phase 2 debug
outputs. It is not a replacement for a full differentiable PyTorch3D/nvdiffrast
renderer; SDS mode can later swap this interface for a differentiable backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from offset_shell import compute_vertex_normals


@dataclass
class Camera:
    name: str = "front"
    resolution: int = 512
    padding: float = 0.12


def view_transform(vertices: np.ndarray, view: str) -> np.ndarray:
    angle = {"front": 0.0, "back": math.pi, "left": math.pi / 2.0, "right": -math.pi / 2.0}.get(view, 0.0)
    rot = np.asarray(
        [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]],
        dtype=np.float32,
    )
    return vertices @ rot.T


def project_vertices(vertices: np.ndarray, resolution: int, padding: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    xy = vertices[:, [0, 1]].astype(np.float32)
    z = vertices[:, 2].astype(np.float32)
    center = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    extent = max(float((xy.max(axis=0) - xy.min(axis=0)).max()), 1e-8) * (1.0 + padding)
    norm = (xy - center) / extent + 0.5
    pix = np.empty_like(xy)
    pix[:, 0] = norm[:, 0] * (resolution - 1)
    pix[:, 1] = (1.0 - norm[:, 1]) * (resolution - 1)
    return pix, z


def _rasterize(vertices: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    v = view_transform(vertices, camera.name)
    pix, z = project_vertices(v, camera.resolution, camera.padding)
    image = np.zeros((camera.resolution, camera.resolution, 3), dtype=np.float32)
    depth = np.full((camera.resolution, camera.resolution), np.inf, dtype=np.float32)
    face_depth = z[faces].mean(axis=1)
    order = np.argsort(face_depth)[::-1]
    for face_index in order:
        face = faces[face_index]
        pts = np.rint(pix[face]).astype(np.int32)
        if (pts[:, 0].max() < 0 or pts[:, 1].max() < 0 or pts[:, 0].min() >= camera.resolution or pts[:, 1].min() >= camera.resolution):
            continue
        tri_mask = np.zeros(depth.shape, dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, pts, 1)
        candidate = (tri_mask > 0) & (face_depth[face_index] < depth)
        if not np.any(candidate):
            continue
        depth[candidate] = face_depth[face_index]
        image[candidate] = vertex_colors[face].mean(axis=0)
    silhouette = np.isfinite(depth).astype(np.float32)
    return np.clip(image, 0.0, 1.0), silhouette


def render_region(vertices: np.ndarray, faces: np.ndarray, vertex_colors_uint8: np.ndarray, view: str = "front", resolution: int = 512) -> np.ndarray:
    colors = vertex_colors_uint8.astype(np.float32) / 255.0
    image, _ = _rasterize(vertices, faces, colors, Camera(view, resolution))
    return (image * 255.0).astype(np.uint8)


def render_silhouette(vertices: np.ndarray, faces: np.ndarray, view: str = "front", resolution: int = 512) -> np.ndarray:
    colors = np.ones((len(vertices), 3), dtype=np.float32)
    _, sil = _rasterize(vertices, faces, colors, Camera(view, resolution))
    return (sil * 255.0).astype(np.uint8)


def render_normals(vertices: np.ndarray, faces: np.ndarray, view: str = "front", resolution: int = 512) -> np.ndarray:
    normals = compute_vertex_normals(vertices, faces)
    normals = view_transform(normals, view)
    colors = normals * 0.5 + 0.5
    image, _ = _rasterize(vertices, faces, colors, Camera(view, resolution))
    return (image * 255.0).astype(np.uint8)


def render_shaded(vertices: np.ndarray, faces: np.ndarray, base_color=(0.72, 0.72, 0.72), view: str = "front", resolution: int = 512) -> np.ndarray:
    normals = view_transform(compute_vertex_normals(vertices, faces), view)
    light = np.asarray([0.25, 0.35, 1.0], dtype=np.float32)
    light /= np.linalg.norm(light)
    intensity = np.clip(normals @ light, 0.0, 1.0)[:, None]
    colors = np.asarray(base_color, dtype=np.float32)[None, :] * (0.35 + 0.65 * intensity)
    image, _ = _rasterize(vertices, faces, colors, Camera(view, resolution))
    return (image * 255.0).astype(np.uint8)
