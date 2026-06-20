"""Normal-offset shell construction and mesh debug export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from debug_render import LOWER_COLOR, SKIN_COLOR, UPPER_COLOR
from load_smplx import MeshData
from part_mapping import VertexMasks


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(norm, 1e-12)).astype(np.float32)


def apply_normal_offset(vertices: np.ndarray, normals: np.ndarray, offset_scale: np.ndarray) -> np.ndarray:
    return vertices.astype(np.float32) + normals.astype(np.float32) * offset_scale.astype(np.float32)[:, None]


def build_vertex_adjacency(num_vertices: int, faces: np.ndarray) -> list[np.ndarray]:
    neighbors = [set() for _ in range(num_vertices)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [np.asarray(sorted(item), dtype=np.int32) for item in neighbors]


def smooth_offset_scale(
    offset_scale: np.ndarray,
    faces: np.ndarray,
    fixed_zero_mask: np.ndarray,
    max_scale: np.ndarray,
    iterations: int = 5,
    lam: float = 0.5,
) -> np.ndarray:
    smoothed = offset_scale.astype(np.float32).copy()
    adjacency = build_vertex_adjacency(len(smoothed), faces)
    for _ in range(iterations):
        updated = smoothed.copy()
        for index, neighbors in enumerate(adjacency):
            if fixed_zero_mask[index] or len(neighbors) == 0:
                continue
            valid = neighbors[~fixed_zero_mask[neighbors]]
            if len(valid) == 0:
                continue
            updated[index] = (1.0 - lam) * smoothed[index] + lam * float(np.mean(smoothed[valid]))
        updated[fixed_zero_mask] = 0.0
        smoothed = np.clip(updated, 0.0, max_scale.astype(np.float32))
    return smoothed


def export_obj(mesh: MeshData, vertices: np.ndarray, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for uv in mesh.uv_coords:
            handle.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")
        for face, face_uv in zip(mesh.faces, mesh.face_uv_indices):
            corners = [f"{vertex + 1}/{uv + 1}" for vertex, uv in zip(face, face_uv)]
            handle.write(f"f {' '.join(corners)}\n")


def export_offset_vertex_colors_ply(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_masks: VertexMasks,
    offset_scale: np.ndarray,
    path: str | Path,
) -> None:
    colors = np.zeros((len(vertices), 3), dtype=np.float32)
    colors[vertex_masks.skin] = SKIN_COLOR
    colors[vertex_masks.upper] = UPPER_COLOR
    colors[vertex_masks.lower] = LOWER_COLOR
    strength = offset_scale / max(float(offset_scale.max()), 1e-8)
    colors = colors * (0.35 + 0.65 * strength[:, None])
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex, color in zip(vertices, colors):
            handle.write(
                f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")
