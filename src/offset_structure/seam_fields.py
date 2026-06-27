"""Rule-based UV seam fields for garment structure offsets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offset_structure.boundary_fields import mesh_edges, soft_band_from_vertices


@dataclass
class SeamFields:
    shoulder_seam_uv: np.ndarray
    side_seam_uv: np.ndarray
    outer_leg_seam_uv: np.ndarray
    inner_leg_seam_uv: np.ndarray


def _part_edge(labels: np.ndarray, edges: np.ndarray, left: set[str], right: set[str]) -> np.ndarray:
    out = np.zeros(len(labels), dtype=bool)
    for a, b in edges:
        la = labels[a]
        lb = labels[b]
        if (la in left and lb in right) or (la in right and lb in left):
            out[a] = True
            out[b] = True
    return out


def build_seam_fields(mesh, uv_data, vertex_masks, part_labels, resolution: int | None = None) -> SeamFields:
    del uv_data, vertex_masks
    resolution = int(resolution or 1024)
    labels = np.asarray(part_labels)
    edges = mesh_edges(mesh.faces)

    shoulder = _part_edge(labels, edges, {"torso"}, {"left_upper_arm", "right_upper_arm"})
    side = shoulder.copy()
    outer_leg = _part_edge(labels, edges, {"left_thigh", "right_thigh"}, {"left_calf", "right_calf", "pelvis"})
    inner_leg = _part_edge(labels, edges, {"left_thigh", "left_calf"}, {"right_thigh", "right_calf"})

    return SeamFields(
        shoulder_seam_uv=soft_band_from_vertices(shoulder, mesh, resolution, sigma=5.0),
        side_seam_uv=soft_band_from_vertices(side, mesh, resolution, sigma=5.0),
        outer_leg_seam_uv=soft_band_from_vertices(outer_leg, mesh, resolution, sigma=5.0),
        inner_leg_seam_uv=soft_band_from_vertices(inner_leg, mesh, resolution, sigma=5.0),
    )

