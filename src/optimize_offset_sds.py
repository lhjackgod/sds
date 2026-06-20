"""Lightweight bounded offset optimization scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from offset_map import compose_offset_uv, sample_uv_to_vertices
from offset_shell import smooth_offset_scale
from sds_guidance import SDSGuidance


@dataclass
class OptimizationResult:
    offset_uv: np.ndarray
    offset_scale: np.ndarray
    log: dict = field(default_factory=dict)


def total_variation(image: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(image, axis=0))) + np.mean(np.abs(np.diff(image, axis=1))))


def mesh_laplacian_loss(values: np.ndarray, faces: np.ndarray) -> float:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return float(np.mean((values[edges[:, 0]] - values[edges[:, 1]]) ** 2))


def optimize_offset(
    init_uv: np.ndarray,
    max_uv: np.ndarray,
    garment_mask_uv: np.ndarray,
    vertex_uv_coords: np.ndarray,
    faces: np.ndarray,
    fixed_zero_mask: np.ndarray,
    max_vertex_scale: np.ndarray,
    prompt: str,
    mode: str = "none",
    optimize_mode: str = "none",
    sds_mode: str = "dummy",
    steps: int = 0,
    smooth_iterations: int = 5,
) -> OptimizationResult:
    if mode == "init" or optimize_mode == "none" or steps <= 0:
        offset_uv = compose_offset_uv(init_uv, max_uv, garment_mask_uv)
        offset_scale = sample_uv_to_vertices(offset_uv, vertex_uv_coords)
        offset_scale[fixed_zero_mask] = 0.0
        offset_scale = np.clip(offset_scale, 0.0, max_vertex_scale)
        offset_scale = smooth_offset_scale(offset_scale, faces, fixed_zero_mask, max_vertex_scale, iterations=smooth_iterations)
        return OptimizationResult(
            offset_uv=offset_uv,
            offset_scale=offset_scale,
            log={"loss_curves": {}, "final_part_scales": {}, "note": "rule initialization only"},
        )

    guidance = SDSGuidance(prompt=prompt, mode=sds_mode)
    if optimize_mode == "part_scale":
        # First stable implementation: bounded scalar refinement placeholders. Dummy SDS
        # keeps the initialized shell unchanged while recording regularization terms.
        offset_uv = compose_offset_uv(init_uv, max_uv, garment_mask_uv)
        losses = {"reg": [], "tv": [], "mesh_smooth": [], "sds": []}
        for _ in range(steps):
            offset_scale = sample_uv_to_vertices(offset_uv, vertex_uv_coords)
            offset_scale[fixed_zero_mask] = 0.0
            losses["reg"].append(float(np.mean((offset_uv - init_uv) ** 2)))
            losses["tv"].append(total_variation(offset_uv))
            losses["mesh_smooth"].append(mesh_laplacian_loss(offset_scale, faces))
            losses["sds"].append(0.0 if sds_mode == "dummy" else float(guidance.loss(None)))
        offset_scale = sample_uv_to_vertices(offset_uv, vertex_uv_coords)
        offset_scale[fixed_zero_mask] = 0.0
        offset_scale = np.clip(offset_scale, 0.0, max_vertex_scale)
        offset_scale = smooth_offset_scale(offset_scale, faces, fixed_zero_mask, max_vertex_scale, iterations=smooth_iterations)
        return OptimizationResult(
            offset_uv=offset_uv,
            offset_scale=offset_scale,
            log={"loss_curves": losses, "final_part_scales": {"global": 1.0}, "note": "part_scale scaffold; dummy SDS leaves init unchanged"},
        )

    if optimize_mode == "lowres_uv":
        raise NotImplementedError("lowres_uv residual optimization scaffold is reserved for the next refinement pass")
    raise ValueError(f"Unknown optimize_mode: {optimize_mode}")
