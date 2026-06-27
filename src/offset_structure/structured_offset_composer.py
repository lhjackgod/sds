"""Compose structure-aware UV offset maps from interpretable components."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from offset_structure.boundary_fields import BoundaryFields, build_boundary_fields
from offset_structure.overlap_fields import OverlapFields, build_overlap_fields
from offset_structure.seam_fields import SeamFields, build_seam_fields
from offset_structure.template_bank import COMPONENT_NAMES
from offset_structure.template_retrieval import retrieve_structure_template
from offset_structure.wrinkle_fields import WrinkleFields, build_wrinkle_fields


@dataclass
class StructuredOffsetBasis:
    base_uv: np.ndarray
    max_uv: np.ndarray
    garment_mask_uv: np.ndarray
    boundary: BoundaryFields
    seams: SeamFields
    overlap: OverlapFields
    wrinkles: WrinkleFields
    template_weights: dict[str, float]
    components: dict[str, np.ndarray]


def _norm(component: np.ndarray) -> np.ndarray:
    value = component.astype(np.float32)
    max_value = float(value.max())
    if max_value > 1e-8:
        value = value / max_value
    return np.clip(value, 0.0, 1.0)


def build_structured_offset_basis(mesh, vertex_masks, part_labels, spec: dict, init_uv: np.ndarray, max_uv: np.ndarray, garment_mask_uv: np.ndarray) -> StructuredOffsetBasis:
    resolution = int(init_uv.shape[0])
    boundary = build_boundary_fields(mesh, None, vertex_masks, part_labels, resolution)
    seams = build_seam_fields(mesh, None, vertex_masks, part_labels, resolution)
    overlap = build_overlap_fields(boundary, garment_mask_uv, vertex_masks, part_labels)
    wrinkles = build_wrinkle_fields(spec, vertex_masks, garment_mask_uv)
    template = retrieve_structure_template(spec)

    boundary_map = np.maximum.reduce([
        boundary.neck_band_uv,
        boundary.sleeve_cuff_band_uv,
        boundary.bottom_hem_band_uv,
        boundary.waistband_uv,
        boundary.pant_leg_opening_uv,
    ])
    seam_map = np.maximum.reduce([
        seams.shoulder_seam_uv,
        seams.side_seam_uv,
        seams.outer_leg_seam_uv,
        seams.inner_leg_seam_uv,
    ])
    overlap_map = np.maximum(overlap.upper_over_lower_uv, overlap.lower_under_upper_uv)
    wrinkle_map = np.maximum.reduce([
        wrinkles.upper_wrinkle_uv,
        wrinkles.sleeve_wrinkle_uv,
        wrinkles.lower_wrinkle_uv,
    ])

    components = {
        "base": _norm(init_uv),
        "neck": _norm(boundary.neck_band_uv),
        "cuff": _norm(boundary.sleeve_cuff_band_uv),
        "hem": _norm(np.maximum(boundary.bottom_hem_band_uv, boundary.pant_leg_opening_uv)),
        "waistband": _norm(boundary.waistband_uv),
        "seam": _norm(seam_map),
        "wrinkle": _norm(wrinkle_map),
        "overlap": _norm(overlap_map),
        "boundary_debug": _norm(boundary_map),
    }
    return StructuredOffsetBasis(
        base_uv=init_uv.astype(np.float32),
        max_uv=max_uv.astype(np.float32),
        garment_mask_uv=garment_mask_uv.astype(np.float32),
        boundary=boundary,
        seams=seams,
        overlap=overlap,
        wrinkles=wrinkles,
        template_weights=template,
        components=components,
    )


def compose_structured_offset_uv(basis: StructuredOffsetBasis, weights: dict[str, float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    max_uv = basis.max_uv.astype(np.float32)
    base_step = np.maximum(basis.base_uv, max_uv * 0.35)
    maps = {}
    final = np.zeros_like(max_uv, dtype=np.float32)
    amplitudes = {
        "base": base_step,
        "neck": max_uv * 0.35,
        "cuff": max_uv * 0.45,
        "hem": max_uv * 0.45,
        "waistband": max_uv * 0.42,
        "seam": max_uv * 0.18,
        "wrinkle": np.minimum(max_uv * 0.20, 0.004),
        "overlap": max_uv * 0.35,
    }
    for name in COMPONENT_NAMES:
        component = basis.components[name] * amplitudes[name] * float(weights.get(name, 0.0))
        maps[name] = component.astype(np.float32)
        final += maps[name]
    final = basis.garment_mask_uv * np.minimum(np.clip(final, 0.0, None), max_uv)
    return final.astype(np.float32), maps


def compose_structured_offset_uv_torch(
    component_tensors: dict[str, torch.Tensor],
    max_uv: torch.Tensor,
    garment_mask_uv: torch.Tensor,
    raw_scales: torch.Tensor,
    template_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = torch.sigmoid(raw_scales) * 1.5
    weighted = template_weights * scales
    final = torch.zeros_like(max_uv)
    names = list(COMPONENT_NAMES)
    for index, name in enumerate(names):
        final = final + component_tensors[name] * weighted[index]
    final = torch.minimum(torch.clamp(final, min=0.0), max_uv)
    final = final * garment_mask_uv
    return final, scales


def save_component_debug(basis: StructuredOffsetBasis, final_uv: np.ndarray, out_dir) -> None:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    debug_maps = {
        "structured_offset_base.png": basis.components["base"],
        "structured_offset_boundary.png": basis.components["boundary_debug"],
        "structured_offset_seam.png": basis.components["seam"],
        "structured_offset_overlap.png": basis.components["overlap"],
        "structured_offset_wrinkle.png": basis.components["wrinkle"],
        "structured_offset_final.png": final_uv / max(float(final_uv.max()), 1e-8),
    }
    for name, image in debug_maps.items():
        cv2.imwrite(str(out / name), (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))

