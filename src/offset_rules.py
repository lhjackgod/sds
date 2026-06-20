"""Prompt-aware bounded offset initialization rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from load_smplx import MeshData
from offset_map import OffsetUVMaps, rasterize_garment_mask_to_uv, rasterize_vertex_scalar_to_uv
from parse_prompt import GarmentSpec
from part_mapping import VertexMasks


@dataclass
class OffsetRuleResult:
    vertex_init: np.ndarray
    vertex_max: np.ndarray
    uv_maps: OffsetUVMaps
    fixed_zero_mask: np.ndarray


def load_offset_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _rule_value(rule_table: dict, garment_type: str, label: str) -> float:
    return float(rule_table.get(garment_type, {}).get(label, 0.0))


def build_offset_rules(
    spec: GarmentSpec,
    part_labels: list[str],
    vertex_masks: VertexMasks,
    mesh: MeshData,
    config: dict,
    resolution: int = 1024,
) -> OffsetRuleResult:
    labels = np.asarray(part_labels)
    garment = vertex_masks.upper | vertex_masks.lower
    fixed_zero_labels = set(config.get("fixed_zero_labels", []))
    fixed_zero = vertex_masks.skin.copy() | np.isin(labels, list(fixed_zero_labels))

    init = np.zeros(len(labels), dtype=np.float32)
    max_scale = np.zeros(len(labels), dtype=np.float32)
    max_by_label = config.get("max_offset", {})
    for label, value in max_by_label.items():
        max_scale[labels == label] = float(value)

    upper_type = spec["upper"].get("type", "none")
    lower_type = spec["lower"].get("type", "none")
    upper_fit = spec["upper"].get("fit", "normal")
    lower_fit = spec["lower"].get("fit", "normal")
    fit_multiplier = config.get("fit_multiplier", {})
    upper_mul = float(fit_multiplier.get(upper_fit, 1.0))
    lower_mul = float(fit_multiplier.get(lower_fit, 1.0))

    for label in np.unique(labels):
        label_mask = labels == label
        if spec["upper"].get("enabled", False):
            init[label_mask & vertex_masks.upper] = _rule_value(config.get("upper", {}), upper_type, label) * upper_mul
        if spec["lower"].get("enabled", False):
            init[label_mask & vertex_masks.lower] = _rule_value(config.get("lower", {}), lower_type, label) * lower_mul

    init[fixed_zero] = 0.0
    init[~garment] = 0.0
    max_scale[fixed_zero] = 0.0
    max_scale[~garment] = 0.0
    init = np.clip(init, 0.0, max_scale)

    garment_uv = rasterize_garment_mask_to_uv(garment, mesh.faces, mesh.uv_coords, mesh.face_uv_indices, resolution)
    init_uv = rasterize_vertex_scalar_to_uv(init, mesh.faces, mesh.uv_coords, mesh.face_uv_indices, resolution, garment)
    max_uv = rasterize_vertex_scalar_to_uv(max_scale, mesh.faces, mesh.uv_coords, mesh.face_uv_indices, resolution, garment)
    init_uv *= garment_uv
    max_uv *= garment_uv
    return OffsetRuleResult(
        vertex_init=init,
        vertex_max=max_scale,
        uv_maps=OffsetUVMaps(init=init_uv, max=max_uv, garment_mask=garment_uv),
        fixed_zero_mask=fixed_zero,
    )
