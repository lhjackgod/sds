"""Overlap UV fields for layered upper/lower garment interactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OverlapFields:
    upper_over_lower_uv: np.ndarray
    lower_under_upper_uv: np.ndarray


def build_overlap_fields(boundary_fields, garment_mask_uv: np.ndarray, vertex_masks=None, part_labels=None) -> OverlapFields:
    del vertex_masks, part_labels
    garment = garment_mask_uv.astype(np.float32)
    upper_over = np.maximum(boundary_fields.bottom_hem_band_uv, boundary_fields.waistband_uv * 0.55) * garment
    lower_under = boundary_fields.waistband_uv * 0.65 * garment
    return OverlapFields(
        upper_over_lower_uv=np.clip(upper_over.astype(np.float32), 0.0, 1.0),
        lower_under_upper_uv=np.clip(lower_under.astype(np.float32), 0.0, 1.0),
    )

