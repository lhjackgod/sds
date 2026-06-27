"""Procedural wrinkle UV fields."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WrinkleFields:
    upper_wrinkle_uv: np.ndarray
    sleeve_wrinkle_uv: np.ndarray
    lower_wrinkle_uv: np.ndarray


def _wave_field(shape: tuple[int, int], freq_x: float, freq_y: float, phase: float = 0.0) -> np.ndarray:
    h, w = shape
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    field = np.sin((xx * freq_x + yy * freq_y + phase) * np.pi * 2.0)
    field += 0.45 * np.sin((xx * freq_x * 0.37 - yy * freq_y * 1.7 + phase + 0.19) * np.pi * 2.0)
    field = np.maximum(field, 0.0)
    return field.astype(np.float32)


def _smooth(field: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(field.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)


def _amp(spec: dict, upper: bool) -> float:
    item = spec.get("upper" if upper else "lower", {})
    garment_type = item.get("type", "none")
    fit = item.get("fit", "normal")
    if fit == "tight":
        return 0.0
    if garment_type == "hoodie":
        return 1.0
    if garment_type == "sweater":
        return 0.75
    if garment_type in {"tshirt", "shirt"}:
        return 0.35
    if garment_type == "jeans":
        return 0.55
    if garment_type == "leggings":
        return 0.0
    return 0.25


def build_wrinkle_fields(spec: dict, vertex_masks, garment_mask_uv: np.ndarray) -> WrinkleFields:
    del vertex_masks
    mask = garment_mask_uv.astype(np.float32)
    upper_amp = _amp(spec, upper=True)
    lower_amp = _amp(spec, upper=False)
    upper = _smooth(_wave_field(mask.shape, 6.0, 1.6, 0.1)) * mask * upper_amp
    sleeve = _smooth(_wave_field(mask.shape, 9.0, 2.2, 0.37)) * mask * upper_amp
    lower = _smooth(_wave_field(mask.shape, 2.2, 9.0, 0.23)) * mask * lower_amp
    return WrinkleFields(
        upper_wrinkle_uv=np.clip(upper, 0.0, 1.0).astype(np.float32),
        sleeve_wrinkle_uv=np.clip(sleeve, 0.0, 1.0).astype(np.float32),
        lower_wrinkle_uv=np.clip(lower, 0.0, 1.0).astype(np.float32),
    )

