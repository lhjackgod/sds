"""Clean UV masks and build derived soft, boundary, and label maps."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from uv_rasterizer import UVMasks


@dataclass
class ProcessedMasks:
    upper: np.ndarray
    lower: np.ndarray
    skin: np.ndarray
    upper_soft: np.ndarray
    lower_soft: np.ndarray
    skin_soft: np.ndarray
    boundary: np.ndarray
    combined_region_map: np.ndarray


def _odd_size(value: int) -> int:
    return max(1, value if value % 2 else value + 1)


def _clean(mask: np.ndarray, valid: np.ndarray, close_kernel: int, min_area: int) -> np.ndarray:
    kernel_size = _odd_size(close_kernel)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) & valid
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    output = np.zeros_like(cleaned)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            output[labels == index] = 255
    return output


def _soft(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask.copy()
    blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return blurred.astype(np.uint8)


def postprocess_masks(
    masks: UVMasks,
    close_kernel: int = 3,
    min_component_area: int = 16,
    boundary_kernel: int = 7,
    soft_blur_sigma: float = 2.0,
) -> ProcessedMasks:
    valid = masks.valid.astype(np.uint8)
    upper = _clean(masks.upper, valid, close_kernel, min_component_area)
    lower = _clean(masks.lower, valid, close_kernel, min_component_area)
    skin = _clean(masks.skin, valid, close_kernel, min_component_area)

    # Upper wins only if coarse UV triangles overlap after cleanup.
    lower[upper > 0] = 0
    skin[(upper > 0) | (lower > 0)] = 0
    skin[(valid > 0) & (upper == 0) & (lower == 0)] = 255

    garment = cv2.bitwise_or(upper, lower)
    kernel_size = _odd_size(boundary_kernel)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(garment, kernel)
    eroded = cv2.erode(garment, kernel)
    boundary = cv2.subtract(dilated, eroded) & valid

    combined = np.zeros_like(valid)
    combined[skin > 0] = 1
    combined[upper > 0] = 2
    combined[lower > 0] = 3
    combined[boundary > 0] = 4

    return ProcessedMasks(
        upper=upper,
        lower=lower,
        skin=skin,
        upper_soft=_soft(upper, soft_blur_sigma),
        lower_soft=_soft(lower, soft_blur_sigma),
        skin_soft=_soft(skin, soft_blur_sigma),
        boundary=boundary,
        combined_region_map=combined,
    )

