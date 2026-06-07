"""Map garment attributes to SMPL-X vertex-level masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parse_prompt import GarmentSpec


@dataclass
class VertexMasks:
    upper: np.ndarray
    lower: np.ndarray
    skin: np.ndarray


def _labels_mask(labels: np.ndarray, selected: set[str]) -> np.ndarray:
    return np.isin(labels, list(selected))


def _upper_half_thighs(labels: np.ndarray, vertices: np.ndarray, axis: int) -> np.ndarray:
    result = np.zeros(len(labels), dtype=bool)
    for side in ("left_thigh", "right_thigh"):
        part = labels == side
        if np.any(part):
            midpoint = (vertices[part, axis].min() + vertices[part, axis].max()) / 2.0
            result |= part & (vertices[:, axis] >= midpoint)
    return result


def build_vertex_masks(
    spec: GarmentSpec,
    part_labels: list[str],
    vertices: np.ndarray,
    body_up_axis: str = "y",
    include_neck_for_vest: bool = False,
) -> VertexMasks:
    labels = np.asarray(part_labels)
    if len(labels) != len(vertices):
        raise ValueError("There must be one body-part label per mesh vertex")
    try:
        axis = {"x": 0, "y": 1, "z": 2}[body_up_axis.lower()]
    except KeyError as exc:
        raise ValueError("body_up_axis must be x, y, or z") from exc

    upper = np.zeros(len(vertices), dtype=bool)
    upper_spec = spec["upper"]
    if upper_spec["enabled"]:
        upper_parts = {"torso"}
        sleeve = upper_spec["sleeve"]
        if sleeve in {"short", "long"}:
            upper_parts |= {"left_upper_arm", "right_upper_arm"}
        if sleeve == "long":
            upper_parts |= {"left_forearm", "right_forearm"}
        if upper_spec["type"] == "hoodie" or (
            upper_spec["type"] == "vest" and include_neck_for_vest
        ):
            upper_parts.add("neck")
        upper = _labels_mask(labels, upper_parts)

    lower = np.zeros(len(vertices), dtype=bool)
    lower_spec = spec["lower"]
    if lower_spec["enabled"]:
        lower |= labels == "pelvis"
        if lower_spec["type"] == "shorts" and lower_spec["length"] == "short":
            lower |= _upper_half_thighs(labels, vertices, axis)
        else:
            lower |= _labels_mask(labels, {"left_thigh", "right_thigh"})
        if lower_spec["length"] == "long":
            lower |= _labels_mask(labels, {"left_calf", "right_calf"})

    # Resolve any coarse-label overlap consistently before deriving skin.
    lower &= ~upper
    skin = ~(upper | lower)
    forced_skin = _labels_mask(
        labels,
        {"head", "left_hand", "right_hand", "left_foot", "right_foot"},
    )
    upper &= ~forced_skin
    lower &= ~forced_skin
    skin |= forced_skin
    return VertexMasks(upper=upper, lower=lower, skin=skin)

