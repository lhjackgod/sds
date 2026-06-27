"""Boundary-derived UV bands for garment structure offsets."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BoundaryFields:
    neck_band_uv: np.ndarray
    sleeve_cuff_band_uv: np.ndarray
    bottom_hem_band_uv: np.ndarray
    waistband_uv: np.ndarray
    pant_leg_opening_uv: np.ndarray


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def vertices_to_uv_points(vertex_mask: np.ndarray, mesh, resolution: int) -> np.ndarray:
    uv_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    for face, face_uv in zip(mesh.faces, mesh.face_uv_indices):
        for vertex_index, uv_index in zip(face, face_uv):
            if not vertex_mask[int(vertex_index)]:
                continue
            uv = mesh.uv_coords[int(uv_index)]
            x = int(np.clip(round(float(uv[0]) * (resolution - 1)), 0, resolution - 1))
            y = int(np.clip(round((1.0 - float(uv[1])) * (resolution - 1)), 0, resolution - 1))
            cv2.circle(uv_mask, (x, y), 2, 1, -1)
    return uv_mask.astype(np.float32)


def soft_band_from_vertices(vertex_mask: np.ndarray, mesh, resolution: int, sigma: float = 8.0, dilate: int = 1) -> np.ndarray:
    points = vertices_to_uv_points(vertex_mask, mesh, resolution)
    if points.max() <= 0:
        return points
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=np.uint8)
        points = cv2.dilate(points.astype(np.uint8), kernel, iterations=1).astype(np.float32)
    dist = cv2.distanceTransform((points <= 0).astype(np.uint8), cv2.DIST_L2, 3)
    band = np.exp(-(dist**2) / (2.0 * sigma * sigma)).astype(np.float32)
    return np.clip(band, 0.0, 1.0)


def _edge_boundary(labels: np.ndarray, edges: np.ndarray, source: np.ndarray, other_labels: set[str] | None = None, other_mask: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros(len(labels), dtype=bool)
    for a, b in edges:
        a_source = source[a]
        b_source = source[b]
        if a_source == b_source:
            continue
        if a_source:
            other = b
            src = a
        elif b_source:
            other = a
            src = b
        else:
            continue
        label_ok = other_labels is not None and labels[other] in other_labels
        mask_ok = other_mask is not None and other_mask[other]
        if label_ok or mask_ok:
            out[src] = True
    return out


def build_boundary_fields(mesh, uv_data, vertex_masks, part_labels, resolution: int | None = None) -> BoundaryFields:
    del uv_data
    resolution = int(resolution or 1024)
    labels = np.asarray(part_labels)
    edges = mesh_edges(mesh.faces)
    upper = vertex_masks.upper.astype(bool)
    lower = vertex_masks.lower.astype(bool)
    skin = vertex_masks.skin.astype(bool)

    neck = _edge_boundary(labels, edges, upper, {"neck", "head"})
    cuff = _edge_boundary(labels, edges, upper, {"left_hand", "right_hand", "left_forearm", "right_forearm"}, skin)
    hem = _edge_boundary(labels, edges, upper, {"pelvis", "left_thigh", "right_thigh"}, lower | skin)
    waistband = _edge_boundary(labels, edges, lower, {"torso", "neck"}, upper | skin)
    pant_opening = _edge_boundary(labels, edges, lower, {"left_foot", "right_foot"}, skin)

    return BoundaryFields(
        neck_band_uv=soft_band_from_vertices(neck, mesh, resolution, sigma=7.0),
        sleeve_cuff_band_uv=soft_band_from_vertices(cuff, mesh, resolution, sigma=7.0),
        bottom_hem_band_uv=soft_band_from_vertices(hem, mesh, resolution, sigma=8.0),
        waistband_uv=soft_band_from_vertices(waistband, mesh, resolution, sigma=8.0),
        pant_leg_opening_uv=soft_band_from_vertices(pant_opening, mesh, resolution, sigma=8.0),
    )

