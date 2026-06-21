"""Torch UV offset-map composition and differentiable sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compose_offset_uv_torch(init_uv: torch.Tensor, max_uv: torch.Tensor, garment_mask_uv: torch.Tensor, delta_low: torch.Tensor | None = None) -> torch.Tensor:
    offset_uv = init_uv
    if delta_low is not None:
        delta = F.interpolate(delta_low, size=init_uv.shape[-2:], mode="bilinear", align_corners=False)
        offset_uv = offset_uv + delta
    offset_uv = torch.minimum(torch.clamp(offset_uv, min=0.0), max_uv)
    return offset_uv * garment_mask_uv


def _as_padded_uvs(vertex_uvs_per_vertex, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(vertex_uvs_per_vertex, tuple):
        return vertex_uvs_per_vertex[0].to(device), vertex_uvs_per_vertex[1].to(device)
    if torch.is_tensor(vertex_uvs_per_vertex):
        uv = vertex_uvs_per_vertex.to(device)
        valid = torch.isfinite(uv[..., 0])
        return torch.nan_to_num(uv, nan=0.0), valid
    max_count = max((len(item) for item in vertex_uvs_per_vertex), default=1)
    padded = torch.zeros((len(vertex_uvs_per_vertex), max_count, 2), dtype=torch.float32, device=device)
    valid = torch.zeros((len(vertex_uvs_per_vertex), max_count), dtype=torch.bool, device=device)
    for index, coords in enumerate(vertex_uvs_per_vertex):
        if len(coords) == 0:
            continue
        tensor = torch.as_tensor(coords, dtype=torch.float32, device=device)
        padded[index, : len(coords)] = tensor
        valid[index, : len(coords)] = True
    return padded, valid


def sample_uv_to_vertices_torch(offset_uv: torch.Tensor, vertex_uvs_per_vertex) -> torch.Tensor:
    uv, valid = _as_padded_uvs(vertex_uvs_per_vertex, offset_uv.device)
    uv = uv.clamp(0.0, 1.0)
    grid = torch.empty_like(uv)
    grid[..., 0] = uv[..., 0] * 2.0 - 1.0
    grid[..., 1] = (1.0 - uv[..., 1]) * 2.0 - 1.0
    samples = F.grid_sample(offset_uv, grid.view(1, -1, 1, 2), mode="bilinear", padding_mode="border", align_corners=True)
    samples = samples.view(-1, uv.shape[0], uv.shape[1]).squeeze(0)
    weights = valid.to(samples.dtype)
    return (samples * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def tv_loss(offset_uv: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(offset_uv[..., :, 1:] - offset_uv[..., :, :-1]).mean()
    dy = torch.abs(offset_uv[..., 1:, :] - offset_uv[..., :-1, :]).mean()
    return dx + dy


def build_vertex_uvs_per_vertex(num_vertices: int, faces, uv_coords, face_uv_indices) -> list[list[list[float]]]:
    out: list[list[list[float]]] = [[] for _ in range(num_vertices)]
    seen: list[set[tuple[float, float]]] = [set() for _ in range(num_vertices)]
    for face, face_uv in zip(faces, face_uv_indices):
        for vertex_index, uv_index in zip(face, face_uv):
            uv = uv_coords[int(uv_index)]
            key = (round(float(uv[0]), 7), round(float(uv[1]), 7))
            if key not in seen[int(vertex_index)]:
                seen[int(vertex_index)].add(key)
                out[int(vertex_index)].append([float(uv[0]), float(uv[1])])
    return out

