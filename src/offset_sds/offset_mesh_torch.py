"""Differentiable normal-offset mesh forward pass."""

from __future__ import annotations

import torch

from offset_sds.offset_map_torch import sample_uv_to_vertices_torch


class DifferentiableOffsetMesh:
    def __init__(self, vertices, faces, vertex_normals, vertex_uvs_per_vertex, fixed_zero_mask, max_vertex_offset, device: str = "cuda") -> None:
        self.vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
        self.faces = torch.as_tensor(faces, dtype=torch.long, device=device)
        self.vertex_normals = torch.as_tensor(vertex_normals, dtype=torch.float32, device=device)
        self.vertex_uvs_per_vertex = vertex_uvs_per_vertex
        self.fixed_zero_mask = torch.as_tensor(fixed_zero_mask, dtype=torch.bool, device=device)
        self.max_vertex_offset = torch.as_tensor(max_vertex_offset, dtype=torch.float32, device=device)

    def forward(self, offset_uv: torch.Tensor):
        offset_scale = sample_uv_to_vertices_torch(offset_uv, self.vertex_uvs_per_vertex)
        offset_scale = torch.minimum(torch.clamp(offset_scale, min=0.0), self.max_vertex_offset)
        offset_scale = torch.where(self.fixed_zero_mask, torch.zeros_like(offset_scale), offset_scale)
        vertices_offset = self.vertices + self.vertex_normals * offset_scale[:, None]
        return vertices_offset, offset_scale

