"""Differentiable torch renderer for offset optimization debug/SDS images.

This is a lightweight Gaussian vertex splatter. It is not a full replacement
for nvdiffrast, but it keeps gradients flowing from image losses to vertices and
matches the stable bridge interface so a CasTex/nvdiffrast backend can be
plugged in later.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from castex_bridge.camera_bridge import OrbitCameras


def _face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    tri = vertices[faces.long()]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    return F.normalize(normals, dim=-1, eps=1e-8)


def compute_vertex_normals_torch(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    normals = torch.zeros_like(vertices)
    fn = _face_normals(vertices, faces)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner].long(), fn)
    return F.normalize(normals, dim=-1, eps=1e-8)


class CasTexOffsetRenderer:
    def __init__(self, faces, uvs=None, face_uv_indices=None, resolution: int = 256, device: str = "cuda") -> None:
        self.faces = torch.as_tensor(faces, dtype=torch.long, device=device)
        self.resolution = resolution
        self.device = device

    def _rotate(self, points: torch.Tensor, azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        ca, sa = torch.cos(azimuth), torch.sin(azimuth)
        ce, se = torch.cos(elevation), torch.sin(elevation)
        row0 = torch.stack([ca, torch.zeros_like(ca), sa], dim=-1)
        row1 = torch.stack([sa * se, ce, -ca * se], dim=-1)
        row2 = torch.stack([-sa * ce, se, ca * ce], dim=-1)
        rot = torch.stack([row0, row1, row2], dim=-2)
        return torch.einsum("bij,nj->bni", rot, points)

    def render(self, vertices: torch.Tensor, vertex_colors=None, mode: str = "shaded", cameras: OrbitCameras | None = None):
        if cameras is None:
            from castex_bridge.camera_bridge import fixed_cameras

            cameras = fixed_cameras(("front",), self.resolution, self.device)
        resolution = cameras.resolution
        batch = int(cameras.azimuth.shape[0])
        v = vertices.to(self.device)
        centered = v - (v.max(dim=0).values + v.min(dim=0).values) * 0.5
        scale = torch.clamp((centered[:, :2].max(dim=0).values - centered[:, :2].min(dim=0).values).max(), min=1e-6)
        view_v = self._rotate(centered / (scale * 0.62), cameras.azimuth, cameras.elevation)
        xy = view_v[..., :2].clamp(-1.4, 1.4)
        z = view_v[..., 2]

        normals = compute_vertex_normals_torch(v, self.faces)
        view_n = self._rotate(normals, cameras.azimuth, cameras.elevation)
        light = torch.tensor([0.25, 0.35, 1.0], dtype=torch.float32, device=self.device)
        light = F.normalize(light, dim=0)
        intensity = torch.clamp(torch.einsum("bni,i->bn", view_n, light), 0.0, 1.0)
        if vertex_colors is None:
            base = torch.tensor([0.72, 0.72, 0.72], dtype=torch.float32, device=self.device)
            colors = base.view(1, 1, 3) * (0.35 + 0.65 * intensity[..., None])
        else:
            colors = torch.as_tensor(vertex_colors, dtype=torch.float32, device=self.device)
            if colors.max() > 1.0:
                colors = colors / 255.0
            colors = colors.view(1, -1, 3).expand(batch, -1, -1) * (0.35 + 0.65 * intensity[..., None])

        lin = torch.linspace(-1.0, 1.0, resolution, device=self.device)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).view(1, 1, resolution, resolution, 2)
        sigma = 2.25 / float(resolution)
        denom = torch.zeros((batch, 1, resolution, resolution), dtype=torch.float32, device=self.device)
        shaded_accum = torch.zeros((batch, resolution, resolution, 3), dtype=torch.float32, device=self.device)
        normal_accum = torch.zeros_like(shaded_accum)
        normal_colors = view_n * 0.5 + 0.5
        chunk_size = 768
        z_center = z.mean(dim=1, keepdim=True)
        for start in range(0, xy.shape[1], chunk_size):
            end = min(start + chunk_size, xy.shape[1])
            xy_chunk = xy[:, start:end]
            diff = grid - xy_chunk.view(batch, -1, 1, 1, 2)
            weights = torch.exp(-(diff.square().sum(dim=-1)) / (2.0 * sigma * sigma))
            depth_weight = torch.sigmoid((z[:, start:end] - z_center) * 8.0).view(batch, -1, 1, 1)
            weights = weights * (0.35 + 0.65 * depth_weight)
            denom = denom + weights.sum(dim=1, keepdim=True)
            shaded_accum = shaded_accum + (weights[..., None] * colors[:, start:end].view(batch, -1, 1, 1, 3)).sum(dim=1)
            normal_accum = normal_accum + (weights[..., None] * normal_colors[:, start:end].view(batch, -1, 1, 1, 3)).sum(dim=1)
        denom = denom.clamp_min(1e-6)
        shaded = shaded_accum / denom.permute(0, 2, 3, 1)
        normal_rgb = normal_accum / denom.permute(0, 2, 3, 1)
        silhouette = torch.clamp(denom, 0.0, 1.0)
        shaded = shaded.permute(0, 3, 1, 2).clamp(0.0, 1.0)
        normal_rgb = normal_rgb.permute(0, 3, 1, 2).clamp(0.0, 1.0)
        return shaded, normal_rgb, silhouette

