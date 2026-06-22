"""Minimal nvdiffrast renderer for SDS offset optimization."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from castex_bridge.camera_bridge import OrbitCameras


def _face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    tri = vertices[faces.long()]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    return F.normalize(normals, dim=-1, eps=1e-8)


def compute_vertex_normals_torch(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    normals = torch.zeros_like(vertices)
    face_normals = _face_normals(vertices, faces)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner].long(), face_normals)
    return F.normalize(normals, dim=-1, eps=1e-8)


class OffsetNvdiffRenderer:
    def __init__(self, faces, uvs=None, face_uv_indices=None, resolution: int = 256, device: str = "cuda") -> None:
        if not str(device).startswith("cuda"):
            raise ValueError("OffsetNvdiffRenderer requires a CUDA device because nvdiffrast rasterization is CUDA-backed.")
        try:
            import nvdiffrast.torch as dr
        except Exception as exc:
            raise RuntimeError(
                "nvdiffrast is required for SDS offset rendering. Install/use the CasTex environment with nvdiffrast available."
            ) from exc
        self.dr = dr
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.faces = torch.as_tensor(faces, dtype=torch.int32, device=device)
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

    def _clip_vertices(self, vertices: torch.Tensor, cameras: OrbitCameras) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(cameras.azimuth.shape[0])
        center = (vertices.max(dim=0).values + vertices.min(dim=0).values) * 0.5
        centered = vertices - center
        extent = (centered.max(dim=0).values - centered.min(dim=0).values).max().clamp_min(1e-6)
        view_vertices = self._rotate(centered / (extent * 0.58), cameras.azimuth, cameras.elevation)
        xy = view_vertices[..., :2]
        z = view_vertices[..., 2:3]
        z_min = z.amin(dim=1, keepdim=True)
        z_max = z.amax(dim=1, keepdim=True)
        z_norm = (z - z_min) / (z_max - z_min).clamp_min(1e-6)
        ones = torch.ones((batch, vertices.shape[0], 1), dtype=vertices.dtype, device=self.device)
        clip = torch.cat([xy, z_norm * 2.0 - 1.0, ones], dim=-1)
        return clip.contiguous(), view_vertices

    def render(
        self,
        vertices: torch.Tensor,
        vertex_colors=None,
        cameras: OrbitCameras | None = None,
        mode: str = "shaded",
    ) -> dict[str, torch.Tensor]:
        if cameras is None:
            from castex_bridge.camera_bridge import fixed_cameras

            cameras = fixed_cameras(("front",), self.resolution, self.device)
        resolution = cameras.resolution
        vertices = vertices.to(self.device)
        clip_vertices, _ = self._clip_vertices(vertices, cameras)
        rast, _ = self.dr.rasterize(self.glctx, clip_vertices, self.faces, (resolution, resolution))
        mask = torch.clamp(rast[..., 3:4], 0.0, 1.0)

        normals = compute_vertex_normals_torch(vertices, self.faces.long())
        view_normals = self._rotate(normals, cameras.azimuth, cameras.elevation).contiguous()
        interp_normals, _ = self.dr.interpolate(view_normals, rast, self.faces)
        interp_normals = F.normalize(interp_normals, dim=-1, eps=1e-8)
        normal_rgb = interp_normals * 0.5 + 0.5

        light = torch.tensor([0.25, 0.35, 1.0], dtype=torch.float32, device=self.device)
        light = F.normalize(light, dim=0)
        diffuse = torch.clamp((interp_normals * light.view(1, 1, 1, 3)).sum(dim=-1, keepdim=True), 0.0, 1.0)
        if vertex_colors is None:
            albedo = torch.tensor([0.72, 0.72, 0.72], dtype=torch.float32, device=self.device).view(1, 1, 1, 3)
        else:
            colors = torch.as_tensor(vertex_colors, dtype=torch.float32, device=self.device)
            if colors.max() > 1.0:
                colors = colors / 255.0
            colors = colors[None].expand(cameras.azimuth.shape[0], -1, -1).contiguous()
            albedo, _ = self.dr.interpolate(colors, rast, self.faces)
        shaded = albedo * (0.45 + 0.55 * diffuse)
        background = torch.ones_like(shaded) * 0.8
        shaded = torch.where(mask > 0.0, shaded, background)
        normal_rgb = torch.where(mask > 0.0, normal_rgb, torch.zeros_like(normal_rgb))

        return {
            "shaded": shaded.permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0),
            "normal": normal_rgb.permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0),
            "silhouette": mask.permute(0, 3, 1, 2).contiguous(),
        }


CasTexOffsetRenderer = OffsetNvdiffRenderer
