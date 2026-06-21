"""Optimization loops for SDS-guided bounded UV offset refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from castex_bridge.camera_bridge import fixed_cameras, random_cameras
from castex_bridge.guidance_bridge import CasTexSDSGuidance
from castex_bridge.renderer_bridge import CasTexOffsetRenderer
from offset_sds.offset_losses import laplacian_offset_loss, mask_loss, mesh_edges
from offset_sds.offset_map_torch import compose_offset_uv_torch, tv_loss
from offset_sds.offset_mesh_torch import DifferentiableOffsetMesh


PART_GROUPS = {
    "upper_torso": {"torso", "neck"},
    "upper_arm": {"left_upper_arm", "right_upper_arm"},
    "forearm": {"left_forearm", "right_forearm"},
    "pelvis": {"pelvis"},
    "thigh": {"left_thigh", "right_thigh"},
    "calf": {"left_calf", "right_calf"},
}


@dataclass
class OffsetSDSResult:
    vertices: np.ndarray
    offset_scale: np.ndarray
    offset_uv: np.ndarray
    log: dict = field(default_factory=dict)


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().float().clamp(0.0, 1.0).cpu()[0]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _save_debug(out_dir: Path, step: int, shaded: torch.Tensor, normal: torch.Tensor, silhouette: torch.Tensor, offset_uv: torch.Tensor) -> None:
    Image.fromarray(_to_image(shaded)).save(out_dir / f"step_{step:04d}_shaded.png")
    Image.fromarray(_to_image(normal)).save(out_dir / f"step_{step:04d}_normal.png")
    Image.fromarray(_to_image(silhouette)).save(out_dir / f"step_{step:04d}_silhouette.png")
    uv = offset_uv.detach().float().cpu()[0, 0].numpy()
    denom = max(float(uv.max()), 1e-8)
    Image.fromarray(np.clip(uv / denom * 255.0, 0, 255).astype(np.uint8)).save(out_dir / f"step_{step:04d}_offset_uv.png")


def _part_index(part_labels: list[str], device: str) -> torch.Tensor:
    indices = []
    for label in part_labels:
        found = 0
        for idx, names in enumerate(PART_GROUPS.values()):
            if label in names:
                found = idx
                break
        indices.append(found)
    return torch.tensor(indices, dtype=torch.long, device=device)


def _part_scale_values(raw: torch.Tensor) -> torch.Tensor:
    return 0.5 + torch.sigmoid(raw) * 1.0


def optimize_uv_offset_sds(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray,
    vertex_uvs_per_vertex,
    uv_coords: np.ndarray,
    face_uv_indices: np.ndarray,
    init_uv: np.ndarray,
    max_uv: np.ndarray,
    garment_mask_uv: np.ndarray,
    init_offset_scale: np.ndarray,
    fixed_zero_mask: np.ndarray,
    max_vertex_offset: np.ndarray,
    part_labels: list[str],
    prompt: str,
    geometry_prompt: str,
    negative_prompt: str,
    castex_root: str | Path,
    out_dir: str | Path,
    optimize_mode: str = "part_scale",
    stage: str = "i",
    steps: int = 500,
    batch_size: int = 4,
    render_resolution: int = 256,
    guidance_scale: float = 20.0,
    device: str = "cuda:0",
    sds_mode: str = "dummy",
    lowres_size: int = 64,
    save_interval: int = 50,
    lambda_sds: float = 1.0,
    lambda_reg: float = 0.1,
    lambda_tv: float = 0.01,
    lambda_mask: float = 1.0,
    lambda_lap: float = 0.1,
    lambda_sil: float = 0.0,
    lambda_seam: float = 0.0,
    lr: float = 0.03,
) -> OffsetSDSResult:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    init_uv_t = torch.as_tensor(init_uv, dtype=torch.float32, device=device_obj)[None, None]
    max_uv_t = torch.as_tensor(max_uv, dtype=torch.float32, device=device_obj)[None, None]
    mask_t = torch.as_tensor(garment_mask_uv, dtype=torch.float32, device=device_obj)[None, None]
    init_scale_t = torch.as_tensor(init_offset_scale, dtype=torch.float32, device=device_obj)
    fixed_t = torch.as_tensor(fixed_zero_mask, dtype=torch.bool, device=device_obj)
    max_vertex_t = torch.as_tensor(max_vertex_offset, dtype=torch.float32, device=device_obj)
    base_vertices_t = torch.as_tensor(vertices, dtype=torch.float32, device=device_obj)
    normals_t = torch.as_tensor(vertex_normals, dtype=torch.float32, device=device_obj)
    edges = mesh_edges(faces, device)

    renderer = CasTexOffsetRenderer(faces, uv_coords, face_uv_indices, render_resolution, device)
    guidance = CasTexSDSGuidance(geometry_prompt, negative_prompt, castex_root, stage=stage, device=device, guidance_scale=guidance_scale, mode=sds_mode)
    offset_mesh = DifferentiableOffsetMesh(vertices, faces, vertex_normals, vertex_uvs_per_vertex, fixed_zero_mask, max_vertex_offset, device=device)

    curves: dict[str, list[float]] = {
        "loss_sds": [],
        "loss_reg": [],
        "loss_tv": [],
        "loss_mask": [],
        "loss_lap": [],
        "loss_total": [],
        "mean_offset": [],
        "max_offset": [],
    }

    if optimize_mode == "part_scale":
        raw_scale = torch.nn.Parameter(torch.zeros(len(PART_GROUPS), dtype=torch.float32, device=device_obj))
        optimizer = torch.optim.Adam([raw_scale], lr=lr)
        part_ids = _part_index(part_labels, device)
        params = [raw_scale]
    elif optimize_mode == "lowres_uv":
        delta_low = torch.nn.Parameter(torch.zeros((1, 1, lowres_size, lowres_size), dtype=torch.float32, device=device_obj))
        optimizer = torch.optim.Adam([delta_low], lr=lr)
        params = [delta_low]
    else:
        raise ValueError(f"Unknown optimize_mode: {optimize_mode}")

    final_offset_uv = init_uv_t * mask_t
    final_scale = init_scale_t
    final_vertices = base_vertices_t

    for step in range(max(steps, 0) + 1):
        optimizer.zero_grad(set_to_none=True)
        if optimize_mode == "part_scale":
            part_scales = _part_scale_values(raw_scale)
            scale = init_scale_t * part_scales[part_ids]
            scale = torch.minimum(torch.clamp(scale, min=0.0), max_vertex_t)
            scale = torch.where(fixed_t, torch.zeros_like(scale), scale)
            vertices_offset = base_vertices_t + normals_t * scale[:, None]
            offset_uv = init_uv_t * mask_t
            loss_reg = (part_scales - 1.0).square().mean()
            loss_tv = tv_loss(offset_uv)
            loss_mask = mask_loss(offset_uv, mask_t)
        else:
            offset_uv = compose_offset_uv_torch(init_uv_t, max_uv_t, mask_t, delta_low)
            vertices_offset, scale = offset_mesh.forward(offset_uv)
            loss_reg = (offset_uv - init_uv_t).square().mean()
            loss_tv = tv_loss(offset_uv)
            loss_mask = mask_loss(offset_uv, mask_t)
            part_scales = None

        cams = fixed_cameras(("front",), render_resolution, device) if step == 0 else random_cameras(batch_size, render_resolution, device)
        shaded, normal, silhouette = renderer.render(vertices_offset, cameras=cams)
        loss_sds = guidance.loss(shaded)
        loss_lap = laplacian_offset_loss(scale, edges)
        loss_sil = silhouette.sum() * 0.0
        loss_seam = scale.sum() * 0.0
        loss = (
            lambda_sds * loss_sds
            + lambda_reg * loss_reg
            + lambda_tv * loss_tv
            + lambda_mask * loss_mask
            + lambda_lap * loss_lap
            + lambda_sil * loss_sil
            + lambda_seam * loss_seam
        )

        for key, value in (
            ("loss_sds", loss_sds),
            ("loss_reg", loss_reg),
            ("loss_tv", loss_tv),
            ("loss_mask", loss_mask),
            ("loss_lap", loss_lap),
            ("loss_total", loss),
            ("mean_offset", scale.mean()),
            ("max_offset", scale.max()),
        ):
            curves[key].append(float(value.detach().cpu()))

        final_offset_uv = offset_uv.detach()
        final_scale = scale.detach()
        final_vertices = vertices_offset.detach()

        if save_interval > 0 and (step % save_interval == 0 or step == steps):
            _save_debug(out_path, step, shaded, normal, silhouette, offset_uv)

        if step == steps:
            break
        loss.backward()
        optimizer.step()
        for param in params:
            if param.grad is not None:
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)

    final_part_scales = {}
    if optimize_mode == "part_scale":
        values = _part_scale_values(raw_scale).detach().cpu().numpy()
        final_part_scales = {name: float(values[index]) for index, name in enumerate(PART_GROUPS)}

    log = {
        "prompt": prompt,
        "geometry_prompt": geometry_prompt,
        "negative_prompt": negative_prompt,
        "optimize_mode": optimize_mode,
        "stage": stage,
        "steps": steps,
        "loss_curves": curves,
        "final_part_scales": final_part_scales,
        "sds_mode": sds_mode,
    }
    return OffsetSDSResult(
        vertices=final_vertices.cpu().numpy().astype(np.float32),
        offset_scale=final_scale.cpu().numpy().astype(np.float32),
        offset_uv=final_offset_uv.cpu().numpy()[0, 0].astype(np.float32),
        log=log,
    )

