"""Create colored debug textures and meshes for region inspection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from load_smplx import MeshData
from mask_postprocess import ProcessedMasks
from part_mapping import VertexMasks
from uv_rasterizer import UVMasks


SKIN_COLOR = (150, 150, 150)
UPPER_COLOR = (220, 50, 50)
LOWER_COLOR = (50, 90, 220)
BOUNDARY_COLOR = (250, 220, 40)
INVALID_COLOR = (0, 0, 0)


def build_debug_texture(masks: ProcessedMasks | UVMasks, show_boundary: bool = True) -> np.ndarray:
    texture = np.zeros((*masks.skin.shape, 3), dtype=np.uint8)
    texture[masks.skin > 0] = SKIN_COLOR
    texture[masks.upper > 0] = UPPER_COLOR
    texture[masks.lower > 0] = LOWER_COLOR
    if show_boundary and isinstance(masks, ProcessedMasks):
        texture[masks.boundary > 0] = BOUNDARY_COLOR
    return texture


def build_vertex_colors(vertex_masks: VertexMasks) -> np.ndarray:
    colors = np.zeros((len(vertex_masks.skin), 3), dtype=np.uint8)
    colors[:] = INVALID_COLOR
    colors[vertex_masks.skin] = SKIN_COLOR
    colors[vertex_masks.upper] = UPPER_COLOR
    colors[vertex_masks.lower] = LOWER_COLOR
    return colors


def export_debug_obj(mesh: MeshData, out_dir: str | Path, texture_name: str = "debug_texture.png") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "debug_mesh.obj"
    mtl_path = out_dir / "debug_mesh.mtl"

    with mtl_path.open("w", encoding="utf-8") as handle:
        handle.write("newmtl garment_regions\n")
        handle.write("Ka 1.000 1.000 1.000\n")
        handle.write("Kd 1.000 1.000 1.000\n")
        handle.write(f"map_Kd {texture_name}\n")

    with obj_path.open("w", encoding="utf-8") as handle:
        handle.write("mtllib debug_mesh.mtl\n")
        handle.write("usemtl garment_regions\n")
        for vertex in mesh.vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for uv in mesh.uv_coords:
            handle.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")
        for face, face_uv in zip(mesh.faces, mesh.face_uv_indices):
            corners = [f"{vertex + 1}/{uv + 1}" for vertex, uv in zip(face, face_uv)]
            handle.write(f"f {' '.join(corners)}\n")


def export_vertex_color_ply(mesh: MeshData, vertex_masks: VertexMasks, out_path: str | Path) -> None:
    colors = build_vertex_colors(vertex_masks)
    with Path(out_path).open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(mesh.vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write(f"element face {len(mesh.faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex, color in zip(mesh.vertices, colors):
            handle.write(
                f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in mesh.faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def save_debug_assets(
    mesh: MeshData,
    masks: ProcessedMasks,
    out_dir: str | Path,
    vertex_masks: VertexMasks | None = None,
    raw_uv_masks: UVMasks | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(build_debug_texture(masks), mode="RGB").save(out_dir / "debug_texture.png")
    if raw_uv_masks is not None:
        Image.fromarray(build_debug_texture(raw_uv_masks, show_boundary=False), mode="RGB").save(
            out_dir / "raw_vertex_debug_texture.png"
        )
    if vertex_masks is not None:
        export_vertex_color_ply(mesh, vertex_masks, out_dir / "vertex_region_mesh.ply")
    export_debug_obj(mesh, out_dir)
