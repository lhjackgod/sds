"""Run SDS-guided UV offset-map optimization for bounded garment shells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from castex_bridge.prompt_bridge import build_geometry_prompt, build_negative_prompt
from debug_render import build_vertex_colors
from differentiable_renderer import render_normals, render_shaded, render_silhouette
from load_smplx import MeshData, load_part_labels, load_smplx
from offset_map import rasterize_vertex_scalar_to_uv, save_offset_uv_png
from offset_rules import build_offset_rules, load_offset_config
from offset_sds.offset_map_torch import build_vertex_uvs_per_vertex
from offset_sds.offset_sds_optimizer import optimize_uv_offset_sds
from offset_shell import compute_vertex_normals, export_obj, export_offset_vertex_colors_ply
from parse_prompt import parse_prompt
from part_mapping import VertexMasks, build_vertex_masks


PROJECT_DIR = Path(__file__).resolve().parents[1]




_COLOR_TABLE = {
    "black": (0.03, 0.03, 0.03),
    "white": (0.88, 0.88, 0.84),
    "gray": (0.48, 0.48, 0.48),
    "grey": (0.48, 0.48, 0.48),
    "red": (0.82, 0.12, 0.10),
    "blue": (0.08, 0.18, 0.72),
    "green": (0.12, 0.48, 0.18),
    "yellow": (0.86, 0.74, 0.12),
    "orange": (0.90, 0.40, 0.08),
    "purple": (0.42, 0.18, 0.66),
    "pink": (0.90, 0.36, 0.60),
    "brown": (0.35, 0.20, 0.10),
    "dark": (0.08, 0.08, 0.10),
}


def _color_from_spec(value, fallback):
    if value is None:
        return np.asarray(fallback, dtype=np.float32)
    return np.asarray(_COLOR_TABLE.get(str(value).lower(), fallback), dtype=np.float32)


def build_prompt_vertex_colors(vertex_masks: VertexMasks, spec: dict) -> np.ndarray:
    colors = np.zeros((len(vertex_masks.skin), 3), dtype=np.float32)
    skin = np.asarray((0.72, 0.66, 0.60), dtype=np.float32)
    upper = _color_from_spec(spec.get("upper", {}).get("color"), (0.78, 0.10, 0.08))
    lower = _color_from_spec(spec.get("lower", {}).get("color"), (0.08, 0.16, 0.55))
    colors[:] = 0.55
    colors[vertex_masks.skin] = skin
    colors[vertex_masks.upper] = upper
    colors[vertex_masks.lower] = lower
    return colors

def _load_vertex_masks(mask_dir: Path) -> VertexMasks | None:
    paths = [mask_dir / "vertex_upper_mask.npy", mask_dir / "vertex_lower_mask.npy", mask_dir / "vertex_skin_mask.npy"]
    if not all(path.exists() for path in paths):
        return None
    return VertexMasks(*(np.load(path).astype(bool) for path in paths))


def _save_offset_debug_texture(offset_uv: np.ndarray, path: Path) -> None:
    denom = max(float(offset_uv.max()), 1e-8)
    color = cv2.applyColorMap((np.clip(offset_uv / denom, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), color)


def _load_init_scale(init_dir: Path | None, mesh: MeshData, fallback_scale: np.ndarray) -> tuple[np.ndarray, str]:
    if init_dir is None:
        return fallback_scale.astype(np.float32), "rule_init"
    candidates = ("optimized_offset_scale.npy", "offset_scale.npy")
    for name in candidates:
        path = init_dir / name
        if path.exists():
            data = np.load(path).astype(np.float32)
            if len(data) != len(mesh.vertices):
                raise ValueError(f"{path} has {len(data)} values, expected {len(mesh.vertices)} mesh vertices")
            return data, str(path)
    return fallback_scale.astype(np.float32), "rule_init"


def _stats(offset_scale: np.ndarray) -> dict:
    nonzero = offset_scale[offset_scale > 0]
    return {
        "max": float(offset_scale.max()) if len(offset_scale) else 0.0,
        "mean": float(offset_scale.mean()) if len(offset_scale) else 0.0,
        "nonzero_mean": float(nonzero.mean()) if len(nonzero) else 0.0,
        "nonzero_vertices": int(len(nonzero)),
    }


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir)
    init_dir = Path(args.init_offset_dir) if args.init_offset_dir else None

    with Path(args.garment_config).open("r", encoding="utf-8") as handle:
        garment_config = yaml.safe_load(handle) or {}
    offset_config = load_offset_config(args.offset_config)

    spec = parse_prompt(args.prompt)
    mesh = load_smplx(args.mesh, args.uv)
    part_labels = load_part_labels(args.part_labels, len(mesh.vertices))
    vertex_masks = _load_vertex_masks(mask_dir)
    if vertex_masks is None:
        vertex_masks = build_vertex_masks(spec, part_labels, mesh.vertices, body_up_axis=garment_config.get("body_up_axis", "y"))

    resolution = args.resolution
    if resolution is None and (mask_dir / "upper_mask.png").exists():
        resolution = int(Image.open(mask_dir / "upper_mask.png").size[0])
    resolution = resolution or 1024

    rule_result = build_offset_rules(spec, part_labels, vertex_masks, mesh, offset_config, resolution=resolution)
    init_scale, init_source = _load_init_scale(init_dir, mesh, rule_result.vertex_init)
    init_uv = rasterize_vertex_scalar_to_uv(init_scale, mesh.faces, mesh.uv_coords, mesh.face_uv_indices, resolution, mask_values=~rule_result.fixed_zero_mask)
    init_uv = np.clip(init_uv, 0.0, rule_result.uv_maps.max) * rule_result.uv_maps.garment_mask

    normals = compute_vertex_normals(mesh.vertices, mesh.faces)
    init_vertices = mesh.vertices + normals * init_scale[:, None]
    export_obj(mesh, init_vertices, out_dir / "init_offset_mesh.obj")
    np.save(out_dir / "init_offset_scale.npy", init_scale.astype(np.float32))
    save_offset_uv_png(init_uv, str(out_dir / "init_offset_scale_uv.png"), max_value=float(rule_result.uv_maps.max.max()))
    vertex_uvs = build_vertex_uvs_per_vertex(len(mesh.vertices), mesh.faces, mesh.uv_coords, mesh.face_uv_indices)
    geometry_prompt = build_geometry_prompt(args.prompt)
    negative_prompt = build_negative_prompt()

    vertex_colors = build_prompt_vertex_colors(vertex_masks, spec)

    result = optimize_uv_offset_sds(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_normals=normals,
        vertex_uvs_per_vertex=vertex_uvs,
        uv_coords=mesh.uv_coords,
        face_uv_indices=mesh.face_uv_indices,
        init_uv=init_uv,
        max_uv=rule_result.uv_maps.max,
        garment_mask_uv=rule_result.uv_maps.garment_mask,
        init_offset_scale=init_scale,
        fixed_zero_mask=rule_result.fixed_zero_mask,
        max_vertex_offset=rule_result.vertex_max,
        part_labels=part_labels,
        prompt=args.prompt,
        geometry_prompt=geometry_prompt,
        negative_prompt=negative_prompt,
        castex_root=args.castex_root,
        out_dir=out_dir,
        optimize_mode=args.optimize_mode,
        stage=args.stage,
        steps=args.steps,
        batch_size=args.batch_size,
        render_resolution=args.render_resolution,
        guidance_scale=args.guidance_scale,
        device=args.device,
        sds_mode=args.sds_mode,
        lowres_size=args.lowres_size,
        save_interval=args.save_interval,
        lambda_sds=args.lambda_sds,
        lambda_reg=args.lambda_reg,
        lambda_tv=args.lambda_tv,
        lambda_mask=args.lambda_mask,
        lambda_lap=args.lambda_lap,
        lambda_sil=args.lambda_sil,
        lambda_seam=args.lambda_seam,
        lr=args.lr,
        vertex_colors=vertex_colors,
        fp16=args.fp16,
    )

    out_mesh = MeshData(result.vertices, mesh.faces, mesh.uv_coords, mesh.face_uv_indices)
    export_obj(mesh, result.vertices, out_dir / "optimized_offset_mesh.obj")
    np.save(out_dir / "optimized_offset_scale.npy", result.offset_scale)
    np.save(out_dir / "optimized_offset_scale_uv.npy", result.offset_uv)
    save_offset_uv_png(result.offset_uv, str(out_dir / "optimized_offset_scale_uv.png"), max_value=float(rule_result.uv_maps.max.max()))
    _save_offset_debug_texture(result.offset_uv, out_dir / "optimized_offset_debug_texture.png")
    export_offset_vertex_colors_ply(result.vertices, mesh.faces, vertex_masks, result.offset_scale, out_dir / "optimized_vertex_colors.ply")
    Image.fromarray(render_shaded(result.vertices, mesh.faces, view="front", resolution=args.render_resolution)).save(out_dir / "optimized_render_shaded_front.png")
    Image.fromarray(render_normals(result.vertices, mesh.faces, view="front", resolution=args.render_resolution)).save(out_dir / "optimized_render_normal_front.png")
    Image.fromarray(render_silhouette(result.vertices, mesh.faces, view="front", resolution=args.render_resolution)).save(out_dir / "optimized_render_silhouette_front.png")

    log = {**result.log, "init_offset_source": init_source, "offset_stats": _stats(result.offset_scale)}
    with (out_dir / "optimization_log.json").open("w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mesh", default="data/smplx/generated/smplx_template.obj")
    parser.add_argument("--uv", default=None)
    parser.add_argument("--part-labels", default="data/smplx/generated/part_labels.json")
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--init-offset-dir", default=None)
    parser.add_argument("--castex-root", default="../CasTex")
    parser.add_argument("--out", required=True)
    parser.add_argument("--optimize-mode", choices=("part_scale", "lowres_uv"), default="part_scale")
    parser.add_argument("--stage", choices=("i", "ii"), default="i")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--render-resolution", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=15.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sds-mode", choices=("dummy", "castex"), default="castex")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--lowres-size", type=int, default=64)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--lambda-sds", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=100.0)
    parser.add_argument("--lambda-tv", type=float, default=0.0)
    parser.add_argument("--lambda-mask", type=float, default=100.0)
    parser.add_argument("--lambda-lap", type=float, default=20.0)
    parser.add_argument("--lambda-sil", type=float, default=0.0)
    parser.add_argument("--lambda-seam", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--garment-config", default=str(PROJECT_DIR / "configs" / "garment_rules.yaml"))
    parser.add_argument("--offset-config", default=str(PROJECT_DIR / "configs" / "offset_rules.yaml"))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

