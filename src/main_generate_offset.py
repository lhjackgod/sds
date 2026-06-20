"""Generate SO-SMPL-inspired bounded garment normal-offset shells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from debug_render import build_vertex_colors
from differentiable_renderer import render_normals, render_region, render_shaded, render_silhouette
from load_smplx import load_part_labels, load_smplx
from offset_map import build_vertex_uv_coords, save_offset_uv_png
from offset_rules import build_offset_rules, load_offset_config
from offset_shell import apply_normal_offset, compute_vertex_normals, export_obj, export_offset_vertex_colors_ply
from optimize_offset_sds import optimize_offset
from parse_prompt import parse_prompt
from part_mapping import VertexMasks, build_vertex_masks


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load_phase1_vertex_masks(mask_dir: Path) -> VertexMasks | None:
    paths = [mask_dir / "vertex_upper_mask.npy", mask_dir / "vertex_lower_mask.npy", mask_dir / "vertex_skin_mask.npy"]
    if not all(path.exists() for path in paths):
        return None
    return VertexMasks(upper=np.load(paths[0]).astype(bool), lower=np.load(paths[1]).astype(bool), skin=np.load(paths[2]).astype(bool))


def _save_phase1_vertex_masks(mask_dir: Path, vertex_masks: VertexMasks) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.save(mask_dir / "vertex_upper_mask.npy", vertex_masks.upper.astype(bool))
    np.save(mask_dir / "vertex_lower_mask.npy", vertex_masks.lower.astype(bool))
    np.save(mask_dir / "vertex_skin_mask.npy", vertex_masks.skin.astype(bool))
    labels = np.zeros(len(vertex_masks.skin), dtype=np.uint8)
    labels[vertex_masks.skin] = 1
    labels[vertex_masks.upper] = 2
    labels[vertex_masks.lower] = 3
    np.save(mask_dir / "vertex_region_labels.npy", labels)


def _mask_resolution(mask_dir: Path, fallback: int) -> int:
    upper = mask_dir / "upper_mask.png"
    if not upper.exists():
        return fallback
    return int(Image.open(upper).size[0])


def _save_offset_debug_texture(offset_uv: np.ndarray, path: Path) -> None:
    denom = max(float(offset_uv.max()), 1e-8)
    gray = np.clip(offset_uv / denom, 0.0, 1.0)
    color = cv2.applyColorMap((gray * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), color)


def _stats(offset_scale: np.ndarray) -> dict:
    nonzero = offset_scale[offset_scale > 0]
    return {
        "min": float(offset_scale.min()) if len(offset_scale) else 0.0,
        "max": float(offset_scale.max()) if len(offset_scale) else 0.0,
        "mean": float(offset_scale.mean()) if len(offset_scale) else 0.0,
        "nonzero_mean": float(nonzero.mean()) if len(nonzero) else 0.0,
        "nonzero_vertices": int(len(nonzero)),
    }


def generate_offset(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir)

    with Path(args.garment_config).open("r", encoding="utf-8") as handle:
        garment_config = yaml.safe_load(handle) or {}
    offset_config = load_offset_config(args.offset_config)

    spec = parse_prompt(args.prompt)
    mesh = load_smplx(args.mesh, args.uv)
    part_labels = load_part_labels(args.part_labels, len(mesh.vertices))
    vertex_masks = _load_phase1_vertex_masks(mask_dir)
    if vertex_masks is None:
        vertex_masks = build_vertex_masks(
            spec,
            part_labels,
            mesh.vertices,
            body_up_axis=garment_config.get("body_up_axis", "y"),
            include_neck_for_vest=garment_config.get("include_neck_for_vest", False),
        )
        _save_phase1_vertex_masks(mask_dir, vertex_masks)

    resolution = args.resolution or _mask_resolution(mask_dir, 1024)
    rule_result = build_offset_rules(spec, part_labels, vertex_masks, mesh, offset_config, resolution=resolution)
    vertex_uv_coords = build_vertex_uv_coords(len(mesh.vertices), mesh.faces, mesh.uv_coords, mesh.face_uv_indices)
    opt_result = optimize_offset(
        rule_result.uv_maps.init,
        rule_result.uv_maps.max,
        rule_result.uv_maps.garment_mask,
        vertex_uv_coords,
        mesh.faces,
        rule_result.fixed_zero_mask,
        rule_result.vertex_max,
        prompt=args.prompt,
        mode=args.mode,
        optimize_mode=args.optimize_mode,
        sds_mode=args.sds_mode,
        steps=args.steps,
        smooth_iterations=args.smooth_iterations,
    )

    normals = compute_vertex_normals(mesh.vertices, mesh.faces)
    offset_vertices = apply_normal_offset(mesh.vertices, normals, opt_result.offset_scale)

    export_obj(mesh, offset_vertices, out_dir / "offset_mesh.obj")
    np.save(out_dir / "offset_scale.npy", opt_result.offset_scale.astype(np.float32))
    save_offset_uv_png(opt_result.offset_uv, str(out_dir / "offset_scale_uv.png"), max_value=float(rule_result.uv_maps.max.max()))
    _save_offset_debug_texture(opt_result.offset_uv, out_dir / "offset_debug_texture.png")
    export_offset_vertex_colors_ply(offset_vertices, mesh.faces, vertex_masks, opt_result.offset_scale, out_dir / "offset_vertex_colors.ply")

    vertex_colors = build_vertex_colors(vertex_masks)
    Image.fromarray(render_region(mesh.vertices, mesh.faces, vertex_colors, "front", args.render_resolution)).save(out_dir / "render_body_front.png")
    Image.fromarray(render_region(offset_vertices, mesh.faces, vertex_colors, "front", args.render_resolution)).save(out_dir / "render_offset_front.png")
    Image.fromarray(render_normals(offset_vertices, mesh.faces, "front", args.render_resolution)).save(out_dir / "render_normal_front.png")
    Image.fromarray(render_silhouette(offset_vertices, mesh.faces, "front", args.render_resolution)).save(out_dir / "render_silhouette_front.png")
    Image.fromarray(render_shaded(offset_vertices, mesh.faces, view="front", resolution=args.render_resolution)).save(out_dir / "render_shaded_front.png")

    log = {
        "prompt": args.prompt,
        "garment_spec": spec,
        "mode": args.mode,
        "optimize_mode": args.optimize_mode,
        "sds_mode": args.sds_mode,
        "steps": args.steps,
        "offset_stats": _stats(opt_result.offset_scale),
        **opt_result.log,
    }
    with (out_dir / "optimization_log.json").open("w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mesh", default="data/smplx/generated/smplx_template.obj")
    parser.add_argument("--uv", default=None)
    parser.add_argument("--part-labels", default="data/smplx/generated/part_labels.json")
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("init", "sds"), default="init")
    parser.add_argument("--optimize-mode", choices=("none", "part_scale", "lowres_uv"), default="none")
    parser.add_argument("--sds-mode", choices=("dummy", "diffusers"), default="dummy")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--smooth-iterations", type=int, default=5)
    parser.add_argument("--garment-config", default=str(PROJECT_DIR / "configs" / "garment_rules.yaml"))
    parser.add_argument("--offset-config", default=str(PROJECT_DIR / "configs" / "offset_rules.yaml"))
    return parser


if __name__ == "__main__":
    generate_offset(build_parser().parse_args())
