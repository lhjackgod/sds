"""Command-line entrypoint for phase-1 prompt-aware UV mask generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image
import numpy as np

from debug_render import save_debug_assets
from load_smplx import load_part_labels, load_smplx
from mask_postprocess import ProcessedMasks, postprocess_masks
from parse_prompt import parse_prompt
from part_mapping import build_vertex_masks
from uv_rasterizer import rasterize_vertex_masks_to_uv


def _save_vertex_masks(vertex_masks, out_dir: Path) -> None:
    np.save(out_dir / "vertex_upper_mask.npy", vertex_masks.upper.astype(bool))
    np.save(out_dir / "vertex_lower_mask.npy", vertex_masks.lower.astype(bool))
    np.save(out_dir / "vertex_skin_mask.npy", vertex_masks.skin.astype(bool))
    labels = np.zeros(len(vertex_masks.skin), dtype=np.uint8)
    labels[vertex_masks.skin] = 1
    labels[vertex_masks.upper] = 2
    labels[vertex_masks.lower] = 3
    np.save(out_dir / "vertex_region_labels.npy", labels)


def _save_masks(masks: ProcessedMasks, out_dir: Path) -> None:
    images = {
        "upper_mask.png": masks.upper,
        "lower_mask.png": masks.lower,
        "skin_mask.png": masks.skin,
        "upper_soft_mask.png": masks.upper_soft,
        "lower_soft_mask.png": masks.lower_soft,
        "skin_soft_mask.png": masks.skin_soft,
        "boundary_mask.png": masks.boundary,
        "combined_region_map.png": masks.combined_region_map,
    }
    for name, image in images.items():
        Image.fromarray(image).save(out_dir / name)


def generate_masks(args: argparse.Namespace) -> None:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    postprocess_config = config.get("postprocess", {})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = parse_prompt(args.prompt)
    mesh = load_smplx(args.mesh, args.uv)
    labels = load_part_labels(args.part_labels, len(mesh.vertices))
    vertex_masks = build_vertex_masks(
        spec,
        labels,
        mesh.vertices,
        body_up_axis=config.get("body_up_axis", "y"),
        include_neck_for_vest=config.get("include_neck_for_vest", False),
    )
    uv_masks = rasterize_vertex_masks_to_uv(
        vertex_masks,
        mesh.faces,
        mesh.uv_coords,
        mesh.face_uv_indices,
        resolution=args.resolution,
    )
    masks = postprocess_masks(uv_masks, **postprocess_config)
    _save_masks(masks, out_dir)
    _save_vertex_masks(vertex_masks, out_dir)
    save_debug_assets(mesh, masks, out_dir, vertex_masks=vertex_masks, raw_uv_masks=uv_masks)
    with (out_dir / "garment_spec.json").open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mesh", required=True, help="SMPL-X triangular template OBJ")
    parser.add_argument("--uv", help="Optional NPZ containing uv_coords and face_uv_indices")
    parser.add_argument("--part-labels", required=True, help="Per-vertex body-part label JSON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "garment_rules.yaml"),
    )
    return parser


if __name__ == "__main__":
    generate_masks(build_parser().parse_args())

