"""Generate an SMPL-X template OBJ and normalized part labels for phase 1."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from load_smplx import MeshData, load_obj


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_SEGMENTATION = (
    WORKSPACE_DIR / "PARTE_RELEASE" / "data" / "body_data" / "smplx_vert_segmentation.json"
)
VENDORED_SMPLX_PARENT = WORKSPACE_DIR / "PARTE_RELEASE" / "lib" / "body_utils"

# PARTE uses SMPL-X joint names. The mask pipeline uses garment-oriented names.
LABEL_PRIORITY = {
    "torso": 10,
    "pelvis": 10,
    "neck": 20,
    "left_upper_arm": 20,
    "right_upper_arm": 20,
    "left_thigh": 20,
    "right_thigh": 20,
    "left_forearm": 30,
    "right_forearm": 30,
    "left_calf": 30,
    "right_calf": 30,
    "head": 40,
    "left_hand": 40,
    "right_hand": 40,
    "left_foot": 40,
    "right_foot": 40,
}

RAW_TO_GARMENT_LABEL = {
    "head": "head",
    "leftEye": "head",
    "rightEye": "head",
    "eyeballs": "head",
    "neck": "neck",
    "spine": "torso",
    "spine1": "torso",
    "spine2": "torso",
    "leftShoulder": "torso",
    "rightShoulder": "torso",
    "hips": "pelvis",
    "leftArm": "left_upper_arm",
    "rightArm": "right_upper_arm",
    "leftForeArm": "left_forearm",
    "rightForeArm": "right_forearm",
    "leftHand": "left_hand",
    "leftHandIndex1": "left_hand",
    "rightHand": "right_hand",
    "rightHandIndex1": "right_hand",
    "leftUpLeg": "left_thigh",
    "rightUpLeg": "right_thigh",
    "leftLeg": "left_calf",
    "rightLeg": "right_calf",
    "leftFoot": "left_foot",
    "leftToeBase": "left_foot",
    "rightFoot": "right_foot",
    "rightToeBase": "right_foot",
}


def import_smplx() -> Any:
    """Import installed smplx, falling back to PARTE's vendored implementation."""
    try:
        return importlib.import_module("smplx")
    except ModuleNotFoundError:
        if not VENDORED_SMPLX_PARENT.exists():
            raise ModuleNotFoundError(
                "Install smplx or provide PARTE_RELEASE/lib/body_utils/smplx"
            ) from None
        sys.path.insert(0, str(VENDORED_SMPLX_PARENT))
        return importlib.import_module("smplx")


def generate_neutral_mesh(model_path: str | Path, gender: str = "neutral") -> MeshData:
    """Run SMPL-X with zero-valued parameters and return its template geometry."""
    smplx = import_smplx()
    model = smplx.create(
        model_path=str(model_path),
        model_type="smplx",
        gender=gender,
        use_pca=False,
        flat_hand_mean=True,
    )
    output = model(return_verts=True)
    vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int32)
    return MeshData(
        vertices=vertices,
        faces=faces,
        uv_coords=np.empty((0, 2), dtype=np.float32),
        face_uv_indices=np.empty((0, 3), dtype=np.int32),
    )


def normalize_part_labels(segmentation_path: str | Path, num_vertices: int) -> list[str]:
    """Convert PARTE-style SMPL-X segmentation into garment-oriented labels."""
    with Path(segmentation_path).open("r", encoding="utf-8") as handle:
        segmentation = json.load(handle)
    if not isinstance(segmentation, dict):
        raise ValueError("SMPL-X segmentation must be a JSON object of vertex-index lists")

    labels: list[str | None] = [None] * num_vertices
    for raw_label, garment_label in RAW_TO_GARMENT_LABEL.items():
        for vertex_index in segmentation.get(raw_label, []):
            if not 0 <= vertex_index < num_vertices:
                raise ValueError(f"Vertex index {vertex_index} is outside the SMPL-X mesh")
            existing = labels[vertex_index]
            if existing is None or LABEL_PRIORITY[garment_label] > LABEL_PRIORITY[existing]:
                labels[vertex_index] = garment_label

    missing = [index for index, label in enumerate(labels) if label is None]
    if missing:
        preview = ", ".join(str(index) for index in missing[:8])
        warnings.warn(
            f"Segmentation leaves {len(missing)} vertices unlabeled ({preview}); assigning them to head",
            stacklevel=2,
        )
        for vertex_index in missing:
            labels[vertex_index] = "head"
    return [str(label) for label in labels]


def attach_uv_from_template(mesh: MeshData, uv_template_path: str | Path) -> MeshData:
    """Copy UV coordinates from an official SMPL-X UV OBJ with matching topology."""
    uv_template = load_obj(uv_template_path)
    if mesh.faces.shape != uv_template.faces.shape or not np.array_equal(mesh.faces, uv_template.faces):
        raise ValueError("UV template faces must exactly match the generated SMPL-X topology")
    if not len(uv_template.uv_coords) or uv_template.face_uv_indices.shape != mesh.faces.shape:
        raise ValueError("UV template OBJ must contain vt entries for every face corner")
    mesh.uv_coords = uv_template.uv_coords
    mesh.face_uv_indices = uv_template.face_uv_indices
    return mesh


def write_obj(mesh: MeshData, path: str | Path) -> None:
    """Write geometry and optional UV coordinates without discarding UV seams."""
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("# Generated neutral SMPL-X template\n")
        for vertex in mesh.vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for uv in mesh.uv_coords:
            handle.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")
        if len(mesh.face_uv_indices):
            for face, face_uv in zip(mesh.faces, mesh.face_uv_indices):
                corners = [f"{vertex + 1}/{uv + 1}" for vertex, uv in zip(face, face_uv)]
                handle.write(f"f {' '.join(corners)}\n")
        else:
            for face in mesh.faces:
                handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def generate_assets(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = generate_neutral_mesh(args.model_path, args.gender)
    labels = normalize_part_labels(args.segmentation, len(mesh.vertices))
    if args.uv_template:
        mesh = attach_uv_from_template(mesh, args.uv_template)

    write_obj(mesh, out_dir / "smplx_template.obj")
    with (out_dir / "part_labels.json").open("w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)
    if len(mesh.uv_coords):
        np.savez(
            out_dir / "uv_data.npz",
            uv_coords=mesh.uv_coords,
            face_uv_indices=mesh.face_uv_indices,
        )
    else:
        print("Generated geometry and labels without UV data. Re-run with --uv-template.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="SMPL-X model root or SMPLX_NEUTRAL.npz file downloaded under its license",
    )
    parser.add_argument("--gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument(
        "--segmentation",
        default=str(DEFAULT_SEGMENTATION),
        help="PARTE-style SMPL-X vertex segmentation JSON",
    )
    parser.add_argument(
        "--uv-template",
        help="Official SMPL-X UV template OBJ with matching face topology",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_DIR / "data" / "smplx" / "generated"),
    )
    return parser


if __name__ == "__main__":
    generate_assets(build_parser().parse_args())
