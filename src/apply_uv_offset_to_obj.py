"""Apply a UV scalar offset map to an OBJ mesh along vertex normals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from load_smplx import MeshData, load_part_labels, load_smplx
from offset_sds.offset_map_torch import build_vertex_uvs_per_vertex


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(norm, 1e-12)).astype(np.float32)


def apply_normal_offset(vertices: np.ndarray, normals: np.ndarray, offset_scale: np.ndarray) -> np.ndarray:
    return vertices.astype(np.float32) + normals.astype(np.float32) * offset_scale.astype(np.float32)[:, None]


def build_vertex_adjacency(num_vertices: int, faces: np.ndarray) -> list[np.ndarray]:
    neighbors = [set() for _ in range(num_vertices)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [np.asarray(sorted(item), dtype=np.int32) for item in neighbors]


def smooth_offset_scale(
    offset_scale: np.ndarray,
    faces: np.ndarray,
    fixed_zero_mask: np.ndarray,
    max_scale: np.ndarray,
    iterations: int = 5,
    lam: float = 0.5,
) -> np.ndarray:
    smoothed = offset_scale.astype(np.float32).copy()
    adjacency = build_vertex_adjacency(len(smoothed), faces)
    for _ in range(iterations):
        updated = smoothed.copy()
        for index, neighbors in enumerate(adjacency):
            if fixed_zero_mask[index] or len(neighbors) == 0:
                continue
            valid = neighbors[~fixed_zero_mask[neighbors]]
            if len(valid) == 0:
                continue
            updated[index] = (1.0 - lam) * smoothed[index] + lam * float(np.mean(smoothed[valid]))
        updated[fixed_zero_mask] = 0.0
        smoothed = np.clip(updated, 0.0, max_scale.astype(np.float32))
    return smoothed


def export_obj(mesh: MeshData, vertices: np.ndarray, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for uv in mesh.uv_coords:
            handle.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")
        for face, face_uv in zip(mesh.faces, mesh.face_uv_indices):
            corners = [f"{vertex + 1}/{uv + 1}" for vertex, uv in zip(face, face_uv)]
            handle.write(f"f {' '.join(corners)}\n")


def load_offset_uv(path: str | Path, max_offset: float = 1.0) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        data = np.load(path).astype(np.float32)
        if data.ndim == 4:
            data = data[0, 0]
        elif data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D offset map in {path}, got shape {data.shape}")
        return data

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("Reading image offset maps requires OpenCV/cv2. Use a .npy offset map or run in the castex environment.") from exc

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denom = 65535.0 if image.dtype == np.uint16 else 255.0
    image = image.astype(np.float32) / denom
    return image * float(max_offset)


def sample_uv_map_to_vertices(offset_uv: np.ndarray, vertex_uvs_per_vertex: list[list[list[float]]]) -> np.ndarray:
    h, w = offset_uv.shape
    offset_scale = np.zeros(len(vertex_uvs_per_vertex), dtype=np.float32)
    for vertex_index, uv_list in enumerate(vertex_uvs_per_vertex):
        if not uv_list:
            continue
        samples = []
        for uv in uv_list:
            u = np.clip(float(uv[0]), 0.0, 1.0) * (w - 1)
            v = (1.0 - np.clip(float(uv[1]), 0.0, 1.0)) * (h - 1)
            x0 = int(np.floor(u))
            y0 = int(np.floor(v))
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)
            dx = u - x0
            dy = v - y0
            top = offset_uv[y0, x0] * (1.0 - dx) + offset_uv[y0, x1] * dx
            bottom = offset_uv[y1, x0] * (1.0 - dx) + offset_uv[y1, x1] * dx
            samples.append(top * (1.0 - dy) + bottom * dy)
        offset_scale[vertex_index] = float(np.mean(samples))
    return offset_scale


def fixed_zero_mask_from_labels(labels: list[str]) -> np.ndarray:
    forced = {"head", "left_hand", "right_hand", "left_foot", "right_foot"}
    return np.asarray([label in forced for label in labels], dtype=bool)


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_smplx(args.mesh, args.uv)
    offset_uv = load_offset_uv(args.offset_uv, max_offset=args.max_offset)
    vertex_uvs = build_vertex_uvs_per_vertex(len(mesh.vertices), mesh.faces, mesh.uv_coords, mesh.face_uv_indices)
    offset_scale = sample_uv_map_to_vertices(offset_uv, vertex_uvs)
    offset_scale *= float(args.scale_multiplier)

    if args.max_vertex_offset is not None:
        offset_scale = np.clip(offset_scale, 0.0, float(args.max_vertex_offset))
    else:
        offset_scale = np.maximum(offset_scale, 0.0)

    fixed_zero_mask = np.zeros(len(mesh.vertices), dtype=bool)
    if args.part_labels is not None:
        fixed_zero_mask = fixed_zero_mask_from_labels(load_part_labels(args.part_labels, len(mesh.vertices)))
        offset_scale[fixed_zero_mask] = 0.0

    if args.smooth_iterations > 0:
        max_scale = np.full(len(mesh.vertices), np.inf, dtype=np.float32)
        if args.max_vertex_offset is not None:
            max_scale[:] = float(args.max_vertex_offset)
        max_scale[fixed_zero_mask] = 0.0
        offset_scale = smooth_offset_scale(
            offset_scale,
            mesh.faces,
            fixed_zero_mask,
            max_scale,
            iterations=args.smooth_iterations,
            lam=args.smooth_lambda,
        )

    normals = compute_vertex_normals(mesh.vertices, mesh.faces)
    offset_vertices = apply_normal_offset(mesh.vertices, normals, offset_scale)
    export_obj(mesh, offset_vertices, out_dir / args.out_name)
    np.save(out_dir / "applied_offset_scale.npy", offset_scale.astype(np.float32))

    stats = {
        "mesh": str(args.mesh),
        "offset_uv": str(args.offset_uv),
        "out_obj": str(out_dir / args.out_name),
        "max_offset": float(offset_scale.max()) if len(offset_scale) else 0.0,
        "mean_offset": float(offset_scale.mean()) if len(offset_scale) else 0.0,
        "nonzero_vertices": int(np.count_nonzero(offset_scale > 0.0)),
        "fixed_zero_vertices": int(np.count_nonzero(fixed_zero_mask)),
    }
    with (out_dir / "apply_uv_offset_log.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Input OBJ with vertices, faces, and vt UVs.")
    parser.add_argument("--uv", default=None, help="Optional uv_data.npz overriding OBJ vt/face UV indices.")
    parser.add_argument("--offset-uv", required=True, help="Offset UV map, either .npy with real values or grayscale .png.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-name", default="uv_offset_mesh.obj")
    parser.add_argument("--max-offset", type=float, default=1.0, help="Physical offset represented by white pixels when --offset-uv is an image.")
    parser.add_argument("--max-vertex-offset", type=float, default=None, help="Optional clamp after sampling UV to vertices.")
    parser.add_argument("--scale-multiplier", type=float, default=1.0, help="Extra multiplier for quick visual exaggeration.")
    parser.add_argument("--part-labels", default=None, help="Optional part_labels.json; head/hands/feet are forced to zero.")
    parser.add_argument("--smooth-iterations", type=int, default=0)
    parser.add_argument("--smooth-lambda", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

