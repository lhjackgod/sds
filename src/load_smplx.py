"""Load a triangular OBJ template, UV data, and per-vertex body labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    uv_coords: np.ndarray
    face_uv_indices: np.ndarray


def _obj_index(raw_index: str, size: int) -> int:
    index = int(raw_index)
    return index - 1 if index > 0 else size + index


def _triangulate(items: list[tuple[int, int | None]]) -> Iterable[list[tuple[int, int | None]]]:
    for index in range(1, len(items) - 1):
        yield [items[0], items[index], items[index + 1]]


def load_obj(path: str | Path) -> MeshData:
    vertices: list[list[float]] = []
    uv_coords: list[list[float]] = []
    faces: list[list[int]] = []
    face_uv_indices: list[list[int]] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields or fields[0] == "#":
                continue
            if fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "vt":
                uv_coords.append([float(value) for value in fields[1:3]])
            elif fields[0] == "f":
                corners: list[tuple[int, int | None]] = []
                for field in fields[1:]:
                    indices = field.split("/")
                    vertex_index = _obj_index(indices[0], len(vertices))
                    uv_index = (
                        _obj_index(indices[1], len(uv_coords))
                        if len(indices) > 1 and indices[1]
                        else None
                    )
                    corners.append((vertex_index, uv_index))
                for triangle in _triangulate(corners):
                    faces.append([corner[0] for corner in triangle])
                    if all(corner[1] is not None for corner in triangle):
                        face_uv_indices.append([int(corner[1]) for corner in triangle])

    return MeshData(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        uv_coords=np.asarray(uv_coords, dtype=np.float32),
        face_uv_indices=np.asarray(face_uv_indices, dtype=np.int32),
    )


def load_smplx(mesh_path: str | Path, uv_path: str | Path | None = None) -> MeshData:
    mesh = load_obj(mesh_path)
    if uv_path is not None:
        uv_data = np.load(uv_path)
        mesh.uv_coords = np.asarray(uv_data["uv_coords"], dtype=np.float32)
        mesh.face_uv_indices = np.asarray(uv_data["face_uv_indices"], dtype=np.int32)

    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("Mesh must contain vertices and faces")
    if mesh.faces.shape != mesh.face_uv_indices.shape:
        raise ValueError("UV indices must contain one UV triangle for every mesh face")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("Mesh faces must be triangular after loading")
    if not len(mesh.uv_coords):
        raise ValueError("UV coordinates are required in OBJ vt entries or --uv NPZ")
    return mesh


def load_part_labels(path: str | Path, num_vertices: int | None = None) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("vertex_labels", data.get("labels"))
    if not isinstance(data, list) or not all(isinstance(label, str) for label in data):
        raise ValueError("Part labels must be a JSON list or an object containing labels")
    if num_vertices is not None and len(data) != num_vertices:
        raise ValueError(f"Expected {num_vertices} part labels, got {len(data)}")
    return data

