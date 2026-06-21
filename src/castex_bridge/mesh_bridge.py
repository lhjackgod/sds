"""Mesh helpers for bridge compatibility."""

from __future__ import annotations

import torch


def tensors_from_mesh(mesh, device: str):
    return {
        "vertices": torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device),
        "faces": torch.as_tensor(mesh.faces, dtype=torch.long, device=device),
        "uvs": torch.as_tensor(mesh.uv_coords, dtype=torch.float32, device=device),
        "face_uv_indices": torch.as_tensor(mesh.face_uv_indices, dtype=torch.long, device=device),
    }

