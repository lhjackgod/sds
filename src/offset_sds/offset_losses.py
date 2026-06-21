"""Regularization losses for bounded offset refinement."""

from __future__ import annotations

import torch


def mesh_edges(faces, device: str) -> torch.Tensor:
    f = torch.as_tensor(faces, dtype=torch.long, device=device)
    edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
    return torch.unique(torch.sort(edges, dim=1).values, dim=0)


def laplacian_offset_loss(offset_scale: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.numel() == 0:
        return offset_scale.sum() * 0.0
    return (offset_scale[edges[:, 0]] - offset_scale[edges[:, 1]]).square().mean()


def mask_loss(offset_uv: torch.Tensor, garment_mask_uv: torch.Tensor) -> torch.Tensor:
    return (((1.0 - garment_mask_uv) * offset_uv) ** 2).mean()

