"""Camera sampling utilities used by the differentiable debug renderer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class OrbitCameras:
    azimuth: torch.Tensor
    elevation: torch.Tensor
    distance: torch.Tensor
    resolution: int


def fixed_cameras(names: tuple[str, ...] = ("front", "back", "left", "right"), resolution: int = 256, device: str = "cuda") -> OrbitCameras:
    lookup = {"front": 0.0, "back": math.pi, "left": math.pi / 2.0, "right": -math.pi / 2.0}
    azimuth = torch.tensor([lookup[name] for name in names], dtype=torch.float32, device=device)
    elevation = torch.zeros_like(azimuth)
    distance = torch.full_like(azimuth, 2.7)
    return OrbitCameras(azimuth=azimuth, elevation=elevation, distance=distance, resolution=resolution)


def random_cameras(batch_size: int, resolution: int = 256, device: str = "cuda") -> OrbitCameras:
    azimuth = torch.rand(batch_size, device=device) * (2.0 * math.pi) - math.pi
    elevation = (torch.rand(batch_size, device=device) - 0.5) * (math.pi / 6.0)
    distance = torch.full((batch_size,), 2.7, dtype=torch.float32, device=device)
    return OrbitCameras(azimuth=azimuth, elevation=elevation, distance=distance, resolution=resolution)

