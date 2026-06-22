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
    base = torch.tensor([0.0, math.pi, math.pi / 2.0, -math.pi / 2.0], dtype=torch.float32, device=device)
    choices = torch.randint(0, len(base), (batch_size,), device=device)
    jitter = (torch.rand(batch_size, device=device) - 0.5) * (math.pi / 12.0)
    azimuth = base[choices] + jitter
    min_elev = -10.0 * math.pi / 180.0
    max_elev = 15.0 * math.pi / 180.0
    elevation = min_elev + torch.rand(batch_size, device=device) * (max_elev - min_elev)
    distance = torch.full((batch_size,), 2.7, dtype=torch.float32, device=device)
    return OrbitCameras(azimuth=azimuth, elevation=elevation, distance=distance, resolution=resolution)

