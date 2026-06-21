"""Prompt helpers for geometry-focused offset refinement."""

from __future__ import annotations


DEFAULT_NEGATIVE_PROMPT = (
    "deformed body, broken arms, broken legs, extra limbs, spikes, holes, "
    "floating cloth, detached cloth, distorted mesh, noisy surface, bad anatomy"
)


def build_geometry_prompt(prompt: str) -> str:
    return (
        f"a full body 3D render of {prompt}, clean body-aligned clothing geometry, "
        "realistic clothing thickness, simple studio lighting"
    )


def build_negative_prompt() -> str:
    return DEFAULT_NEGATIVE_PROMPT

