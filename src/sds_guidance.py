"""Pluggable SDS guidance interface for bounded offset refinement."""

from __future__ import annotations


class SDSGuidance:
    def __init__(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        device: str = "cuda",
        model_name: str = "stabilityai/stable-diffusion-2-1-base",
        guidance_scale: float = 50.0,
        mode: str = "dummy",
    ) -> None:
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.device = device
        self.model_name = model_name
        self.guidance_scale = guidance_scale
        self.mode = mode
        self.pipe = None
        if mode == "dummy":
            return
        if mode == "diffusers":
            try:
                import diffusers  # noqa: F401
                import torch  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    "sds-mode=diffusers requires diffusers and torch with a usable diffusion setup. "
                    "Use --sds-mode dummy to test the offset pipeline without diffusion."
                ) from exc
            raise NotImplementedError(
                "Diffusers SDS is intentionally left as a plug-in backend in this phase. "
                "The pipeline supports --sds-mode dummy and stable init outputs now."
            )
        raise ValueError(f"Unknown SDS mode: {mode}")

    def loss(self, images):
        if self.mode == "dummy":
            try:
                import torch

                if hasattr(images, "sum"):
                    return images.sum() * 0.0
                return torch.tensor(0.0)
            except Exception:
                return 0.0
        raise NotImplementedError("Only dummy SDS loss is available in this lightweight backend")
