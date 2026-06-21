"""CasTex SDS wrapper with a deterministic differentiable fallback."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


class CasTexSDSGuidance:
    def __init__(
        self,
        prompt: str,
        negative_prompt: str | None,
        castex_root: str | Path,
        stage: str = "i",
        device: str = "cuda",
        guidance_scale: float = 20.0,
        model_i: str = "DeepFloyd/IF-I-XL-v1.0",
        model_ii: str = "DeepFloyd/IF-II-L-v1.0",
        mode: str = "dummy",
    ) -> None:
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.stage = stage
        self.device = device
        self.guidance_scale = guidance_scale
        self.mode = mode
        self.backend = None
        if stage != "i":
            raise NotImplementedError('Only stage="i" is wired for the first offset SDS pass')
        if mode in {"dummy", "none"}:
            return

        root = Path(castex_root).resolve()
        sys.path.insert(0, str(root))
        try:
            from src.guidance import SDSLoss  # type: ignore
            from src.prompt_processing import encode_prompt  # type: ignore

            model_name = model_i if stage == "i" else model_ii
            self.backend = SDSLoss(stage=stage, model_name=model_name, device=device)
            cache_dir = root / "cached_prompts"
            self.prompt_embeddings = encode_prompt(
                prompt,
                directional=False,
                deepfloyd_model=model_name,
                cache_dir=str(cache_dir),
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize CasTex SDS. Use --sds-mode dummy to validate the differentiable offset pipeline."
            ) from exc

    def loss(self, images: torch.Tensor, view_dirs=None) -> torch.Tensor:
        if self.backend is not None:
            return self.backend(images, prompt_embeddings=self.prompt_embeddings, guidance_scale=self.guidance_scale)
        # Dummy mode is intentionally non-zero so optimization plumbing can be
        # tested: it rewards larger visible shaded clothing while regularizers
        # keep the bounded shell sane.
        return -images.mean()

