"""OpenCLIP text-only retrieval for structure templates."""

from __future__ import annotations

import torch

from offset_structure.template_bank import COMPONENT_NAMES
from progress_log import append_progress


class OpenCLIPTemplateRetriever:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda:0",
        cache_dir=None,
        temperature: float = 0.07,
        progress_log_path=None,
    ) -> None:
        self.progress_log_path = progress_log_path
        append_progress(progress_log_path, "openclip: importing open_clip", {"model": model_name, "pretrained": pretrained})
        try:
            import open_clip
        except Exception as exc:
            raise RuntimeError(
                "OpenCLIP retrieval requested but open_clip_torch is not installed. "
                "Install with pip install open_clip_torch or use --template-retrieval rule."
            ) from exc

        self.open_clip = open_clip
        self.device = device
        self.temperature = float(temperature)
        kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
        append_progress(progress_log_path, "openclip: loading model weights", {"model": model_name, "pretrained": pretrained, "device": device})
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, **kwargs)
        self.model = self.model.to(device).eval()
        append_progress(progress_log_path, "openclip: model loaded")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        append_progress(progress_log_path, "openclip: tokenizer ready")

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> torch.Tensor:
        append_progress(self.progress_log_path, "openclip: encoding text", {"num_texts": len(texts)})
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def retrieve(
        self,
        prompt: str,
        templates: dict,
        category: str | None = None,
        top_k: int = 3,
    ) -> tuple[dict[str, float], dict]:
        append_progress(self.progress_log_path, "openclip: retrieving templates", {"prompt": prompt, "category": category, "top_k": top_k})
        candidates = [(name, item) for name, item in templates.items() if category is None or item.get("category") == category]
        if not candidates:
            raise ValueError(f"No structure templates available for category={category!r}")

        names = [name for name, _ in candidates]
        descriptions = [item.get("description", name) for name, item in candidates]
        prompt_feature = self._encode([prompt])
        template_features = self._encode(descriptions)
        scores = (prompt_feature @ template_features.T).squeeze(0)
        k = min(max(int(top_k), 1), len(candidates))
        top_scores, top_indices = torch.topk(scores, k=k)
        weights = torch.softmax(top_scores / max(self.temperature, 1e-6), dim=0)

        blended = {name: 0.0 for name in COMPONENT_NAMES}
        top_debug = []
        for score, weight, index in zip(top_scores, weights, top_indices):
            template_name = names[int(index)]
            template = templates[template_name]
            component_weights = template["weights"]
            for component in COMPONENT_NAMES:
                blended[component] += float(weight) * float(component_weights.get(component, 0.0))
            top_debug.append(
                {
                    "template": template_name,
                    "score": float(score.detach().cpu()),
                    "weight": float(weight.detach().cpu()),
                    "description": template.get("description", ""),
                }
            )

        append_progress(self.progress_log_path, "openclip: retrieval complete", {"category": category, "top_k": top_debug})
        return blended, {"method": "openclip", "prompt": prompt, "category": category, "top_k": top_debug}
