"""Template retrieval for structure-aware offsets."""

from __future__ import annotations

from offset_structure.openclip_retriever import OpenCLIPTemplateRetriever
from progress_log import append_progress
from offset_structure.template_bank import COMPONENT_NAMES, LEGACY_TEMPLATE_ALIASES, STRUCTURE_TEMPLATES, get_template_weights


def _empty_weights() -> dict[str, float]:
    return {name: 0.0 for name in COMPONENT_NAMES}


def _rule_template_name(garment_type: str, category: str) -> str:
    if garment_type in LEGACY_TEMPLATE_ALIASES:
        return LEGACY_TEMPLATE_ALIASES[garment_type]
    return "default_upper" if category == "upper" else "default_lower"


def _merge_category_weights(upper: dict[str, float] | None, lower: dict[str, float] | None) -> dict[str, float]:
    if upper is None and lower is None:
        return get_template_weights("default")
    if upper is None:
        return {name: float(lower.get(name, 0.0)) for name in COMPONENT_NAMES}  # type: ignore[union-attr]
    if lower is None:
        return {name: float(upper.get(name, 0.0)) for name in COMPONENT_NAMES}
    # TODO: replace this first-pass mean with mask-area/category-aware merging.
    return {name: (float(upper.get(name, 0.0)) + float(lower.get(name, 0.0))) * 0.5 for name in COMPONENT_NAMES}


def _rule_retrieval(spec: dict) -> tuple[dict[str, float], dict]:
    selected = []
    upper_weights = None
    lower_weights = None
    upper = spec.get("upper", {})
    if upper.get("enabled", False):
        name = _rule_template_name(str(upper.get("type", "default")), "upper")
        upper_weights = get_template_weights(name)
        selected.append({"category": "upper", "template": name, "garment_type": upper.get("type")})
    lower = spec.get("lower", {})
    if lower.get("enabled", False):
        name = _rule_template_name(str(lower.get("type", "default")), "lower")
        lower_weights = get_template_weights(name)
        selected.append({"category": "lower", "template": name, "garment_type": lower.get("type")})
    weights = _merge_category_weights(upper_weights, lower_weights)
    return weights, {"method": "rule", "selected": selected, "merged_weights": weights}


def retrieve_structure_template(
    spec: dict,
    prompt: str | None = None,
    method: str = "rule",
    openclip_model: str = "ViT-B-32",
    openclip_pretrained: str = "laion2b_s34b_b79k",
    device: str = "cuda:0",
    top_k: int = 3,
    temperature: float = 0.07,
    progress_log_path=None,
) -> tuple[dict[str, float], dict]:
    append_progress(progress_log_path, "template retrieval: start", {"method": method})
    if method == "rule":
        weights, debug = _rule_retrieval(spec)
        append_progress(progress_log_path, "template retrieval: rule complete", {"selected": debug.get("selected", [])})
        return weights, debug
    if method != "openclip":
        raise ValueError(f"Unknown template retrieval method: {method}")
    if prompt is None:
        raise ValueError("OpenCLIP template retrieval requires a prompt")

    retriever = OpenCLIPTemplateRetriever(
        model_name=openclip_model,
        pretrained=openclip_pretrained,
        device=device,
        temperature=temperature,
        progress_log_path=progress_log_path,
    )
    upper_weights = None
    lower_weights = None
    debug: dict = {"method": "openclip", "prompt": prompt}
    if spec.get("upper", {}).get("enabled", False):
        upper_prompt = f"{prompt}, upper garment clothing structure"
        upper_weights, upper_debug = retriever.retrieve(upper_prompt, STRUCTURE_TEMPLATES, category="upper", top_k=top_k)
        debug["upper"] = upper_debug
    if spec.get("lower", {}).get("enabled", False):
        lower_prompt = f"{prompt}, lower garment pants structure"
        lower_weights, lower_debug = retriever.retrieve(lower_prompt, STRUCTURE_TEMPLATES, category="lower", top_k=top_k)
        debug["lower"] = lower_debug
    weights = _merge_category_weights(upper_weights, lower_weights)
    debug["merged_weights"] = weights
    append_progress(progress_log_path, "template retrieval: openclip complete", {"merged_weights": weights})
    return weights, debug
