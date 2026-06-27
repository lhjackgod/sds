"""Template retrieval for structure-aware offsets."""

from __future__ import annotations

from offset_structure.template_bank import COMPONENT_NAMES, STRUCTURE_TEMPLATES


def retrieve_structure_template(spec: dict, prompt: str | None = None) -> dict[str, float]:
    del prompt
    weights = {name: 0.0 for name in COMPONENT_NAMES}
    count = 0
    for section in ("upper", "lower"):
        item = spec.get(section, {})
        if not item.get("enabled", False):
            continue
        garment_type = item.get("type", "default")
        template = STRUCTURE_TEMPLATES.get(garment_type, STRUCTURE_TEMPLATES["default"])
        for name in COMPONENT_NAMES:
            weights[name] += float(template.get(name, 0.0))
        count += 1
    if count == 0:
        return {name: float(STRUCTURE_TEMPLATES["default"].get(name, 0.0)) for name in COMPONENT_NAMES}
    return {name: value / count for name, value in weights.items()}

