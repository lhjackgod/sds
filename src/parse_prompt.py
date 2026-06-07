"""Rule-based prompt parser for body-aligned garments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


GarmentSpec = Dict[str, Dict[str, Any]]

_COLORS = (
    "black",
    "white",
    "gray",
    "grey",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "dark",
)

_DEFAULT_SPEC: GarmentSpec = {
    "upper": {
        "enabled": False,
        "type": "none",
        "sleeve": "none",
        "length": "normal",
        "fit": "normal",
        "color": None,
    },
    "lower": {
        "enabled": False,
        "type": "none",
        "length": "long",
        "fit": "normal",
        "color": None,
    },
}

_UNSUPPORTED = ("skirt", "dress", "open jacket", "cape", "scarf")


def _contains(text: str, *phrases: str) -> bool:
    return any(phrase in text for phrase in phrases)


def _tokenize(text: str) -> list[str]:
    cleaned = "".join(char if char.isalnum() else " " for char in text)
    return cleaned.split()


def _nearby_color(text: str, garment_terms: tuple[str, ...]) -> str | None:
    tokens = _tokenize(text.replace("t-shirt", "tshirt").replace("short-sleeve", "short sleeve"))
    normalized_terms = [term.replace("-", " ").split() for term in garment_terms]
    colors = set(_COLORS)

    for index in range(len(tokens)):
        for term_tokens in normalized_terms:
            if tokens[index : index + len(term_tokens)] != term_tokens:
                continue
            start = max(0, index - 4)
            window = tokens[start:index]
            for separator in ("and", "with", "wearing"):
                if separator in window:
                    window = window[window.index(separator) + 1 :]
            for token in reversed(window):
                if token in colors:
                    return token
    return None


def parse_prompt(prompt: str) -> GarmentSpec:
    """Parse a prompt into deterministic upper and lower garment attributes."""
    text = prompt.lower().replace("_", " ")
    spec = deepcopy(_DEFAULT_SPEC)

    unsupported = [term for term in _UNSUPPORTED if term in text]
    if unsupported:
        spec["unsupported"] = {"items": unsupported}  # type: ignore[assignment]

    upper = spec["upper"]
    if _contains(text, "hoodie"):
        upper.update(enabled=True, type="hoodie", sleeve="long", fit="loose")
    elif _contains(text, "sweater"):
        upper.update(enabled=True, type="sweater", sleeve="long")
    elif _contains(text, "vest", "sleeveless"):
        upper.update(enabled=True, type="vest", sleeve="none")
    elif _contains(text, "shirt", "t-shirt", "tshirt", "tee"):
        sleeve = "long" if _contains(text, "long sleeve", "long-sleeve") else "short"
        upper.update(enabled=True, type="shirt" if sleeve == "long" else "tshirt", sleeve=sleeve)

    if upper["enabled"]:
        if _contains(text, "cropped", "crop top"):
            upper["length"] = "cropped"
        elif _contains(text, "longline", "long shirt", "long hoodie"):
            upper["length"] = "long"
        if _contains(text, "tight", "fitted", "slim fit"):
            upper["fit"] = "tight"
        elif _contains(text, "loose", "oversized"):
            upper["fit"] = "loose"
        upper["color"] = _nearby_color(
            text, ("hoodie", "sweater", "vest", "t-shirt", "tshirt", "shirt", "tee")
        )

    lower = spec["lower"]
    if _contains(text, "leggings"):
        lower.update(enabled=True, type="leggings", length="long", fit="tight")
    elif _contains(text, "jeans"):
        lower.update(enabled=True, type="jeans", length="long")
    elif _contains(text, "shorts", "short pants"):
        lower.update(enabled=True, type="shorts", length="short")
    elif _contains(text, "pants", "trousers"):
        lower.update(enabled=True, type="pants", length="long")

    if lower["enabled"]:
        if _contains(text, "knee shorts", "knee-length shorts"):
            lower["length"] = "knee"
        if _contains(text, "tight pants", "tight trousers"):
            lower["fit"] = "tight"
        elif _contains(text, "loose pants", "loose trousers", "baggy"):
            lower["fit"] = "loose"
        lower["color"] = _nearby_color(
            text, ("leggings", "jeans", "shorts", "short pants", "pants", "trousers")
        )

    return spec

