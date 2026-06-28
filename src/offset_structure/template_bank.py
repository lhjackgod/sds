"""Template bank for structure-aware offset composition.

OpenCLIP retrieval uses descriptions only to choose/blend these templates.
The templates remain hand-authored structure weights; no model generates UV
offsets directly.
"""

from __future__ import annotations


COMPONENT_NAMES = ("base", "neck", "cuff", "hem", "waistband", "seam", "wrinkle", "overlap")


STRUCTURE_TEMPLATES = {
    "loose_hoodie": {
        "category": "upper",
        "aliases": ["hoodie", "loose hoodie", "oversized hoodie"],
        "description": "a loose hoodie with thick cuffs, raised bottom hem, soft wrinkles, and a slightly bulky torso",
        "weights": {"base": 1.25, "neck": 0.7, "cuff": 1.0, "hem": 1.0, "waistband": 0.3, "seam": 0.4, "wrinkle": 0.6, "overlap": 1.0},
    },
    "tight_long_sleeve": {
        "category": "upper",
        "aliases": ["tight shirt", "tight long sleeve shirt", "shirt"],
        "description": "a tight long sleeve shirt with thin fabric, minimal wrinkles, small cuffs, and body-hugging shape",
        "weights": {"base": 0.65, "neck": 0.35, "cuff": 0.25, "hem": 0.3, "waistband": 0.2, "seam": 0.2, "wrinkle": 0.05, "overlap": 0.2},
    },
    "tshirt": {
        "category": "upper",
        "aliases": ["tshirt", "t-shirt", "short sleeve shirt", "tee"],
        "description": "a short sleeve t-shirt with a visible neck opening, sleeve cuffs, bottom hem, and light fabric thickness",
        "weights": {"base": 1.0, "neck": 0.4, "cuff": 0.55, "hem": 0.6, "waistband": 0.25, "seam": 0.3, "wrinkle": 0.2, "overlap": 0.4},
    },
    "sweater": {
        "category": "upper",
        "aliases": ["sweater", "pullover"],
        "description": "a soft sweater with thicker torso, ribbed cuffs, bottom hem, and moderate wrinkles",
        "weights": {"base": 1.15, "neck": 0.5, "cuff": 0.8, "hem": 0.8, "waistband": 0.3, "seam": 0.3, "wrinkle": 0.45, "overlap": 0.7},
    },
    "denim_jeans": {
        "category": "lower",
        "aliases": ["jeans", "denim jeans", "blue jeans"],
        "description": "denim jeans with a raised waistband, side seams, pant leg hems, and subtle vertical fabric folds",
        "weights": {"base": 1.0, "neck": 0.0, "cuff": 0.0, "hem": 0.0, "waistband": 0.85, "seam": 0.65, "wrinkle": 0.3, "overlap": 0.35},
    },
    "leggings": {
        "category": "lower",
        "aliases": ["leggings", "tight pants"],
        "description": "tight leggings with very thin fabric, minimal seams, no bulky wrinkles, and body-hugging shape",
        "weights": {"base": 0.45, "neck": 0.0, "cuff": 0.0, "hem": 0.0, "waistband": 0.35, "seam": 0.1, "wrinkle": 0.0, "overlap": 0.1},
    },
    "shorts": {
        "category": "lower",
        "aliases": ["shorts", "short pants"],
        "description": "shorts with a waistband, leg openings above the knees, side seams, and light fabric thickness",
        "weights": {"base": 0.9, "neck": 0.0, "cuff": 0.0, "hem": 0.0, "waistband": 0.75, "seam": 0.45, "wrinkle": 0.2, "overlap": 0.3},
    },
    "default_upper": {
        "category": "upper",
        "aliases": ["upper garment", "shirt"],
        "description": "a generic upper garment with modest thickness, small seams, and light boundary details",
        "weights": {"base": 1.0, "neck": 0.35, "cuff": 0.45, "hem": 0.55, "waistband": 0.25, "seam": 0.25, "wrinkle": 0.2, "overlap": 0.35},
    },
    "default_lower": {
        "category": "lower",
        "aliases": ["lower garment", "pants"],
        "description": "a generic lower garment with a waistband, side seams, and moderate fabric thickness",
        "weights": {"base": 0.9, "neck": 0.0, "cuff": 0.0, "hem": 0.0, "waistband": 0.45, "seam": 0.3, "wrinkle": 0.15, "overlap": 0.25},
    },
}


LEGACY_TEMPLATE_ALIASES = {
    "hoodie": "loose_hoodie",
    "tshirt": "tshirt",
    "shirt": "tight_long_sleeve",
    "sweater": "sweater",
    "jeans": "denim_jeans",
    "leggings": "leggings",
    "shorts": "shorts",
    "pants": "default_lower",
    "default": "default_upper",
}


def get_template_weights(template_name: str) -> dict[str, float]:
    resolved = LEGACY_TEMPLATE_ALIASES.get(template_name, template_name)
    template = STRUCTURE_TEMPLATES.get(resolved, STRUCTURE_TEMPLATES["default_upper"])
    weights = template.get("weights", template)
    return {name: float(weights.get(name, 0.0)) for name in COMPONENT_NAMES}
