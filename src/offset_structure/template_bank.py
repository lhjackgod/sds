"""Rule template bank for structure-aware offset composition."""

from __future__ import annotations


STRUCTURE_TEMPLATES = {
    "hoodie": {
        "base": 1.2,
        "neck": 0.7,
        "cuff": 1.0,
        "hem": 1.0,
        "waistband": 0.5,
        "seam": 0.4,
        "wrinkle": 0.6,
        "overlap": 1.0,
    },
    "tshirt": {
        "base": 1.0,
        "neck": 0.4,
        "cuff": 0.5,
        "hem": 0.6,
        "waistband": 0.4,
        "seam": 0.3,
        "wrinkle": 0.2,
        "overlap": 0.4,
    },
    "sweater": {
        "base": 1.15,
        "neck": 0.5,
        "cuff": 0.8,
        "hem": 0.8,
        "waistband": 0.5,
        "seam": 0.3,
        "wrinkle": 0.4,
        "overlap": 0.7,
    },
    "jeans": {
        "base": 1.0,
        "neck": 0.0,
        "cuff": 0.0,
        "hem": 0.0,
        "waistband": 0.8,
        "seam": 0.6,
        "wrinkle": 0.3,
        "overlap": 0.4,
        "pant_hem": 0.5,
    },
    "leggings": {
        "base": 0.5,
        "neck": 0.0,
        "cuff": 0.0,
        "hem": 0.0,
        "waistband": 0.4,
        "seam": 0.1,
        "wrinkle": 0.0,
        "overlap": 0.1,
        "pant_hem": 0.2,
    },
    "default": {
        "base": 1.0,
        "neck": 0.35,
        "cuff": 0.45,
        "hem": 0.55,
        "waistband": 0.45,
        "seam": 0.25,
        "wrinkle": 0.2,
        "overlap": 0.35,
        "pant_hem": 0.35,
    },
}


COMPONENT_NAMES = ("base", "neck", "cuff", "hem", "waistband", "seam", "wrinkle", "overlap")

