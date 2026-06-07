"""Generate phase-1 UV masks for a small prompt set using one SMPL-X asset bundle."""

from __future__ import annotations

import argparse
import re
from argparse import Namespace
from pathlib import Path

from main_generate_masks import generate_masks


DEFAULT_PROMPTS = [
    "a person wearing a red short sleeve T-shirt and blue jeans",
    "a person wearing a black sleeveless vest and gray shorts",
    "a person wearing a green long sleeve sweater and black pants",
    "a person wearing a white hoodie and dark jeans",
    "a person wearing a tight black long sleeve shirt and leggings",
]


def slugify(text: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return f"{index:02d}_{slug[:72]}"


def read_prompts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def run_batch(args: argparse.Namespace) -> None:
    prompts = read_prompts(args.prompts_file)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts, start=1):
        out_dir = out_root / slugify(prompt, index)
        print(f"[{index}/{len(prompts)}] {prompt}")
        generate_masks(
            Namespace(
                prompt=prompt,
                mesh=args.mesh,
                uv=args.uv,
                part_labels=args.part_labels,
                out=str(out_dir),
                resolution=args.resolution,
                config=args.config,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default="data/smplx/generated/smplx_template.obj")
    parser.add_argument("--uv", default=None)
    parser.add_argument("--part-labels", default="data/smplx/generated/part_labels.json")
    parser.add_argument("--out-root", default="outputs/masks/prompt_batch")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "garment_rules.yaml"),
    )
    parser.add_argument("--prompts-file", help="Optional txt file, one prompt per line")
    return parser


if __name__ == "__main__":
    run_batch(build_parser().parse_args())
