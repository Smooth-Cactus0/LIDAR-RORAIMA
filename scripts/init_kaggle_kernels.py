from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kaggle kernel metadata for all notebooks.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root.")
    parser.add_argument("--username", required=True, help="Kaggle username.")
    parser.add_argument("--dataset-slug", default="lidar-roraima-parime-research", help="Kaggle input dataset slug.")
    parser.add_argument("--language", default="python")
    parser.add_argument("--kernel-type", default="notebook")
    parser.add_argument("--is-private", action="store_true", help="Create private kernels first.")
    return parser.parse_args()


def notebook_title(file_name: str) -> str:
    stem = file_name.replace(".ipynb", "")
    return stem.replace("_", " ").title()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    out_root = root / "kaggle" / "kernels"
    out_root.mkdir(parents=True, exist_ok=True)

    for nb in notebooks:
        slug = nb.stem.replace("_", "-")
        ref = f"{args.username}/{slug}"
        folder = out_root / slug
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {
            "id": ref,
            "title": notebook_title(nb.name),
            "code_file": nb.name,
            "language": args.language,
            "kernel_type": args.kernel_type,
            "is_private": bool(args.is_private),
            "enable_gpu": False,
            "enable_internet": True,
            "dataset_sources": [f"{args.username}/{args.dataset_slug}"],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Generated metadata for {len(notebooks)} notebooks in: {out_root}")


if __name__ == "__main__":
    main()
