from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_utils import notebook_slug, project_root, run_kaggle_cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download outputs for all project Kaggle kernels.")
    parser.add_argument("--root", type=Path, default=None, help="Project root.")
    parser.add_argument("--username", required=True, help="Kaggle username.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder for downloaded artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve() if args.root else project_root(__file__)
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError("No notebooks found.")

    out_dir = args.out_dir.resolve() if args.out_dir else (root / "kaggle" / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for nb in notebooks:
        slug = notebook_slug(nb)
        ref = f"{args.username}/{slug}"
        target = out_dir / slug
        target.mkdir(parents=True, exist_ok=True)
        res = run_kaggle_cmd(["kernels", "output", ref, "-p", str(target)])
        if res.returncode == 0:
            downloaded += 1
            print(f"- downloaded: {ref} -> {target}")
        else:
            err = res.stderr.strip() or res.stdout.strip() or "download failed"
            print(f"- failed: {ref} -> {err}")

    print(f"Done. Downloaded={downloaded}/{len(notebooks)}")


if __name__ == "__main__":
    main()
