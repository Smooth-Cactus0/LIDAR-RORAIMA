from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_utils import assert_kaggle_credentials, ensure_kernel_folder, project_root, run_kaggle_cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push all project notebooks to Kaggle kernels.")
    parser.add_argument("--root", type=Path, default=None, help="Project root.")
    parser.add_argument("--username", required=True, help="Kaggle username.")
    parser.add_argument("--dataset-slug", default="lidar-roraima-parime-research", help="Kaggle dataset slug.")
    parser.add_argument("--private", action="store_true", help="Publish kernels as private.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare folders but do not push.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve() if args.root else project_root(__file__)
    notebooks_dir = root / "notebooks"
    out_root = root / "kaggle" / "kernels"
    out_root.mkdir(parents=True, exist_ok=True)
    assert_kaggle_credentials()

    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(f"No notebooks found in {notebooks_dir}")

    print(f"Preparing {len(notebooks)} notebooks from: {notebooks_dir}")
    pushed = 0
    for nb in notebooks:
        folder = ensure_kernel_folder(
            notebook_path=nb,
            out_dir=out_root,
            username=args.username,
            dataset_slug=args.dataset_slug,
            is_private=args.private,
        )
        print(f"- Prepared: {folder.name}")
        if args.dry_run:
            continue
        res = run_kaggle_cmd(["kernels", "push", "-p", str(folder)])
        if res.returncode == 0:
            pushed += 1
            print(f"  push ok: {folder.name}")
        else:
            print(f"  push failed: {folder.name}")
            if res.stdout.strip():
                print(res.stdout.strip())
            if res.stderr.strip():
                print(res.stderr.strip())

    print(f"Done. Prepared={len(notebooks)}, Pushed={pushed}, DryRun={args.dry_run}")


if __name__ == "__main__":
    main()
