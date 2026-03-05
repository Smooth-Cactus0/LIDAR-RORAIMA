from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_utils import notebook_slug, project_root, run_kaggle_cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check status of all Kaggle kernels for project notebooks.")
    parser.add_argument("--root", type=Path, default=None, help="Project root.")
    parser.add_argument("--username", required=True, help="Kaggle username.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve() if args.root else project_root(__file__)
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError("No notebooks found.")

    print("Kernel status:")
    for nb in notebooks:
        ref = f"{args.username}/{notebook_slug(nb)}"
        res = run_kaggle_cmd(["kernels", "status", ref])
        if res.returncode == 0:
            line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "ok"
            print(f"- {ref}: {line}")
        else:
            err = res.stderr.strip() or res.stdout.strip() or "status failed"
            print(f"- {ref}: {err}")


if __name__ == "__main__":
    main()
