from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ORDER = [
    "13_canopy_ensemble.ipynb",
    "20_landcover_baseline.ipynb",
    "21_landcover_random_forest.ipynb",
    "22_landcover_boosting.ipynb",
    "23_landcover_ensemble.ipynb",
    "90_portfolio_inference_showcase.ipynb",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execute selected notebooks locally in order.")
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Project root.")
    p.add_argument("--timeout", type=int, default=1800, help="Per-notebook execution timeout in seconds.")
    p.add_argument("--python", default="py -3.11", help="Python launcher prefix.")
    p.add_argument("--kernel", default="py311", help="Jupyter kernel name to execute notebooks with.")
    return p.parse_args()


def run_nb(root: Path, nb_path: Path, timeout: int, python_cmd: str, kernel: str) -> int:
    cmd = [
        *python_cmd.split(),
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(nb_path),
        f"--ExecutePreprocessor.timeout={timeout}",
        f"--ExecutePreprocessor.kernel_name={kernel}",
    ]
    proc = subprocess.run(cmd, cwd=str(root), text=True)
    return proc.returncode


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    nb_dir = root / "notebooks"
    failed: list[str] = []

    for name in ORDER:
        nb = nb_dir / name
        print(f"\n=== Running {name} ===")
        if not nb.exists():
            print("missing notebook, skipping")
            failed.append(name)
            continue
        code = run_nb(root, nb, args.timeout, args.python, args.kernel)
        if code == 0:
            print(f"OK: {name}")
        else:
            print(f"FAILED: {name} (exit={code})")
            failed.append(name)

    if failed:
        print("\nFailed notebooks:")
        for name in failed:
            print(f"- {name}")
        raise SystemExit(1)
    print("\nAll notebooks completed.")


if __name__ == "__main__":
    main()
