from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from lidar_roraima.config import ProjectConfig
from lidar_roraima.manifest import build_manifest, save_manifest
from lidar_roraima.validation import validate_manifest_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LAS/LAZ tile manifest.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root directory.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output parquet path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig.from_root(args.root)
    cfg.ensure_dirs()
    manifest = build_manifest(cfg.raw_data_dir)
    out_path = args.output or (cfg.manifests_dir / "tile_manifest.parquet")
    save_manifest(manifest, out_path)
    report = validate_manifest_schema(manifest)
    print(f"Manifest rows: {len(manifest)}")
    print(f"Output: {out_path}")
    print(f"Validation passed: {report.passed}")
    if report.details:
        for line in report.details:
            print(f"- {line}")


if __name__ == "__main__":
    main()
