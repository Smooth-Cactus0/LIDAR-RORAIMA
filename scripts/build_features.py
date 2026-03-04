from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from lidar_roraima.config import ProjectConfig
from lidar_roraima.features import extract_features_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grid-level feature tables from LAZ tiles.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest parquet path.")
    parser.add_argument("--cell-size", type=float, default=10.0, help="Grid cell size in projected units.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Chunk size for LAZ streaming.")
    parser.add_argument("--max-points-per-tile", type=int, default=None, help="Optional cap for notebook prototyping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig.from_root(args.root)
    cfg.ensure_dirs()
    manifest_path = args.manifest or (cfg.manifests_dir / "tile_manifest.parquet")
    manifest = pd.read_parquet(manifest_path)
    written = extract_features_from_manifest(
        manifest=manifest,
        output_dir=cfg.features_dir,
        cell_size=args.cell_size,
        include_targets=True,
        chunk_size=args.chunk_size,
        max_points_per_tile=args.max_points_per_tile,
    )
    print(f"Wrote {len(written)} feature tables:")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
