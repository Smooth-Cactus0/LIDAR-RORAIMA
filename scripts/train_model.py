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
from lidar_roraima.models import train_canopy_model, train_landcover_model
from lidar_roraima.registry import append_model_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train canopy or land-cover model.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root directory.")
    parser.add_argument("--project", choices=["canopy", "landcover"], required=True)
    parser.add_argument("--family", choices=["baseline", "random_forest", "boosting"], required=True)
    parser.add_argument("--features", type=Path, default=None, help="Optional merged features parquet path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional deterministic cap on training rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig.from_root(args.root)
    cfg.ensure_dirs()
    if args.features is None:
        features_files = sorted(cfg.features_dir.glob("features_*m.parquet"))
        if not features_files:
            raise FileNotFoundError("No feature tables found. Run scripts/build_features.py first.")
        features_df = pd.concat([pd.read_parquet(path) for path in features_files], ignore_index=True)
    else:
        features_df = pd.read_parquet(args.features)

    if args.project == "canopy":
        result = train_canopy_model(
            features_df=features_df,
            model_family=args.family,
            output_dir=cfg.models_dir,
            seed=args.seed,
            n_splits=args.splits,
            max_rows=args.max_rows,
        )
    else:
        result = train_landcover_model(
            features_df=features_df,
            model_family=args.family,
            output_dir=cfg.models_dir,
            seed=args.seed,
            n_splits=args.splits,
            max_rows=args.max_rows,
        )
    registry = append_model_result(result, registry_path=cfg.models_dir / "model_results.csv")
    print(result)
    print(f"Registry rows: {len(registry)}")


if __name__ == "__main__":
    main()
