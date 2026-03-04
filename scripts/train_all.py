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
from lidar_roraima.runtime import get_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full model suite with a runtime profile.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root directory.")
    parser.add_argument(
        "--profile",
        choices=["kaggle_full", "colab_full", "local_smoke"],
        default="kaggle_full",
        help="Runtime profile controlling row caps and CV splits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig.from_root(args.root)
    cfg.ensure_dirs()
    profile = get_profile(args.profile)

    features_files = sorted(cfg.features_dir.glob("features_*m.parquet"))
    if not features_files:
        raise FileNotFoundError("No feature tables found in artifacts/features.")
    features_df = pd.concat([pd.read_parquet(path) for path in features_files], ignore_index=True)

    jobs = [
        ("canopy", "baseline"),
        ("canopy", "random_forest"),
        ("canopy", "boosting"),
        ("landcover", "baseline"),
        ("landcover", "random_forest"),
        ("landcover", "boosting"),
    ]

    for project, family in jobs:
        print(f"Training {project}/{family} with profile={profile.name}")
        if project == "canopy":
            result = train_canopy_model(
                features_df=features_df,
                model_family=family,
                output_dir=cfg.models_dir,
                seed=args.seed,
                n_splits=profile.n_splits,
                max_rows=profile.max_rows,
            )
        else:
            result = train_landcover_model(
                features_df=features_df,
                model_family=family,
                output_dir=cfg.models_dir,
                seed=args.seed,
                n_splits=profile.n_splits,
                max_rows=profile.max_rows,
            )
        append_model_result(result, registry_path=cfg.models_dir / "model_results.csv")
        print(result)


if __name__ == "__main__":
    main()
