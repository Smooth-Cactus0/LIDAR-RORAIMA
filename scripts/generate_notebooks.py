from __future__ import annotations

import json
from pathlib import Path


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code_cell(code: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = """from pathlib import Path
import sys
import pandas as pd
import numpy as np
import importlib

# Resolve project root in local or Kaggle runtime.
if Path("/kaggle/working").exists():
    ROOT = Path("/kaggle/working")
else:
    ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()

if (ROOT / "src").exists():
    SRC = ROOT / "src"
else:
    SRC = Path.cwd() / "src"
if str(SRC) not in sys.path and SRC.exists():
    sys.path.append(str(SRC))

if importlib.util.find_spec("laspy") is None:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "laspy[lazrs]"])

from lidar_roraima.config import ProjectConfig
cfg = ProjectConfig.from_root(ROOT)
cfg.ensure_dirs()

# Kaggle dataset fallback path.
RAW_DATA_DIR = cfg.raw_data_dir
for candidate in [
    Path("/kaggle/input/lidar-roraima-parime-research/lidar_data"),
    Path("/kaggle/input/lidar-roraima-parime-research"),
]:
    if candidate.exists():
        RAW_DATA_DIR = candidate
        break

print("ROOT:", ROOT)
print("RAW_DATA_DIR:", RAW_DATA_DIR)
TRAIN_MAX_ROWS = None  # Set integer for constrained runtime, e.g. 1200000
cfg
"""


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")


def build_00_metadata_eda(path: Path) -> None:
    cells = [
        md_cell("# 00 Metadata EDA\n\nAudit LAS/LAZ headers, projection zones, duplicates, and QA flags."),
        code_cell(COMMON_SETUP),
        code_cell(
            """from lidar_roraima.manifest import build_manifest, save_manifest
from lidar_roraima.validation import validate_manifest_schema

manifest = build_manifest(RAW_DATA_DIR)
save_manifest(manifest, cfg.manifests_dir / "tile_manifest.parquet")
report = validate_manifest_schema(manifest)
print(report)
manifest.head()
"""
        ),
        code_cell(
            """manifest["epsg"].value_counts(dropna=False)
"""
        ),
        code_cell(
            """manifest["is_duplicate"].value_counts()
"""
        ),
        code_cell(
            """import json
import matplotlib.pyplot as plt

bbox = manifest["bbox"].apply(json.loads).apply(pd.Series)
plt.figure(figsize=(10, 6))
plt.scatter((bbox["min_x"] + bbox["max_x"]) / 2, (bbox["min_y"] + bbox["max_y"]) / 2, s=30)
plt.title("Tile centroid distribution (projected coordinates)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""
        ),
    ]
    write_notebook(path, cells)


def build_01_features(path: Path) -> None:
    cells = [
        md_cell("# 01 Shared Feature Engineering\n\nBuild grid-level features and targets for both sub-projects."),
        code_cell(COMMON_SETUP),
        code_cell(
            """from lidar_roraima.features import extract_features_from_manifest, load_feature_tables

manifest = pd.read_parquet(cfg.manifests_dir / "tile_manifest.parquet")
written = extract_features_from_manifest(
    manifest=manifest,
    output_dir=cfg.features_dir,
    cell_size=10.0,
    include_targets=True,
    chunk_size=1_000_000,
    max_points_per_tile=2_000_000,  # remove cap for full local production run
)
written
"""
        ),
        code_cell(
            """features = load_feature_tables(cfg.features_dir)
features.shape
"""
        ),
        code_cell(
            """features.head()
"""
        ),
    ]
    write_notebook(path, cells)


def build_model_notebook(path: Path, title: str, project: str, family: str) -> None:
    if project == "canopy":
        train_code = """from lidar_roraima.features import load_feature_tables
from lidar_roraima.models import train_canopy_model
from lidar_roraima.registry import append_model_result

features = load_feature_tables(cfg.features_dir)
result = train_canopy_model(features_df=features, model_family="{family}", output_dir=cfg.models_dir, seed=cfg.random_seed, n_splits=5, max_rows=TRAIN_MAX_ROWS)
registry = append_model_result(result, cfg.models_dir / "model_results.csv")
result
""".replace("{family}", family)
    else:
        train_code = """from lidar_roraima.features import load_feature_tables
from lidar_roraima.models import train_landcover_model
from lidar_roraima.registry import append_model_result

features = load_feature_tables(cfg.features_dir)
result = train_landcover_model(features_df=features, model_family="{family}", output_dir=cfg.models_dir, seed=cfg.random_seed, n_splits=5, max_rows=TRAIN_MAX_ROWS)
registry = append_model_result(result, cfg.models_dir / "model_results.csv")
result
""".replace("{family}", family)

    cells = [
        md_cell(f"# {title}\n\nModel family: `{family}`"),
        code_cell(COMMON_SETUP),
        code_cell(train_code),
        code_cell("""pd.read_csv(cfg.models_dir / "model_results.csv").tail(10)"""),
    ]
    write_notebook(path, cells)


def build_ensemble_notebook(path: Path, title: str, project: str) -> None:
    if project == "canopy":
        infer_cell = """import pandas as pd
from lidar_roraima.features import load_feature_tables
from lidar_roraima.inference import run_inference
from lidar_roraima.ensemble import blend_regression_predictions

features = load_feature_tables(cfg.features_dir)
model_paths = [
    cfg.models_dir / "canopy_baseline.joblib",
    cfg.models_dir / "canopy_random_forest.joblib",
    cfg.models_dir / "canopy_boosting.joblib",
]
preds = [run_inference(path, features, prediction_column="pred_chm", uncertainty_column="uncertainty_chm_single") for path in model_paths if path.exists()]
ensemble = blend_regression_predictions(preds, pred_col="pred_chm")
ensemble.to_parquet(cfg.inference_dir / "canopy_ensemble.parquet", index=False)
ensemble.head()
"""
    else:
        infer_cell = """import pandas as pd
from lidar_roraima.features import load_feature_tables
from lidar_roraima.inference import run_inference
from lidar_roraima.ensemble import majority_vote_classification

features = load_feature_tables(cfg.features_dir)
model_paths = [
    cfg.models_dir / "landcover_baseline.joblib",
    cfg.models_dir / "landcover_random_forest.joblib",
    cfg.models_dir / "landcover_boosting.joblib",
]
preds = [run_inference(path, features, prediction_column="pred_landcover", uncertainty_column="uncertainty_landcover_single") for path in model_paths if path.exists()]
ensemble = majority_vote_classification(preds, pred_col="pred_landcover")
ensemble.to_parquet(cfg.inference_dir / "landcover_ensemble.parquet", index=False)
ensemble.head()
"""

    cells = [
        md_cell(f"# {title}\n\nEnsemble from baseline + random forest + boosting notebooks."),
        code_cell(COMMON_SETUP),
        code_cell(infer_cell),
    ]
    write_notebook(path, cells)


def build_showcase_notebook(path: Path) -> None:
    cells = [
        md_cell("# 90 Portfolio Inference Showcase\n\nEnd-to-end output for recruiters and Kaggle readers."),
        code_cell(COMMON_SETUP),
        code_cell(
            """canopy_path = cfg.inference_dir / "canopy_ensemble.parquet"
landcover_path = cfg.inference_dir / "landcover_ensemble.parquet"
canopy = pd.read_parquet(canopy_path) if canopy_path.exists() else pd.DataFrame()
landcover = pd.read_parquet(landcover_path) if landcover_path.exists() else pd.DataFrame()
canopy.shape, landcover.shape
"""
        ),
        code_cell(
            """import matplotlib.pyplot as plt

if not canopy.empty:
    sample = canopy.sample(min(len(canopy), 20000), random_state=42)
    plt.figure(figsize=(9, 6))
    plt.scatter(sample["grid_x"], sample["grid_y"], c=sample["pred_chm"], s=1, cmap="viridis")
    plt.colorbar(label="Predicted CHM")
    plt.title("Canopy height predictions")
    plt.show()
"""
        ),
        code_cell(
            """if not landcover.empty:
    sample = landcover.sample(min(len(landcover), 20000), random_state=42)
    plt.figure(figsize=(9, 6))
    plt.scatter(sample["grid_x"], sample["grid_y"], c=sample["pred_landcover"], s=1, cmap="tab20")
    plt.colorbar(label="Predicted class")
    plt.title("Land-cover predictions")
    plt.show()
"""
        ),
    ]
    write_notebook(path, cells)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    nb = root / "notebooks"
    build_00_metadata_eda(nb / "00_metadata_eda.ipynb")
    build_01_features(nb / "01_feature_engineering_shared.ipynb")

    build_model_notebook(nb / "10_canopy_baseline.ipynb", "10 Canopy Baseline", "canopy", "baseline")
    build_model_notebook(nb / "11_canopy_random_forest.ipynb", "11 Canopy Random Forest", "canopy", "random_forest")
    build_model_notebook(nb / "12_canopy_boosting.ipynb", "12 Canopy Boosting", "canopy", "boosting")
    build_ensemble_notebook(nb / "13_canopy_ensemble.ipynb", "13 Canopy Ensemble", "canopy")

    build_model_notebook(nb / "20_landcover_baseline.ipynb", "20 Land-Cover Baseline", "landcover", "baseline")
    build_model_notebook(nb / "21_landcover_random_forest.ipynb", "21 Land-Cover Random Forest", "landcover", "random_forest")
    build_model_notebook(nb / "22_landcover_boosting.ipynb", "22 Land-Cover Boosting", "landcover", "boosting")
    build_ensemble_notebook(nb / "23_landcover_ensemble.ipynb", "23 Land-Cover Ensemble", "landcover")

    build_showcase_notebook(nb / "90_portfolio_inference_showcase.ipynb")
    print("Notebooks generated.")


if __name__ == "__main__":
    main()
