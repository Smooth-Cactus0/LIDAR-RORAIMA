# LIDAR RORAIMA

Portfolio-grade LiDAR ML repository built from the Kaggle dataset:
`rogriofmeireles/lidar-roraima-parime-research`

This repo contains two subprojects built on a shared geospatial pipeline:

1. Canopy height prediction (CHM regression)
2. Land-cover prediction (LAS class supervision)

## Why this repo exists

- Demonstrate geospatial/ML engineering competence end-to-end.
- Publish a coherent Kaggle notebook series linked to GitHub commits.
- Track benchmarks and visuals in a recruiter-friendly format.

## Repository layout

- `lidar_data/` raw LAZ tiles (local only)
- `src/lidar_roraima/` reusable package (manifest, features, CV, training, inference, ensemble)
- `scripts/` reproducible CLI entrypoints
- `notebooks/` Kaggle notebook sequence (00 to 90)
- `subprojects/canopy-height/` canopy-focused docs and benchmark context
- `subprojects/land-cover/` land-cover-focused docs and benchmark context
- `benchmarks/` exported model metrics
- `images/` EDA and benchmark figures
- `docs/notebook_index.md` Kaggle/GitHub publication tracker

## Current verified status

- Full manifest generated from local dataset:
  - `artifacts/manifests/tile_manifest.parquet`
- Full 10m feature extraction completed:
  - 4 zone partitions
  - 3,673,706 engineered grid cells
- Baseline models trained:
  - canopy baseline RMSE/MAE/R2 logged
  - land-cover baseline macro-F1/precision/recall logged

## Quickstart (Windows, Python 3.11)

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 scripts/build_manifest.py --root .
py -3.11 scripts/build_features.py --root . --cell-size 10 --chunk-size 1000000
py -3.11 scripts/train_model.py --project canopy --family baseline
py -3.11 scripts/train_model.py --project landcover --family baseline
```

## Heavy Training Policy

Run heavy model training on remote runtimes only.

- Kaggle / Colab full run:
```powershell
py -3.11 scripts/train_all.py --profile kaggle_full
```
- Local smoke test only:
```powershell
py -3.11 scripts/train_all.py --profile local_smoke
```

Profiles are defined in:
- `src/lidar_roraima/runtime.py`

## Notebook publication order

1. `00_metadata_eda.ipynb`
2. `01_feature_engineering_shared.ipynb`
3. `10_canopy_baseline.ipynb`
4. `11_canopy_random_forest.ipynb`
5. `12_canopy_boosting.ipynb`
6. `13_canopy_ensemble.ipynb`
7. `20_landcover_baseline.ipynb`
8. `21_landcover_random_forest.ipynb`
9. `22_landcover_boosting.ipynb`
10. `23_landcover_ensemble.ipynb`
11. `90_portfolio_inference_showcase.ipynb`

## Data contracts

- `tile_manifest.parquet`: tile metadata, CRS, QA flags, duplicates
- `features_{zone}_{cellsize}m.parquet`: grid features + optional targets
- `model_results.csv`: benchmark registry
- inference outputs: grid predictions + uncertainty + model version

## Kaggle runtime hardening

Notebooks now detect:
- local repo paths
- `/kaggle/input/lidar-roraima-parime-research[/lidar_data]` dataset mounts

This keeps one notebook codebase usable both locally and on Kaggle.
