# Project Overview

## One-minute summary

This project turns raw airborne LiDAR tiles into two geospatial ML products:

1. Grid-level canopy height prediction (`target_chm` regression).
2. Grid-level land-cover prediction (LAS class supervision).

The pipeline is CRS-aware, duplicate-aware, and evaluated with tile-blocked validation to reduce spatial leakage.

## Dataset in practice

- Source: Kaggle `rogriofmeireles/lidar-roraima-parime-research`
- Local inventory: 45 LAZ tiles, including one COPC duplicate representation.
- Total points (headers): 2,249,752,652.
- Mixed projected zones: EPSG `31974`, `31975`, `31980`.

These constraints drive core engineering decisions:
- never blindly merge across zones,
- deduplicate equivalent tile representations,
- report metrics per project with reproducible schemas.

## Pipeline

1. `build_manifest.py`: tile metadata audit and QA flags.
2. `build_features.py`: 10m grid feature generation + targets.
3. `train_model.py`: model-family training and registry updates.
4. Notebook series `00` to `90`: EDA to showcase.

## Visual diagnostics

Coverage profile:

![Coverage by zone](../images/eda/coverage_by_zone.png)

Canopy signal:

![CHM distribution](../images/canopy/chm_target_distribution.png)
![Feature signal vs CHM](../images/eda/feature_signal_vs_chm.png)

Land-cover supervision balance:

![Land-cover class distribution](../images/landcover/landcover_class_distribution.png)

Baseline metrics snapshot:

![Baseline benchmark snapshot](../images/eda/baseline_benchmark_snapshot.png)

## Repro commands

```powershell
py -3.11 scripts/build_manifest.py --root .
py -3.11 scripts/build_features.py --root . --cell-size 10 --chunk-size 1000000
py -3.11 scripts/train_model.py --project canopy --family baseline
py -3.11 scripts/train_model.py --project landcover --family baseline
py -3.11 scripts/build_story_plots.py
```
