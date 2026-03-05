# Land-Cover Subproject

## Objective

Predict grid-level land-cover classes from LiDAR feature vectors using LAS classification as supervision.

## Data usage

- Input: `artifacts/features/features_*_10m.parquet`
- Label source: LAS `classification` mapped by `configs/landcover_class_map.json`
- Focus: class-wise behavior and macro metrics (not only accuracy)

## Visuals

![Land-cover class distribution](../../images/landcover/landcover_class_distribution.png)

## Model Track

1. `20_landcover_baseline.ipynb`
2. `21_landcover_random_forest.ipynb`
3. `22_landcover_boosting.ipynb`
4. `23_landcover_ensemble.ipynb`

## Inputs

- `artifacts/features/features_*_10m.parquet`
- `configs/landcover_class_map.json`

## Benchmark Artifacts

- `benchmarks/landcover/metrics.csv`
- `images/landcover/macro_f1_baseline.png`

## Success Criteria

- Improve macro-F1 against baseline.
- Prevent minority class collapse using per-class evaluation.
