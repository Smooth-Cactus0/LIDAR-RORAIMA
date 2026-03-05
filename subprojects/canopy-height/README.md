# Canopy Height Subproject

## Objective

Predict grid-level canopy height (`target_chm`) from LiDAR-derived features.

## Data usage

- Input: `artifacts/features/features_*_10m.parquet`
- Signal features: height percentiles, return behavior, intensity, roughness, density
- Target: `target_chm` built from canopy-ground elevation logic

## Visuals

![CHM distribution](../../images/canopy/chm_target_distribution.png)
![Feature signal vs CHM](../../images/eda/feature_signal_vs_chm.png)

## Model Track

1. `10_canopy_baseline.ipynb`
2. `11_canopy_random_forest.ipynb`
3. `12_canopy_boosting.ipynb`
4. `13_canopy_ensemble.ipynb`

## Inputs

- `artifacts/features/features_*_10m.parquet`

## Benchmark Artifacts

- `benchmarks/canopy/metrics.csv`
- `images/canopy/rmse_baseline.png`

## Success Criteria

- Improve RMSE and MAE against baseline.
- Keep tile-blocked CV to avoid spatial leakage.
