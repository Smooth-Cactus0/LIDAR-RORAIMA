# Canopy Height Subproject

## Objective

Predict grid-level canopy height (`target_chm`) from LiDAR-derived features.

## Model Track

1. `10_canopy_baseline.ipynb`
2. `11_canopy_random_forest.ipynb`
3. `12_canopy_boosting.ipynb`
4. `13_canopy_ensemble.ipynb`

## Inputs

- `artifacts/features/features_*_10m.parquet`
- target from canopy-ground class elevation deltas

## Benchmark Artifacts

- `benchmarks/canopy/metrics.csv`
- `images/canopy/rmse_baseline.png`

## Success Criteria

- Improve RMSE and MAE against baseline.
- Keep tile-blocked CV to avoid spatial leakage.
