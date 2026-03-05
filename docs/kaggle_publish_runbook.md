# Kaggle Publish Runbook

## Goal

Publish the notebook series in order and link each notebook to the matching GitHub commit.

## Runtime rule

- Use Kaggle or Colab for heavy model training.
- Keep local execution to manifest/EDA/smoke checks.

## Publication order

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

## Per-notebook checklist

1. Sync latest code from GitHub.
2. Confirm notebook executes top-to-bottom on Kaggle.
3. Add references to benchmark artifacts in markdown.
4. Publish notebook publicly.
5. Record Kaggle URL + Git commit in `docs/notebook_index.md`.

## Metadata bootstrap (optional, recommended)

Generate Kaggle `kernel-metadata.json` templates for all notebooks:

```powershell
py -3.11 scripts/init_kaggle_kernels.py --username <your-kaggle-username>
```

Output folder:
- `kaggle/kernels/<notebook-slug>/kernel-metadata.json`

## Fully automated CLI workflow (recommended)

1. Push all notebooks to Kaggle:

```powershell
py -3.11 scripts/kaggle_bulk_push.py --username <your-kaggle-username>
```

To chain outputs notebook-to-notebook (Kaggle Notebook Output), use:

```powershell
py -3.11 scripts/kaggle_bulk_push.py --username <your-kaggle-username> --chain-previous
```

2. Check execution status for all notebooks:

```powershell
py -3.11 scripts/kaggle_bulk_status.py --username <your-kaggle-username>
```

3. Download all notebook outputs:

```powershell
py -3.11 scripts/kaggle_bulk_output.py --username <your-kaggle-username>
```

## Suggested commit strategy

1. `chore: initialize lidar roraima portfolio scaffold`
2. `feat: add shared lidar pipeline and notebook series`
3. `docs: add benchmark visuals and kaggle publication runbook`
