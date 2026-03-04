from __future__ import annotations

import numpy as np
import pandas as pd


def blend_regression_predictions(predictions: list[pd.DataFrame], pred_col: str = "pred_chm") -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    base = predictions[0][["tile_id", "grid_x", "grid_y"]].copy()
    stacked = np.vstack([df[pred_col].to_numpy() for df in predictions])
    base[pred_col] = np.mean(stacked, axis=0)
    base["uncertainty_chm"] = np.std(stacked, axis=0)
    return base


def majority_vote_classification(predictions: list[pd.DataFrame], pred_col: str = "pred_landcover") -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    base = predictions[0][["tile_id", "grid_x", "grid_y"]].copy()
    stacked = np.vstack([df[pred_col].to_numpy(dtype=int) for df in predictions]).T
    voted = []
    agreement = []
    for row in stacked:
        vals, counts = np.unique(row, return_counts=True)
        idx = np.argmax(counts)
        voted.append(int(vals[idx]))
        agreement.append(float(counts[idx] / len(row)))
    base[pred_col] = voted
    base["agreement_landcover"] = agreement
    return base
