from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def run_inference(
    model_path: Path,
    features_df: pd.DataFrame,
    prediction_column: str,
    uncertainty_column: str = "uncertainty",
) -> pd.DataFrame:
    payload = joblib.load(model_path)
    model = payload["model"]
    cols = payload["feature_columns"]

    pred = model.predict(features_df[cols])
    out = features_df[["tile_id", "grid_x", "grid_y"]].copy()
    out[prediction_column] = pred

    uncertainty = np.nan
    model_step = model.named_steps.get("model")
    if hasattr(model_step, "predict_proba"):
        proba = model.predict_proba(features_df[cols])
        uncertainty = 1.0 - np.max(proba, axis=1)
    elif hasattr(model_step, "estimators_"):
        try:
            preds = np.vstack([est.predict(features_df[cols]) for est in model_step.estimators_])
            uncertainty = np.std(preds, axis=0)
        except Exception:
            uncertainty = np.nan
    out[uncertainty_column] = uncertainty
    out["model_version"] = model_path.stem
    out["timestamp_utc"] = datetime.now(tz=timezone.utc).isoformat()
    return out
