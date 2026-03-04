from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .models import TrainResult


def append_model_result(result: TrainResult, registry_path: Path, kaggle_notebook_url: str = "") -> pd.DataFrame:
    row = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "project": result.project,
        "model_name": result.model_name,
        "params_hash": result.params_hash,
        "fold_scheme": result.fold_scheme,
        "artifact_path": result.artifact_path,
        "kaggle_notebook_url": kaggle_notebook_url,
    }
    row.update(result.metrics)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        df = pd.read_csv(registry_path)
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(registry_path, index=False)
    return df
