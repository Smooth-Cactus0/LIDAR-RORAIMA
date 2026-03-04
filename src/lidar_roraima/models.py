from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cv import build_tile_blocked_folds


CANOPY_TARGET = "target_chm"
LANDCOVER_TARGET = "target_landcover"
COMMON_EXCLUDE = {"tile_id", "zone_epsg", "grid_x", "grid_y", CANOPY_TARGET, LANDCOVER_TARGET}


@dataclass
class TrainResult:
    project: str
    model_name: str
    params_hash: str
    fold_scheme: str
    metrics: dict[str, float]
    artifact_path: str


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col not in COMMON_EXCLUDE]
    usable: list[str] = []
    for col in cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            if series.notna().sum() == 0:
                continue
            usable.append(col)
    return usable


def _hash_params(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _build_regressor(model_family: str, seed: int):
    if model_family == "baseline":
        return DummyRegressor(strategy="mean"), {"strategy": "mean"}
    if model_family == "random_forest":
        params = {
            "n_estimators": 200,
            "max_depth": 20,
            "random_state": seed,
            "n_jobs": -1,
        }
        return RandomForestRegressor(**params), params
    if model_family == "boosting":
        try:
            from xgboost import XGBRegressor

            params = {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": seed,
                "n_jobs": -1,
            }
            return XGBRegressor(**params), params
        except Exception:
            params = {"n_estimators": 300, "learning_rate": 0.05, "random_state": seed}
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(**params), params
    raise ValueError(f"Unsupported regressor family: {model_family}")


def _build_classifier(model_family: str, seed: int):
    if model_family == "baseline":
        return DummyClassifier(strategy="most_frequent"), {"strategy": "most_frequent"}
    if model_family == "random_forest":
        params = {
            "n_estimators": 300,
            "max_depth": 25,
            "random_state": seed,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        return RandomForestClassifier(**params), params
    if model_family == "boosting":
        try:
            from xgboost import XGBClassifier

            params = {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": seed,
                "n_jobs": -1,
                "eval_metric": "mlogloss",
            }
            return XGBClassifier(**params), params
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier

            params = {"max_depth": 10, "learning_rate": 0.05, "random_state": seed}
            return HistGradientBoostingClassifier(**params), params
    raise ValueError(f"Unsupported classifier family: {model_family}")


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "macro_f1": macro_f1,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1_recomputed": float(f1),
    }


def _mean_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in metrics_list for key in row})
    return {key: float(np.mean([row[key] for row in metrics_list])) for key in keys}


def train_canopy_model(
    features_df: pd.DataFrame,
    model_family: str,
    output_dir: Path,
    seed: int = 42,
    n_splits: int = 5,
    max_rows: int | None = None,
) -> TrainResult:
    df = features_df.dropna(subset=[CANOPY_TARGET]).copy()
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    folds = build_tile_blocked_folds(df, n_splits=n_splits, seed=seed)
    cols = _feature_columns(df)
    estimator, params = _build_regressor(model_family=model_family, seed=seed)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    fold_metrics: list[dict[str, float]] = []
    for fold in folds:
        train_df = df[df["tile_id"].isin(fold.train_tiles)]
        valid_df = df[df["tile_id"].isin(fold.valid_tiles)]
        if train_df.empty or valid_df.empty:
            continue
        pipeline.fit(train_df[cols], train_df[CANOPY_TARGET])
        pred = pipeline.predict(valid_df[cols])
        fold_metrics.append(_regression_metrics(valid_df[CANOPY_TARGET].to_numpy(), np.asarray(pred)))

    pipeline.fit(df[cols], df[CANOPY_TARGET])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"canopy_{model_family}.joblib"
    joblib.dump({"model": pipeline, "feature_columns": cols, "params": params}, model_path)

    metrics = _mean_metrics(fold_metrics) if fold_metrics else {}
    return TrainResult(
        project="canopy",
        model_name=model_family,
        params_hash=_hash_params(params),
        fold_scheme=f"tile_blocked_{n_splits}",
        metrics=metrics,
        artifact_path=str(model_path.resolve()),
    )


def train_landcover_model(
    features_df: pd.DataFrame,
    model_family: str,
    output_dir: Path,
    seed: int = 42,
    n_splits: int = 5,
    max_rows: int | None = None,
) -> TrainResult:
    df = features_df.dropna(subset=[LANDCOVER_TARGET]).copy()
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    df[LANDCOVER_TARGET] = df[LANDCOVER_TARGET].astype(int)
    folds = build_tile_blocked_folds(df, n_splits=n_splits, seed=seed)
    cols = _feature_columns(df)
    estimator, params = _build_classifier(model_family=model_family, seed=seed)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    fold_metrics: list[dict[str, float]] = []
    for fold in folds:
        train_df = df[df["tile_id"].isin(fold.train_tiles)]
        valid_df = df[df["tile_id"].isin(fold.valid_tiles)]
        if train_df.empty or valid_df.empty:
            continue
        pipeline.fit(train_df[cols], train_df[LANDCOVER_TARGET])
        pred = pipeline.predict(valid_df[cols])
        fold_metrics.append(_classification_metrics(valid_df[LANDCOVER_TARGET].to_numpy(), np.asarray(pred)))

    pipeline.fit(df[cols], df[LANDCOVER_TARGET])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"landcover_{model_family}.joblib"
    joblib.dump({"model": pipeline, "feature_columns": cols, "params": params}, model_path)

    metrics = _mean_metrics(fold_metrics) if fold_metrics else {}
    return TrainResult(
        project="landcover",
        model_name=model_family,
        params_hash=_hash_params(params),
        fold_scheme=f"tile_blocked_{n_splits}",
        metrics=metrics,
        artifact_path=str(model_path.resolve()),
    )
