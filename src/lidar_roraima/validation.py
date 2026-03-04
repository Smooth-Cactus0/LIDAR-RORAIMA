from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    name: str
    passed: bool
    details: list[str]


def validate_manifest_schema(manifest: pd.DataFrame) -> ValidationReport:
    required = {
        "tile_id",
        "file_name",
        "is_copc",
        "is_duplicate",
        "las_version",
        "point_format",
        "epsg",
        "bbox",
        "point_count",
        "size_bytes",
        "qa_flags",
    }
    details: list[str] = []
    missing = sorted(required - set(manifest.columns))
    if missing:
        details.append(f"Missing columns: {missing}")
    if "point_count" in manifest.columns and (manifest["point_count"] <= 0).any():
        details.append("Found non-positive point_count values.")
    if "bbox" in manifest.columns:
        try:
            manifest["bbox"].apply(json.loads)
        except Exception:
            details.append("bbox column is not valid JSON for all rows.")
    return ValidationReport(name="manifest_schema", passed=not details, details=details)


def validate_feature_schema(features: pd.DataFrame) -> ValidationReport:
    required = {"tile_id", "zone_epsg", "grid_x", "grid_y", "point_density", "z_mean"}
    details: list[str] = []
    missing = sorted(required - set(features.columns))
    if missing:
        details.append(f"Missing columns: {missing}")
    numeric_cols = [col for col in features.columns if col not in {"tile_id"}]
    if numeric_cols:
        has_inf = np.isinf(features[numeric_cols].select_dtypes(include=[np.number])).any().any()
        if bool(has_inf):
            details.append("Found inf values in numeric feature columns.")
    return ValidationReport(name="feature_schema", passed=not details, details=details)


def validate_no_tile_leakage(train_tiles: list[str], valid_tiles: list[str]) -> ValidationReport:
    overlap = sorted(set(train_tiles).intersection(valid_tiles))
    details: list[str] = []
    if overlap:
        details.append(f"Leakage detected; overlapping tiles: {overlap}")
    return ValidationReport(name="tile_leakage", passed=not details, details=details)
