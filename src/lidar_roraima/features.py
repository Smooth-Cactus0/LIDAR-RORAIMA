from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _require_laspy() -> None:
    try:
        import laspy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "laspy is required for LAZ feature extraction. Install with: pip install laspy[lazrs]"
        ) from exc


def _weighted_majority(values: np.ndarray) -> int:
    values = values.astype(np.int32, copy=False)
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[np.argmax(counts)])


def _safe_quantile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return np.nan
    return float(np.quantile(arr, q))


def _aggregate_cell(df: pd.DataFrame, include_targets: bool) -> dict[str, float | int]:
    z = df["z"].to_numpy(dtype=np.float64)
    intensity = df["intensity"].to_numpy(dtype=np.float64)
    returns = df["return_number"].to_numpy(dtype=np.float64)
    n_returns = df["number_of_returns"].to_numpy(dtype=np.float64)

    result: dict[str, float | int] = {
        "point_density": float(len(df)),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "z_p10": _safe_quantile(z, 0.10),
        "z_p50": _safe_quantile(z, 0.50),
        "z_p90": _safe_quantile(z, 0.90),
        "intensity_mean": float(np.mean(intensity)),
        "intensity_std": float(np.std(intensity)),
        "return_number_mean": float(np.mean(returns)),
        "number_of_returns_mean": float(np.mean(n_returns)),
        "single_return_ratio": float(np.mean(n_returns == 1)),
        "last_return_ratio": float(np.mean(returns == n_returns)),
    }
    result["z_range"] = float(result["z_max"] - result["z_min"])
    result["roughness"] = float(result["z_std"])

    if include_targets:
        cls = df["classification"].to_numpy(dtype=np.int32)
        ground_mask = cls == 2
        canopy_mask = np.isin(cls, [3, 4, 5])
        if np.any(ground_mask) and np.any(canopy_mask):
            chm = float(np.mean(z[canopy_mask]) - np.mean(z[ground_mask]))
        else:
            chm = np.nan
        result["target_chm"] = chm
        result["target_landcover"] = _weighted_majority(cls)

    return result


def _iter_points_chunked(file_path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    _require_laspy()
    import laspy

    with laspy.open(str(file_path)) as reader:
        for points in reader.chunk_iterator(chunk_size):
            yield pd.DataFrame(
                {
                    "x": np.asarray(points.x, dtype=np.float64),
                    "y": np.asarray(points.y, dtype=np.float64),
                    "z": np.asarray(points.z, dtype=np.float64),
                    "intensity": np.asarray(points.intensity),
                    "return_number": np.asarray(points.return_number),
                    "number_of_returns": np.asarray(points.number_of_returns),
                    "classification": np.asarray(points.classification),
                }
            )


def extract_grid_features_for_tile(
    file_path: Path,
    tile_id: str,
    zone_epsg: int | None,
    cell_size: float = 10.0,
    chunk_size: int = 1_000_000,
    include_targets: bool = True,
    max_points: int | None = None,
) -> pd.DataFrame:
    loaded = 0
    # Streaming accumulators by grid cell key.
    stats: dict[tuple[int, int], dict[str, float | int | Counter]] = defaultdict(
        lambda: {
            "count": 0,
            "z_sum": 0.0,
            "z_sumsq": 0.0,
            "z_min": float("inf"),
            "z_max": float("-inf"),
            "int_sum": 0.0,
            "int_sumsq": 0.0,
            "ret_sum": 0.0,
            "nret_sum": 0.0,
            "single_count": 0,
            "last_count": 0,
            "class_counts": Counter(),
            "ground_sum": 0.0,
            "ground_count": 0,
            "canopy_sum": 0.0,
            "canopy_count": 0,
        }
    )

    for chunk in _iter_points_chunked(file_path=file_path, chunk_size=chunk_size):
        if max_points is not None and loaded >= max_points:
            break
        if max_points is not None:
            remaining = max_points - loaded
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]
        loaded += len(chunk)
        gx = np.floor(chunk["x"].to_numpy(dtype=np.float64) / cell_size).astype(np.int64)
        gy = np.floor(chunk["y"].to_numpy(dtype=np.float64) / cell_size).astype(np.int64)
        z = chunk["z"].to_numpy(dtype=np.float64)
        intensity = chunk["intensity"].to_numpy(dtype=np.float64)
        ret = chunk["return_number"].to_numpy(dtype=np.float64)
        nret = chunk["number_of_returns"].to_numpy(dtype=np.float64)
        cls = chunk["classification"].to_numpy(dtype=np.int32)

        for i in range(len(chunk)):
            key = (int(gx[i]), int(gy[i]))
            cell = stats[key]
            zi = float(z[i])
            ii = float(intensity[i])
            ri = float(ret[i])
            nri = float(nret[i])
            ci = int(cls[i])

            cell["count"] += 1
            cell["z_sum"] += zi
            cell["z_sumsq"] += zi * zi
            if zi < cell["z_min"]:
                cell["z_min"] = zi
            if zi > cell["z_max"]:
                cell["z_max"] = zi
            cell["int_sum"] += ii
            cell["int_sumsq"] += ii * ii
            cell["ret_sum"] += ri
            cell["nret_sum"] += nri
            if nri == 1:
                cell["single_count"] += 1
            if ri == nri:
                cell["last_count"] += 1
            cell["class_counts"][ci] += 1
            if ci == 2:
                cell["ground_sum"] += zi
                cell["ground_count"] += 1
            if ci in (3, 4, 5):
                cell["canopy_sum"] += zi
                cell["canopy_count"] += 1

    records: list[dict[str, float | int | str | None]] = []
    for (grid_x, grid_y), cell in stats.items():
        n = int(cell["count"])
        if n < 5:
            continue
        z_mean = float(cell["z_sum"] / n)
        z_var = max(float(cell["z_sumsq"] / n - z_mean * z_mean), 0.0)
        z_std = float(np.sqrt(z_var))

        int_mean = float(cell["int_sum"] / n)
        int_var = max(float(cell["int_sumsq"] / n - int_mean * int_mean), 0.0)
        int_std = float(np.sqrt(int_var))

        class_counts: Counter = cell["class_counts"]  # type: ignore[assignment]
        target_landcover = int(class_counts.most_common(1)[0][0]) if class_counts else np.nan

        g_count = int(cell["ground_count"])
        c_count = int(cell["canopy_count"])
        if include_targets and g_count > 0 and c_count > 0:
            target_chm = float((cell["canopy_sum"] / c_count) - (cell["ground_sum"] / g_count))
        else:
            target_chm = np.nan

        record: dict[str, float | int | str | None] = {
            "tile_id": tile_id,
            "zone_epsg": int(zone_epsg) if zone_epsg is not None else np.nan,
            "grid_x": int(grid_x),
            "grid_y": int(grid_y),
            "point_density": float(n),
            "z_min": float(cell["z_min"]),
            "z_max": float(cell["z_max"]),
            "z_mean": z_mean,
            "z_std": z_std,
            "z_p10": np.nan,
            "z_p50": np.nan,
            "z_p90": np.nan,
            "intensity_mean": int_mean,
            "intensity_std": int_std,
            "return_number_mean": float(cell["ret_sum"] / n),
            "number_of_returns_mean": float(cell["nret_sum"] / n),
            "single_return_ratio": float(cell["single_count"] / n),
            "last_return_ratio": float(cell["last_count"] / n),
            "z_range": float(cell["z_max"] - cell["z_min"]),
            "roughness": z_std,
        }
        if include_targets:
            record["target_chm"] = target_chm
            record["target_landcover"] = target_landcover
        records.append(record)

    return pd.DataFrame(records)


def extract_features_from_manifest(
    manifest: pd.DataFrame,
    output_dir: Path,
    cell_size: float,
    include_targets: bool = True,
    chunk_size: int = 1_000_000,
    max_points_per_tile: int | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    non_duplicate = manifest[manifest["is_duplicate"] == False].copy()

    for zone_epsg, zone_df in non_duplicate.groupby("epsg", dropna=False):
        zone_rows: list[pd.DataFrame] = []
        for _, row in zone_df.iterrows():
            tile_features = extract_grid_features_for_tile(
                file_path=Path(row["file_path"]),
                tile_id=row["tile_id"],
                zone_epsg=None if pd.isna(zone_epsg) else int(zone_epsg),
                cell_size=cell_size,
                chunk_size=chunk_size,
                include_targets=include_targets,
                max_points=max_points_per_tile,
            )
            if not tile_features.empty:
                zone_rows.append(tile_features)
        if not zone_rows:
            continue
        zone_features = pd.concat(zone_rows, ignore_index=True)
        zone_label = "unknown" if pd.isna(zone_epsg) else str(int(zone_epsg))
        file_name = f"features_{zone_label}_{int(cell_size)}m.parquet"
        out_path = output_dir / file_name
        zone_features.to_parquet(out_path, index=False)
        written.append(out_path)

    return written


def load_feature_tables(features_dir: Path) -> pd.DataFrame:
    paths = sorted(features_dir.glob("features_*m.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def write_class_remap(remap: dict[int, str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"class_id": class_id, "class_name": class_name} for class_id, class_name in sorted(remap.items())]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
