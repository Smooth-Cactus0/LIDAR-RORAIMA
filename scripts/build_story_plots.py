from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_feature_frame(features_dir: Path) -> pd.DataFrame:
    cols = ["zone_epsg", "target_chm", "target_landcover", "z_p90", "point_density", "z_range", "z_std"]
    frames: list[pd.DataFrame] = []
    for path in sorted(features_dir.glob("features_*_10m.parquet")):
        frames.append(pd.read_parquet(path, columns=cols))
    if not frames:
        raise FileNotFoundError(f"No feature parquet files found in {features_dir}")
    return pd.concat(frames, ignore_index=True)


def save_tile_coverage(manifest: pd.DataFrame, out: Path) -> None:
    plot_df = manifest.copy()
    plot_df["epsg"] = plot_df["epsg"].fillna("Unknown").astype(str)
    summary = (
        plot_df.groupby("epsg", dropna=False)
        .agg(tiles=("tile_id", "count"), points=("point_count", "sum"))
        .reset_index()
        .sort_values("points", ascending=False)
    )
    summary["points_b"] = summary["points"] / 1e9

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=summary, x="epsg", y="tiles", hue="epsg", palette="crest", legend=False, ax=axes[0])
    axes[0].set_title("Tile count by EPSG zone")
    axes[0].set_xlabel("EPSG")
    axes[0].set_ylabel("Tiles")

    sns.barplot(data=summary, x="epsg", y="points_b", hue="epsg", palette="mako", legend=False, ax=axes[1])
    axes[1].set_title("Point volume by EPSG zone")
    axes[1].set_xlabel("EPSG")
    axes[1].set_ylabel("Points (billions)")

    fig.suptitle("Spatial coverage profile", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_chm_distribution(features: pd.DataFrame, out: Path) -> None:
    chm = features["target_chm"].dropna()
    chm = chm[chm.between(0, chm.quantile(0.995))]
    if len(chm) > 350_000:
        chm = chm.sample(350_000, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(chm, bins=80, kde=True, color="#2f855a", ax=ax)
    ax.set_title("Canopy height (CHM) distribution")
    ax.set_xlabel("CHM target")
    ax.set_ylabel("Grid-cell frequency")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_landcover_distribution(features: pd.DataFrame, class_map: dict[int, str], out: Path) -> None:
    lc = features["target_landcover"].dropna().astype(int)
    counts = lc.value_counts().sort_values(ascending=False).reset_index()
    counts.columns = ["class_id", "count"]
    counts["class_name"] = counts["class_id"].map(class_map).fillna("Unknown")
    counts["label"] = counts["class_id"].astype(str) + " - " + counts["class_name"]
    counts["share_pct"] = counts["count"] / counts["count"].sum() * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=counts, x="share_pct", y="label", hue="label", palette="viridis", legend=False, ax=ax)
    ax.set_title("Land-cover supervision distribution")
    ax.set_xlabel("Share of labeled grid cells (%)")
    ax.set_ylabel("LAS class")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_feature_signal(features: pd.DataFrame, out: Path) -> None:
    plot_df = features[["z_p90", "point_density", "target_chm"]].dropna()
    plot_df = plot_df[plot_df["target_chm"].between(0, plot_df["target_chm"].quantile(0.995))]
    if len(plot_df) > 250_000:
        plot_df = plot_df.sample(250_000, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hexbin(plot_df["z_p90"], plot_df["target_chm"], gridsize=70, cmap="YlGnBu", mincnt=1)
    axes[0].set_title("z_p90 vs CHM target")
    axes[0].set_xlabel("z_p90")
    axes[0].set_ylabel("target_chm")

    axes[1].hexbin(plot_df["point_density"], plot_df["target_chm"], gridsize=70, cmap="YlOrBr", mincnt=1)
    axes[1].set_title("Point density vs CHM target")
    axes[1].set_xlabel("point_density")
    axes[1].set_ylabel("target_chm")

    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_benchmark_snapshot(benchmarks: pd.DataFrame, out: Path) -> None:
    rows: list[dict[str, object]] = []
    for _, row in benchmarks.iterrows():
        if row["project"] == "canopy":
            rows.append({"project": "Canopy", "metric": "RMSE", "value": float(row["rmse"])})
            rows.append({"project": "Canopy", "metric": "MAE", "value": float(row["mae"])})
        if row["project"] == "landcover":
            rows.append({"project": "Land-cover", "metric": "Macro-F1", "value": float(row["macro_f1"])})
            rows.append({"project": "Land-cover", "metric": "Macro-Precision", "value": float(row["macro_precision"])})
            rows.append({"project": "Land-cover", "metric": "Macro-Recall", "value": float(row["macro_recall"])})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="metric", y="value", hue="project", palette="deep", ax=ax)
    ax.set_title("Baseline benchmark snapshot")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "artifacts" / "manifests" / "tile_manifest.parquet"
    features_dir = root / "artifacts" / "features"
    benchmarks_path = root / "benchmarks" / "all_model_results.csv"
    class_map_path = root / "configs" / "landcover_class_map.json"

    out_eda = root / "images" / "eda"
    out_canopy = root / "images" / "canopy"
    out_land = root / "images" / "landcover"
    out_eda.mkdir(parents=True, exist_ok=True)
    out_canopy.mkdir(parents=True, exist_ok=True)
    out_land.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    manifest = pd.read_parquet(manifest_path)
    features = load_feature_frame(features_dir)
    benchmarks = pd.read_csv(benchmarks_path)
    class_map_raw = json.loads(class_map_path.read_text(encoding="utf-8"))
    class_map = {int(x["class_id"]): x["class_name"] for x in class_map_raw}

    save_tile_coverage(manifest, out_eda / "coverage_by_zone.png")
    save_chm_distribution(features, out_canopy / "chm_target_distribution.png")
    save_landcover_distribution(features, class_map, out_land / "landcover_class_distribution.png")
    save_feature_signal(features, out_eda / "feature_signal_vs_chm.png")
    save_benchmark_snapshot(benchmarks, out_eda / "baseline_benchmark_snapshot.png")

    print("Story plots generated.")


if __name__ == "__main__":
    main()
