from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "artifacts" / "models" / "model_results.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing model registry: {src}")
    out = root / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    canopy = out / "canopy"
    land = out / "landcover"
    canopy.mkdir(parents=True, exist_ok=True)
    land.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    if "timestamp_utc" in df.columns:
        df = df.sort_values("timestamp_utc").drop_duplicates(
            subset=["project", "model_name", "params_hash", "fold_scheme"], keep="last"
        )
    df.to_csv(out / "all_model_results.csv", index=False)
    df[df["project"] == "canopy"].to_csv(canopy / "metrics.csv", index=False)
    df[df["project"] == "landcover"].to_csv(land / "metrics.csv", index=False)
    print("Benchmarks exported.")


if __name__ == "__main__":
    main()
