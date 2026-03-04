from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    max_rows: int | None
    n_splits: int


PROFILES: dict[str, RuntimeProfile] = {
    "kaggle_full": RuntimeProfile(name="kaggle_full", max_rows=None, n_splits=5),
    "colab_full": RuntimeProfile(name="colab_full", max_rows=None, n_splits=5),
    "local_smoke": RuntimeProfile(name="local_smoke", max_rows=250_000, n_splits=3),
}


def get_profile(name: str) -> RuntimeProfile:
    if name not in PROFILES:
        supported = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unsupported runtime profile '{name}'. Supported: {supported}")
    return PROFILES[name]
