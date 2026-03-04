from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FoldAssignment:
    fold_id: int
    train_tiles: list[str]
    valid_tiles: list[str]


def build_tile_blocked_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> list[FoldAssignment]:
    tiles = sorted(df["tile_id"].dropna().unique().tolist())
    if len(tiles) < n_splits:
        n_splits = max(2, len(tiles))
    rng = np.random.default_rng(seed)
    shuffled = tiles.copy()
    rng.shuffle(shuffled)

    buckets: list[list[str]] = [[] for _ in range(n_splits)]
    for idx, tile in enumerate(shuffled):
        buckets[idx % n_splits].append(tile)

    folds: list[FoldAssignment] = []
    for fold_id in range(n_splits):
        valid_tiles = sorted(buckets[fold_id])
        train_tiles = sorted([tile for i, bucket in enumerate(buckets) if i != fold_id for tile in bucket])
        folds.append(FoldAssignment(fold_id=fold_id, train_tiles=train_tiles, valid_tiles=valid_tiles))
    return folds
