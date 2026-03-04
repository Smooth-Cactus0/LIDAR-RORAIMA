from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    root_dir: Path
    raw_data_dir: Path
    artifacts_dir: Path
    manifests_dir: Path
    features_dir: Path
    models_dir: Path
    inference_dir: Path
    random_seed: int = 42

    @staticmethod
    def from_root(root_dir: str | Path) -> "ProjectConfig":
        root = Path(root_dir).resolve()
        artifacts = root / "artifacts"
        return ProjectConfig(
            root_dir=root,
            raw_data_dir=root / "lidar_data",
            artifacts_dir=artifacts,
            manifests_dir=artifacts / "manifests",
            features_dir=artifacts / "features",
            models_dir=artifacts / "models",
            inference_dir=artifacts / "inference",
        )

    def ensure_dirs(self) -> None:
        for path in [
            self.artifacts_dir,
            self.manifests_dir,
            self.features_dir,
            self.models_dir,
            self.inference_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
