from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def project_root(from_file: str) -> Path:
    return Path(from_file).resolve().parents[1]


def kaggle_json_path() -> Path:
    cfg_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if cfg_dir:
        return Path(cfg_dir) / "kaggle.json"
    return Path.home() / ".kaggle" / "kaggle.json"


def assert_kaggle_credentials() -> Path:
    path = kaggle_json_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Kaggle credentials file: {path}. "
            "Create it from your Kaggle account API token."
        )
    return path


def notebook_slug(nb_path: Path) -> str:
    return nb_path.stem.replace("_", "-")


def notebook_title(nb_path: Path) -> str:
    return nb_path.stem.replace("_", " ").title()


def ensure_kernel_folder(
    notebook_path: Path,
    out_dir: Path,
    username: str,
    dataset_ref: str,
    is_private: bool,
    kernel_id_override: str | None = None,
    kernel_sources: list[str] | None = None,
) -> Path:
    root = notebook_path.resolve().parents[1]
    slug = notebook_slug(notebook_path)
    folder = out_dir / slug
    folder.mkdir(parents=True, exist_ok=True)
    target_nb = folder / notebook_path.name
    shutil.copy2(notebook_path, target_nb)
    src_dir = root / "src"
    if src_dir.exists():
        target_src = folder / "src"
        if target_src.exists():
            shutil.rmtree(target_src)
        shutil.copytree(src_dir, target_src)

    metadata = {
        "id": kernel_id_override or f"{username}/{slug}",
        "title": notebook_title(notebook_path),
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": bool(is_private),
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": kernel_sources or [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return folder


def run_kaggle_cmd(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    kaggle_exe = shutil.which("kaggle")
    if kaggle_exe:
        cmd = [kaggle_exe] + args
    else:
        cmd = [sys.executable, "-m", "kaggle.cli"] + args
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )
