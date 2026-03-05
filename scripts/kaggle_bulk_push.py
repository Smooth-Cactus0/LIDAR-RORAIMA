from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_utils import assert_kaggle_credentials, ensure_kernel_folder, project_root, run_kaggle_cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push all project notebooks to Kaggle kernels.")
    parser.add_argument("--root", type=Path, default=None, help="Project root.")
    parser.add_argument("--username", required=True, help="Kaggle username.")
    parser.add_argument(
        "--dataset-ref",
        default="rogriofmeireles/lidar-roraima-parime-research",
        help="Kaggle dataset source in owner/slug format.",
    )
    parser.add_argument("--private", action="store_true", help="Publish kernels as private.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare folders but do not push.")
    parser.add_argument(
        "--chain-previous",
        action="store_true",
        help="Set kernel_sources so each notebook depends on previous notebook in sorted order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve() if args.root else project_root(__file__)
    notebooks_dir = root / "notebooks"
    out_root = root / "kaggle" / "kernels"
    out_root.mkdir(parents=True, exist_ok=True)
    assert_kaggle_credentials()

    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(f"No notebooks found in {notebooks_dir}")

    print(f"Preparing {len(notebooks)} notebooks from: {notebooks_dir}")
    pushed = 0
    prev_ref: str | None = None
    for nb in notebooks:
        ks = [prev_ref] if (args.chain_previous and prev_ref) else []
        folder = ensure_kernel_folder(
            notebook_path=nb,
            out_dir=out_root,
            username=args.username,
            dataset_ref=args.dataset_ref,
            is_private=args.private,
            kernel_sources=ks,
        )
        print(f"- Prepared: {folder.name}")
        prev_ref = f"{args.username}/{folder.name}"
        if args.dry_run:
            continue
        res = run_kaggle_cmd(["kernels", "push", "-p", str(folder)])
        combined = (res.stdout or "") + "\n" + (res.stderr or "")
        has_error_text = "Kernel push error" in combined or "error:" in combined.lower()
        if res.returncode == 0 and not has_error_text:
            pushed += 1
            print(f"  push ok: {folder.name}")
        else:
            # Kaggle CLI occasionally returns "Notebook not found" for a new slug.
            # Retry once with a suffixed slug to avoid stale slug state.
            if "Notebook not found" in combined:
                fallback_id = f"{args.username}/{folder.name}-r1"
                meta_path = folder / "kernel-metadata.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["id"] = fallback_id
                meta["title"] = f"{meta.get('title', folder.name)} r1"
                meta["dataset_sources"] = [args.dataset_ref]
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                retry = run_kaggle_cmd(["kernels", "push", "-p", str(folder)])
                retry_combined = (retry.stdout or "") + "\n" + (retry.stderr or "")
                retry_has_error = "Kernel push error" in retry_combined or "error:" in retry_combined.lower()
                if retry.returncode == 0 and not retry_has_error:
                    pushed += 1
                    print(f"  push ok after retry: {folder.name} -> {fallback_id}")
                    continue

            print(f"  push failed: {folder.name}")
            if res.stdout.strip():
                print(res.stdout.strip())
            if res.stderr.strip():
                print(res.stderr.strip())

    print(f"Done. Prepared={len(notebooks)}, Pushed={pushed}, DryRun={args.dry_run}")


if __name__ == "__main__":
    main()
