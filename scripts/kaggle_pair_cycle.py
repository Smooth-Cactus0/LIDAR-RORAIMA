from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kaggle_utils import ensure_kernel_folder, project_root, run_kaggle_cmd


ORDER = [
    "00_metadata_eda.ipynb",
    "01_feature_engineering_shared.ipynb",
    "10_canopy_baseline.ipynb",
    "11_canopy_random_forest.ipynb",
    "12_canopy_boosting.ipynb",
    "13_canopy_ensemble.ipynb",
    "20_landcover_baseline.ipynb",
    "21_landcover_random_forest.ipynb",
    "22_landcover_boosting.ipynb",
    "23_landcover_ensemble.ipynb",
    "90_portfolio_inference_showcase.ipynb",
]

SLUG_OVERRIDES: dict[str, str] = {
    "13_canopy_ensemble.ipynb": "roraima-13-canopy-ensemble-v1",
    "20_landcover_baseline.ipynb": "roraima-20-landcover-baseline-v1",
    "21_landcover_random_forest.ipynb": "roraima-21-landcover-rf-v1",
    "22_landcover_boosting.ipynb": "roraima-22-landcover-boosting-v1",
    "23_landcover_ensemble.ipynb": "roraima-23-landcover-ensemble-v1",
    "90_portfolio_inference_showcase.ipynb": "roraima-90-showcase-v1",
}

DEPENDENCIES: dict[str, list[str]] = {
    "00_metadata_eda.ipynb": [],
    "01_feature_engineering_shared.ipynb": ["00_metadata_eda.ipynb"],
    "10_canopy_baseline.ipynb": ["01_feature_engineering_shared.ipynb"],
    "11_canopy_random_forest.ipynb": ["01_feature_engineering_shared.ipynb"],
    "12_canopy_boosting.ipynb": ["01_feature_engineering_shared.ipynb"],
    "13_canopy_ensemble.ipynb": [
        "10_canopy_baseline.ipynb",
        "11_canopy_random_forest.ipynb",
        "12_canopy_boosting.ipynb",
    ],
    "20_landcover_baseline.ipynb": ["01_feature_engineering_shared.ipynb"],
    "21_landcover_random_forest.ipynb": ["01_feature_engineering_shared.ipynb"],
    "22_landcover_boosting.ipynb": ["01_feature_engineering_shared.ipynb"],
    "23_landcover_ensemble.ipynb": [
        "20_landcover_baseline.ipynb",
        "21_landcover_random_forest.ipynb",
        "22_landcover_boosting.ipynb",
    ],
    "90_portfolio_inference_showcase.ipynb": [
        "13_canopy_ensemble.ipynb",
        "23_landcover_ensemble.ipynb",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Kaggle notebook push/status in pairs.")
    p.add_argument("--username", required=True)
    p.add_argument("--dataset-ref", default="rogriofmeireles/lidar-roraima-parime-research")
    p.add_argument("--start", type=int, default=0, help="Start index in ordered notebook list.")
    p.add_argument("--count", type=int, default=2, help="How many notebooks to process this cycle.")
    p.add_argument("--polls", type=int, default=8, help="Number of status polls after push.")
    p.add_argument("--sleep-seconds", type=int, default=25, help="Sleep between polls.")
    p.add_argument("--cooldown-seconds", type=int, default=120, help="Wait before starting cycle.")
    p.add_argument("--private", action="store_true")
    p.add_argument(
        "--chain-previous",
        action="store_true",
        help="Set each notebook kernel_sources to previous notebook ref in ORDER.",
    )
    return p.parse_args()


def slug_from_name(name: str) -> str:
    return name.replace(".ipynb", "").replace("_", "-")


def slug_for_name(name: str) -> str:
    return SLUG_OVERRIDES.get(name, slug_from_name(name))


def main() -> None:
    args = parse_args()
    root = project_root(__file__)
    nb_dir = root / "notebooks"
    out_root = root / "kaggle" / "kernels"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.cooldown_seconds > 0:
        print(f"Cooling down for {args.cooldown_seconds}s ...")
        time.sleep(args.cooldown_seconds)

    selected = ORDER[args.start : args.start + args.count]
    if not selected:
        print("No notebooks selected.")
        return

    refs: list[str] = []
    print("Pushing pair:")
    for name in selected:
        nb_path = nb_dir / name
        if not nb_path.exists():
            print(f"- missing local notebook: {name}")
            continue
        prev_source: list[str] = []
        if args.chain_previous:
            prev_source = [f"{args.username}/{slug_for_name(dep)}" for dep in DEPENDENCIES.get(name, [])]
        folder = ensure_kernel_folder(
            notebook_path=nb_path,
            out_dir=out_root,
            username=args.username,
            dataset_ref=args.dataset_ref,
            is_private=args.private,
            kernel_sources=prev_source,
        )
        slug = slug_for_name(name)
        ref = f"{args.username}/{slug}"
        refs.append(ref)
        res = run_kaggle_cmd(["kernels", "push", "-p", str(folder)])
        combined = (res.stdout or "") + "\n" + (res.stderr or "")
        ok = res.returncode == 0 and "Kernel push error" not in combined and "error:" not in combined.lower()
        if not ok and "Notebook not found" in combined:
            fallback_ref = f"{args.username}/{slug}-r1"
            meta_path = folder / "kernel-metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["id"] = fallback_ref
            meta["title"] = f"{meta.get('title', slug)} r1"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            retry = run_kaggle_cmd(["kernels", "push", "-p", str(folder)])
            retry_combined = (retry.stdout or "") + "\\n" + (retry.stderr or "")
            ok = retry.returncode == 0 and "Kernel push error" not in retry_combined and "error:" not in retry_combined.lower()
            if ok:
                ref = fallback_ref
        print(f"- {ref}: {'push ok' if ok else 'push failed'}")
        if not ok and combined.strip():
            print(combined.strip())

    for poll in range(1, args.polls + 1):
        print(f"\nPoll {poll}/{args.polls}:")
        for ref in refs:
            status = run_kaggle_cmd(["kernels", "status", ref])
            if status.returncode == 0:
                line = status.stdout.strip().splitlines()[-1] if status.stdout.strip() else "ok"
                print(f"- {ref}: {line}")
            else:
                err = status.stderr.strip() or status.stdout.strip() or "status failed"
                print(f"- {ref}: {err}")
        if poll < args.polls:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
