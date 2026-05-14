from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


SAFE_CHARS = re.compile(r"[^A-Za-z0-9._/-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Kaggle-safe DeepFashion export folder.")
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("kaggle_export"))
    parser.add_argument("--selected-attributes", type=Path, default=Path("data/selected_attributes.txt"))
    return parser.parse_args()


def sanitize_path(path: str) -> str:
    parts = Path(path).parts
    return "/".join(SAFE_CHARS.sub("_", part).strip("_") for part in parts)


def copy_image(src_root: Path, dst_root: Path, old_rel_path: str, new_rel_path: str) -> None:
    src = src_root / old_rel_path
    dst = dst_root / new_rel_path
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata not found: {args.metadata}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metadata)

    if "image_path" not in df.columns:
        raise ValueError("metadata.csv must contain an image_path column.")

    old_paths = df["image_path"].astype(str).tolist()
    new_paths = [sanitize_path(path) for path in old_paths]

    if len(set(new_paths)) != len(new_paths):
        raise ValueError("Sanitized image paths produced duplicates. Use a stricter naming strategy.")

    for old_path, new_path in tqdm(list(zip(old_paths, new_paths)), desc="copy images"):
        copy_image(args.data_root, args.output_dir, old_path, new_path)

    df["image_path"] = new_paths
    if "abs_image_path" in df.columns:
        df = df.drop(columns=["abs_image_path"])
    df.to_csv(args.output_dir / "metadata.csv", index=False)

    if args.selected_attributes.exists():
        shutil.copy2(args.selected_attributes, args.output_dir / "selected_attributes.txt")

    print(f"Kaggle export written to: {args.output_dir}")
    print("Upload this folder or zip its contents for Kaggle.")


if __name__ == "__main__":
    main()
