import argparse
import csv
import re
from collections import Counter
from pathlib import Path


def find_one(pattern: str) -> Path:
    matches = sorted(Path().glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No matches for pattern: {pattern}")
    return matches[0]


def read_attribute_names(attr_cloth_file: Path) -> list[str]:
    names: list[str] = []
    with attr_cloth_file.open("r", encoding="utf-8", errors="ignore") as f:
        next(f, None)
        next(f, None)
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            name, _attr_type = stripped.rsplit(maxsplit=1)
            names.append(name)
    return names


def read_eval_splits(eval_file: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    with eval_file.open("r", encoding="utf-8", errors="ignore") as f:
        next(f, None)
        next(f, None)
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            split_map[parts[0]] = parts[1]
    return split_map


def count_positive_attributes(attr_img_file: Path, n_attrs: int) -> tuple[list[int], int]:
    positives = [0] * n_attrs
    n_images = 0
    with attr_img_file.open("r", encoding="utf-8", errors="ignore") as f:
        next(f, None)
        next(f, None)
        for line in f:
            parts = line.split()
            if len(parts) < n_attrs + 1:
                continue
            labels = parts[1 : 1 + n_attrs]
            for idx, value in enumerate(labels):
                if value == "1":
                    positives[idx] += 1
            n_images += 1
    return positives, n_images


def sanitize_column(name: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"attr_{clean or 'unnamed'}"


def unique_column_names(attr_names: list[str]) -> list[str]:
    columns: list[str] = []
    seen: Counter[str] = Counter()
    for name in attr_names:
        base = sanitize_column(name)
        seen[base] += 1
        if seen[base] == 1:
            columns.append(base)
        else:
            columns.append(f"{base}_{seen[base]}")
    return columns


def read_selected_attributes(attrs_file: Path) -> list[str]:
    selected: list[str] = []
    with attrs_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            selected.append(name)
    return selected


def build_metadata(
    data_root: Path,
    attr_img_file: Path,
    eval_file: Path,
    attr_names: list[str],
    output_file: Path,
    max_attrs: int,
    selected_names: list[str] | None = None,
) -> None:
    split_map = read_eval_splits(eval_file)
    positives, n_images = count_positive_attributes(attr_img_file, len(attr_names))

    name_to_idx = {name: idx for idx, name in enumerate(attr_names)}

    if selected_names is not None:
        missing = [name for name in selected_names if name not in name_to_idx]
        if missing:
            raise ValueError(
                "Attributes from --attrs-file not found in dataset: "
                + ", ".join(missing)
            )
        selected_indices = [name_to_idx[name] for name in selected_names]
    elif max_attrs <= 0 or max_attrs >= len(attr_names):
        selected_indices = list(range(len(attr_names)))
    else:
        selected_indices = sorted(
            sorted(range(len(attr_names)), key=lambda i: positives[i], reverse=True)[:max_attrs]
        )

    selected_attr_names = [attr_names[i] for i in selected_indices]
    selected_columns = unique_column_names(selected_attr_names)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "image_path",
        "abs_image_path",
        "split",
        "num_positive_total",
        "num_positive_selected",
        "active_attrs_selected",
        *selected_columns,
    ]

    split_counts: Counter[str] = Counter()
    unknown_splits = 0

    with attr_img_file.open("r", encoding="utf-8", errors="ignore") as src, output_file.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        writer = csv.writer(dst)
        writer.writerow(header)

        next(src, None)
        next(src, None)

        for line in src:
            parts = line.split()
            if len(parts) < len(attr_names) + 1:
                continue

            image_path = parts[0]
            labels = parts[1 : 1 + len(attr_names)]

            split = split_map.get(image_path, "unknown")
            if split == "unknown":
                unknown_splits += 1
            split_counts[split] += 1

            selected_values = [1 if labels[idx] == "1" else 0 for idx in selected_indices]
            active_selected = [
                selected_attr_names[i] for i, value in enumerate(selected_values) if value == 1
            ]

            row = [
                image_path,
                str((data_root / image_path).resolve()),
                split,
                sum(1 for value in labels if value == "1"),
                sum(selected_values),
                "|".join(active_selected),
                *selected_values,
            ]
            writer.writerow(row)

    print(f"Images processed: {n_images}")
    print(f"Attributes available: {len(attr_names)}")
    print(f"Attributes selected: {len(selected_indices)}")
    if selected_names is not None:
        print("Selection mode: attrs-file")
    else:
        print("Selection mode: top-k frequency")
    print(f"Output: {output_file}")
    print(f"Split counts: {dict(split_counts)}")
    print(f"Unknown split rows: {unknown_splits}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata.csv from DeepFashion annotations.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--anno-dir", type=Path, default=None)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument(
        "--attrs-file",
        type=Path,
        default=None,
        help="Path to a text file with one attribute name per line.",
    )
    parser.add_argument(
        "--max-attrs",
        type=int,
        default=50,
        help="How many most frequent attributes to include. Ignored when --attrs-file is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    anno_dir = args.anno_dir or find_one("data/Anno_coarse*/Anno_coarse")
    eval_file = args.eval_file or find_one("data/Eval*/Eval/list_eval_partition.txt")
    attr_img_file = anno_dir / "list_attr_img.txt"
    attr_cloth_file = anno_dir / "list_attr_cloth.txt"

    if not attr_img_file.exists():
        raise FileNotFoundError(f"Missing file: {attr_img_file}")
    if not attr_cloth_file.exists():
        raise FileNotFoundError(f"Missing file: {attr_cloth_file}")
    if not eval_file.exists():
        raise FileNotFoundError(f"Missing file: {eval_file}")

    attr_names = read_attribute_names(attr_cloth_file)
    selected_names = None
    if args.attrs_file is not None:
        selected_names = read_selected_attributes(args.attrs_file)
        if not selected_names:
            raise ValueError(f"No attribute names found in: {args.attrs_file}")

    build_metadata(
        data_root=args.data_root,
        attr_img_file=attr_img_file,
        eval_file=eval_file,
        attr_names=attr_names,
        output_file=args.output,
        max_attrs=args.max_attrs,
        selected_names=selected_names,
    )


if __name__ == "__main__":
    main()
