from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import build_transforms
from src.models import create_resnet50_multilabel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained garment attribute classifier.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_resnet50.pt"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Print predictions as JSON.")
    return parser.parse_args()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_attribute_name(column_name: str) -> str:
    name = column_name.removeprefix("attr_")
    return name.replace("_", " ")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def predict(
    checkpoint_path: Path,
    image_path: Path,
    image_size: int | None,
    threshold: float,
    top_k: int,
) -> list[dict[str, float | str]]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not 0 < threshold < 1:
        raise ValueError("--threshold must be in the interval (0, 1).")
    if top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")

    device = get_device()
    checkpoint = load_checkpoint(checkpoint_path, device)
    attr_columns = checkpoint["attr_columns"]
    train_args = checkpoint.get("args", {})
    resolved_image_size = image_size or int(train_args.get("image_size", 224))

    model = create_resnet50_multilabel(
        num_labels=len(attr_columns),
        pretrained=False,
        dropout=float(train_args.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transforms(image_size=resolved_image_size, train=False)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    logits = model(image_tensor)
    probabilities = torch.sigmoid(logits).squeeze(0).cpu()

    ranked_indices = torch.argsort(probabilities, descending=True).tolist()
    selected_indices = [idx for idx in ranked_indices if probabilities[idx].item() >= threshold]
    if not selected_indices:
        selected_indices = ranked_indices[:top_k]
    else:
        selected_indices = selected_indices[:top_k]

    return [
        {
            "attribute": clean_attribute_name(attr_columns[idx]),
            "probability": round(float(probabilities[idx].item()), 4),
        }
        for idx in selected_indices
    ]


def main() -> None:
    args = parse_args()
    predictions = predict(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        image_size=args.image_size,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(predictions, indent=2))
        return

    print(f"Image: {args.image}")
    for item in predictions:
        print(f"{item['attribute']}: {item['probability']:.4f}")


if __name__ == "__main__":
    main()
