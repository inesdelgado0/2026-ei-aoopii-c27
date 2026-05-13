from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import create_dataloader
from src.models import create_resnet50_multilabel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet-50 multi-label garment classifier.")
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-pos-weight", action="store_true")
    return parser.parse_args()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_pos_weight(train_loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    labels = train_loader.dataset.labels
    positives = labels.sum(dim=0)
    negatives = len(labels) - positives
    pos_weight = negatives / positives.clamp(min=1)
    return pos_weight.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    samples_seen = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        samples_seen += images.size(0)
        if max_batches is not None and batch_idx >= max_batches:
            break

    return running_loss / samples_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    samples_seen = 0
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="val", leave=False), start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).int()

        running_loss += loss.item() * images.size(0)
        samples_seen += images.size(0)
        all_targets.append(labels.cpu().int())
        all_predictions.append(predictions.cpu())
        if max_batches is not None and batch_idx >= max_batches:
            break

    targets = torch.cat(all_targets).numpy()
    predictions = torch.cat(all_predictions).numpy()

    return {
        "loss": running_loss / samples_seen,
        "f1_micro": f1_score(targets, predictions, average="micro", zero_division=0),
        "f1_macro": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    attr_columns: list[str],
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "attr_columns": attr_columns,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be in the interval (0, 1).")

    device = get_device()
    print(f"Device: {device}")

    train_loader, attr_columns = create_dataloader(
        args.metadata,
        split="train",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    val_loader, val_attr_columns = create_dataloader(
        args.metadata,
        split="val",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    if attr_columns != val_attr_columns:
        raise ValueError("Train and validation attribute columns do not match.")

    model = create_resnet50_multilabel(
        num_labels=len(attr_columns),
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    pos_weight = None if args.no_pos_weight else compute_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_path = args.output_dir / "best_resnet50.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.threshold,
            max_batches=args.max_val_batches,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"f1_micro={val_metrics['f1_micro']:.4f} "
            f"f1_macro={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            save_checkpoint(best_path, model, attr_columns, epoch, val_metrics, args)
            print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
