from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class DeepFashionMultiLabelDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
        split: str,
        transform: Callable | None = None,
        use_abs_path: bool = True,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.split = split
        self.transform = transform
        self.use_abs_path = use_abs_path

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_csv}")

        df = pd.read_csv(self.metadata_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows found for split='{split}' in {self.metadata_csv}")

        self.attr_columns = [col for col in df.columns if col.startswith("attr_")]
        if not self.attr_columns:
            raise ValueError("No attribute columns found. Expected columns prefixed with 'attr_'.")

        path_column = "abs_image_path" if use_abs_path and "abs_image_path" in df.columns else "image_path"
        self.image_paths = df[path_column].astype(str).tolist()
        self.labels = torch.tensor(df[self.attr_columns].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def create_dataloader(
    metadata_csv: str | Path,
    split: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    use_abs_path: bool = True,
) -> tuple[DataLoader, list[str]]:
    is_train = split == "train"
    transform = build_transforms(image_size=image_size, train=is_train)
    dataset = DeepFashionMultiLabelDataset(
        metadata_csv=metadata_csv,
        split=split,
        transform=transform,
        use_abs_path=use_abs_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.attr_columns
