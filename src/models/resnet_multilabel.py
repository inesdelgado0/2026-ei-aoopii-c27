from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_resnet50_multilabel(
    num_labels: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    if num_labels <= 0:
        raise ValueError("num_labels must be greater than 0.")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in the interval [0, 1).")

    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_labels),
    )

    return model
