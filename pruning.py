"""Utilities for applying pruning to MinimalRFDETR checkpoints.

This module centralizes pruning so it can be reused by model export scripts
without duplicating logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class MinimalRFDETR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024),
            num_layers=6,
        )

        self.classifier = nn.Linear(256, 2)
        self.regressor = nn.Linear(256, 4)

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(2).transpose(1, 2)
        encoded = self.transformer(features)
        logits = self.classifier(encoded)
        boxes = self.regressor(encoded)
        return logits, boxes


def load_pytorch_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = MinimalRFDETR().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def prune_model(model: nn.Module, prune_percent: float) -> nn.Module:
    """Apply global-style unstructured pruning to Conv/Linear weights."""
    model_copy = MinimalRFDETR()
    model_copy.load_state_dict(model.state_dict())

    for _, module in model_copy.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name="weight", amount=prune_percent)
            prune.remove(module, "weight")

    return model_copy


def save_pruned_checkpoint(input_checkpoint: str, output_checkpoint: str, percent: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pytorch_model(input_checkpoint, device)
    pruned = prune_model(model, percent)

    Path(output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": pruned.state_dict()}, output_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune a MinimalRFDETR checkpoint")
    parser.add_argument("--input", required=True, help="Input .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output .pth checkpoint")
    parser.add_argument(
        "--percent",
        required=True,
        type=float,
        help="Pruning fraction (e.g. 0.3 for 30%%)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_pruned_checkpoint(args.input, args.output, args.percent)
    print(f"Saved pruned checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
