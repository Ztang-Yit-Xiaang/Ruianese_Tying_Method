from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models


@dataclass
class ClassSets:
    initials: list[str]
    finals: list[str]
    tones: list[str]

    @classmethod
    def from_labels(
        cls,
        labels: list[dict],
        initial_key: str = "initial",
        final_key: str = "final",
    ) -> "ClassSets":
        return cls(
            initials=sorted({str(row[initial_key]) for row in labels}),
            finals=sorted({str(row[final_key]) for row in labels}),
            tones=sorted({str(row["tone"]) for row in labels}, key=lambda value: int(value)),
        )

    def as_dict(self) -> dict:
        return {"initials": self.initials, "finals": self.finals, "tones": self.tones}

    @classmethod
    def from_dict(cls, data: dict) -> "ClassSets":
        return cls(
            initials=list(data["initials"]),
            finals=list(data["finals"]),
            tones=list(data["tones"]),
        )


class MultiHeadClassifier(nn.Module):
    def __init__(self, arch: str, classes: ClassSets) -> None:
        super().__init__()
        self.arch = arch
        self.classes = classes
        if arch == "resnet18":
            backbone = models.resnet18(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif arch == "resnet34":
            backbone = models.resnet34(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif arch == "convnext_tiny":
            backbone = models.convnext_tiny(weights=None)
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        self.backbone = backbone
        self.initial_head = nn.Linear(feature_dim, len(classes.initials))
        self.final_head = nn.Linear(feature_dim, len(classes.finals))
        self.tone_head = nn.Linear(feature_dim, len(classes.tones))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {
            "initial": self.initial_head(feat),
            "final": self.final_head(feat),
            "tone": self.tone_head(feat),
        }


def save_checkpoint(path: Path, model: MultiHeadClassifier, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": model.arch,
            "classes": model.classes.as_dict(),
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> tuple[MultiHeadClassifier, dict]:
    checkpoint = torch.load(path, map_location=map_location)
    classes = ClassSets.from_dict(checkpoint["classes"])
    model = MultiHeadClassifier(checkpoint["arch"], classes)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint.get("metadata", {})
