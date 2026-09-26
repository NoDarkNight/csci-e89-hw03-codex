"""Define the multilayer perceptron used for Fashion-MNIST classification."""

import torch
from torch import nn


class FashionMNISTClassifier(nn.Module):
    """Classify 28-by-28 Fashion-MNIST images into one of 10 classes."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized logit per Fashion-MNIST class."""
        return self.network(images)
