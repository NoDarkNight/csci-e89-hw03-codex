"""Define the multilayer perceptron used for Fashion-MNIST classification."""

import torch
from torch import nn


class FashionMNISTClassifier(nn.Module):
    """Classify 28-by-28 Fashion-MNIST images into one of 10 classes."""

    def __init__(self, hidden_layer_one: int = 300, hidden_layer_two: int = 100) -> None:
        """Initialize the classifier with configurable hidden-layer widths."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_layer_one),
            nn.ReLU(),
            nn.Linear(hidden_layer_one, hidden_layer_two),
            nn.ReLU(),
            nn.Linear(hidden_layer_two, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized logit per Fashion-MNIST class."""
        return self.network(images)
