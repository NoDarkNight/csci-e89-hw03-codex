"""Configurable multilayer perceptron for Fashion-MNIST."""

import torch.nn as nn


class FashionMNISTMLP(nn.Module):
    """Flatten 28x28 images and classify them with two ReLU hidden layers."""

    def __init__(self, hidden_size1=300, hidden_size2=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 10),
        )

    def forward(self, images):
        return self.network(images)

