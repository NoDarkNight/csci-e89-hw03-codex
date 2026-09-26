"""Fashion-MNIST data preparation for Problem 5.

Run this file first when using the scripts as consecutive Jupyter cells.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 32
DATA_DIR = "data"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ToTensor converts the uint8 images to float32 tensors in [0, 1].
# The Fashion-MNIST training-set mean and standard deviation normalize them.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
)

full_train_dataset = datasets.FashionMNIST(
    root=DATA_DIR, train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root=DATA_DIR, train=False, download=True, transform=transform
)

split_generator = torch.Generator().manual_seed(SEED)
train_dataset, validation_dataset = random_split(
    full_train_dataset, [55_000, 5_000], generator=split_generator
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

