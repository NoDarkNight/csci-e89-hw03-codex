"""Download, normalize, split, and load the Fashion-MNIST dataset.

Run this file (or paste its sections into Jupyter Notebook cells) before
training a model.  It exposes ``train_loader``, ``validation_loader``, and
``test_loader`` for the three dataset splits.
"""

import random

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 32
DATA_DIR = "data"
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)


def set_seed(seed: int = SEED) -> None:
    """Seed random number generators so the validation split is repeatable."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()

# ToTensor converts grayscale image values to float32 tensors in [0, 1], and
# Normalize applies the Fashion-MNIST training-set statistics channel-wise.
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
    ]
)

full_training_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)
test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)

# Use a separately seeded generator so the 55,000/5,000 split is stable even
# if random values are consumed elsewhere before this cell is run.
split_generator = torch.Generator().manual_seed(SEED)
train_dataset, validation_dataset = random_split(
    full_training_dataset,
    [55_000, 5_000],
    generator=split_generator,
)

loader_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=loader_generator,
    num_workers=0,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)
