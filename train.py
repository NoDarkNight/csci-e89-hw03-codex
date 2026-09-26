"""Train and evaluate the Fashion-MNIST multilayer perceptron classifier."""

import json
from pathlib import Path

import torch
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy

from data import test_loader, train_loader, validation_loader
from model import FashionMNISTClassifier
from plot import plot_accuracy


LEARNING_RATE = 0.1
EPOCHS = 20
NUM_CLASSES = 10
OUTPUT_DIR = Path("artifacts")
MODEL_PATH = OUTPUT_DIR / "fashion_mnist_mlp.pt"
HISTORY_PATH = OUTPUT_DIR / "fashion_mnist_history.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_accuracy(
    model: nn.Module, loader: torch.utils.data.DataLoader, metric: MulticlassAccuracy
) -> float:
    """Compute multiclass accuracy for every example in a data loader."""
    model.eval()
    metric.reset()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            metric.update(model(images), labels)

    return metric.compute().item()


def train_model() -> tuple[FashionMNISTClassifier, dict[str, object]]:
    """Train for ``EPOCHS``, save artifacts, and return the model and history."""
    model = FashionMNISTClassifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    accuracy_metric = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)

    history: dict[str, object] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        accuracy_metric.reset()
        total_loss = 0.0
        total_examples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            accuracy_metric.update(logits, labels)

        mean_training_loss = total_loss / total_examples
        training_accuracy = accuracy_metric.compute().item()
        validation_accuracy = evaluate_accuracy(
            model, validation_loader, accuracy_metric
        )

        history["train_loss"].append(mean_training_loss)
        history["train_accuracy"].append(training_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"Epoch {epoch:02d}/{EPOCHS}: "
            f"train loss={mean_training_loss:.4f}, "
            f"train accuracy={training_accuracy:.4f}, "
            f"validation accuracy={validation_accuracy:.4f}"
        )

    test_accuracy = evaluate_accuracy(model, test_loader, accuracy_metric)
    history["test_accuracy"] = test_accuracy
    print(f"Test accuracy={test_accuracy:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with HISTORY_PATH.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)
    plot_accuracy(HISTORY_PATH)

    return model, history


if __name__ == "__main__":
    model, history = train_model()
