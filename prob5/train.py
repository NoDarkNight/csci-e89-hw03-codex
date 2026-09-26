"""Train and evaluate the baseline Fashion-MNIST MLP."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torchmetrics.classification import MulticlassAccuracy

if "train_loader" not in globals():
    from data import test_loader, train_loader, validation_loader
if "FashionMNISTMLP" not in globals():
    from model import FashionMNISTMLP


EPOCHS = 20
LEARNING_RATE = 0.1
ARTIFACT_DIR = Path("artifacts")
WEIGHTS_PATH = ARTIFACT_DIR / "fashion_mnist_mlp.pt"
HISTORY_PATH = ARTIFACT_DIR / "fashion_mnist_history.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_for_loader(model, loader, metric, device):
    """Compute micro multiclass accuracy without retaining gradients."""
    model.eval()
    metric.reset()
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device))
            metric.update(predictions, labels.to(device))
    return metric.compute().item()


def train_model(model, train_loader, validation_loader, epochs, learning_rate, device,
                momentum=0.0, print_progress=True):
    """Train a model and return epoch-level loss and accuracy history."""
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    train_metric = MulticlassAccuracy(num_classes=10, average="micro").to(device)
    validation_metric = MulticlassAccuracy(num_classes=10, average="micro").to(device)
    result = {"train_loss": [], "train_accuracy": [], "validation_accuracy": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_metric.reset()
        loss_total = 0.0
        sample_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            loss_total += loss.item() * batch_size
            sample_count += batch_size
            train_metric.update(logits, labels)

        mean_loss = loss_total / sample_count
        train_accuracy = train_metric.compute().item()
        validation_accuracy = accuracy_for_loader(
            model, validation_loader, validation_metric, device
        )
        result["train_loss"].append(mean_loss)
        result["train_accuracy"].append(train_accuracy)
        result["validation_accuracy"].append(validation_accuracy)
        if print_progress:
            print(
                f"Epoch {epoch:02d}/{epochs}: loss={mean_loss:.4f}, "
                f"train_accuracy={train_accuracy:.4f}, "
                f"validation_accuracy={validation_accuracy:.4f}"
            )
    return result


print(f"Training on {device}.")
model = FashionMNISTMLP().to(device)
history = train_model(
    model, train_loader, validation_loader, EPOCHS, LEARNING_RATE, device
)
test_metric = MulticlassAccuracy(num_classes=10, average="micro").to(device)
test_accuracy = accuracy_for_loader(model, test_loader, test_metric, device)
history["test_accuracy"] = test_accuracy
print(f"Test accuracy: {test_accuracy:.4f}")

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), WEIGHTS_PATH)
with HISTORY_PATH.open("w", encoding="utf-8") as history_file:
    json.dump(history, history_file, indent=2)

# Direct script execution imports plot.py.  In pasted cells, plot.py can run next
# and will find ``history`` in the shared notebook namespace.
if "plot_history" in globals():
    plot_history(history)
elif "__file__" in globals():
    from plot import plot_history
    plot_history(history)
