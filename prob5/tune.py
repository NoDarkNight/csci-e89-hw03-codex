"""Tune the Fashion-MNIST MLP with a seeded Optuna study."""

import json
import random
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import nn
from torch.optim import SGD
from torchmetrics.classification import MulticlassAccuracy

if "train_loader" not in globals():
    from data import test_loader, train_loader, validation_loader
if "FashionMNISTMLP" not in globals():
    from model import FashionMNISTMLP


SEED = 42
TRIALS = 10
TUNING_EPOCHS = 5
FINAL_EPOCHS = 20
ARTIFACT_DIR = Path("artifacts")
BASELINE_HISTORY_PATH = ARTIFACT_DIR / "fashion_mnist_history.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_accuracy(model, loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for images, labels in loader:
            metric.update(model(images.to(device)), labels.to(device))
    return metric.compute().item()


def fit(model, epochs, learning_rate, momentum, report=False):
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    train_metric = MulticlassAccuracy(num_classes=10, average="micro").to(device)
    validation_metric = MulticlassAccuracy(num_classes=10, average="micro").to(device)
    history = {"train_loss": [], "train_accuracy": [], "validation_accuracy": []}
    for epoch in range(1, epochs + 1):
        model.train()
        train_metric.reset()
        loss_sum, example_count = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            example_count += labels.size(0)
            train_metric.update(logits, labels)
        history["train_loss"].append(loss_sum / example_count)
        history["train_accuracy"].append(train_metric.compute().item())
        validation_accuracy = loader_accuracy(model, validation_loader, validation_metric)
        history["validation_accuracy"].append(validation_accuracy)
        if report:
            print(f"Epoch {epoch:02d}/{epochs}: validation_accuracy={validation_accuracy:.4f}")
    return history


def objective(trial):
    set_seed(SEED + trial.number)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_size1 = trial.suggest_int("hidden_size1", 100, 500, step=50)
    hidden_size2 = trial.suggest_int("hidden_size2", 50, 300, step=50)
    momentum = trial.suggest_float("momentum", 0.0, 0.9)
    candidate = FashionMNISTMLP(hidden_size1, hidden_size2).to(device)
    trial_history = fit(candidate, TUNING_EPOCHS, learning_rate, momentum)
    return trial_history["validation_accuracy"][-1]


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
)
study.optimize(objective, n_trials=TRIALS)

print("\nTrials")
print("number  validation_accuracy  learning_rate  hidden_size1  hidden_size2  momentum")
for trial in study.trials:
    parameters = trial.params
    print(
        f"{trial.number:>6}  {trial.value:>19.4f}  "
        f"{parameters['learning_rate']:<13.6g}  {parameters['hidden_size1']:>12}  "
        f"{parameters['hidden_size2']:>12}  {parameters['momentum']:.4f}"
    )

best_parameters = study.best_params
print(f"\nBest parameters: {best_parameters}")
set_seed(SEED)
best_model = FashionMNISTMLP(
    best_parameters["hidden_size1"], best_parameters["hidden_size2"]
).to(device)
best_history = fit(
    best_model, FINAL_EPOCHS, best_parameters["learning_rate"],
    best_parameters["momentum"], report=True
)
test_metric = MulticlassAccuracy(num_classes=10, average="micro").to(device)
best_test_accuracy = loader_accuracy(best_model, test_loader, test_metric)

if "history" in globals() and "test_accuracy" in history:
    baseline_test_accuracy = history["test_accuracy"]
else:
    with BASELINE_HISTORY_PATH.open(encoding="utf-8") as baseline_file:
        baseline_test_accuracy = json.load(baseline_file)["test_accuracy"]

print(f"\nBaseline test accuracy: {baseline_test_accuracy:.4f}")
print(f"Tuned test accuracy:    {best_test_accuracy:.4f}")
print(f"Difference (tuned - baseline): {best_test_accuracy - baseline_test_accuracy:+.4f}")

