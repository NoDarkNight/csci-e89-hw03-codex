"""Tune Fashion-MNIST MLP hyperparameters with Optuna.

Run after ``train.py`` so the baseline test accuracy is available in its saved
history. The module can also be pasted into Jupyter Notebook cells.
"""

import json
from pathlib import Path

import optuna
import torch
from torch import nn, optim

from data import test_loader, train_loader, validation_loader
from model import FashionMNISTClassifier


OUTPUT_DIR = Path("artifacts")
BASELINE_HISTORY_PATH = OUTPUT_DIR / "fashion_mnist_history.json"
TUNING_RESULTS_PATH = OUTPUT_DIR / "fashion_mnist_tuning_results.json"
NUM_TRIALS = 10
TUNING_EPOCHS = 5
RETRAINING_EPOCHS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
    """Return the fraction of correctly predicted examples in ``loader``."""
    model.eval()
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            labels = labels.to(device)
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return correct_predictions / total_examples


def train_for_one_epoch(
    model: nn.Module, optimizer: optim.Optimizer, loss_function: nn.Module
) -> None:
    """Train ``model`` for one epoch using the Fashion-MNIST training split."""
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(images), labels)
        loss.backward()
        optimizer.step()


def build_model(parameters: dict[str, float | int]) -> FashionMNISTClassifier:
    """Build a classifier from Optuna's sampled hidden-layer parameters."""
    return FashionMNISTClassifier(
        hidden_layer_one=int(parameters["hidden_layer_one"]),
        hidden_layer_two=int(parameters["hidden_layer_two"]),
    ).to(device)


def objective(trial: optuna.Trial) -> float:
    """Train one sampled configuration briefly and return validation accuracy."""
    parameters = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "hidden_layer_one": trial.suggest_int("hidden_layer_one", 64, 512, step=64),
        "hidden_layer_two": trial.suggest_int("hidden_layer_two", 32, 256, step=32),
        "momentum": trial.suggest_float("momentum", 0.0, 0.95),
    }
    model = build_model(parameters)
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        momentum=float(parameters["momentum"]),
    )
    loss_function = nn.CrossEntropyLoss()

    for _ in range(TUNING_EPOCHS):
        train_for_one_epoch(model, optimizer, loss_function)

    return evaluate_accuracy(model, validation_loader)


def print_trial_table(study: optuna.Study) -> None:
    """Print each completed trial and its sampled hyperparameters."""
    print("Trial | Validation accuracy | Learning rate | Hidden 1 | Hidden 2 | Momentum")
    print("-" * 76)
    for trial in study.trials:
        if trial.value is None:
            continue
        parameters = trial.params
        print(
            f"{trial.number:5d} | {trial.value:19.4f} | "
            f"{parameters['learning_rate']:.6f} | "
            f"{parameters['hidden_layer_one']:8d} | "
            f"{parameters['hidden_layer_two']:8d} | {parameters['momentum']:.3f}"
        )


def load_baseline_accuracy(history_path: Path = BASELINE_HISTORY_PATH) -> float:
    """Read the baseline test accuracy produced by ``train.py``."""
    with history_path.open(encoding="utf-8") as history_file:
        return float(json.load(history_file)["test_accuracy"])


def retrain_best_configuration(parameters: dict[str, float | int]) -> float:
    """Train the best sampled configuration for 20 epochs and test it."""
    model = build_model(parameters)
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        momentum=float(parameters["momentum"]),
    )
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, RETRAINING_EPOCHS + 1):
        train_for_one_epoch(model, optimizer, loss_function)
        validation_accuracy = evaluate_accuracy(model, validation_loader)
        print(f"Retraining epoch {epoch:02d}/{RETRAINING_EPOCHS}: validation accuracy={validation_accuracy:.4f}")

    return evaluate_accuracy(model, test_loader)


def run_tuning() -> None:
    """Tune, summarize, retrain, and compare the best configuration."""
    baseline_accuracy = load_baseline_accuracy()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=NUM_TRIALS)

    print_trial_table(study)
    print(f"Best parameters: {study.best_params}")
    best_test_accuracy = retrain_best_configuration(study.best_params)
    print(f"Baseline test accuracy: {baseline_accuracy:.4f}")
    print(f"Tuned test accuracy: {best_test_accuracy:.4f}")
    print(f"Test accuracy difference: {best_test_accuracy - baseline_accuracy:+.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "best_parameters": study.best_params,
        "best_validation_accuracy": study.best_value,
        "baseline_test_accuracy": baseline_accuracy,
        "tuned_test_accuracy": best_test_accuracy,
        "trials": [
            {"number": trial.number, "parameters": trial.params, "value": trial.value}
            for trial in study.trials
        ],
    }
    with TUNING_RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)


if __name__ == "__main__":
    run_tuning()
