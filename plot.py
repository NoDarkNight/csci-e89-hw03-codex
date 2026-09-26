"""Plot training and validation accuracy saved by :mod:`train`.

Run this file after training, or paste it into a Jupyter Notebook cell.  The
call to ``plt.show()`` displays the figure inline when the notebook is using
Matplotlib's inline backend.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("artifacts")
HISTORY_PATH = OUTPUT_DIR / "fashion_mnist_history.json"
PLOT_PATH = OUTPUT_DIR / "fashion_mnist_accuracy.png"


def plot_accuracy(
    history_path: Path = HISTORY_PATH, output_path: Path = PLOT_PATH
) -> None:
    """Save and display training and validation accuracy by epoch."""
    with history_path.open(encoding="utf-8") as history_file:
        history = json.load(history_file)

    training_accuracy = history["train_accuracy"]
    validation_accuracy = history["validation_accuracy"]
    if len(training_accuracy) != len(validation_accuracy):
        raise ValueError("Training and validation accuracy must have equal lengths.")
    if not training_accuracy:
        raise ValueError("History does not contain any epoch accuracy values.")

    epochs = range(1, len(training_accuracy) + 1)
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(epochs, training_accuracy, marker="o", label="Training accuracy")
    axes.plot(epochs, validation_accuracy, marker="o", label="Validation accuracy")
    axes.set_title("Fashion-MNIST Accuracy by Epoch")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Accuracy")
    axes.set_xticks(list(epochs))
    axes.set_ylim(0, 1)
    axes.grid(visible=True, alpha=0.3)
    axes.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    # In a notebook this cell runs before training (train.py calls
    # plot_accuracy), so only regenerate the chart once a history exists.
    if HISTORY_PATH.exists():
        plot_accuracy()
    else:
        print(f"{HISTORY_PATH} not found; run train.py to create it and plot.")
