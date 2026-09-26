"""Plot Fashion-MNIST accuracy history."""

from pathlib import Path

import matplotlib.pyplot as plt


ARTIFACT_DIR = Path("artifacts")


def plot_history(history, output_path=ARTIFACT_DIR / "fashion_mnist_accuracy.png"):
    """Save and display training and validation accuracy on one chart."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_accuracy"]) + 1)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history["train_accuracy"], label="Training accuracy")
    axis.plot(epochs, history["validation_accuracy"], label="Validation accuracy")
    axis.set(xlabel="Epoch", ylabel="Accuracy", title="Fashion-MNIST MLP Accuracy")
    axis.set_xticks(list(epochs))
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.show()  # Displays inline when this is run in Jupyter.
    return figure


# When this source is pasted after train.py, use the history that train.py made.
if "history" in globals():
    plot_history(history)

