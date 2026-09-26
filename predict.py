"""Print Fashion-MNIST predictions for the first validation images.

Run this file after ``train.py`` has saved model weights, or paste its
sections into Jupyter Notebook cells.
"""

from pathlib import Path

import torch

from data import validation_dataset
from model import FashionMNISTClassifier


MODEL_PATH = Path("artifacts") / "fashion_mnist_mlp.pt"
CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
NUM_PREDICTIONS = 3
TOP_K = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: Path = MODEL_PATH) -> FashionMNISTClassifier:
    """Load the trained Fashion-MNIST classifier in evaluation mode."""
    model = FashionMNISTClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_validation_images(
    model_path: Path = MODEL_PATH, num_predictions: int = NUM_PREDICTIONS
) -> None:
    """Print predictions, probabilities, and top classes for validation images."""
    if num_predictions > len(validation_dataset):
        raise ValueError("num_predictions exceeds the validation dataset size.")

    model = load_model(model_path)
    images, labels = zip(*(validation_dataset[index] for index in range(num_predictions)))
    image_batch = torch.stack(images).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(image_batch), dim=1).cpu()

    for index, (label, image_probabilities) in enumerate(zip(labels, probabilities), start=1):
        top_probabilities, top_indices = torch.topk(image_probabilities, TOP_K)
        class_probabilities = {
            class_name: round(probability, 3)
            for class_name, probability in zip(CLASS_NAMES, image_probabilities.tolist())
        }

        print(f"Image {index}")
        print(f"Predicted class: {CLASS_NAMES[image_probabilities.argmax().item()]}")
        print(f"True class: {CLASS_NAMES[label]}")
        print(f"Softmax probabilities: {class_probabilities}")
        print("Top 4 classes:")
        for class_index, probability in zip(top_indices.tolist(), top_probabilities.tolist()):
            print(f"  {CLASS_NAMES[class_index]}: {probability:.3f}")
        print()


if __name__ == "__main__":
    predict_validation_images()
