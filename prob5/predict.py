"""Print detailed predictions for the first three validation images."""

from pathlib import Path

import torch

if "validation_loader" not in globals():
    from data import class_names, validation_loader
if "FashionMNISTMLP" not in globals():
    from model import FashionMNISTMLP


WEIGHTS_PATH = Path("artifacts") / "fashion_mnist_mlp.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reuse train.py's in-memory model in a notebook; otherwise load saved weights.
if "model" not in globals():
    model = FashionMNISTMLP().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))

model.eval()
images, labels = next(iter(validation_loader))
with torch.no_grad():
    probabilities = torch.softmax(model(images[:3].to(device)), dim=1).cpu()

for index, (probability, true_label) in enumerate(zip(probabilities, labels[:3]), start=1):
    predicted_label = probability.argmax().item()
    top_probabilities, top_labels = probability.topk(4)
    formatted_probabilities = ", ".join(
        f"{name}={score:.3f}" for name, score in zip(class_names, probability.tolist())
    )
    print(f"Image {index}: predicted={class_names[predicted_label]}, true={class_names[true_label.item()]}")
    print(f"  Probabilities: {formatted_probabilities}")
    print("  Top 4:")
    for rank, (score, label) in enumerate(zip(top_probabilities, top_labels), start=1):
        print(f"    {rank}. {class_names[label.item()]} ({score.item():.3f})")

