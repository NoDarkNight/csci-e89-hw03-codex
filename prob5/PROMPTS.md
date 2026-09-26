Build a PyTorch image classifier for Fashion MNIST in a new folder prob5/ (ignore all other files in this repo). Do not run code or install packages; I will run everything locally. Just write the files carefully so they run the first time, and commit each script separately.

Scripts must also work when pasted into consecutive Jupyter cells (no argparse, plots shown inline, later cells reuse names from earlier cells if the .py files aren't importable). Add a .gitignore for the dataset and __pycache__/.

1. data.py: download Fashion MNIST with torchvision, convert to normalized float32 tensors, seed 42, split the 60,000 training images into 55,000 train / 5,000 validation, DataLoaders with batch size 32 (only train shuffled).
2. model.py: MLP nn.Module: flatten 28x28, hidden layers of 300 and 100 with ReLU (configurable sizes), 10 output logits.
3. train.py: SGD (lr 0.1), cross-entropy, 20 epochs, torchmetrics multiclass accuracy (plain accuracy, average="micro"). Record and print mean train loss, train accuracy, and validation accuracy per epoch. Use a GPU if available. Then evaluate test accuracy, save the weights, and save the history (with test accuracy) as JSON.
4. plot.py: plot training and validation accuracy per epoch on one chart; save as PNG and show inline in Jupyter. train.py calls it after training.
5. predict.py: for the first 3 validation images, print predicted vs. true class names, softmax probabilities for all 10 classes (3 decimals), and the top 4 classes.
6. tune.py: seeded Optuna search over the learning rate (log scale 1e-5 to 1e-1), both hidden layer sizes, and SGD momentum; 10 trials of 5 epochs each. Print a table of trials and the best parameters, retrain the best configuration for 20 epochs, and compare its test accuracy with train.py's baseline.

Then:
7. prob5/PROMPTS.md: this prompt word for word.
8. prob5/SUMMARY.md: a summary of this session: what was built in each step, a table of scripts with their commit and run order, and a note that the code was not executed in this environment.
9. prob5/e89_Li_Ethan_HW03_Prob5.ipynb: one self-contained notebook (no imports from the .py files), unexecuted. First cell is markdown with "Ethan Li", "CSCI E-89 Deep Learning, Assignment 03, Problem 5", and a short description, then this prompt quoted. Before each code cell, a markdown cell naming the source script and explaining it; comments in the code. It must run top to bottom with Restart & Run All.

Finish with the exact local commands to run, in order.

