# Repository Summary

## Project purpose

This repository implements a complete PyTorch workflow for classifying
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) images with a
multilayer perceptron (MLP). The work is split into small Python scripts so each
stage is easy to understand and reuse. The code is also written so its sections
can be pasted into Jupyter Notebook cells; each script performs its work only
when run directly, except for `data.py`, which prepares the datasets and loaders
when it is imported or executed.

## Prompt-to-code walkthrough

The original requests are retained in full in [`PROMPTS.md`](PROMPTS.md). This
section groups each request with the file it produced, so the outcome of every
prompt is clear without having to cross-reference separate sections.

### 1. Dataset preparation → `data.py`

> “The first step is to write `data.py`. This should handle downloading Fashion
> MNIST with torchvision, converting and normalizing vectors to float32 tensors,
> setting the seed (42), splitting the 60,000 training images into 55,000 for
> training and 5,000 for validation, creating dataloaders with batch size 32,
> and whatever else is necessary.”

`data.py` is the shared starting point for all model-related scripts. It sets the
seed to 42, downloads the train and test portions of Fashion-MNIST into `data/`,
and transforms every grayscale image into a normalized `float32` tensor. The
60,000-image training portion is deterministically split into 55,000 training
examples and 5,000 validation examples. It exports `train_dataset`,
`validation_dataset`, `test_dataset`, and their corresponding `DataLoader`
instances. Training batches are shuffled; validation and test batches retain a
fixed order.

### 2. MLP classifier → `model.py`

> “Write `model.py` with a MLP classifier as an `nn.Module`. Flatten the 28x28
> image, then use two hidden layers of 300 and 100 neurons with ReLU. The output
> layer should be 10 logits (one per class).”

`model.py` defines `FashionMNISTClassifier`. Its sequential network flattens a
single 28×28 image to 784 values, applies linear layers with ReLU activations,
and returns 10 logits—one for each Fashion-MNIST class. The default hidden-layer
widths are 300 and 100, while constructor arguments allow `tune.py` to build
models with sampled widths.

### 3. Baseline training → `train.py`

> “Write `train.py` -- train the model with SGD (learning rate 0.1) and
> cross-entropy loss for 20 epochs, using torchmetrics to compute multiclass
> accuracy. For each epoch, record the mean training loss, training accuracy, and
> validation accuracy in a variable and print them. Use a GPU if one is available,
> otherwise CPU is fine. After training, evaluate accuracy on the test set, save
> the model weights, and save the history (including test accuracy).”

`train.py` imports the loaders and default classifier, chooses CUDA when it is
available (otherwise CPU), and trains for 20 epochs with SGD at a learning rate
of 0.1. During every epoch it calculates the mean per-example training loss and
uses `torchmetrics` multiclass accuracy for training and validation accuracy. It
prints those three values, evaluates the finished model on the test split, saves
the artifacts below, and then calls `plot_accuracy` to generate the chart.

### 4. Accuracy chart → `plot.py` and `train.py`

> “Write `plot.py` to plot the training accuracy per epoch. Put validation
> accuracy on the same chart for comparison. Save it as png, but make it display
> them inline when running in Jupyter. Have `train.py` call the plotting after
> training. Regenerate the plots from history.”

`plot.py` reads the JSON history produced by `train.py`. It validates that the
training and validation series have matching, nonzero lengths, draws both
accuracy series against epoch number, writes a PNG image, displays the figure for
notebook use, and closes it afterwards. `train.py` invokes this function after
writing history, and `plot.py` can be run independently after a training run to
regenerate the chart without retraining.

### 5. Validation predictions → `predict.py`

> “Write `predict.py` to use the trained model: take the first 3 images from the
> validation set, print the predicted and true class names, the softmax
> probabilities for all 10 classes (rounded to 3 decimals), and the top 4 most
> likely classes for each image.”

`predict.py` loads the state dictionary saved by baseline training into the
default model architecture and switches the model to evaluation mode. It selects
the first three examples in the validation split, calculates softmax
probabilities, and prints the predicted and true class names, the complete
10-class probability mapping rounded to three decimal places, and the top four
classes for each image.

### 6. Hyperparameter tuning → `tune.py`

> “Use Optuna to search the learning rate (log scale from about 1e-5 to 1e-1),
> the number of neurons in each hidden layer, and the optimizer momentum. Set it
> up to keep things fast; maybe with 10 trials of 5 epochs each. Set up code for
> printing a table of the trials and the best parameters, then for retraining the
> best configuration for 20 epochs and report its test accuracy compared with the
> baseline from `train.py`.”

`tune.py` first reads the baseline test accuracy from the history JSON. Optuna
then runs 10 trials; each trains a newly sampled MLP for 5 epochs and uses its
validation accuracy as the objective. The search samples a log-scale learning
rate from 1e-5 through 1e-1, first and second hidden-layer widths, and SGD
momentum. After printing the completed trials and winning parameters, it
rebuilds the best model, trains it for 20 epochs, reports validation accuracy at
each retraining epoch, evaluates it on the test set, compares that result with
the baseline, and saves a JSON record of the experiment.

## Full execution order

1. Install the Python dependencies used by the scripts: PyTorch, torchvision,
   torchmetrics, matplotlib, and Optuna.
2. From the repository root, run `python train.py`. This also downloads
   Fashion-MNIST automatically on its first run and creates the baseline model,
   metric history, and accuracy plot.
3. Optionally run `python plot.py` to recreate the baseline plot from the saved
   history without retraining.
4. Run `python predict.py` to inspect predictions from the saved baseline model.
5. Optionally run `python tune.py` to search the configured hyperparameters and
   compare the retrained best configuration with the saved baseline. This step
   requires `train.py` to have completed successfully because it reads the
   baseline history file.

In a Jupyter Notebook, paste and execute the corresponding script sections in
the same dependency order: data preparation, model definition, training, then
plotting, prediction, or tuning as needed.

## Generated files and directories

The scripts create files that are intentionally not source code:

- `data/` contains the downloaded Fashion-MNIST data.
- `artifacts/fashion_mnist_mlp.pt` contains the baseline model state dictionary.
- `artifacts/fashion_mnist_history.json` contains per-epoch baseline training
  loss, training accuracy, validation accuracy, and final test accuracy.
- `artifacts/fashion_mnist_accuracy.png` contains the baseline accuracy chart.
- `artifacts/fashion_mnist_tuning_results.json` contains the best sampled
  parameters, baseline and tuned test accuracies, and all trial results.

The baseline model and history are inputs to later stages: `predict.py` needs the
model state dictionary, and `tune.py` needs the history JSON. Training can be
re-run to overwrite the baseline model, history, and plot with a new run.
