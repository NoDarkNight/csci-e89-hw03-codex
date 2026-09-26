# Problem 5 session summary

This session created a complete Fashion-MNIST MLP workflow in `prob5/`. The workflow downloads and splits the data, defines the model, trains and evaluates a baseline, plots its accuracy, reports predictions, and tunes hyperparameters. It also includes a standalone, unexecuted notebook that duplicates the workflow rather than importing the Python scripts.

| Step | Script or file | What was built | Commit | Run order |
| --- | --- | --- | --- | --- |
| 1 | `.gitignore`, `data.py` | Dataset exclusions; deterministic normalized data splits and loaders. | `9da46c7` | 1 |
| 2 | `model.py` | Configurable two-hidden-layer MLP. | `f1f1ac8` | 2 |
| 3 | `plot.py` | Saved and inline training/validation accuracy plot. | `02ed810` | Called by 4 |
| 4 | `train.py` | 20-epoch baseline training, evaluation, weights, history, and plot. | `17be615` | 3 |
| 5 | `predict.py` | Detailed predictions for three validation images. | `a3765dc` | 4 |
| 6 | `tune.py` | Seeded 10-trial Optuna search and final baseline comparison. | `6f830b9` | 5 |
| 7 | `PROMPTS.md` | Verbatim task prompt. | This documentation commit | N/A |
| 8 | `SUMMARY.md` | This implementation and execution guide. | This documentation commit | N/A |
| 9 | `e89_Li_Ethan_HW03_Prob5.ipynb` | Self-contained notebook version. | This documentation commit | Alternative |

## Execution note

The code was **not executed in this environment**, per the request. The only checks performed were non-execution repository checks (such as `git diff --check`). Run the commands from the repository root in the order shown below:

```bash
cd prob5
python train.py
python predict.py
python tune.py
```

For a notebook workflow, open `prob5/e89_Li_Ethan_HW03_Prob5.ipynb` and select **Kernel → Restart Kernel and Run All Cells**.

