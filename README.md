#ML-distribution-shift-CS725CourseProject
## Python environment

Install the project environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python your_script.py
```

For notebooks, select the project's `.venv` Python interpreter as the kernel.

## Data shifts

`DataLoader` downloads the AI4I dataset to `data/ai4i2020.csv` if it is not already present. Use the top-level `get()` function to return `X_train, X_test, y_train, y_test`.

Available splits:

- `get()` or `get(type="iid")`: random train/test split.
- `get(type="temp", temp_delta=2.0)`: adds the same temperature offset in kelvin to both temperature features in the test set. Labels stay unchanged.
- `get(type="wear")`: resamples the test set to favor higher tool-wear rows.
- `get(type="product")`: resamples the test set for an approximately balanced L/M/H product mix.
- `get(type="joint")`: resamples the test set to favor higher torque-speed combinations.
- `get(type="label", failure_rate=0.2)`: resamples the test set to target the requested machine-failure rate.
- `get(type="corruption", feature="Torque [Nm]", corruption_pct=20)`: corrupts the named feature in the requested percentage of test rows. Numeric values are sampled from the training range; categorical values are replaced by another observed category.

Resampled shifts keep each row with its original label and sample with replacement. Feature corruption leaves labels unchanged. `random_state` controls the split and random sampling.

```python
from src.data.data import get

X_train, X_test, y_train, y_test = get(
    type="corruption", feature="Torque [Nm]", corruption_pct=20, random_state=42
)
```

## Neural network experiments

`DeepNN` has more hidden layers, while `WideNN` has more neurons in one hidden layer. `ShiftRobustNN` keeps input/hidden dropout and LayerNorm. `ShiftAwareNN` adds a temperature difference and torque interactions, uses product-specific output heads, and uses dropout. These shift-aware choices are hypotheses, not guaranteed improvements.

```python
from src.data.data import get
from src.models.nn import DeepNN, ShiftAwareNN, ShiftRobustNN, Trainer, WideNN

X_train, X_test, y_train, y_test = get(type="iid")
trainer = Trainer(ShiftRobustNN, epochs=50, batch_size=64, learning_rate=0.001)
trainer.fit(X_train, y_train)
scores = trainer.evaluate(X_test, y_test, shift_name="iid")
```

For a label shift with a known target failure rate, pass it to `evaluate()` to adjust the output prior, for example `trainer.evaluate(X_test, y_test, shift_name="label", target_failure_rate=0.2)`.

Evaluation saves accuracy, positive-class F1/recall/precision, ROC-AUC, ECE, a confusion matrix, and per-class precision/recall/F1/support to JSON. `run_nn.py` saves both a source-validation `*_th_change_<threshold>.json` and a `*_shift_adapted_th_<threshold>.json` per shift. The shift-adapted threshold uses one half of each shifted set for calibration and the other half for evaluation. Individual and combined ROC/calibration plots are saved under `plots/nn/`.

Run all six neural-network shift experiments on CUDA device 2 with:

```bash
uv run python run_nn.py
```

## Classical model experiments (Naive Bayes vs. Logistic Regression)

`GaussianNBModel` is a generative baseline treating standardized numeric and one-hot encoded `Type` features as Gaussian, while `MixedNBModel` separates continuous sensor features (`GaussianNB`) from the categorical `Type` feature (`CategoricalNB`) without double-counting the class prior. `LogisticRegressionModel` is the discriminative linear baseline, and `ShiftAwareLogisticRegression` adds the same temperature difference (`Process - Air`) and torque interactions (`Torque x wear`, `Torque x speed`) used by `ShiftAwareNN`.

```python
from src.data.data import get
from src.models.classical import (
    ClassicalTrainer,
    GaussianNBModel,
    LogisticRegressionModel,
    MixedNBModel,
    ShiftAwareLogisticRegression,
)

X_train, X_test, y_train, y_test = get(type="iid")
trainer = ClassicalTrainer(MixedNBModel, random_state=42)
trainer.fit(X_train, y_train)
scores = trainer.evaluate(X_test, y_test, shift_name="iid")
```

`ClassicalTrainer` mirrors the neural-network `Trainer` API and supports three shift mitigations:

- **Prior re-weighting (label shift)**: pass `target_failure_rate` to `evaluate()` or `predict_proba()`, e.g. `trainer.evaluate(X_test, y_test, shift_name="label", target_failure_rate=0.2)`.
- **Temperature scaling (calibration)**: fit a scalar temperature on validation logits via `T = trainer.fit_temperature(X_val, y_val)` and pass `temperature=T` to `evaluate()` or `predict_proba()`.
- **Importance weighting (covariate shift)**: estimate unlabeled density ratios with `weights = estimate_importance_weights(X_train, X_test)` and pass `sample_weight=weights` to `trainer.fit(X_train, y_train, sample_weight=weights)`.

`run_classical.py` evaluates all four models across IID and all six shifts using the same train/validation split and group-deduplicated shift-adaptation split as `run_nn.py`, saving metrics JSONs under `metrics/classical/` (plus `metrics/classical/nb_vs_lr_summary.json`) and individual, per-model, and head-to-head ROC/calibration SVG plots under `plots/classical/`.

Run all classical Naive Bayes vs. Logistic Regression shift experiments with:

```bash
uv run python run_classical.py
```
(On macOS without CUDA `cu130` wheels, run with the local virtual environment: `.venv/bin/python run_classical.py`.)
