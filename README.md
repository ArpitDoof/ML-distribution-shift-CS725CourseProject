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



## SVM and Decision Tree experiments

The project also evaluates **Support Vector Machines (SVMs)** and **Decision Trees** to study how margin-based and tree-based models behave under distribution shifts.

The SVM implementation uses `SVC` with configurable kernels, including Linear and RBF kernels. Numerical sensor features are standardized using `StandardScaler`, while the categorical `Type` feature is one-hot encoded using `OneHotEncoder`. Since `SVC` does not directly provide probability estimates by default, `CalibratedClassifierCV` with sigmoid calibration and 5-fold cross-validation is used to obtain calibrated class probabilities.

The Decision Tree implementation uses `DecisionTreeClassifier` with configurable maximum depth and minimum samples per leaf. Class weighting is set to `balanced` to account for the imbalance between machine failure and non-failure examples. Numerical features are not standardized because Decision Trees do not require feature scaling. The categorical `Type` feature is still one-hot encoded.

Both models are implemented as scikit-learn pipelines, combining preprocessing and classification into a single workflow.

```python
from src.data.data import get
from src.models.svmdecisiontree import build_svm, build_decision_tree

X_train, X_test, y_train, y_test = get(
    type="iid",
    random_state=42
)

# SVM
svm = build_svm(kernel="rbf")
svm.fit(X_train, y_train)
svm_prob = svm.predict_proba(X_test)[:, 1]

# Decision Tree
tree = build_decision_tree(max_depth=5)
tree.fit(X_train, y_train)
tree_prob = tree.predict_proba(X_test)[:, 1]
```

The `build_svm()` function supports configurable kernel, regularization parameter (`C`), and kernel coefficient (`gamma`). The `build_decision_tree()` function supports configurable maximum depth and minimum samples per leaf.

The experiments evaluate the following configurations:

- **Linear SVM**
- **RBF SVM**
- **Decision Tree (maximum depth = 3)**
- **Decision Tree (maximum depth = 5)**
- **Decision Tree (maximum depth = 10)**

Predicted positive-class probabilities are extracted using `predict_proba()`. These probabilities can subsequently be used to calculate classification metrics, generate ROC curves, and evaluate model performance under different distribution shifts.

Run the SVM and Decision Tree experiments with:

```bash
uv run python run_svmDecisionTree.py
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


## Cross-family comparison

`run_cross_family.py` trains and benchmarks all 13 models across all three paradigms—Generative vs. Discriminative (`GaussianNBModel`, `MixedNBModel`, `LogisticRegressionModel`, `ShiftAwareLogisticRegression`), Parametric vs. Non-Parametric (`SVM_Linear`, `SVM_RBF`, `DecisionTree_Depth3`, `DecisionTree_Depth5`, `DecisionTree_Depth10`), and Neural Networks (`DeepNN`, `WideNN`, `ShiftRobustNN`, `ShiftAwareNN`)—on identical train/validation splits (`RANDOM_STATE=42`), all seven regimes (`iid` + 6 shifts), exact Bayes label-shift prior re-weighting, and all four Subbaswamy et al. (2021) stability partitions. For balanced Decision Trees (`class_weight="balanced"`), it applies exact Bayes prior restoration to recover source-posterior probabilities for fair ECE and BCE comparison while recording raw balanced ECE alongside.

Outputs are saved under `metrics/cross_family/` (`cross_family_benchmark.json`, `cross_family_stability.json`) and `plots/cross_family/` (`heatmap_roc_auc.svg`, `heatmap_f1_src_th.svg`, `heatmap_ece.svg`, per-shift ROC/calibration overlays, and cross-family stability curve overlays).

Run the full cross-family comparison with:

```bash
uv run python run_cross_family.py
```
(On macOS without CUDA `cu130` wheels: `.venv/bin/python run_cross_family.py`.)

## Advanced stability evaluation (Subbaswamy et al., 2021)

`src/stability.py` implements the model-agnostic $\sqrt{N}$-consistent Debiased Machine Learning (DML) **Worst-Case Sampler** (Algorithms 1 & 2 from *Evaluating Model Robustness and Stability to Dataset Shift*, AISTATS 2021). Given any held-out test set and model probabilities, it partitions variables into mutable ($W$) and immutable ($Z$) sets, fits conditional expected loss $\hat{\mu}_k(W, Z)$ via RBF `KernelRidge` and conditional $\alpha$-quantiles $\hat{\eta}_k(Z)$ via discrete strata or B-spline `QuantileRegressor`, and estimates worst-case $(1 - \alpha)$ risk with 95% confidence intervals and subpopulation diagnostics.

```python
from src.stability import compute_loss, estimate_worst_case_risk

losses = compute_loss(y_test, probabilities, loss="bce")
result, masks = estimate_worst_case_risk(
    df_eval,
    losses,
    mutable_cols=["Tool wear [min]", "Torque [Nm]", "Rotational speed [rpm]"],
    immutable_cols=["Type", "Machine failure"],
    stratify_col="Machine failure",
)
```

`run_stability.py` runs both a synthetic Figure 2 sanity-check validation ($X_1, X_2 \sim \mathcal{N}(0, 1)$, $Y = \mathbb{I}[X_1 > \sin(2X_2)]$) and four targeted AI4I $(W, Z)$ partitions (`marginal_all_features`, `cond_mechanical_given_type_y`, `cond_thermal_given_type_y`, and discrete-$W$ `cond_product_given_sensors`) across BCE, Brier, and 0–1 loss. JSON outputs are saved under `metrics/stability/` and vector SVG stability curves and subpopulation profiles under `plots/stability/`.

Run the stability analysis with:

```bash
uv run python run_stability.py
```
(On macOS without CUDA `cu130` wheels: `.venv/bin/python run_stability.py`.)