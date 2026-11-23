#ML-distribution-shift-CS725CourseProject
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

`SimpleNN` is the baseline. `ShiftRobustNN` adds input/hidden dropout and LayerNorm to encourage reliance on multiple features and avoid batch-running statistics. This is a hypothesis for improved shift robustness, especially under feature corruption; it is not guaranteed to win on every shift.

```python
from src.data.data import get
from src.models.nn import ShiftRobustNN, Trainer

X_train, X_test, y_train, y_test = get(type="iid")
trainer = Trainer(ShiftRobustNN, epochs=50, batch_size=64, learning_rate=0.001)
trainer.fit(X_train, y_train)
scores = trainer.evaluate(X_test, y_test, shift_name="iid")
```

Evaluation saves `accuracy`, `f1_score`, `recall`, `precision`, `roc_auc`, and `expected_calibration_error` to `results/<shift_name>/<ModelClass>.json`.
