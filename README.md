#ML-distribution-shift-CS725CourseProject
## Data shifts

`DataLoader` downloads the AI4I dataset to `data/ai4i2020.csv` if it is not already present. Use the top-level `get()` function to return `X_train, X_test, y_train, y_test`.

Available splits:

- `get()` or `get(type="iid")`: random train/test split.
- `get(type="temp", temp_delta=2.0)`: adds the same temperature offset in kelvin to both temperature features in the test set. Labels stay unchanged.
- `get(type="wear")`: resamples the test set to favor higher tool-wear rows.
- `get(type="product")`: resamples the test set for an approximately balanced L/M/H product mix.
- `get(type="joint")`: resamples the test set to favor higher torque-speed combinations.

Resampled shifts keep each row with its original label and sample with replacement. `random_state` controls the split and resampling.

```python
from src.data.data import get

X_train, X_test, y_train, y_test = get(type="wear", random_state=42)
```
