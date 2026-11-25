from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier


NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

CATEGORICAL_FEATURES = ["Type"]


def build_preprocessor(scale_numeric=True):

    numeric_steps = []

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(numeric_steps) if numeric_steps else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


def build_svm(kernel="rbf", C=1.0, gamma="scale"):

    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=42
    )

    calibrated_svm = CalibratedClassifierCV(
        estimator=svm,
        method="sigmoid",
        cv=5,
        ensemble=False
    )

    return Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=True)),
        ("classifier", calibrated_svm)
    ])

def build_decision_tree(max_depth=None, min_samples_leaf=1):

    return Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=False)),
        ("classifier", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=42
        ))
    ])

