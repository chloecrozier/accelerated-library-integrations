"""Compare CPU scikit-learn and GPU cuML on the wine quality dataset.

The original CSV is small, so this script repeats the train and test splits with
--scale to make the timing difference easier to see on a GPU instance. The
split happens before scaling so duplicate rows do not leak across train/test.
"""

import argparse
import time
from pathlib import Path


try:
    import cudf
    import pandas as pd
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForest
    from cuml.metrics import accuracy_score as cuml_accuracy_score
    from cuml.model_selection import train_test_split as cuml_train_test_split
    from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
    from sklearn.metrics import accuracy_score as sklearn_accuracy_score
    from sklearn.model_selection import train_test_split as sklearn_train_test_split
except ImportError as exc:
    raise SystemExit(
        "This benchmark needs pandas, scikit-learn, RAPIDS cuDF, and RAPIDS cuML "
        "installed in the active environment."
    ) from exc


DATASET = Path(__file__).resolve().parents[1] / "dataset" / "winequality-white.csv"


def to_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        type=int,
        default=25,
        help="Repeat the wine dataset this many times before benchmarking.",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=100,
        help="Number of Random Forest trees to train.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=16,
        help="Maximum tree depth for both Random Forest models.",
    )
    return parser


def run_sklearn(scale, trees, max_depth):
    base_df = pd.read_csv(DATASET, sep=";")
    X_base = base_df.drop(columns=["quality"]).astype("float32")
    y_base = base_df["quality"].astype("int32")

    X_train_base, X_test_base, y_train_base, y_test_base = sklearn_train_test_split(
        X_base, y_base, test_size=0.2, random_state=42
    )
    X_train = pd.concat([X_train_base] * scale, ignore_index=True)
    X_test = pd.concat([X_test_base] * scale, ignore_index=True)
    y_train = pd.concat([y_train_base] * scale, ignore_index=True)
    y_test = pd.concat([y_test_base] * scale, ignore_index=True)

    model = SklearnRandomForest(
        n_estimators=trees,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    elapsed = time.perf_counter() - start

    return len(X_train) + len(X_test), elapsed, sklearn_accuracy_score(y_test, predictions)


def run_cuml(scale, trees, max_depth):
    base_df = cudf.read_csv(str(DATASET), sep=";")
    X_base = base_df.drop(columns=["quality"]).astype("float32")
    y_base = base_df["quality"].astype("int32")

    X_train_base, X_test_base, y_train_base, y_test_base = cuml_train_test_split(
        X_base, y_base, test_size=0.2, random_state=42
    )
    X_train = cudf.concat([X_train_base] * scale, ignore_index=True)
    X_test = cudf.concat([X_test_base] * scale, ignore_index=True)
    y_train = cudf.concat([y_train_base] * scale, ignore_index=True)
    y_test = cudf.concat([y_test_base] * scale, ignore_index=True)

    model = CuMLRandomForest(
        n_estimators=trees,
        max_depth=max_depth,
        random_state=42,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    elapsed = time.perf_counter() - start

    return len(X_train) + len(X_test), elapsed, to_float(cuml_accuracy_score(y_test, predictions))


def main():
    args = make_parser().parse_args()

    rows, sklearn_seconds, sklearn_accuracy = run_sklearn(
        args.scale, args.trees, args.max_depth
    )
    _, cuml_seconds, cuml_accuracy = run_cuml(args.scale, args.trees, args.max_depth)

    print(f"Rows: {rows:,}")
    print(f"Trees: {args.trees}")
    print(f"Max depth: {args.max_depth}")
    print()
    print("Timing: train + predict")
    print(f"scikit-learn CPU: {sklearn_seconds:.3f}s, accuracy {sklearn_accuracy:.3f}")
    print(f"cuML GPU:         {cuml_seconds:.3f}s, accuracy {cuml_accuracy:.3f}")
    print()
    print(f"Speedup: {sklearn_seconds / cuml_seconds:.2f}x")


if __name__ == "__main__":
    main()
