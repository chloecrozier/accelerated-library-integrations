"""Compare CPU scikit-learn and GPU cuML on the wine quality dataset.

The original CSV is small, so this script repeats the rows with --scale to make
the timing difference easier to see on a GPU instance.
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
    df = pd.concat([base_df] * scale, ignore_index=True)

    X = df.drop(columns=["quality"]).astype("float32")
    y = df["quality"].astype("int32")

    X_train, X_test, y_train, y_test = sklearn_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    return len(df), elapsed, sklearn_accuracy_score(y_test, predictions)


def run_cuml(scale, trees, max_depth):
    base_df = cudf.read_csv(str(DATASET), sep=";")
    df = cudf.concat([base_df] * scale, ignore_index=True)

    X = df.drop(columns=["quality"]).astype("float32")
    y = df["quality"].astype("int32")

    X_train, X_test, y_train, y_test = cuml_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CuMLRandomForest(
        n_estimators=trees,
        max_depth=max_depth,
        random_state=42,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    elapsed = time.perf_counter() - start

    return len(df), elapsed, to_float(cuml_accuracy_score(y_test, predictions))


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
