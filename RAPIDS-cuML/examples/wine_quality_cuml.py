"""
Simple cuML wine quality demo.

Goal: predict the wine's quality score from lab measurements like acidity,
sugar, pH, and alcohol.

Run this on a Brev GPU instance or DGX Spark with RAPIDS installed.
"""

from pathlib import Path


try:
    import cudf
    from cuml.ensemble import RandomForestClassifier
    from cuml.metrics import accuracy_score
    from cuml.model_selection import train_test_split
except ImportError as exc:
    raise SystemExit(
        "This example needs RAPIDS cuDF/cuML on a CUDA-capable Linux or WSL2 "
        "machine. Run it on Brev or DGX Spark, not local macOS."
    ) from exc


DATASET = Path(__file__).resolve().parents[1] / "dataset" / "winequality-white.csv"


def to_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def main():
    # 1. Load the CSV into a GPU DataFrame.
    df = cudf.read_csv(str(DATASET), sep=";")

    print("First 3 rows:")
    print(df.head(3))
    print()

    # 2. Split the table into inputs (X) and known answers (y).
    X = df.drop(columns=["quality"]).astype("float32")
    y = df["quality"].astype("int32")

    # 3. Keep 20% of the rows hidden for testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train one GPU model. Random Forest is a strong beginner-friendly
    #    classifier because it combines many simple decision trees.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Ask the model to predict the hidden test rows and score the result.
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {to_float(score):.3f}")
    print()

    print("One example prediction:")
    print("Input measurements:")
    print(X_test.head(1))
    print()
    print("Predicted quality:")
    print(model.predict(X_test.head(1)))
    print()
    print("Actual quality from the dataset:")
    print(y_test.head(1))


if __name__ == "__main__":
    main()
