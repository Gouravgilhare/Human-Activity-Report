
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "pose_dataset.csv"
MODEL_PATH = ROOT / "models" / "activity_classifier.joblib"
REPORT_PATH = ROOT / "reports" / "metrics.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run src/collect_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"session_id", "activity"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'session_id' and 'activity'. "
            "Use the updated collect_data.py."
        )

    # Keep only the 132 pose features, in their existing order.
    feature_names = [
        col for col in df.columns
        if col not in ("session_id", "activity")
    ]

    if len(feature_names) != 132:
        raise ValueError(
            f"Expected 132 pose features, found {len(feature_names)}."
        )

    df = df.dropna(subset=["session_id", "activity", *feature_names])
    df = df[df["activity"].isin(["standing", "sitting", "walking"])]

    if df.empty:
        raise ValueError("No valid samples found in the dataset.")

    print(f"Dataset shape: {df.shape}")
    print("\nSamples per activity:")
    print(df["activity"].value_counts().to_string())

    session_count = df["session_id"].nunique()
    print(f"\nNumber of sessions: {session_count}")

    if session_count < 2:
        raise ValueError(
            "At least 2 different recording sessions are required "
            "for session-based evaluation. Collect more sessions first."
        )

    X = df[feature_names].astype("float32")
    y = df["activity"].astype(str)
    groups = df["session_id"].astype(str)

    # Try group splits until the test set contains every activity
    # and the training set contains every activity.
    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    split = None

    for train_idx, test_idx in splitter.split(X, y, groups):
        train_classes = set(y.iloc[train_idx])
        test_classes = set(y.iloc[test_idx])

        if train_classes == set(y.unique()) and test_classes == set(y.unique()):
            split = (train_idx, test_idx)
            break

    if split is None:
        raise ValueError(
            "Could not find a session-based split containing every activity "
            "in both sets. Collect more sessions, and ensure each session "
            "contains standing, sitting, and walking."
        )

    train_idx, test_idx = split

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_sessions = sorted(groups.iloc[train_idx].unique())
    test_sessions = sorted(groups.iloc[test_idx].unique())

    print(f"\nTraining sessions: {train_sessions}")
    print(f"Testing sessions: {test_sessions}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Testing samples: {len(test_idx)}")

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining model...")
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    labels = ["sitting", "standing", "walking"]

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print(f"\nSession-based test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)

    print("Confusion matrix (rows=true, columns=predicted):")
    print(f"Labels: {labels}")
    print(matrix)

    # Save the model with the same keys expected by predict.py.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": classifier,
            "feature_names": feature_names,
            "classes": list(classifier.classes_),
            "evaluation": {
                "accuracy": float(accuracy),
                "train_sessions": train_sessions,
                "test_sessions": test_sessions,
            },
        },
        MODEL_PATH,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("Human Activity Recognition - Session-Based Evaluation\n")
        file.write("=" * 55 + "\n\n")
        file.write(f"Total samples: {len(df)}\n")
        file.write(f"Total sessions: {session_count}\n")
        file.write(f"Training samples: {len(train_idx)}\n")
        file.write(f"Testing samples: {len(test_idx)}\n")
        file.write(f"Training sessions: {train_sessions}\n")
        file.write(f"Testing sessions: {test_sessions}\n")
        file.write(f"Test accuracy: {accuracy:.4f}\n\n")
        file.write("Classification report:\n")
        file.write(report + "\n")
        file.write("Confusion matrix (rows=true, columns=predicted):\n")
        file.write(f"Labels: {labels}\n")
        file.write(str(matrix) + "\n")

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
