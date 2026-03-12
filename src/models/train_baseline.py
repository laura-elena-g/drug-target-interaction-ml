import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def main():
    root = Path(__file__).resolve().parents[2]
    proc = root / "data" / "processed"

    X_drug = np.load(proc / "X_drug.npy")
    X_prot = np.load(proc / "X_protein.npy")
    df = pd.read_csv(proc / "kiba_kept.csv")

    # Label (Active if affinity >= 12)
    y = (df["affinity"].values >= 12).astype(int)

    # Combine features
    X = np.hstack([X_drug, X_prot]).astype(np.float32)
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Active rate:", y.mean())

    # Random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features for logistic regression
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Baseline model
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)

    # Probabilities for metrics
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    # Confusion matrix at default threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Logistic Regression (Random split) ===")
    print("ROC-AUC:", roc)
    print("PR-AUC :", pr)
    print("Confusion matrix (threshold=0.5):")
    print(cm)

    out_dir = root / "reports" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_score": y_prob
    })

    pred_df.to_csv(out_dir / "logistic_random_split_predictions.csv", index=False)
    
    print("Saved predictions to:", out_dir / "logistic_random_split_predictions.csv")


if __name__ == "__main__":
    main()