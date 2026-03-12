import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def main():
    root = Path(__file__).resolve().parents[2]
    proc = root / "data" / "processed"

    X_drug = np.load(proc / "X_drug.npy")
    X_prot = np.load(proc / "X_protein.npy")
    df = pd.read_csv(proc / "kiba_kept.csv")

    y = (df["affinity"].values >= 12).astype(int)
    X = np.hstack([X_drug, X_prot]).astype(np.float32)

    print("Total samples:", len(df))

    # --- Drug-level split ---
    unique_drugs = df["compound_iso_smiles"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_drugs)

    split_idx = int(0.8 * len(unique_drugs))
    train_drugs = set(unique_drugs[:split_idx])
    test_drugs = set(unique_drugs[split_idx:])

    train_mask = df["compound_iso_smiles"].isin(train_drugs)
    test_mask = df["compound_iso_smiles"].isin(test_drugs)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print("Train samples:", len(y_train))
    print("Test samples:", len(y_test))
    print("Train active rate:", y_train.mean())
    print("Test active rate:", y_test.mean())

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
            )
    clf.fit(X_train_s, y_train)

    y_prob = clf.predict_proba(X_test_s)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Logistic Regression (Drug-level split) ===")
    print("ROC-AUC:", roc)
    print("PR-AUC :", pr)
    print("Confusion matrix:")
    print(cm)

    out_dir = root / "reports" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_score": y_prob
    })

    pred_df.to_csv(out_dir / "logistic_drug_split_predictions.csv", index=False)
    
    print("Saved predictions to:", out_dir / "logistic_drug_split_predictions.csv")


if __name__ == "__main__":
    main()