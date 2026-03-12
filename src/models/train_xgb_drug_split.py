import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def main():
    root = Path(__file__).resolve().parents[2]
    proc = root / "data" / "processed"

    X_drug = np.load(proc / "X_drug.npy")
    X_prot = np.load(proc / "X_protein.npy")
    df = pd.read_csv(proc / "kiba_kept.csv")

    y = (df["affinity"].values >= 12).astype(int)
    X = np.hstack([X_drug, X_prot]).astype(np.float32)

    # --- Drug-level split ---
    unique_drugs = df["compound_iso_smiles"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_drugs)

    split_idx = int(0.8 * len(unique_drugs))
    train_drugs = set(unique_drugs[:split_idx])
    test_drugs = set(unique_drugs[split_idx:])

    train_mask = df["compound_iso_smiles"].isin(train_drugs)
    test_mask = df["compound_iso_smiles"].isin(test_drugs)

    train_smiles = set(df.loc[train_mask, "compound_iso_smiles"])
    test_smiles = set(df.loc[test_mask, "compound_iso_smiles"])
    overlap = train_smiles.intersection(test_smiles)
    print("SMILES overlap count:", len(overlap))

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print("Train samples:", len(y_train))
    print("Test samples:", len(y_test))

    # XGBoost model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== XGBoost (Drug-level split) ===")
    print("ROC-AUC:", roc)
    print("PR-AUC :", pr)
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()