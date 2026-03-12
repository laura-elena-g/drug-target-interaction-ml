import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier


def main():
    root = Path(__file__).resolve().parents[2]
    proc = root / "data" / "processed"

    X_drug = np.load(proc / "X_drug.npy")
    X_prot = np.load(proc / "X_protein.npy")
    df = pd.read_csv(proc / "kiba_kept.csv")

    y = (df["affinity"].values >= 12).astype(int)
    X = np.hstack([X_drug, X_prot]).astype(np.float32)

    # --- Random split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== XGBoost (Random split) ===")
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

    pred_df.to_csv(out_dir / "xgboost_random_split_predictions.csv", index=False)

    print("Saved predictions to:", out_dir / "xgboost_random_split_predictions.csv")


if __name__ == "__main__":
    main()