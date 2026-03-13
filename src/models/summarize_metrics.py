from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    root = Path(__file__).resolve().parents[2]
    metrics_dir = root / "reports" / "metrics"

    model_files = {
        "Logistic Random Split": metrics_dir / "logistic_random_split_predictions.csv",
        "Logistic Drug Split": metrics_dir / "logistic_drug_split_predictions.csv",
        "XGBoost Random Split": metrics_dir / "xgboost_random_split_predictions.csv",
        "XGBoost Drug Split": metrics_dir / "xgboost_drug_split_predictions.csv",
    }

    rows = []

    for model_name, file_path in model_files.items():
        df = pd.read_csv(file_path)
        y_true = df["y_true"].values
        y_score = df["y_score"].values

        rows.append({
            "Model": model_name,
            "ROC-AUC": roc_auc_score(y_true, y_score),
            "PR-AUC": average_precision_score(y_true, y_score),
        })

    results = pd.DataFrame(rows)
    print(results)

    tables_dir = root / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "model_summary.csv"

    results.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()