from pathlib import Path
import numpy as np
import pandas as pd


def precision_at_fraction(y_true, y_score, fraction):
    n = len(y_true)
    k = max(1, int(n * fraction))

    order = np.argsort(y_score)[::-1]
    top_idx = order[:k]

    return y_true[top_idx].mean()


def recall_at_fraction(y_true, y_score, fraction):
    n = len(y_true)
    k = max(1, int(n * fraction))

    order = np.argsort(y_score)[::-1]
    top_idx = order[:k]

    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0

    return y_true[top_idx].sum() / total_positives


def enrichment_factor(y_true, y_score, fraction):
    baseline_rate = y_true.mean()
    if baseline_rate == 0:
        return 0.0

    return precision_at_fraction(y_true, y_score, fraction) / baseline_rate


def main():
    root = Path(__file__).resolve().parents[2]
    metrics_dir = root / "reports" / "metrics"

    model_files = {
        "Logistic Random Split": metrics_dir / "logistic_random_split_predictions.csv",
        "Logistic Drug Split": metrics_dir / "logistic_drug_split_predictions.csv",
        "XGBoost Random Split": metrics_dir / "xgboost_random_split_predictions.csv",
        "XGBoost Drug Split": metrics_dir / "xgboost_drug_split_predictions.csv",
    }

    fractions = [0.05, 0.10, 0.20]
    rows = []

    for model_name, file_path in model_files.items():
        df = pd.read_csv(file_path)
        y_true = df["y_true"].values
        y_score = df["y_score"].values

        row = {"Model": model_name}

        for fraction in fractions:
            pct = int(fraction * 100)
            row[f"Precision@{pct}%"] = precision_at_fraction(y_true, y_score, fraction)
            row[f"Recall@{pct}%"] = recall_at_fraction(y_true, y_score, fraction)
            row[f"EF@{pct}%"] = enrichment_factor(y_true, y_score, fraction)

        rows.append(row)

    results = pd.DataFrame(rows)
    print(results)

    tables_dir = root / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "screening_metrics.csv"

    results.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()