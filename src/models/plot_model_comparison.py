from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve, auc


def main():
    root = Path(__file__).resolve().parents[2]
    metrics_dir = root / "reports" / "metrics"
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_files = {
        "Logistic Random Split": metrics_dir / "logistic_random_split_predictions.csv",
        "Logistic Drug Split": metrics_dir / "logistic_drug_split_predictions.csv",
        "XGBoost Random Split": metrics_dir / "xgboost_random_split_predictions.csv",
        "XGBoost Drug Split": metrics_dir / "xgboost_drug_split_predictions.csv",
    }

    # --- ROC curve plot ---
    plt.figure(figsize=(8, 6))

    for model_name, file_path in model_files.items():
        df = pd.read_csv(file_path)
        y_true = df["y_true"].values
        y_score = df["y_score"].values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_comparison.png", dpi=300)
    plt.close()

    # --- Precision-Recall curve plot ---
    plt.figure(figsize=(8, 6))

    for model_name, file_path in model_files.items():
        df = pd.read_csv(file_path)
        y_true = df["y_true"].values
        y_score = df["y_score"].values

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, label=f"{model_name} (AUC = {pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "pr_comparison.png", dpi=300)
    plt.close()

    print("Saved:", figures_dir / "roc_comparison.png")
    print("Saved:", figures_dir / "pr_comparison.png")


if __name__ == "__main__":
    main()