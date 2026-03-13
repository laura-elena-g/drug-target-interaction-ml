from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[2]
    model_path = root / "reports" / "models" / "xgboost_drug_split_model.pkl"
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_path)

    importances = model.feature_importances_
    feature_idx = np.arange(len(importances))

    df = pd.DataFrame({
        "feature_idx": feature_idx,
        "importance": importances
    })

    df["feature_group"] = np.where(df["feature_idx"] < 1029, "Drug", "Protein")

    group_summary.to_csv(tables_dir / "feature_group_importance.csv", index=False)
    top20.to_csv(tables_dir / "top20_feature_importance.csv", index=False)

    print("\n=== Importance by feature group ===")
    print(group_summary)

    print("\n=== Top 20 features ===")
    print(top20)

    group_summary.to_csv(reports_dir / "feature_group_importance.csv", index=False)
    top20.to_csv(reports_dir / "top20_feature_importance.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.bar(group_summary["feature_group"], group_summary["importance"])
    plt.ylabel("Total Importance")
    plt.title("Feature Importance by Group")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_group_importance.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    top20_plot = top20.sort_values("importance", ascending=True)
    plt.barh(top20_plot["feature_idx"].astype(str), top20_plot["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature Index")
    plt.title("Top 20 XGBoost Features")
    plt.tight_layout()
    plt.savefig(figures_dir / "top20_feature_importance.png", dpi=300)
    plt.close()

    print("Saved:", tables_dir / "feature_group_importance.csv")
    print("Saved:", tables_dir / "top20_feature_importance.csv")
    print("Saved:", figures_dir / "feature_group_importance.png")
    print("Saved:", figures_dir / "top20_feature_importance.png")


if __name__ == "__main__":
    main()