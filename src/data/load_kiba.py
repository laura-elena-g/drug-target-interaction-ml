import pandas as pd
from pathlib import Path


def load_kiba():
    data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "kiba_all.csv"
    df = pd.read_csv(data_path)
    return df


if __name__ == "__main__":
    df = load_kiba()
    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)

    print("\nKIBA score statistics:")
    print(df["affinity"].describe())

    threshold = 12
    df["label"] = (df["affinity"] >= threshold).astype(int)

    print("\nClass distribution (threshold = 12):")
    print(df["label"].value_counts(normalize=True))

    print("\nMissing values:")
    print(df.isnull().sum())
