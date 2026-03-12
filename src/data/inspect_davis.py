import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "raw" / "davis_all.csv"
    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nHead:")
    print(df.head(3))

    if "affinity" in df.columns:
        print("\nAffinity stats:")
        print(df["affinity"].describe())

threshold = 7.0
df["label"] = (df["affinity"] >= threshold).astype(int)

print("\nClass distribution (pKd ≥ 7):")
print(df["label"].value_counts(normalize=True))