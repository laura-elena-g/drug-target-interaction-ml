import numpy as np
import pandas as pd
from pathlib import Path
from featurize_proteins import featurize_sequence

def main():
    root = Path(__file__).resolve().parents[2]
    in_path = root / "data" / "processed" / "kiba_kept.csv"   # aligned with X_drug
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    comps = []
    lens = []

    for seq in df["target_sequence"].astype(str).values:
        comp, length = featurize_sequence(seq)
        comps.append(comp)
        lens.append(length)

    X_comp = np.vstack(comps)         # (N, 20)
    X_len = np.vstack(lens)           # (N, 1)
    X_protein = np.hstack([X_comp, X_len]).astype(np.float32)  # (N, 21)

    np.save(out_dir / "X_protein.npy", X_protein)

    print("Rows:", len(df))
    print("X_protein shape:", X_protein.shape)
    print("Saved:", out_dir / "X_protein.npy")


if __name__ == "__main__":
    main()