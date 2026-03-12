import numpy as np
import pandas as pd
from pathlib import Path
from featurize_drugs import featurize_smiles


def main():
    root = Path(__file__).resolve().parents[2]
    in_path = root / "data" / "raw" / "kiba_all.csv"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    fps = []
    descs = []
    keep_idx = []
    failed = 0

    for i, s in enumerate(df["compound_iso_smiles"].astype(str).values):
        fp, desc = featurize_smiles(s)
        if fp is None:
            failed += 1
            continue
        fps.append(fp)
        descs.append(desc)
        keep_idx.append(i)

    X_fp = np.vstack(fps)          # (N, 1024)
    X_desc = np.vstack(descs)      # (N, 5)
    X_drug = np.hstack([X_fp, X_desc]).astype(np.float32)  # (N, 1029)

    df_kept = df.iloc[keep_idx].reset_index(drop=True)

    np.save(out_dir / "X_drug.npy", X_drug)
    df_kept.to_csv(out_dir / "kiba_kept.csv", index=False)

    print("Rows in raw:", len(df))
    print("Rows kept:", len(df_kept))
    print("Failed SMILES:", failed)
    print("X_drug shape:", X_drug.shape)
    print("Saved:")
    print(" -", out_dir / "X_drug.npy")
    print(" -", out_dir / "kiba_kept.csv")


if __name__ == "__main__":
    main()