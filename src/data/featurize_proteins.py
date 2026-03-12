import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA)}


def featurize_sequence(seq: str):
    """
    Returns:
      - comp: (20,) float32 amino acid fraction vector (sums to 1 if length>0)
      - length: (1,) float32 sequence length
    """
    seq = str(seq)
    n = len(seq)
    comp = np.zeros((20,), dtype=np.float32)

    if n == 0:
        return comp, np.array([0.0], dtype=np.float32)

    for ch in seq:
        idx = AA_TO_IDX.get(ch)
        if idx is not None:
            comp[idx] += 1.0

    comp /= n
    return comp, np.array([float(n)], dtype=np.float32)


if __name__ == "__main__":
    test = "MKTFFVLLLFLTLATYYT"
    comp, length = featurize_sequence(test)
    print("Comp shape:", comp.shape, "sum:", float(comp.sum()))
    print("Length:", float(length[0]))