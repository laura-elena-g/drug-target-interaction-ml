import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def featurize_smiles(smiles: str, n_bits: int = 1024, radius: int = 2):
    """
    Returns:
      - fp: (n_bits,) uint8 array
      - desc: (5,) float32 array [MolWt, LogP, HBD, HBA, TPSA]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp_arr = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)

    desc = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
        ],
        dtype=np.float32,
    )

    return fp_arr, desc


if __name__ == "__main__":
    test = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
    fp, desc = featurize_smiles(test)
    print("FP shape:", fp.shape, "FP sum(bits):", int(fp.sum()))
    print("Descriptors:", desc)