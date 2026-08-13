"""Validates generated SMILES and produces 3D conformers. Runs entirely locally via RDKit."""

from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import rdMolDescriptors
from src.schemas import Candidate


def sanitize_and_dedupe(candidates: list[Candidate]) -> list[Candidate]:
    seen = set()
    valid = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c.smiles)
        if mol is None:
            c.is_valid = False
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)
        c.smiles = canonical
        c.is_valid = True
        c.qed = round(QED.qed(mol), 3)
        valid.append(c)
    return valid


def embed_3d(candidates: list[Candidate], out_dir: str = "data/conformers") -> list[Candidate]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    for i, c in enumerate(candidates):
        mol = Chem.MolFromSmiles(c.smiles)
        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status != 0:
            c.is_valid = False
            continue
        AllChem.MMFFOptimizeMolecule(mol)
        path = f"{out_dir}/candidate_{i}.sdf"
        writer = Chem.SDWriter(path)
        writer.write(mol)
        writer.close()
        c.conformer_path = path
    return [c for c in candidates if c.is_valid]


def synthetic_accessibility(candidates: list[Candidate]) -> list[Candidate]:
    # placeholder proxy until sascorer.py (from RDKit contrib) is vendored in;
    # ring count + rotatable bonds as a rough stand-in, replace before Review 3
    for c in candidates:
        mol = Chem.MolFromSmiles(c.smiles)
        rings = rdMolDescriptors.CalcNumRings(mol)
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        c.sa_score = round(1 + 0.3 * rings + 0.2 * rot, 2)
    return candidates
