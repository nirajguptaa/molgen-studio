"""Wraps REINVENT4 for target-conditioned generation.

Training runs on Kaggle (GPU required, not available locally). This module
loads the CSV exported from a completed Kaggle run. To regenerate:
see kaggle_setup/ notes -- sample from the trained checkpoint
(egfr_stage1.chkpt), download the CSV, drop it in data/.
"""

import csv
from pathlib import Path
from src.schemas import ProteinFeatures, Candidate

_FALLBACK_SEEDS = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cc1ccc(cc1)S(=O)(=O)N",
    "COc1ccc(cc1)C(=O)Nc1ccccc1",
]


def generate(features: ProteinFeatures, n: int = 50, source_csv: str = "data/egfr_generated.csv") -> list[Candidate]:
    path = Path(source_csv)
    if not path.exists():
        return [Candidate(smiles=s, source="fallback_seed") for s in _FALLBACK_SEEDS]

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)[:n]

    return [
        Candidate(
            smiles=row["SMILES"],
            source="reinvent4_egfr_trained",
        )
        for row in rows
    ]