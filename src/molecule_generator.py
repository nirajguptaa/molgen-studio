"""Generates candidate molecules for a target.

Two tiers, chosen automatically per target -- never hardcoded to one file:

1. TRAINED: if data/<target>_generated.csv exists (exported from a completed
   REINVENT4 Kaggle run for that specific target), use it. This is the
   highest-quality path: molecules were RL-trained against that target's
   ChEMBL actives.

2. EXPLORATORY (fallback): if no trained CSV exists for the target yet,
   generate a fresh combinatorial library with RDKit's BRICS algorithm
   instead of returning a fixed set of molecules. A curated pool of
   fragment-donor seed molecules is decomposed into BRICS fragments; the
   subset of fragments used and the traversal order are both seeded from
   the target name, so EGFR, BRAF, ACE2, DRD2, JAK2, or any other typed-in
   target each draw a different fragment subset and reassemble into a
   different set of valid, unique molecules. This is real molecule
   construction (not curve-fitted to any target's actual binding data) and
   is always labeled clearly in `Candidate.source` so it's never confused
   with the trained output.

To add a trained tier for a new target: run the Kaggle REINVENT4 notebook
against that target's ChEMBL actives, download the resulting CSV, and drop
it in data/<target>_generated.csv (lowercase target name, e.g.
data/braf_generated.csv). No code changes needed -- generate() picks it up
automatically.
"""

import csv
import hashlib
import itertools
import random
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BRICS

from src.schemas import ProteinFeatures, Candidate

# Diverse curated seed molecules spanning several real drug chemotypes
# (kinase-hinge binders, sulfonamides, biaryl amides, piperazines, etc.)
# so the BRICS fragment pool isn't narrow. These are only *starting*
# material for decomposition/recombination, not the final output.
_SEED_MOLECULES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cc1ccc(cc1)S(=O)(=O)N",
    "COc1ccc(cc1)C(=O)Nc1ccccc1",
    "O=C(Nc1ccc(cc1)N1CCOCC1)c1ccc(cc1)-c1ccncn1",
    "CN1CCN(CC1)c1ccc(cc1)C(=O)Nc1ccc(F)cc1",
    "Fc1ccc(cc1)C(=O)Nc1ccc(cc1)N1CCNCC1",
    "O=C1CCC(=O)N1c1ccc(Cl)cc1",
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC",
    "CS(=O)(=O)c1ccc(cc1)-c1cc(nc(n1)N)N1CCOCC1",
    "O=C(Nc1ccccc1)c1ccc(cc1)N1CCNCC1",
    "Clc1ccc(cc1)C(=O)Nc1nc2ccccc2s1",
    "O=C(c1ccc(F)cc1)N1CCN(CC1)c1ccccn1",
]

_FALLBACK_SEEDS = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cc1ccc(cc1)S(=O)(=O)N",
    "COc1ccc(cc1)C(=O)Nc1ccccc1",
]

_MAX_BRICS_ITERATIONS = 4000


def _target_key(features: ProteinFeatures) -> str:
    """The stable per-target key used both for the trained-CSV lookup and
    as the RNG seed for exploratory generation."""
    t = features.target
    raw = (t.raw_input or t.pdb_id or "target").strip().lower()
    return raw


def _trained_csv_path(target_key: str) -> Path:
    return Path("data") / f"{target_key}_generated.csv"


def _load_trained(path: Path, n: int) -> list[Candidate]:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)[:n]
    return [
        Candidate(smiles=row["SMILES"], source="reinvent4_trained")
        for row in rows
        if row.get("SMILES")
    ]


def _fragment_pool() -> list[Chem.Mol]:
    frags = set()
    for smi in _SEED_MOLECULES:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            frags |= BRICS.BRICSDecompose(mol)
    mols = [Chem.MolFromSmiles(f) for f in frags]
    return [m for m in mols if m is not None]


def _generate_exploratory(target_key: str, n: int) -> list[Candidate]:
    """BRICS-recombine a target-seeded subset of fragments into `n` unique,
    valid, novel molecules. Different target_key -> different subset and
    traversal order -> different molecules out."""
    seed_int = int(hashlib.sha256(target_key.encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(seed_int)

    pool = _fragment_pool()
    if not pool:
        return [Candidate(smiles=s, source="fallback_seed") for s in _FALLBACK_SEEDS]

    rng.shuffle(pool)
    # take a target-dependent slice size (60-90% of the pool) so different
    # targets don't all just use "the whole pool" and converge anyway
    frac = 0.6 + (seed_int % 31) / 100.0
    subset_size = max(4, int(len(pool) * frac))
    subset = pool[:subset_size]

    builder = BRICS.BRICSBuild(subset)
    seen_smiles = set()
    out = []
    for mol in itertools.islice(builder, _MAX_BRICS_ITERATIONS):
        try:
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if smi in seen_smiles:
            continue
        seen_smiles.add(smi)
        out.append(Candidate(smiles=smi, source=f"brics_exploratory[{target_key}]"))
        if len(out) >= n:
            break

    if not out:
        return [Candidate(smiles=s, source="fallback_seed") for s in _FALLBACK_SEEDS]
    return out


def generate(features: ProteinFeatures, n: int = 50) -> list[Candidate]:
    target_key = _target_key(features)
    trained_path = _trained_csv_path(target_key)

    if trained_path.exists():
        candidates = _load_trained(trained_path, n)
        if candidates:
            return candidates

    return _generate_exploratory(target_key, n)