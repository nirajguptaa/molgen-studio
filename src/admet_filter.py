"""Filters candidates on toxicity/pharmacokinetics via ADMETlab 3.0, with a
local RDKit-based Lipinski pre-filter to cut down on API calls (see risk
register: ADMET API rate limits)."""

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from src.schemas import Candidate

ADMETLAB_BATCH_URL = "https://admetlab3.scbdd.com/api/screening/batch"  # confirm exact path in their docs before use

THRESHOLDS = {
    "herg_risk_max": 0.5,
    "hepatotoxicity_max": 0.5,
    "oral_bioavailability_min": 0.3,
}


def lipinski_prefilter(candidates: list[Candidate]) -> list[Candidate]:
    passed = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c.smiles)
        violations = sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            Lipinski.NumHDonors(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10,
        ])
        if violations <= 1:  # Lipinski allows one violation
            passed.append(c)
    return passed


def admet_screen(candidates: list[Candidate]) -> list[Candidate]:
    smiles_list = [c.smiles for c in candidates]
    resp = requests.post(ADMETLAB_BATCH_URL, json={"smiles": smiles_list}, timeout=120)
    resp.raise_for_status()
    scores = resp.json()["results"]

    for c, s in zip(candidates, scores):
        c.admet = s
        c.admet_pass = (
            s.get("herg", 1.0) <= THRESHOLDS["herg_risk_max"]
            and s.get("hepatotoxicity", 1.0) <= THRESHOLDS["hepatotoxicity_max"]
            and s.get("oral_bioavailability", 0.0) >= THRESHOLDS["oral_bioavailability_min"]
        )
    return [c for c in candidates if c.admet_pass]
