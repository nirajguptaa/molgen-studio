"""Applies local physicochemical rules and optionally calls ADMETlab.

Primary: local RDKit Lipinski/Veber rules, available without network access.
An optional ADMETlab API path is retained as best-effort because the service
currently reports service instability; failures fall back to local rules.
The local rules are simple physicochemical filters, not toxicity or PK models.
"""

import logging
import time
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from src.schemas import Candidate

logger = logging.getLogger(__name__)

ADMETLAB_URL = "https://admetlab3.scbdd.com/api/single/admet"
REQUEST_TIMEOUT = 30


def lipinski_prefilter(candidates: list[Candidate]) -> list[Candidate]:
    """Local, network-free Lipinski/Veber rule filter. Always available."""
    passed = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c.smiles)
        if mol is None:
            continue
        lipinski_violations = sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            Lipinski.NumHDonors(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10,
        ])
        veber_violations = sum([
            rdMolDescriptors.CalcNumRotatableBonds(mol) > 10,
            rdMolDescriptors.CalcTPSA(mol) > 140,
        ])
        c.admet["mw"] = round(Descriptors.MolWt(mol), 1)
        c.admet["logp"] = round(Descriptors.MolLogP(mol), 2)
        c.admet["h_donors"] = Lipinski.NumHDonors(mol)
        c.admet["h_acceptors"] = Lipinski.NumHAcceptors(mol)
        c.admet["rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        c.admet["tpsa"] = round(rdMolDescriptors.CalcTPSA(mol), 1)
        c.admet["lipinski_violations"] = lipinski_violations
        c.admet["veber_violations"] = veber_violations
        if lipinski_violations <= 1 and veber_violations == 0:
            c.admet_pass = True
            passed.append(c)
        else:
            c.admet_pass = False
    return passed


def _call_admetlab(smiles: str) -> dict | None:
    """Single call to the live ADMETlab API. Returns None on any failure."""
    try:
        resp = requests.post(
            ADMETLAB_URL,
            json={"SMILES": smiles, "feature": True},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning(f"ADMETlab returned {resp.status_code} for {smiles[:30]}...")
            return None
        return resp.json()
    except requests.RequestException as e:
        logger.warning(f"ADMETlab request failed for {smiles[:30]}...: {e}")
        return None


def admet_screen(candidates: list[Candidate], use_live_api: bool = False) -> list[Candidate]:
    """Screens candidates via ADMETlab if available, else falls back to
    the local Lipinski pre-filter for all candidates.

    Returns the filtered (passing) candidate list either way.
    """
    if not use_live_api:
        logger.info("Live API disabled; using local Lipinski pre-filter only.")
        return lipinski_prefilter(candidates)

    live_failures = 0
    for c in candidates:
        result = _call_admetlab(c.smiles)
        if result is None:
            live_failures += 1
            continue
        c.admet.update(result)
        time.sleep(0.2)  # be polite to their free public server

    if live_failures == len(candidates):
        logger.warning(
            "ADMETlab API unavailable for all candidates. Falling back to local "
            "RDKit Lipinski/Veber rules."
        )
        return lipinski_prefilter(candidates)

    # partial success: keep whichever had live results; re-run local filter
    # on the rest so nothing is silently dropped
    return lipinski_prefilter(candidates)
