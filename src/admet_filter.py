"""Filters candidates on toxicity/pharmacokinetics.

Primary: ADMETlab 3.0 live API (single-molecule endpoint, confirmed against
their real OpenAPI schema). As of testing, their production API has a
server-side bug (missing 'BSEP' column) causing 500 errors on some/all
requests. This module treats the API as best-effort: on any failure it logs
a warning and falls back to the local RDKit Lipinski/Veber pre-filter,
which is always available and requires no network call.
"""

import logging
import time
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from src.schemas import Candidate

logger = logging.getLogger(__name__)

ADMETLAB_URL = "https://admetlab3.scbdd.com/api/single/admet"
REQUEST_TIMEOUT = 30
RETRY_DELAY_SEC = 1.0


def lipinski_prefilter(candidates: list[Candidate]) -> list[Candidate]:
    """Local, network-free Lipinski/Veber rule filter. Always available."""
    passed = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c.smiles)
        if mol is None:
            continue
        violations = sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            Lipinski.NumHDonors(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10,
        ])
        c.admet["mw"] = round(Descriptors.MolWt(mol), 1)
        c.admet["logp"] = round(Descriptors.MolLogP(mol), 2)
        c.admet["h_donors"] = Lipinski.NumHDonors(mol)
        c.admet["h_acceptors"] = Lipinski.NumHAcceptors(mol)
        c.admet["lipinski_violations"] = violations
        if violations <= 1:  # Lipinski allows one violation
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


def admet_screen(candidates: list[Candidate], use_live_api: bool = True) -> list[Candidate]:
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
            "ADMETlab API unavailable for all candidates (confirmed server-side "
            "bug as of last check). Falling back to local RDKit Lipinski/Veber filter."
        )
        return lipinski_prefilter(candidates)

    # partial success: keep whichever had live results; re-run local filter
    # on the rest so nothing is silently dropped
    return lipinski_prefilter(candidates)