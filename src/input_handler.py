"""Resolves a raw user input (PDB ID or target name) into a TargetDescriptor."""

import re
from src.schemas import TargetDescriptor

# demo lookup for the 5 evaluation targets; replace with a real UniProt/ChEMBL query later
KNOWN_TARGETS = {
    "EGFR": "1M17",
    "BRAF": "1UWH",
    "ACE2": "1R42",
    "DRD2": "6CM4",
    "JAK2": "4C61",
}

PDB_ID_PATTERN = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


def resolve_target(raw_input: str) -> TargetDescriptor:
    cleaned = raw_input.strip()
    if not cleaned:
        raise ValueError("empty target input")

    if PDB_ID_PATTERN.match(cleaned):
        return TargetDescriptor(raw_input=raw_input, pdb_id=cleaned.upper(), is_resolved_structure=True)

    name = cleaned.upper()
    if name in KNOWN_TARGETS:
        return TargetDescriptor(raw_input=raw_input, pdb_id=KNOWN_TARGETS[name], is_resolved_structure=True)

    # unresolved name -> needs ESMFold from sequence later in the protein pipeline stage
    return TargetDescriptor(raw_input=raw_input, is_resolved_structure=False)
