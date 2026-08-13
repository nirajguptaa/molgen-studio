"""Data contracts passed between pipeline modules. Every stage speaks these types only."""

from dataclasses import dataclass, field


@dataclass
class TargetDescriptor:
    raw_input: str
    pdb_id: str | None = None
    sequence: str | None = None
    is_resolved_structure: bool = False


@dataclass
class ProteinFeatures:
    target: TargetDescriptor
    structure_path: str | None = None
    embedding: list[float] = field(default_factory=list)


@dataclass
class Candidate:
    smiles: str
    source: str  # which generator produced it, e.g. "reinvent4"
    qed: float | None = None
    sa_score: float | None = None
    is_valid: bool | None = None
    conformer_path: str | None = None
    admet: dict = field(default_factory=dict)
    admet_pass: bool | None = None
    explanation: str | None = None
