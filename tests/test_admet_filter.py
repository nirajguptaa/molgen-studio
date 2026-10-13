from src.admet_filter import lipinski_prefilter
from src.schemas import Candidate


def test_veber_violation_fails_local_rule_screen():
    candidate = Candidate(smiles="COCCOCCOCCOCCOCCOCCOCCOCCO", source="test")
    result = lipinski_prefilter([candidate])

    assert candidate.admet["veber_violations"] > 0
    assert candidate.admet_pass is False
    assert result == []
