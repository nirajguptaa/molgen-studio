from src.schemas import Candidate
from src.validator import sanitize_and_dedupe, embed_3d, synthetic_accessibility


def test_invalid_smiles_dropped():
    candidates = [Candidate(smiles="not_a_molecule", source="test")]
    result = sanitize_and_dedupe(candidates)
    assert result == []


def test_valid_smiles_kept_with_qed():
    candidates = [Candidate(smiles="CC(=O)Oc1ccccc1C(=O)O", source="test")]  # aspirin
    result = sanitize_and_dedupe(candidates)
    assert len(result) == 1
    assert result[0].qed is not None
    assert 0 <= result[0].qed <= 1


def test_dedup_removes_duplicates():
    candidates = [
        Candidate(smiles="CCO", source="test"),
        Candidate(smiles="OCC", source="test"),  # same molecule, different SMILES
    ]
    result = sanitize_and_dedupe(candidates)
    assert len(result) == 1


def test_3d_embedding_produces_conformer_file(tmp_path):
    candidates = [Candidate(smiles="CCO", source="test", is_valid=True)]
    result = embed_3d(candidates, out_dir=str(tmp_path))
    assert len(result) == 1
    assert result[0].conformer_path is not None


def test_sa_score_assigned():
    candidates = [Candidate(smiles="c1ccccc1", source="test", is_valid=True)]
    result = synthetic_accessibility(candidates)
    assert result[0].sa_score is not None
