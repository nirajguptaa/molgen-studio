from src.chatbot import build_run_summary
from src.schemas import Candidate


def test_run_summary_contains_every_candidates_properties():
    candidates = [
        Candidate(smiles="CCO", source="test", qed=0.5, sa_score=2.1,
                  admet={"mw": 46, "logp": -0.1, "tpsa": 20, "rotatable_bonds": 0,
                         "lipinski_violations": 0, "veber_violations": 0}),
        Candidate(smiles="c1ccccc1", source="test", qed=0.4, sa_score=2.5,
                  admet={"mw": 78, "logp": 1.7, "vina_score_kcal_mol": -5.3}),
    ]
    summary = build_run_summary("EGFR", "1M17", candidates)

    assert "CCO" in summary
    assert "c1ccccc1" in summary
    assert "LipinskiViol=0" in summary
    assert "Vina=-5.3 kcal/mol" in summary
