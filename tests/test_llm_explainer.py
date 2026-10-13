from pathlib import Path

from src.llm_explainer import build_index_from_chembl_csv


def test_all_configured_targets_have_retrieval_corpora():
    targets = ("egfr", "braf", "ace2", "drd2", "jak2")
    for target in targets:
        path = Path("data") / f"{target}_bioactivity_text.csv"
        index = build_index_from_chembl_csv(str(path))
        assert index.retrieve("protein inhibitor bioactivity", k=3)
