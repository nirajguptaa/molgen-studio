from src.input_handler import resolve_target
from src.molecule_generator import generate
from src.schemas import ProteinFeatures


def _generate_for(target_name):
    return generate(ProteinFeatures(target=resolve_target(target_name)), n=15)


def test_exploratory_generation_is_reproducible_per_target():
    first = _generate_for("BRAF")
    second = _generate_for("BRAF")

    assert [c.smiles for c in first] == [c.smiles for c in second]
    assert all(c.source == "brics_exploratory[braf]" for c in first)


def test_exploratory_candidate_library_changes_with_target():
    braf = {c.smiles for c in _generate_for("BRAF")}
    jak2 = {c.smiles for c in _generate_for("JAK2")}

    assert braf != jak2
