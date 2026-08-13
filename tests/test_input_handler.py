from src.input_handler import resolve_target


def test_known_target_name():
    t = resolve_target("EGFR")
    assert t.pdb_id == "1M17"
    assert t.is_resolved_structure


def test_raw_pdb_id():
    t = resolve_target("6lu7")
    assert t.pdb_id == "6LU7"
    assert t.is_resolved_structure


def test_unknown_target_falls_through():
    t = resolve_target("SOME_UNKNOWN_PROTEIN")
    assert t.pdb_id is None
    assert not t.is_resolved_structure


def test_empty_input_raises():
    try:
        resolve_target("   ")
        assert False, "expected ValueError"
    except ValueError:
        pass
