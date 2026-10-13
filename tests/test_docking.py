from pathlib import Path

import pytest

from src.docking import DOCKING_SITES, DockingUnavailable, _find_site
from src.input_handler import resolve_target
from src.schemas import Candidate
from src.validator import embed_3d
from src.docking import dock_candidates


@pytest.mark.parametrize("pdb_id", sorted(DOCKING_SITES))
def test_native_ligand_defines_a_bounded_site(pdb_id):
    path = Path("data/structures") / f"{pdb_id}.pdb"
    site = _find_site(path, pdb_id)

    assert site.ligand_id == DOCKING_SITES[pdb_id]
    assert all(18 <= edge <= 30 for edge in site.size)
    assert site.receptor_chain


def test_ace2_has_no_claimed_co_crystal_docking_site():
    from src.docking import prepare_site

    with pytest.raises(DockingUnavailable, match="No co-crystal ligand site"):
        prepare_site(resolve_target("ACE2"))


@pytest.mark.skipif(not Path("tools/vina/vina_1.2.7_win.exe").is_file(), reason="install Vina first")
def test_single_egfr_candidate_can_be_docked(tmp_path):
    candidate = Candidate(smiles="CCOc1cc2ncnc(Nc3ccccc3)c2cc1", source="test")
    embed_3d([candidate], out_dir=str(tmp_path / "conformers"))

    result = dock_candidates([candidate], resolve_target("EGFR"), exhaustiveness=1, cpu=1)

    assert result[0].admet["vina_score_kcal_mol"] < 0
    assert Path(result[0].admet["docked_pose_sdf"]).is_file()
