"""Optional, experimental AutoDock Vina docking for supported co-crystal sites.

Docking scores are approximate scoring-function outputs, not measured affinities.
Only PDB structures with a known bound-ligand reference are enabled so the search
box is anchored to an observed site rather than guessed from the protein name.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate

from src.schemas import Candidate, TargetDescriptor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKING_SITES = {
    # RCSB PDB native ligand identifiers used to define each search site.
    "1M17": "AQ4",  # EGFR / erlotinib
    "1UWH": "BAX",  # BRAF / sorafenib
    "4C61": "LMM",  # JAK2 inhibitor
    "6CM4": "8NU",  # DRD2 / risperidone
}
BOX_PADDING_ANGSTROM = 7.0
MAX_BOX_EDGE_ANGSTROM = 30.0


class DockingUnavailable(RuntimeError):
    """Raised when a target has no defensible site or docking runtime."""


@dataclass(frozen=True)
class DockingSite:
    pdb_id: str
    ligand_id: str
    ligand_chain: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    receptor_chain: str


def _vina_executable() -> str:
    configured = os.environ.get("VINA_EXECUTABLE")
    candidates = [configured] if configured else []
    candidates += [
        str(PROJECT_ROOT / "tools" / "vina" / "vina_1.2.7_win.exe"),
        str(PROJECT_ROOT / "tools" / "vina" / "vina_1.2.7_linux_x86_64"),
        str(PROJECT_ROOT / "tools" / "vina" / "vina_1.2.7_mac_x86_64"),
        shutil.which("vina"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise DockingUnavailable(
        "AutoDock Vina executable not found. See scripts/setup_vina.ps1 or set "
        "VINA_EXECUTABLE to the path of a Vina 1.2.7 binary."
    )


def _atom_coordinates(line: str) -> tuple[float, float, float]:
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def _heavy_atom_line(line: str) -> bool:
    element = line[76:78].strip().upper()
    if element:
        return element not in {"H", "D"}
    return line[12:16].strip()[0:1].upper() not in {"H", "D"}


def _find_site(pdb_path: Path, pdb_id: str) -> DockingSite:
    ligand_id = DOCKING_SITES[pdb_id]
    lines = pdb_path.read_text(errors="replace").splitlines()
    ligand_groups: dict[tuple[str, str, str, str], list[tuple[float, float, float]]] = {}
    protein_by_chain: dict[str, list[tuple[float, float, float]]] = {}

    for line in lines:
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"} or not _heavy_atom_line(line):
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A", "1"}:
            continue
        try:
            xyz = _atom_coordinates(line)
        except ValueError:
            continue
        if record == "HETATM" and line[17:20].strip().upper() == ligand_id:
            key = (line[21:22], line[22:26], line[26:27], line[17:20].strip())
            ligand_groups.setdefault(key, []).append(xyz)
        elif record == "ATOM":
            chain = line[21:22]
            protein_by_chain.setdefault(chain, []).append(xyz)

    if not ligand_groups:
        raise DockingUnavailable(
            f"PDB {pdb_id} does not contain the expected native ligand {ligand_id}; "
            "the docking site cannot be anchored safely."
        )

    ligand_key, ligand_coords = max(ligand_groups.items(), key=lambda item: len(item[1]))
    lig = np.asarray(ligand_coords, dtype=np.float32)
    center = lig.mean(axis=0)
    low = lig.min(axis=0)
    high = lig.max(axis=0)
    size = np.maximum(high - low + 2 * BOX_PADDING_ANGSTROM, 18.0)
    if np.any(size > MAX_BOX_EDGE_ANGSTROM):
        raise DockingUnavailable(
            f"Native ligand {ligand_id} requires a search box larger than "
            f"{MAX_BOX_EDGE_ANGSTROM:g} Å; set up a reviewed site-specific box."
        )

    ligand_chain = ligand_key[0]
    best_chain = None
    best_contact_count = -1
    for chain, coords in protein_by_chain.items():
        atoms = np.asarray(coords, dtype=np.float32)
        # Count receptor heavy atoms in contact with the native ligand pocket.
        nearest_sq = np.square(atoms[:, None, :] - lig[None, :, :]).sum(axis=2).min(axis=1)
        contact_count = int(np.count_nonzero(nearest_sq <= 12.0**2))
        if contact_count > best_contact_count:
            best_chain, best_contact_count = chain, contact_count
    if best_chain is None or best_contact_count == 0:
        raise DockingUnavailable(f"No protein chain found around the {ligand_id} site in {pdb_id}.")

    return DockingSite(
        pdb_id=pdb_id,
        ligand_id=ligand_id,
        ligand_chain=ligand_chain,
        center=tuple(float(x) for x in center),
        size=tuple(float(x) for x in size),
        receptor_chain=best_chain,
    )


def _prepare_receptor(pdb_path: Path, site: DockingSite) -> Path:
    out_dir = PROJECT_ROOT / "data" / "docking" / site.pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)
    receptor_pdb = out_dir / "receptor_protein_only.pdb"
    receptor_pdbqt = out_dir / "receptor.pdbqt"
    box_path = out_dir / "receptor.box.txt"
    if receptor_pdbqt.is_file() and box_path.is_file():
        return receptor_pdbqt

    protein_lines = []
    for line in pdb_path.read_text(errors="replace").splitlines():
        if line.startswith("ATOM  ") and line[21:22] == site.receptor_chain:
            if line[16:17] in {" ", "A", "1"}:
                protein_lines.append(line)
    if not protein_lines:
        raise DockingUnavailable(f"No protein atoms found for chain {site.receptor_chain} in {pdb_path}.")
    receptor_pdb.write_text("\n".join(protein_lines + ["TER", "END", ""]) )

    prep_exe = PROJECT_ROOT / "venv" / "Scripts" / "mk_prepare_receptor.exe"
    if not prep_exe.is_file():
        prep_exe = Path(shutil.which("mk_prepare_receptor") or "")
    if not prep_exe.is_file():
        raise DockingUnavailable("Meeko receptor preparation command is missing; install project requirements.")

    command = [
        str(prep_exe), "--read_pdb", str(receptor_pdb), "--output_basename",
        str(out_dir / "receptor"), "--write_pdbqt", "--write_vina_box",
        "--box_center", *(f"{v:.3f}" for v in site.center),
        "--box_size", *(f"{v:.3f}" for v in site.size),
        "--delete_bad_res", "--default_altloc", "A",
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not receptor_pdbqt.is_file():
        details = (result.stderr or result.stdout)[-2400:]
        raise DockingUnavailable(f"Meeko could not prepare receptor {site.pdb_id}: {details}")
    box_path.write_text(
        f"reference_ligand = {site.ligand_id}\n"
        f"reference_chain = {site.ligand_chain}\n"
        f"receptor_chain = {site.receptor_chain}\n"
        f"center_x = {site.center[0]:.3f}\ncenter_y = {site.center[1]:.3f}\ncenter_z = {site.center[2]:.3f}\n"
        f"size_x = {site.size[0]:.3f}\nsize_y = {site.size[1]:.3f}\nsize_z = {site.size[2]:.3f}\n"
    )
    return receptor_pdbqt


def prepare_site(target: TargetDescriptor) -> tuple[DockingSite, Path, str]:
    pdb_id = (target.pdb_id or "").upper()
    if pdb_id not in DOCKING_SITES:
        supported = ", ".join(DOCKING_SITES)
        raise DockingUnavailable(
            f"No co-crystal ligand site is configured for {pdb_id or target.raw_input}. "
            f"Current site-supported PDBs: {supported}."
        )
    pdb_path = PROJECT_ROOT / "data" / "structures" / f"{pdb_id}.pdb"
    if not pdb_path.is_file():
        from src.protein_pipeline import _fetch_pdb_structure
        _fetch_pdb_structure(pdb_id)
    site = _find_site(pdb_path, pdb_id)
    receptor = _prepare_receptor(pdb_path, site)
    vina = _vina_executable()
    return site, receptor, vina


def dock_candidates(
    candidates: list[Candidate],
    target: TargetDescriptor,
    *,
    exhaustiveness: int = 4,
    cpu: int = 2,
) -> list[Candidate]:
    """Dock candidates into the co-crystal site and attach score/path metadata.

    Vina scores are stored in ``candidate.admet['vina_score_kcal_mol']`` and
    pose paths in ``candidate.admet['docked_pose_path']``. Failures are attached
    per candidate so a single unsupported molecule does not discard a run.
    """
    if not candidates:
        return candidates
    site, receptor, vina = prepare_site(target)
    out_dir = PROJECT_ROOT / "data" / "docking" / site.pdb_id / "poses"
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = MoleculePreparation()
    for i, candidate in enumerate(candidates):
        raw_sdf_path = candidate.conformer_path
        mol = Chem.SDMolSupplier(raw_sdf_path, removeHs=False)[0] if raw_sdf_path else None
        if mol is None:
            candidate.admet["docking_error"] = "No usable 3D conformer was available."
            continue
        try:
            setups = prep(mol)
            if not setups:
                raise ValueError("Meeko produced no ligand setup")
            pdbqt, valid, error = PDBQTWriterLegacy.write_string(setups[0])
            if not valid or not pdbqt:
                raise ValueError(error or "Meeko could not write the ligand PDBQT")
            ligand_path = out_dir / f"candidate_{i + 1}.pdbqt"
            pose_path = out_dir / f"candidate_{i + 1}_pose.pdbqt"
            ligand_path.write_text(pdbqt)
            command = [
                vina, "--receptor", str(receptor), "--ligand", str(ligand_path),
                "--center_x", f"{site.center[0]:.3f}",
                "--center_y", f"{site.center[1]:.3f}",
                "--center_z", f"{site.center[2]:.3f}",
                "--size_x", f"{site.size[0]:.3f}",
                "--size_y", f"{site.size[1]:.3f}",
                "--size_z", f"{site.size[2]:.3f}",
                "--exhaustiveness", str(max(1, exhaustiveness)),
                "--num_modes", "1", "--cpu", str(max(1, cpu)),
                "--seed", str(4200 + i), "--out", str(pose_path),
            ]
            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=240)
            if result.returncode != 0 or not pose_path.is_file():
                raise RuntimeError((result.stderr or result.stdout)[-1600:] or "Vina returned no pose")
            match = re.search(r"REMARK VINA RESULT:\s*(-?\d+(?:\.\d+)?)", pose_path.read_text())
            if not match:
                raise RuntimeError("Vina completed but did not report a pose score")
            candidate.admet["vina_score_kcal_mol"] = float(match.group(1))
            candidate.admet["docked_pose_path"] = str(pose_path)
            candidate.admet["docking_site"] = f"{site.pdb_id}:{site.ligand_id}"
            docked_molecule = RDKitMolCreate.from_pdbqt_mol(
                PDBQTMolecule.from_file(str(pose_path), skip_typing=True)
            )
            if docked_molecule:
                sdf_path = out_dir / f"candidate_{i + 1}_pose.sdf"
                writer = Chem.SDWriter(str(sdf_path))
                writer.write(docked_molecule[0])
                writer.close()
                candidate.admet["docked_pose_sdf"] = str(sdf_path)
        except Exception as exc:
            candidate.admet["docking_error"] = str(exc)
    return candidates
