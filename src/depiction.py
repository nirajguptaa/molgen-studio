"""2D and 3D visualization helpers for candidates and protein structures."""

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
import py3Dmol


def mol_to_2d_png_bytes(smiles: str, size: int = 500) -> bytes:
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(size, size))
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mol_info(smiles: str) -> dict:
    """Quick chemistry facts to caption a 2D/3D view with -- real RDKit
    descriptors, not decorative text."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": round(Descriptors.MolWt(mol), 1),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "h_bond_donors": rdMolDescriptors.CalcNumHBD(mol),
        "h_bond_acceptors": rdMolDescriptors.CalcNumHBA(mol),
    }


def conformer_3d_html(sdf_path: str, width: int = 760, height: int = 560, show_labels: bool = False) -> str:
    with open(sdf_path) as f:
        sdf_data = f.read()
    view = py3Dmol.view(width=width, height=height)
    view.addModel(sdf_data, "sdf")
    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.28}})
    if show_labels:
        view.addPropertyLabels(
            "elem", {}, {"fontColor": "black", "fontSize": 11, "showBackground": False}
        )
    view.zoomTo()
    view.setBackgroundColor("0xF7FAFC")
    view.zoom(1.15)
    return view._make_html()


def protein_3d_html(pdb_path: str, width: int = 1050, height: int = 600) -> str:
    with open(pdb_path) as f:
        pdb_data = f.read()
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addSurface(py3Dmol.VDW, {"opacity": 0.08, "color": "white"})
    view.zoomTo()
    view.setBackgroundColor("0xF7FAFC")
    return view._make_html()