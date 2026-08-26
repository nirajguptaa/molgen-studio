"""2D and 3D visualization helpers for candidates and protein structures."""

from rdkit import Chem
from rdkit.Chem import Draw
import py3Dmol


def mol_to_2d_png_bytes(smiles: str, size: int = 350) -> bytes:
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(size, size))
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def conformer_3d_html(sdf_path: str, width: int = 420, height: int = 380) -> str:
    with open(sdf_path) as f:
        sdf_data = f.read()
    view = py3Dmol.view(width=width, height=height)
    view.addModel(sdf_data, "sdf")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
    view.zoomTo()
    view.setBackgroundColor("0xF7FAFC")
    return view._make_html()


def protein_3d_html(pdb_path: str, width: int = 700, height: int = 450) -> str:
    with open(pdb_path) as f:
        pdb_data = f.read()
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    view.setBackgroundColor("0xF7FAFC")
    return view._make_html()