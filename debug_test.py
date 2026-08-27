import sys
sys.path.insert(0, '.')

from src.molecule_generator import generate
from src.validator import sanitize_and_dedupe, synthetic_accessibility, embed_3d
from src.schemas import ProteinFeatures, TargetDescriptor
from src.depiction import mol_to_2d_png_bytes
from src.llm_explainer import build_index_from_chembl_csv, explain

features = ProteinFeatures(target=TargetDescriptor(raw_input='EGFR', pdb_id='1M17'))
candidates = generate(features, n=3)
candidates = sanitize_and_dedupe(candidates)
candidates = synthetic_accessibility(candidates)
candidates = embed_3d(candidates)

c = candidates[0]
print("=== IMAGE TEST ===")
print("SMILES:", c.smiles)
png = mol_to_2d_png_bytes(c.smiles)
print("PNG bytes:", len(png), type(png))
with open("debug_2d.png", "wb") as f:
    f.write(png)
print("Saved debug_2d.png -- open it and confirm it's a real molecule image")

print()
print("=== EXPLANATION TEST ===")
index = build_index_from_chembl_csv('data/egfr_bioactivity_text.csv')
raw = explain(c, features, index)
print("RAW RESPONSE TYPE:", type(raw))
print("RAW RESPONSE REPR:")
print(repr(raw))