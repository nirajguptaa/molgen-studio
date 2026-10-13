"""Fetches resolved PDB structures and optionally computes sequence embeddings."""

import os
import requests
from src.schemas import TargetDescriptor, ProteinFeatures

BIONEMO_ESMFOLD_URL = "https://build.nvidia.com/meta/esmfold"


def get_protein_features(target: TargetDescriptor, embed_sequence: bool = False) -> ProteinFeatures:
    if target.is_resolved_structure:
        structure_path = _fetch_pdb_structure(target.pdb_id)
    else:
        if not target.sequence:
            raise ValueError(
                f"No amino-acid sequence is available for '{target.raw_input}'. "
                "Resolve the name to a PDB ID or provide a sequence before predicting a structure."
            )
        structure_path = _predict_structure_esmfold(target.sequence)
    embedding = _embed_sequence(target.sequence) if embed_sequence and target.sequence else []
    return ProteinFeatures(target=target, structure_path=structure_path, embedding=embedding)


def _fetch_pdb_structure(pdb_id: str) -> str:
    path = f"data/structures/{pdb_id.upper()}.pdb"
    # Reuse structures already fetched for this project, including when the
    # app is running without network access.
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    os.makedirs("data/structures", exist_ok=True)
    with open(path, "w") as f:
        f.write(resp.text)
    return path


def _predict_structure_esmfold(sequence: str) -> str:
    api_key = os.environ.get("NVIDIA_BIONEMO_API_KEY")
    if not api_key:
        raise RuntimeError("set NVIDIA_BIONEMO_API_KEY to call the hosted ESMFold endpoint")
    resp = requests.post(
        BIONEMO_ESMFOLD_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"sequence": sequence},
        timeout=120,
    )
    resp.raise_for_status()
    path = "data/structures/predicted.pdb"
    os.makedirs("data/structures", exist_ok=True)
    with open(path, "w") as f:
        f.write(resp.json()["pdb"])
    return path


def _embed_sequence(sequence: str) -> list[float]:
    # loads Rostlab/prot_t5_xl_uniref50 via transformers; heavy download (~11GB),
    # do this on your own machine/Colab, not in a lightweight dev sandbox
    from transformers import T5Tokenizer, T5EncoderModel
    import torch

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    ids = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids)
    return out.last_hidden_state.mean(dim=1).squeeze().tolist()
