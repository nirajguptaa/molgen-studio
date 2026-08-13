"""Generates a grounded, plain-English explanation for each surviving candidate.
Requires ANTHROPIC_API_KEY and a FAISS index built over ChEMBL literature
(build_index() is a stub — point it at your actual bioactivity text corpus)."""

import os
import numpy as np
from anthropic import Anthropic
from src.schemas import Candidate, ProteinFeatures

SYSTEM_PROMPT = """You explain computational drug candidate predictions to a
non-specialist reader. Ground every claim in the provided literature snippets.
Never state or imply the molecule is safe, effective, or clinically validated —
frame everything as a computational prediction requiring experimental follow-up."""


def build_index(chembl_texts: list[str]):
    import faiss
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embedder.encode(chembl_texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors, dtype="float32"))
    return index, embedder, chembl_texts


def retrieve(query: str, index, embedder, texts: list[str], k: int = 3) -> list[str]:
    query_vec = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_vec, k)
    return [texts[i] for i in indices[0] if i < len(texts)]


def explain(candidate: Candidate, features: ProteinFeatures, retrieved_context: list[str]) -> str:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    context_block = "\n".join(f"- {c}" for c in retrieved_context)

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Candidate SMILES: {candidate.smiles}\n"
                f"Target: {features.target.raw_input}\n"
                f"QED: {candidate.qed}, SA score: {candidate.sa_score}\n"
                f"ADMET flags: {candidate.admet}\n\n"
                f"Relevant literature:\n{context_block}\n\n"
                "Explain in 3-4 sentences why this candidate might be relevant "
                "to the target, and note any ADMET concerns."
            ),
        }],
    )
    return message.content[0].text
