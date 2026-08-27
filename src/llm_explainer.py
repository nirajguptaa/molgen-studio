"""Generates a grounded, plain-English explanation for each surviving candidate.

Retrieval: TF-IDF + FAISS over target-specific ChEMBL bioactivity text (a
standard, lightweight approach for small per-target corpora -- avoids a
heavy embedding-model download for what is typically a few dozen to a few
hundred short text entries per target).

Generation: Google Gemini API (free tier), hard-grounded in retrieved
literature, with an explicit system-level guardrail against
clinical/therapeutic claims. Requires GEMINI_API_KEY in the environment --
get a free key at https://aistudio.google.com/apikey (no card required).
"""

import os
import logging
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.schemas import Candidate, ProteinFeatures

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """You explain computational drug candidate predictions to a \
non-specialist reader. Ground every claim in the provided literature snippets. \
Never state or imply the molecule is safe, effective, or clinically validated \
-- frame everything as a computational prediction requiring experimental \
follow-up. Keep explanations to 3-4 sentences."""


class RetrievalIndex:
    """Wraps a TF-IDF + FAISS index over a target's bioactivity text corpus."""

    def __init__(self, texts: list[str]):
        if not texts:
            raise ValueError("cannot build an index from an empty text corpus")
        self.texts = texts
        self.vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
        vectors = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(vectors)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        query_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        k = min(k, len(self.texts))
        _, indices = self.index.search(query_vec, k)
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]


def build_index_from_chembl_csv(csv_path: str, text_column: str = "description") -> RetrievalIndex:
    """Loads a target's bioactivity text corpus from a CSV (pulled from
    ChEMBL's API -- see scripts/fetch_chembl_text.py) and builds a
    RetrievalIndex over it.
    """
    import csv
    texts = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_column, "").strip()
            if text:
                texts.append(text)
    return RetrievalIndex(texts)


def explain(candidate: Candidate, features: ProteinFeatures, index: RetrievalIndex) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a free key at "
            "https://aistudio.google.com/apikey (no card required) and set "
            "it as an environment variable before calling explain()."
        )

    query = f"{features.target.raw_input} inhibitor binding activity"
    retrieved = index.retrieve(query, k=3)
    context_block = "\n".join(f"- {t}" for t in retrieved)

    prompt = (
        f"Candidate SMILES: {candidate.smiles}\n"
        f"Target: {features.target.raw_input}\n"
        f"QED: {candidate.qed}, SA score: {candidate.sa_score}\n"
        f"ADMET flags: {candidate.admet}\n\n"
        f"Relevant literature:\n{context_block}\n\n"
        "Explain in 3-4 sentences why this candidate might be relevant "
        "to the target, and note any ADMET concerns."
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2048,
            temperature=0.4,
        ),
    )

    finish_reason = response.candidates[0].finish_reason if response.candidates else None
    if response.text is None:
        raise RuntimeError(
            f"Gemini returned no text (finish_reason={finish_reason}). "
            "This model reserves part of its token budget for internal "
            "reasoning before writing the answer; if this happens often, "
            "increase max_output_tokens further."
        )
    return response.text