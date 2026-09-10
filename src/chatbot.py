"""Conversational Q&A assistant grounded in the current pipeline run.

Reuses the same Gemini setup as llm_explainer.py (GEMINI_API_KEY). Every
call is grounded with a fresh summary of the actual target + candidates +
ADMET numbers currently in session state, so the model can't drift into
generic answers that don't match what's on screen -- and it keeps prior
turns so follow-up questions ("what about the second one?") work.
"""

import os
from google import genai
from google.genai import types

from src.schemas import Candidate, ProteinFeatures

GEMINI_MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are the in-app assistant for MolGen Studio, a student \
computational drug-design demo. You answer questions about the CURRENT \
pipeline run only, using the run summary provided in each turn as ground \
truth -- don't invent numbers not present in it. Explain concepts (QED, SA \
score, Lipinski's rule of five, BRICS, docking, ADMET, etc.) in plain \
English suitable for a project viva audience (a teacher/evaluator who may \
not code). Never claim any molecule is safe, effective, or ready for real \
use -- everything here is an unvalidated computational screen. If asked \
something the run summary can't answer, say so plainly and explain what \
additional step (e.g. real docking, wet-lab assay, RL training on that \
target) would be needed. Keep answers to a few short paragraphs, use \
plain text (no markdown tables)."""


def build_run_summary(target_raw: str, pdb_id: str | None, candidates: list[Candidate]) -> str:
    if not candidates:
        return f"Target: {target_raw} (PDB {pdb_id or 'unresolved'}). No candidates survived this run."

    n = len(candidates)
    avg_qed = sum(c.qed or 0 for c in candidates) / n
    avg_sa = sum(c.sa_score or 0 for c in candidates) / n
    sources = {c.source for c in candidates}
    top = sorted(candidates, key=lambda c: c.qed or 0, reverse=True)[:5]
    top_lines = "\n".join(
        f"  {i+1}. {c.smiles} | QED={c.qed} SA={c.sa_score} "
        f"MW={c.admet.get('mw')} LogP={c.admet.get('logp')}"
        for i, c in enumerate(top)
    )
    return (
        f"Target: {target_raw} (PDB {pdb_id or 'unresolved'})\n"
        f"Candidates surviving filters: {n}\n"
        f"Generation source(s): {', '.join(sources)}\n"
        f"Average QED: {avg_qed:.3f} | Average SA score: {avg_sa:.2f}\n"
        f"Top candidates by QED:\n{top_lines}"
    )


def ask(question: str, run_summary: str, history: list[dict]) -> str:
    """history is a list of {'role': 'user'|'model', 'text': str}, oldest first."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a free key at "
            "https://aistudio.google.com/apikey and set it as an "
            "environment variable to enable the chatbot."
        )

    contents = []
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["text"])]))

    grounded_question = f"Current run summary:\n{run_summary}\n\nQuestion: {question}"
    contents.append(types.Content(role="user", parts=[types.Part(text=grounded_question)]))

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
            temperature=0.3,
        ),
    )

    if response.text is None:
        raise RuntimeError("Chatbot returned no text -- try rephrasing the question.")
    return response.text