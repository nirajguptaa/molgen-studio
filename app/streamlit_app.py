import streamlit as st
import sys
import os
from pathlib import Path
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.input_handler import resolve_target
from src.protein_pipeline import get_protein_features
from src.molecule_generator import generate
from src.validator import sanitize_and_dedupe, synthetic_accessibility, embed_3d
from src.admet_filter import admet_screen
from src.depiction import mol_to_2d_png_bytes, conformer_3d_html, protein_3d_html
from src.llm_explainer import build_index_from_chembl_csv, explain as explain_candidate
from src.schemas import ProteinFeatures

st.set_page_config(page_title="MolGen Studio", layout="wide", page_icon="🧬")

st.title("🧬 MolGen Studio")
st.caption("Generative AI Pipeline for Novel Drug Molecule Design — EGFR-conditioned demo")

with st.sidebar:
    st.header("Target")
    target_input = st.text_input("Protein target (name or PDB ID)", value="EGFR")
    n_candidates = st.slider("Number of candidates", 5, 50, 20)
    run_btn = st.button("Run Pipeline", type="primary")

    

# --- Run the pipeline ONLY when the button is freshly clicked, and stash
# everything needed to render results in session_state. Every other widget
# interaction (e.g. an "explain" button inside an expander) triggers a
# Streamlit rerun where run_btn is False -- without session_state that
# rerun would lose all results and fall back to the empty state.
if run_btn:
    with st.spinner("Resolving target..."):
        target = resolve_target(target_input)

    features = ProteinFeatures(target=target)
    structure_msg = None
    if target.is_resolved_structure:
        with st.spinner("Fetching protein structure from RCSB PDB..."):
            try:
                features = get_protein_features(target)
                structure_msg = f"Structure fetched: `{features.structure_path}`"
            except Exception as e:
                structure_msg = f"Structure fetch failed ({e}); continuing without it."
    else:
        structure_msg = "No resolved PDB structure for this target. ESMFold requires a BioNeMo API key (not configured) — skipping structure step."

    with st.spinner("Generating candidates..."):
        candidates = generate(features, n=n_candidates)

    with st.spinner("Validating (RDKit sanitize, dedupe, 3D embed)..."):
        candidates = sanitize_and_dedupe(candidates)
        candidates = synthetic_accessibility(candidates)
        candidates = embed_3d(candidates)

    with st.spinner("Screening ADMET (local Lipinski/Veber pre-filter)..."):
        candidates = admet_screen(candidates, use_live_api=False)

    explainer_index = None
    explainer_error = None
    bioactivity_csv = f"data/{target.raw_input.lower()}_bioactivity_text.csv"
    if not os.environ.get("GEMINI_API_KEY"):
        explainer_error = "GEMINI_API_KEY not set — explanations unavailable this run."
    elif not Path(bioactivity_csv).exists():
        explainer_error = (
            f"No bioactivity text corpus found for {target.raw_input} "
            f"(expected `{bioactivity_csv}`). Run `scripts/fetch_chembl_text.py` first."
        )
    else:
        try:
            explainer_index = build_index_from_chembl_csv(bioactivity_csv)
        except Exception as e:
            explainer_error = f"Could not build retrieval index: {e}"

    # stash everything the results section needs -- this is what survives
    # across reruns triggered by the per-candidate "explain" buttons below
    st.session_state["target"] = target
    st.session_state["features"] = features
    st.session_state["structure_msg"] = structure_msg
    st.session_state["candidates"] = candidates
    st.session_state["explainer_index"] = explainer_index
    st.session_state["explainer_error"] = explainer_error
    st.session_state["explanations"] = {}  # candidate index -> explanation text

# --- Render results from session_state (persists across every rerun) ---
if "candidates" in st.session_state:
    target = st.session_state["target"]
    features = st.session_state["features"]
    candidates = st.session_state["candidates"]
    explainer_index = st.session_state["explainer_index"]
    explainer_error = st.session_state["explainer_error"]

    st.success(f"Target resolved: **{target.raw_input}** → PDB `{target.pdb_id or 'unresolved'}`")
    if st.session_state["structure_msg"]:
        st.info(st.session_state["structure_msg"])

    if features.structure_path:
        st.subheader(f"Target structure — {target.pdb_id}")
        try:
            html = protein_3d_html(features.structure_path)
            components.html(html, height=470)
        except Exception as e:
            st.warning(f"Could not render protein structure: {e}")

    st.success(f"Generated {len(candidates)} candidates (source: `{candidates[0].source if candidates else 'n/a'}`)")
    st.success(f"{len(candidates)} valid candidates with 3D structures, passed ADMET pre-filter")

    if explainer_error:
        st.info(f"LLM Explainer inactive this run: {explainer_error}")

    st.divider()
    st.subheader(f"Results — {len(candidates)} candidates")

    if not candidates:
        st.warning("No candidates survived filtering.")
    else:
        avg_qed = sum(c.qed for c in candidates) / len(candidates)
        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates", len(candidates))
        col2.metric("Avg QED", f"{avg_qed:.3f}")
        col3.metric("Source", "REINVENT4 (trained)" if "trained" in candidates[0].source else "fallback")

        sorted_candidates = sorted(candidates, key=lambda c: c.qed, reverse=True)
        for i, c in enumerate(sorted_candidates):
            with st.expander(f"Candidate {i + 1} — QED {c.qed:.3f}"):
                col_a, col_b, col_c = st.columns([1.1, 1.1, 1.4])

                with col_a:
                    st.caption("2D structure")
                    try:
                        png_bytes = mol_to_2d_png_bytes(c.smiles)
                        st.image(png_bytes, use_container_width=True)
                    except Exception as e:
                        st.warning(f"2D render failed: {e}")

                with col_b:
                    st.caption("3D conformer")
                    if c.conformer_path:
                        try:
                            html3d = conformer_3d_html(c.conformer_path)
                            components.html(html3d, height=380)
                        except Exception as e:
                            st.warning(f"3D render failed: {e}")
                    else:
                        st.info("No 3D conformer available.")

                with col_c:
                    st.caption("Details")
                    st.code(c.smiles, language=None)
                    st.write(f"**SA Score:** {c.sa_score}")
                    st.write(f"**MW:** {c.admet.get('mw', 'n/a')}  |  **LogP:** {c.admet.get('logp', 'n/a')}")
                    st.write(f"**H-Donors:** {c.admet.get('h_donors', 'n/a')}  |  **H-Acceptors:** {c.admet.get('h_acceptors', 'n/a')}")

                    if explainer_index is not None:
                        if st.button("Generate explanation", key=f"explain_{i}"):
                            with st.spinner("Calling Gemini..."):
                                try:
                                    text = explain_candidate(c, features, explainer_index)
                                    st.session_state["explanations"][i] = text
                                except Exception as e:
                                    st.error(f"Explanation failed: {e}")
                        if i in st.session_state["explanations"]:
                            st.markdown(f"**Explanation:**  \n{st.session_state['explanations'][i]}")
                    else:
                        st.info("Explanation: LLM Explainer inactive this run (see note above).")
else:
    st.info("← Set a target and click **Run Pipeline** to generate candidates.")