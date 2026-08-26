import streamlit as st
import sys
from pathlib import Path
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.input_handler import resolve_target
from src.protein_pipeline import get_protein_features
from src.molecule_generator import generate
from src.validator import sanitize_and_dedupe, synthetic_accessibility, embed_3d
from src.admet_filter import admet_screen
from src.depiction import mol_to_2d_png_bytes, conformer_3d_html, protein_3d_html
from src.schemas import ProteinFeatures

st.set_page_config(page_title="MolGen Studio", layout="wide", page_icon="🧬")

st.title("🧬 MolGen Studio")
st.caption("Generative AI Pipeline for Novel Drug Molecule Design — EGFR-conditioned demo")

with st.sidebar:
    st.header("Target")
    target_input = st.text_input("Protein target (name or PDB ID)", value="EGFR")
    n_candidates = st.slider("Number of candidates", 5, 50, 20)
    run_btn = st.button("Run Pipeline", type="primary")

    st.divider()
    st.subheader("Module status")
    st.markdown("""
    - ✅ Input Handler
    - ✅ Protein Pipeline
    - ✅ Molecule Generator *(trained on EGFR)*
    - ✅ Validator
    - ✅ ADMET Filter *(local pre-filter — live ADMETlab API has a confirmed server-side bug)*
    - ⏳ LLM Explainer *(not yet connected)*
    """)

if run_btn:
    with st.spinner("Resolving target..."):
        target = resolve_target(target_input)
    st.success(f"Target resolved: **{target.raw_input}** → PDB `{target.pdb_id or 'unresolved'}`")

    features = ProteinFeatures(target=target)
    if target.is_resolved_structure:
        with st.spinner("Fetching protein structure from RCSB PDB..."):
            try:
                features = get_protein_features(target)
                st.success(f"Structure fetched: `{features.structure_path}`")
            except Exception as e:
                st.warning(f"Structure fetch failed ({e}); continuing without it.")
    else:
        st.info("No resolved PDB structure for this target. ESMFold prediction requires a BioNeMo API key (not configured) — skipping structure step.")

    # --- Protein 3D viewer ---
    if features.structure_path:
        st.subheader(f"Target structure — {target.pdb_id}")
        try:
            html = protein_3d_html(features.structure_path)
            components.html(html, height=470)
        except Exception as e:
            st.warning(f"Could not render protein structure: {e}")

    with st.spinner("Generating candidates..."):
        candidates = generate(features, n=n_candidates)
    st.success(f"Generated {len(candidates)} candidates (source: `{candidates[0].source if candidates else 'n/a'}`)")

    with st.spinner("Validating (RDKit sanitize, dedupe, 3D embed)..."):
        candidates = sanitize_and_dedupe(candidates)
        candidates = synthetic_accessibility(candidates)
        candidates = embed_3d(candidates)
    st.success(f"{len(candidates)} valid candidates with 3D structures")

    with st.spinner("Screening ADMET (local Lipinski/Veber pre-filter)..."):
        candidates = admet_screen(candidates, use_live_api=False)
    st.success(f"{len(candidates)} candidates passed ADMET pre-filter")

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
            # NOTE: never put raw SMILES in an expander/label -- Streamlit's
            # markdown parser reads "[N+](=O)" as link syntax [text](url)
            # and mangles it. SMILES only ever goes in st.code() or st.image().
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
                    st.info("Explanation: not yet connected (LLM Explainer module pending).")
else:
    st.info("← Set a target and click **Run Pipeline** to generate candidates.")