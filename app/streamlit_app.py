import sys
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.input_handler import resolve_target, KNOWN_TARGETS
from src.protein_pipeline import get_protein_features
from src.molecule_generator import generate
from src.validator import sanitize_and_dedupe, synthetic_accessibility, embed_3d
from src.admet_filter import admet_screen
from src.docking import DOCKING_SITES, DockingUnavailable, dock_candidates
from src.depiction import mol_to_2d_png_bytes, conformer_3d_html, protein_3d_html, mol_info
from src.llm_explainer import build_index_from_chembl_csv, explain as explain_candidate
from src.chatbot import build_run_summary, ask as ask_chatbot
from src.schemas import ProteinFeatures

st.set_page_config(page_title="MolGen Studio", layout="wide", page_icon="🧬")

# ---------------------------------------------------------------- styling --
st.markdown("""
<style>
:root {
    --mg-primary: #5B6EF5;
    --mg-primary-dark: #3D4FCF;
    --mg-accent: #17C3B2;
    --mg-bg-soft: #F5F7FF;
}
.block-container { padding-top: 1.6rem; max-width: 1300px; }

.mg-hero {
    background: linear-gradient(120deg, var(--mg-primary) 0%, var(--mg-accent) 100%);
    padding: 1.6rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.2rem;
}
.mg-hero h1 { margin: 0; font-size: 1.9rem; }
.mg-hero p { margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 0.98rem; }

.mg-badge {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.mg-badge-trained { background: #DCFCE7; color: #166534; }
.mg-badge-exploratory { background: #FEF3C7; color: #92400E; }

div[data-testid="stMetric"] {
    background: var(--mg-bg-soft);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    border: 1px solid #E4E8FF;
}
div[data-testid="stMetric"] * {
    color: #1A1D2E !important;
}
div[data-testid="stMetric"] label {
    color: #4A4F6B !important;
}

.mg-card {
    background: white;
    border: 1px solid #ECECF5;
    border-radius: 14px;
    padding: 1rem;
}

section[data-testid="stSidebar"] { background: #10142B; }
section[data-testid="stSidebar"] * { color: #EDEEF7 !important; }
section[data-testid="stSidebar"] .stButton button {
    background: var(--mg-accent); color: #06231F !important; border: none; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mg-hero">
  <h1>🧬 MolGen Studio</h1>
  <p>Computational molecule-design prototype — target structure lookup,
  molecule generation, RDKit validation, property-rule screening, optional docking,
  and AI-assisted explanations.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    st.header("⚙️ Configuration")
    preset_options = ["Custom / PDB ID"] + sorted(KNOWN_TARGETS.keys())
    preset = st.selectbox("Quick target presets", preset_options, index=1)
    if preset == "Custom / PDB ID":
        target_input = st.text_input("Protein target (name or 4-char PDB ID)", value="EGFR")
    else:
        target_input = preset
        st.caption(f"Maps to PDB `{KNOWN_TARGETS[preset]}`")

    n_candidates = st.slider("Number of candidates", 5, 50, 20)
    run_docking = st.checkbox("Run AutoDock Vina docking (experimental)", value=False)
    st.caption(
        f"Docking is available for {', '.join(DOCKING_SITES)}, each with a bound "
        "reference ligand. It may add several minutes."
    )
    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "**How generation works:** if a REINVENT4-format CSV exists "
        "for this target, its SMILES are used (training metadata is not bundled). "
        "Otherwise RDKit recombines fragments "
        "with a reproducible seed based on the target name. This creates an "
        "exploratory, untrained set; it is not conditioned on protein binding."
    )
    if not os.environ.get("GEMINI_API_KEY"):
        st.warning("GEMINI_API_KEY not set — explanations & chatbot disabled this run.", icon="⚠️")

# --- Run the pipeline ONLY when the button is freshly clicked, and stash
# everything needed to render results in session_state. Every other widget
# interaction (e.g. an "explain" button, or a chat message) triggers a
# Streamlit rerun where run_btn is False -- without session_state that
# rerun would lose all results and fall back to the empty state.
if run_btn:
    with st.spinner("Resolving target..."):
        try:
            target = resolve_target(target_input)
        except ValueError as e:
            st.error(str(e))
            st.stop()

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
        structure_msg = (
            "No sequence or PDB mapping is available for this protein name. "
            "Exploratory molecule generation can proceed, but protein structure "
            "visualization and docking are unavailable until a structure is resolved."
        )

    with st.spinner("Generating candidates..."):
        candidates = generate(features, n=n_candidates)

    with st.spinner("Validating (RDKit sanitize, dedupe, 3D embed)..."):
        candidates = sanitize_and_dedupe(candidates)
        candidates = synthetic_accessibility(candidates)
        candidates = embed_3d(candidates)

    with st.spinner("Screening local Lipinski/Veber property rules..."):
        candidates = admet_screen(candidates, use_live_api=False)

    docking_msg = None
    if run_docking:
        with st.spinner("Preparing the reference pocket and docking candidates with AutoDock Vina..."):
            try:
                candidates = dock_candidates(candidates, target)
                n_docked = sum("vina_score_kcal_mol" in c.admet for c in candidates)
                docking_msg = (
                    f"AutoDock Vina returned scores for {n_docked}/{len(candidates)} candidates. "
                    "These are approximate docking scores, not measured binding affinities."
                )
            except DockingUnavailable as e:
                docking_msg = f"Docking unavailable: {e}"
            except Exception as e:
                docking_msg = f"Docking step failed: {e}"

    explainer_index = None
    explainer_error = None
    bioactivity_csv = f"data/{target.raw_input.strip().lower()}_bioactivity_text.csv"
    if not os.environ.get("GEMINI_API_KEY"):
        explainer_error = (
            "GEMINI_API_KEY is not set in this environment, so explanations "
            "and the chatbot are disabled. Add it to a `.env` file in the "
            "project root or `export GEMINI_API_KEY=...` before launching Streamlit."
        )
    elif not Path(bioactivity_csv).exists():
        chembl_ids = {
            "EGFR": "CHEMBL203", "BRAF": "CHEMBL5145", "ACE2": "CHEMBL3736",
            "DRD2": "CHEMBL217", "JAK2": "CHEMBL2971",
        }
        chembl_hint = chembl_ids.get(target.raw_input.strip().upper(), "<CHEMBL_TARGET_ID>")
        explainer_error = (
            f"GEMINI_API_KEY is set, but there's no bioactivity text corpus for "
            f"'{target.raw_input}' yet (expected `{bioactivity_csv}`). Run "
            f"`python scripts/fetch_chembl_text.py {target.raw_input} {chembl_hint}` "
            f"to build it — explanations currently only work out-of-the-box for EGFR."
        )
    else:
        try:
            explainer_index = build_index_from_chembl_csv(bioactivity_csv)
        except Exception as e:
            explainer_error = f"Could not build retrieval index: {e}"

    # stash everything the results section needs -- this is what survives
    # across reruns triggered by the per-candidate "explain" buttons / chat
    st.session_state["target"] = target
    st.session_state["features"] = features
    st.session_state["structure_msg"] = structure_msg
    st.session_state["docking_msg"] = docking_msg
    st.session_state["candidates"] = candidates
    st.session_state["explainer_index"] = explainer_index
    st.session_state["explainer_error"] = explainer_error
    st.session_state["explanations"] = {}
    st.session_state["chat_history"] = []  # reset chat context for the new run

# --------------------------------------------------------- render results --
if "candidates" in st.session_state:
    target = st.session_state["target"]
    features = st.session_state["features"]
    candidates = st.session_state["candidates"]
    explainer_index = st.session_state["explainer_index"]
    explainer_error = st.session_state["explainer_error"]

    tier_badge = ""
    if candidates:
        if "reinvent4" in candidates[0].source:
            tier_badge = '<span class="mg-badge mg-badge-trained">REINVENT4 CSV (training metadata not bundled)</span>'
        else:
            tier_badge = '<span class="mg-badge mg-badge-exploratory">Exploratory (BRICS, untrained)</span>'

    st.markdown(
        f"**Target resolved:** `{target.raw_input}` → PDB `{target.pdb_id or 'unresolved'}` {tier_badge}",
        unsafe_allow_html=True,
    )
    if st.session_state["structure_msg"]:
        st.info(st.session_state["structure_msg"])
    if st.session_state.get("docking_msg"):
        if "unavailable" in st.session_state["docking_msg"].lower() or "failed" in st.session_state["docking_msg"].lower():
            st.warning(st.session_state["docking_msg"])
        else:
            st.info(st.session_state["docking_msg"])

    tab_overview, tab_candidates, tab_chat = st.tabs(
        ["📊 Overview", "🧪 Candidates", "💬 Ask MolGen"]
    )

    # ---------------------------------------------------------- Overview --
    with tab_overview:
        if features.structure_path:
            st.subheader(f"Target structure — {target.pdb_id}")
            try:
                html = protein_3d_html(features.structure_path)
                components.html(html, height=610)
            except Exception as e:
                st.warning(f"Could not render protein structure: {e}")

        if not candidates:
            st.warning("No candidates survived filtering.")
        else:
            df = pd.DataFrame([{
                "SMILES": c.smiles,
                "QED": c.qed,
                "SA Score (RDKit)": c.sa_score,
                "MW": c.admet.get("mw"),
                "LogP": c.admet.get("logp"),
                "H-Donors": c.admet.get("h_donors"),
                "H-Acceptors": c.admet.get("h_acceptors"),
                "Rotatable bonds": c.admet.get("rotatable_bonds"),
                "TPSA": c.admet.get("tpsa"),
                "Lipinski violations": c.admet.get("lipinski_violations"),
                "Veber violations": c.admet.get("veber_violations"),
                "Vina score (kcal/mol)": c.admet.get("vina_score_kcal_mol"),
                "Source": c.source,
            } for c in candidates])

            avg_qed = df["QED"].mean()
            avg_sa = df["SA Score (RDKit)"].mean()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Candidates passing filters", len(candidates))
            col2.metric("Avg QED (drug-likeness)", f"{avg_qed:.3f}")
            col3.metric("Avg SA score (RDKit; lower = easier)", f"{avg_sa:.2f}")
            n_docked = sum("vina_score_kcal_mol" in c.admet for c in candidates)
            col4.metric("Candidates docked" if run_docking else "Passed local rule screen", n_docked if run_docking else len(candidates))

            st.markdown("##### QED distribution")
            st.bar_chart(df.set_index("SMILES")["QED"])

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Molecular weight vs LogP")
                st.scatter_chart(df, x="MW", y="LogP")
            with c2:
                st.markdown("##### SA score by candidate")
                st.bar_chart(df.set_index("SMILES")["SA Score (RDKit)"])

            with st.expander("Full candidate data table"):
                st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download candidate results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"molgen_{target.raw_input.strip().lower().replace(' ', '_')}_candidates.csv",
                mime="text/csv",
            )

    # --------------------------------------------------------- Candidates --
    with tab_candidates:
        if not candidates:
            st.warning("No candidates survived filtering.")
        else:
            if any("vina_score_kcal_mol" in c.admet for c in candidates):
                sorted_candidates = sorted(
                    candidates,
                    key=lambda c: (
                        c.admet.get("vina_score_kcal_mol", float("inf")),
                        -(c.qed or 0),
                    ),
                )
                st.caption("Sorted by AutoDock Vina score (lower is more favorable); undocked candidates follow, then QED.")
            else:
                sorted_candidates = sorted(candidates, key=lambda c: c.qed or 0, reverse=True)
            for i, c in enumerate(sorted_candidates):
                with st.expander(f"Candidate {i + 1} — QED {c.qed:.3f}  |  {c.smiles[:40]}{'...' if len(c.smiles) > 40 else ''}"):
                    info = mol_info(c.smiles)
                    info_line = (
                        f"**Formula:** {info.get('formula','n/a')}  |  "
                        f"**MW:** {info.get('mw','n/a')}  |  "
                        f"**Heavy atoms:** {info.get('heavy_atoms','n/a')}  |  "
                        f"**Rings:** {info.get('rings','n/a')}  |  "
                        f"**Rotatable bonds:** {info.get('rotatable_bonds','n/a')}  |  "
                        f"**H-donors/acceptors:** {info.get('h_bond_donors','n/a')}/{info.get('h_bond_acceptors','n/a')}"
                    )
                    st.markdown(info_line)
                    st.caption(f"Source: `{c.source}`  |  SA score: {c.sa_score}  |  ADMET: MW {c.admet.get('mw','n/a')}, LogP {c.admet.get('logp','n/a')}")

                    sub_2d, sub_3d, sub_pose, sub_details = st.tabs([
                        "🔬 2D structure", "🌀 3D conformer", "🎯 Vina pose", "📋 Details / Explain"
                    ])

                    with sub_2d:
                        try:
                            png_bytes = mol_to_2d_png_bytes(c.smiles, size=560)
                            st.image(png_bytes, width=560)
                        except Exception as e:
                            st.warning(f"2D render failed: {e}")

                    with sub_3d:
                        show_labels = st.checkbox("Show atom element labels", key=f"labels_{i}")
                        if c.conformer_path:
                            try:
                                html3d = conformer_3d_html(c.conformer_path, width=560, height=560, show_labels=show_labels)
                                components.html(html3d, height=580)
                            except Exception as e:
                                st.warning(f"3D render failed: {e}")
                        else:
                            st.info("No 3D conformer available.")

                    with sub_pose:
                        pose_sdf = c.admet.get("docked_pose_sdf")
                        if pose_sdf:
                            st.caption(
                                f"Site: {c.admet.get('docking_site')} · "
                                f"Vina score: {c.admet.get('vina_score_kcal_mol')} kcal/mol"
                            )
                            try:
                                html_pose = conformer_3d_html(pose_sdf, width=560, height=560)
                                components.html(html_pose, height=580)
                            except Exception as e:
                                st.warning(f"Docked pose render failed: {e}")
                        elif c.admet.get("docking_error"):
                            st.warning(c.admet["docking_error"])
                        else:
                            st.info("Enable AutoDock Vina in the sidebar and rerun to generate a docked pose.")

                    with sub_details:
                        st.code(c.smiles, language=None)
                        st.write(f"**Source:** `{c.source}`")
                        st.write(f"**QED (drug-likeness, 0-1):** {c.qed}")
                        st.write(f"**SA Score (RDKit fragment model, 1=easier to 10=harder):** {c.sa_score}")
                        if "vina_score_kcal_mol" in c.admet:
                            st.write(
                                f"**AutoDock Vina score (approximate; kcal/mol):** "
                                f"{c.admet['vina_score_kcal_mol']} · site {c.admet.get('docking_site')}"
                            )
                        st.write(f"**MW:** {c.admet.get('mw', 'n/a')}  |  **LogP:** {c.admet.get('logp', 'n/a')}")
                        st.write(f"**H-Donors:** {c.admet.get('h_donors', 'n/a')}  |  **H-Acceptors:** {c.admet.get('h_acceptors', 'n/a')}")
                        st.write(
                            f"**TPSA:** {c.admet.get('tpsa', 'n/a')} Å²  |  "
                            f"**Rotatable bonds:** {c.admet.get('rotatable_bonds', 'n/a')}  |  "
                            f"**Lipinski / Veber violations:** "
                            f"{c.admet.get('lipinski_violations', 'n/a')} / "
                            f"{c.admet.get('veber_violations', 'n/a')}"
                        )

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
                            st.info(f"Explanation unavailable: {explainer_error}")

    # ---------------------------------------------------------------- Chat --
    with tab_chat:
        st.caption(
            "Ask about this run — e.g. \"why is candidate 3 flagged?\", "
            "\"what does QED mean?\", \"which molecule is most drug-like and why?\". "
            "Answers are grounded in this run's actual data, and remember earlier "
            "questions in this session."
        )
        if not os.environ.get("GEMINI_API_KEY"):
            st.info("Set GEMINI_API_KEY to enable the chatbot.")
        else:
            for turn in st.session_state.get("chat_history", []):
                with st.chat_message("user" if turn["role"] == "user" else "assistant"):
                    st.markdown(turn["text"])

            user_q = st.chat_input("Ask a question about this run...")
            if user_q:
                st.session_state["chat_history"].append({"role": "user", "text": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            summary = build_run_summary(target.raw_input, target.pdb_id, candidates)
                            answer = ask_chatbot(
                                user_q, summary, st.session_state["chat_history"][:-1]
                            )
                        except Exception as e:
                            answer = f"Chatbot error: {e}"
                    st.markdown(answer)
                st.session_state["chat_history"].append({"role": "model", "text": answer})

else:
    st.info("← Set a target and click **Run Pipeline** to generate candidates.")
