import streamlit as st
from src.pipeline import run

st.set_page_config(page_title="MolGen Studio", layout="wide")
st.title("MolGen Studio")

target_input = st.text_input("Target protein (PDB ID or name)", value="EGFR")

if st.button("Generate candidates"):
    with st.spinner("Running pipeline..."):
        candidates = run(target_input)

    if not candidates:
        st.warning("No candidates survived filtering.")
    for c in candidates:
        with st.expander(c.smiles):
            st.write(f"QED: {c.qed}  |  SA score: {c.sa_score}")
            st.write(f"ADMET: {c.admet}")
            if c.explanation:
                st.write(c.explanation)
