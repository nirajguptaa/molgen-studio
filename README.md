# MolGen Studio

MolGen Studio is a Streamlit prototype for exploring a computational molecule-design workflow. It resolves a small set of protein targets to PDB structures, generates candidate SMILES, validates and scores them with RDKit, applies a local drug-likeness filter, and displays 2D/3D structures. Optional Gemini explanations and chat use target-specific ChEMBL assay-description files.

## Run locally

From the project root, activate the existing virtual environment and start Streamlit:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

Or run without activating it:

```powershell
.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). The five configured targets and previously downloaded PDB structures work without a new PDB download. Entering another PDB ID requires network access the first time. Unknown protein names do not resolve to a sequence or structure; the app can still demonstrate its exploratory molecule-generation path for them.

## Candidate generation tiers

- **REINVENT4 CSV:** `data/<target>_generated.csv` is used when present and contains a `SMILES` column. The checked-in EGFR file has a REINVENT4-style output schema, but model weights and training metadata are not bundled, so its training run and target conditioning cannot be independently confirmed from this project.
- **Exploratory BRICS:** otherwise, RDKit recombines fragments from a small curated set of seed molecules. The target name seeds this process so runs are reproducible and targets can receive different candidate sets. This is not protein-conditioned generation and does not estimate binding.

## Scoring and limitations

RDKit computes QED, molecular descriptors, 3D conformers, and the Ertl/Schuffenhauer fragment-based synthetic-accessibility (SA) score. The default local screen applies Lipinski and Veber physicochemical rules; it is not a toxicity or pharmacokinetics assessment. An optional ADMETlab API path remains best-effort because the service reports instability. No live ADMET prediction is enabled by default.

Optional AutoDock Vina docking is available for PDB structures with an observed co-crystal ligand site (1M17, 1UWH, 4C61, and 6CM4). It reports approximate Vina scoring-function values and poses, not experimental binding affinities. The selected ACE2 structure (1R42) has no small-molecule inhibitor to anchor a site, so it is not enabled for this docking path. Vina docking has target-specific accuracy limits and should be benchmarked by redocking known ligands before using results for prioritization. Outputs are computational examples and require experimental validation; they are not drug or clinical recommendations.

The optional explanation and chat features require `GEMINI_API_KEY` and a target-specific `data/<target>_bioactivity_text.csv`. The retriever in `src/llm_explainer.py` currently uses TF-IDF with FAISS, not sentence embeddings. Keep API keys in the ignored `.env` file and do not commit them.

### Enable Vina on Windows

Install the Python dependencies, then fetch the official AutoDock Vina 1.2.7 Windows executable:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\setup_vina.ps1
```

The installer stores the executable under the ignored `tools/vina/` directory. The docking option is experimental and can take several minutes for a batch.

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

## Main modules

- `app/streamlit_app.py` — interactive interface and result views
- `src/input_handler.py` — target-name/PDB-ID resolution
- `src/protein_pipeline.py` — PDB retrieval with local cache reuse
- `src/molecule_generator.py` — trained-set lookup and exploratory BRICS generation
- `src/validator.py` — sanitization, deduplication, QED, SA score, and 3D conformers
- `src/admet_filter.py` — local rule filter and optional live API path
- `src/docking.py` — optional Vina docking for targets with co-crystal ligand sites
- `src/llm_explainer.py`, `src/chatbot.py` — optional Gemini features
- `src/pipeline.py` — non-UI pipeline entry point (live ADMET disabled by default)
