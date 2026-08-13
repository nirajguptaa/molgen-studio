# MolGen Studio — Module Status

Run tests: `PYTHONPATH=. pytest tests/ -v`

| Module | File | Status |
|---|---|---|
| Data contracts | `src/schemas.py` | Done |
| Input Handler | `src/input_handler.py` | **Working** — tested, no external deps |
| Validator (RDKit) | `src/validator.py` | **Working** — tested locally, real RDKit calls |
| Molecule Generator | `src/molecule_generator.py` | Stub — fallback seed list; REINVENT4 call needs GPU install, follow their repo's `examples/` to build a TOML config, then pass `reinvent_config=` |
| Protein Pipeline | `src/protein_pipeline.py` | Stub — PDB fetch works (public REST API); ESMFold needs `NVIDIA_BIONEMO_API_KEY`; ProtTrans embedding needs `transformers`+`torch`, ~11GB model download |
| ADMET Filter | `src/admet_filter.py` | Lipinski pre-filter works locally (RDKit); ADMETlab API call needs their exact endpoint confirmed from docs, untested |
| LLM Explainer | `src/llm_explainer.py` | Needs `ANTHROPIC_API_KEY`; FAISS index needs a real ChEMBL text corpus fed into `build_index()` |
| UI | `app/streamlit_app.py` | Skeleton, wired to `pipeline.run()`, not yet run against a live pipeline |

## Next steps, in priority order
1. Get REINVENT4 running locally/Colab with a target-conditioned TOML config — this is the highest-risk item (see risk register).
2. Confirm ADMETlab 3.0's actual batch endpoint + auth from their docs; the URL in `admet_filter.py` is a placeholder.
3. Get a BioNeMo API key or fall back to `ESMFold` open-source weights if the key is delayed.
4. Build the FAISS index from a real ChEMBL bioactivity text pull, not dummy strings.
5. Only then wire `pipeline.py`'s stub calls into the Streamlit UI and test end-to-end on one real target (EGFR is the best first pick — PDB structure already resolved, no ESMFold dependency needed).
