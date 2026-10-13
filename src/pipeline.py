from src import input_handler, protein_pipeline, molecule_generator, validator, admet_filter
from src.schemas import Candidate, ProteinFeatures


def run(
    raw_target: str,
    n_candidates: int = 50,
    use_live_admet: bool = False,
    dock: bool = False,
) -> list[Candidate]:
    target = input_handler.resolve_target(raw_target)
    # Unknown names have no sequence source yet, but can still use the clearly
    # labeled exploratory generator. Known PDB targets get their structure.
    if target.is_resolved_structure:
        features = protein_pipeline.get_protein_features(target)
    else:
        features = ProteinFeatures(target=target)

    candidates = molecule_generator.generate(features, n=n_candidates)
    candidates = validator.sanitize_and_dedupe(candidates)
    candidates = validator.synthetic_accessibility(candidates)
    candidates = validator.embed_3d(candidates)

    candidates = admet_filter.admet_screen(candidates, use_live_api=use_live_admet)
    if dock:
        from src.docking import dock_candidates
        candidates = dock_candidates(candidates, target)

    return candidates


if __name__ == "__main__":
    import sys
    results = run(sys.argv[1] if len(sys.argv) > 1 else "EGFR")
    for c in results:
        print(c.smiles, c.qed, c.sa_score)
