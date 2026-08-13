from src import input_handler, protein_pipeline, molecule_generator, validator, admet_filter
from src.schemas import Candidate


def run(raw_target: str, n_candidates: int = 50, reinvent_config: str | None = None) -> list[Candidate]:
    target = input_handler.resolve_target(raw_target)
    features = protein_pipeline.get_protein_features(target)

    candidates = molecule_generator.generate(features, n=n_candidates, reinvent_config=reinvent_config)
    candidates = validator.sanitize_and_dedupe(candidates)
    candidates = validator.synthetic_accessibility(candidates)
    candidates = validator.embed_3d(candidates)

    candidates = admet_filter.lipinski_prefilter(candidates)
    candidates = admet_filter.admet_screen(candidates)

    return candidates


if __name__ == "__main__":
    import sys
    results = run(sys.argv[1] if len(sys.argv) > 1 else "EGFR")
    for c in results:
        print(c.smiles, c.qed, c.sa_score)
