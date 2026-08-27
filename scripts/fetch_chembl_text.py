"""Pulls real bioactivity assay descriptions for a target from ChEMBL's
public API and writes them to a CSV for build_index_from_chembl_csv().

Run this once per target (needs internet):

    python3 scripts/fetch_chembl_text.py EGFR CHEMBL203
"""

import sys
import csv
import requests

def fetch_bioactivity_text(target_chembl_id: str, limit: int = 200) -> list[str]:
    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": "IC50",
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    activities = resp.json()["activities"]

    texts = []
    for a in activities:
        desc = a.get("assay_description") or a.get("description")
        if desc:
            texts.append(desc.strip())
    return list(dict.fromkeys(texts))  # dedupe, preserve order


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 fetch_chembl_text.py <TARGET_NAME> <CHEMBL_TARGET_ID>")
        print("Example: python3 fetch_chembl_text.py EGFR CHEMBL203")
        sys.exit(1)

    target_name, chembl_id = sys.argv[1], sys.argv[2]
    texts = fetch_bioactivity_text(chembl_id)
    print(f"Fetched {len(texts)} unique assay descriptions for {target_name} ({chembl_id})")

    out_path = f"data/{target_name.lower()}_bioactivity_text.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["description"])
        for t in texts:
            writer.writerow([t])
    print(f"Saved to {out_path}")