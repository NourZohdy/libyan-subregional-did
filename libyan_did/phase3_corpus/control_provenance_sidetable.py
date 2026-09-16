"""Emit outputs/metadata/control_provenance.csv — one row per control EPISODE with its
provenance ∈ {ADI-17, ADI-20, self-mined} (T016 / D8).

Provenance is NOT reconstructed from memory: it was recorded at harvest time in each
episode's ``source_name`` field (metadata.json). Episodes whose ``source_name`` is
``ADI-17`` or ``ADI-20`` are catalog-sourced; everything else is self-mined. Keyed on
``file_id`` so ``diagnostics.control_provenance`` can join it to the control manifests.

Run (from repo root):
    python -m libyan_did.phase3_corpus.control_provenance_sidetable
"""

from __future__ import annotations

import json

import pandas as pd

from libyan_did.shared import config

OUT = config.METADATA_DIR / "control_provenance.csv"
CATALOGS = {"ADI-17", "ADI-20"}


def main() -> None:
    rows = []
    for meta in sorted(config.CORPUS_DIR.glob("*/*/metadata.json")):
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if d.get("macro") != "Non-Libyan":
            continue
        cls = d.get("region")
        channel = d.get("playlist")
        for e in (d.get("episodes") or []):
            sn = str(e.get("source_name") or "").strip()
            rows.append({
                "class": cls,
                "channel_key": channel,
                "file_id": e.get("file_id"),
                "provenance": sn if sn in CATALOGS else "self-mined",
            })

    prov = pd.DataFrame(rows).dropna(subset=["file_id"]).drop_duplicates("file_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prov.to_csv(OUT, index=False)

    cat = int(prov["provenance"].isin(CATALOGS).sum())
    print(f"control_provenance -> {OUT}")
    print(f"  episodes : {len(prov)}")
    print(f"  by provenance: {prov['provenance'].value_counts().to_dict()}")
    print(f"  catalog-sourced: {cat} / {len(prov)} = {round(100 * cat / len(prov))}%")


if __name__ == "__main__":
    main()
