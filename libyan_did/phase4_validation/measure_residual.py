"""Cross-channel speaker residual for the channel-disjoint v2 split (T7, FR / D5).

The v2 split is channel-disjoint, NOT speaker-disjoint: the same Libyan speaker can
record on two different channels and land on opposite split sides. This measures the
single fraction of Libyan test-split speakers who also appear in the train split
(necessarily via a different channel, since channels are disjoint), and applies the
strict >5% rule (D5): >5% -> a speaker-disjoint modeling arm is recorded; <=5%
(including exactly 5%) -> the residual is bounded.

``speaker_final`` is already the corpus-wide merged identity (post link_actors), so a
same-person cross-channel occurrence shares one id and is caught directly from the split
manifest.
# ponytail: reads speaker_final directly; speaker_links.csv is what PRODUCED speaker_final,
# so re-joining it would not surface any link the merge missed. Add link-table expansion
# only if speaker_final is ever built per-channel instead of corpus-wide.

Run:
    python -m libyan_did.phase4_validation.measure_residual
"""

from __future__ import annotations

import pandas as pd

from libyan_did.shared import config

SPLITS_MANIFEST_V2 = config.MANIFEST_DIR / "splits_manifest_v2.csv"
RESIDUAL_CSV = config.TABLES_DIR / "residual.csv"
THRESHOLD = 0.05


def main() -> None:
    if not SPLITS_MANIFEST_V2.exists():
        raise SystemExit("splits_manifest_v2.csv not found — run "
                         "python -m libyan_did.phase4_validation.build_splits --mode channel-disjoint first")

    sp = pd.read_csv(SPLITS_MANIFEST_V2)
    # Libyan rows only, identified by the 3 region names (robust; no macro column needed).
    lib = sp[sp["final_dialect"].isin(config.REGIONS)]
    test_spk = set(lib[lib["split"] == "test"]["speaker_final"])
    train_spk = set(lib[lib["split"] == "train"]["speaker_final"])
    leaked = test_spk & train_spk
    fraction = len(leaked) / len(test_spk) if test_spk else 0.0

    outcome = "speaker-disjoint-arm" if fraction > THRESHOLD else "bounded"

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "metric": "libyan_test_speaker_residual",
        "value": round(fraction, 4),
        "threshold": THRESHOLD,
        "outcome": outcome,
    }]).to_csv(RESIDUAL_CSV, index=False)

    print(f"Libyan test speakers: {len(test_spk)} | also in train (cross-channel): {len(leaked)}")
    print(f"residual = {fraction:.4f}  (> {THRESHOLD} ?)  -> outcome = {outcome}")
    print(f"residual -> {RESIDUAL_CSV}")


if __name__ == "__main__":
    main()
