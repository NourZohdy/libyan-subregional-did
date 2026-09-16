"""Paper-ready corpus tables from the 21-class v2 corpus (+ v2 channel-disjoint splits).

Builds the combined 21-class kept frame = Libyan verified kept rows
(``verified_manifest.csv``) + Non-Libyan control kept rows (per-resource control
manifests via ``combine.combine_resources``), then writes the Chapter-3 table set
(TB1-TB6) to outputs/tables/ keyed on ``speaker_final`` / ``final_dialect``.

Splits come from ``splits_manifest_v2.csv`` (the channel-disjoint v2 split), never v1.

Run (diarization env, from repo root):
    python -m libyan_did.phase4_validation.make_corpus_tables
"""

from __future__ import annotations

import pandas as pd

from libyan_did.shared import config
from libyan_did.shared import diagnostics
# Reuse the exact combined-frame loaders the v2 split uses (T010) so tables and
# splits see the identical 21-class kept frame.
from libyan_did.phase4_validation.build_splits import _libyan_kept, _control_kept

SPLITS_MANIFEST_V2 = config.MANIFEST_DIR / "splits_manifest_v2.csv"
PROVENANCE_SIDE_TABLE = config.METADATA_DIR / "control_provenance.csv"
FR010_DOC = config.PROJECT_ROOT / "docs" / "CONTROL_LABEL_PROVENANCE.md"


def _combined_kept() -> pd.DataFrame:
    """Libyan verified kept + Non-Libyan control kept, column-aligned (T010 shape)."""
    lib = _libyan_kept()
    ctl = _control_kept()
    return pd.concat([lib, ctl], ignore_index=True)


def _audit_neff_cv(kept: pd.DataFrame) -> pd.DataFrame:
    """Per class: channel-level effective-N and coefficient of variation (TB6, T8 audit).

    ``w`` = per-channel hours within the class; cv = std/mean of that vector,
    n_eff = (sum w)^2 / sum(w^2) (Kish effective sample size on channel mass).
    """
    dcol = diagnostics._dialect_key(kept)
    rows = []
    for cls, g in kept.groupby(dcol):
        w = g.groupby("source")["duration"].sum() / 3600.0
        w = w.to_numpy()
        mean = w.mean()
        cv = float(w.std() / mean) if mean else 0.0
        n_eff = float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() else 0.0
        rows.append({"class": cls, "n_eff": round(n_eff, 3), "cv": round(cv, 4)})
    return pd.DataFrame(rows).sort_values("class").reset_index(drop=True)


def _write_fr010_doc(kept: pd.DataFrame) -> None:
    """FR-010 control-label-provenance justification (standalone-thesis clean)."""
    # (b) magnitude: control segments whose class is one of the 18 controls — the pool
    # in which an untreated Libyan-in-control mislabel could sit. The precise
    # plausibly-Libyan count is a modeling-phase number (needs the classifier); here we
    # report the control-segment pool size as the denominator for that bound.
    dcol = diagnostics._dialect_key(kept)
    ctl_segs = int(kept[dcol].isin(config.CONTROL_CODES).sum())

    # (c) provenance share: read the real catalog fraction from the T016 side-table when it
    # exists (no hand-typed estimate); fall back to a qualitative statement otherwise.
    if PROVENANCE_SIDE_TABLE.exists():
        _p = pd.read_csv(PROVENANCE_SIDE_TABLE)
        _cat = int(_p["provenance"].isin(["ADI-17", "ADI-20"]).sum())
        prov_share = f"{round(100 * _cat / len(_p))}% ({_cat} of {len(_p)} control episodes)"
    else:
        prov_share = "a majority of control episodes (see the control-provenance table)"

    text = f"""# Control label provenance (FR-010)

## Direction of harm

No reverse-Libyan contamination screen is applied to the non-Libyan control classes.
Any Libyan speech that leaks into a control class can only *depress* measured Libyan
performance: a Libyan segment mislabelled as a control adds a hard negative to the
control class and removes signal from the Libyan class, never the reverse. Left
untreated, this leakage is therefore a **conservative lower bound** on Libyan
results, not an inflation — so screening the controls is unnecessary for an honest
Libyan-side claim.

## Magnitude

The control pool against which this bound is taken contains {ctl_segs:,} kept control
segments across the 18 non-Libyan classes. The share of these that could plausibly be
Libyan is finalised at the modelling phase (it requires the dialect classifier's own
confusion output); until then only the pool size above is asserted, and the
direction-of-harm argument makes any residual mislabel conservative.

## Provenance

{prov_share} are catalog-sourced (ADI-17 / ADI-20), the remainder self-mined — this
share is read directly from each episode's recorded ``source_name``, not estimated; the
per-class breakdown is in the control-provenance table. Because a
majority of control labels inherit their dialect assignment from the source catalog
rather than from an independent per-segment check, the 18-way confusion analysis is
reported at **provenance-label confidence only**.
"""
    FR010_DOC.parent.mkdir(parents=True, exist_ok=True)
    FR010_DOC.write_text(text, encoding="utf-8")


def main() -> None:
    kept = _combined_kept()
    tdir = config.TABLES_DIR
    tdir.mkdir(parents=True, exist_ok=True)

    written = {
        "per_dialect_summary": diagnostics.per_dialect_summary(kept),
        "top_source_share": diagnostics.top_source_share(kept),
        "quality_counts": diagnostics.quality_counts(kept),
        "audit_neff_cv": _audit_neff_cv(kept),
    }

    if SPLITS_MANIFEST_V2.exists():
        sp = pd.read_csv(SPLITS_MANIFEST_V2)
        written["split_summary"] = diagnostics.split_summary(sp)
    else:
        print(f"[tables] WARNING: {SPLITS_MANIFEST_V2.name} missing — run build_splits "
              "--mode channel-disjoint first; split_summary skipped.")

    # TB4 control provenance — needs the T016 episode-level side-table.
    if PROVENANCE_SIDE_TABLE.exists():
        prov = pd.read_csv(PROVENANCE_SIDE_TABLE)
        written["control_provenance"] = diagnostics.control_provenance(kept, prov)
    else:
        print(f"[tables] WARNING: {PROVENANCE_SIDE_TABLE} missing (T016 blocked) — "
              "control_provenance table (TB4) skipped.")

    for name, table in written.items():
        path = tdir / f"{name}.csv"
        table.to_csv(path, index=False)

    _write_fr010_doc(kept)

    tot_h = kept["duration"].sum() / 3600.0
    n_classes = kept[diagnostics._dialect_key(kept)].nunique()
    print(f"combined corpus: {len(kept):,} segments | {tot_h:.2f} h | {n_classes} classes\n")
    print("== per-dialect summary ==")
    print(diagnostics.per_dialect_summary(kept).to_string(index=False))
    print("\ntables written:")
    for name in written:
        print(f"  {name:24s} -> {tdir / (name + '.csv')}")
    print(f"  {'FR-010 doc':24s} -> {FR010_DOC}")


if __name__ == "__main__":
    main()
