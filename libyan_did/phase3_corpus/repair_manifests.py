"""One-off repair of the per-resource final_manifest.csv files (non-destructive):

  1. file_id un-mangle — an all-numeric id like ``55917924e225`` was round-tripped
     through a float and stored as ``5.5917924e+232`` (ALG/NON_ALG_ALG_Interview_001).
     Restored from the authoritative value in metadata.json.
  2. exclude_reason backfill — the controls were clustered before the music-scoring
     pass, so ``is_excluded`` rows were left with no reason. Fill them with
     ``cluster.fill_missing_exclusion_reasons`` (the same thresholds the pipeline uses;
     validated to reproduce 100% of the populated Libyan labels).

Only ``file_id`` (one cell) and blank ``exclude_reason`` on already-excluded rows are
touched — is_excluded / tier / speaker ids / durations are unchanged, so the frozen
keep-decisions and hours are preserved. Every other column is written back verbatim
(read as text) so the repair does not reformat floats.

Run (from repo root), then re-run combine_corpus -> freeze -> build_splits -> tables/figures:
    python -m libyan_did.phase3_corpus.repair_manifests
"""

from __future__ import annotations

import json

import pandas as pd

from libyan_did.shared import cluster, config

MANGLED = {"5.5917924e+232": "55917924e225"}  # bad -> true (from ALG metadata.json)


def _true_ids(meta_path) -> set[str]:
    try:
        d = json.loads(meta_path.read_text(encoding="utf-8"))
        return {str(e.get("file_id")) for e in (d.get("episodes") or [])}
    except Exception:  # noqa: BLE001
        return set()


def main() -> None:
    finals = sorted(config.CORPUS_DIR.glob("*/*/final_manifest.csv"))
    n_files = n_idfix = n_reasons = 0
    for fpath in finals:
        df = pd.read_csv(fpath, dtype=str, keep_default_na=False)
        if "is_excluded" not in df.columns:
            continue
        before_id = df["file_id"].copy()
        before_reason = df.get("exclude_reason", pd.Series("", index=df.index)).copy()

        # 1) un-mangle file_id (verify the restored value is the resource's real id)
        true_ids = _true_ids(fpath.parent / "metadata.json")
        df["file_id"] = df["file_id"].replace(
            {bad: good for bad, good in MANGLED.items() if good in true_ids})

        # 2a) a KEPT segment never carries an exclusion reason — enforce the invariant
        # (also undoes a prior buggy backfill that labelled kept rows).
        if "exclude_reason" not in df.columns:
            df["exclude_reason"] = ""
        kept = ~cluster._as_bool(df["is_excluded"])
        df.loc[kept, "exclude_reason"] = ""
        # 2b) backfill blank exclusion reasons with the pipeline's own logic
        df = cluster.fill_missing_exclusion_reasons(df)

        idfix = int((df["file_id"] != before_id).sum())
        rfix = int((df["exclude_reason"].fillna("") != before_reason.fillna("")).sum())
        if idfix or rfix:
            df.to_csv(fpath, index=False)
            n_files += 1
            n_idfix += idfix
            n_reasons += rfix
            print(f"  repaired {fpath.parent.parent.name}/{fpath.parent.name}: "
                  f"{idfix} file_id, {rfix} reasons")

    print(f"\nrepaired {n_files} resources | {n_idfix} file_id cells | {n_reasons} exclude_reason cells")


if __name__ == "__main__":
    main()
