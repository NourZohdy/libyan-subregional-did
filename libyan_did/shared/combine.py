"""Combine every per-resource manifest + metadata into the corpus-level files.

Run AFTER all resources are processed (``scripts/run_resources.py``) and BEFORE the
splits step. Adds ``region`` / ``playlist`` / ``actor_uid`` to every row, where

    actor_uid = "<region>/<playlist>#<global_actor>"

is a corpus-unique speaker key (per-playlist ``global_actor`` IDs restart at 0, so they
collide across playlists — ``actor_uid`` does not). When the cross-playlist linking
pass has run (``scripts/link_actors.py``), a ``speaker_global`` column maps actors of
the same human across playlists to one canonical ID; the splits step must group on
``speaker_global`` (it falls back to actor_uid when linking was not run) so speakers
stay disjoint across the whole corpus.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from libyan_did.shared import config

_TRUE = {"true", "1", "1.0", "yes"}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(_TRUE)


def _read_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def temporal_coverage(metas: list[dict]) -> dict:
    """Upload-date span of the corpus, from per-resource episode metadata.

    Scans every resource's ``episodes[].upload_date`` (``YYYYMMDD``), ignoring blanks /
    ``"unknown"``. Returns ``{n_dated, n_total, date_min, date_max, by_year:{year:count}}`` —
    the basis for the release statement *"collected from videos uploaded between X and Y"*.
    Episodes still missing a date are recoverable via ``phase2_harvest.backfill_ids``.
    """
    dates, n_total = [], 0
    for m in metas:
        for e in (m.get("episodes") or []):
            n_total += 1
            d = str(e.get("upload_date") or "").strip()
            if len(d) == 8 and d.isdigit():
                dates.append(d)
    by_year: dict[str, int] = {}
    for d in dates:
        by_year[d[:4]] = by_year.get(d[:4], 0) + 1
    return {"n_dated": len(dates), "n_total": n_total,
            "date_min": min(dates) if dates else "", "date_max": max(dates) if dates else "",
            "by_year": dict(sorted(by_year.items()))}


def combine_resources(resources_dir: Path | None = None) -> pd.DataFrame:
    """Concatenate all ``corpus/<region>/<playlist>/final_manifest.csv`` into
    ``corpus_manifest.csv`` and aggregate metadata into ``corpus_metadata.json``."""
    resources_dir = resources_dir or config.CORPUS_DIR
    finals = sorted(resources_dir.glob("*/*/final_manifest.csv"))
    if not finals:
        raise FileNotFoundError(
            f"no per-resource final_manifest.csv under {resources_dir} — run run_resources.py first")

    frames, metas = [], []
    for fpath in finals:
        region = fpath.parent.parent.name
        playlist = fpath.parent.name
        # file_id as str: an all-numeric id like "55917924e225" is otherwise parsed as
        # float (5.59e+232) and permanently corrupted on the next round-trip.
        df = pd.read_csv(fpath, dtype={"file_id": str})
        if len(df) == 0:
            continue
        df["is_excluded"] = _as_bool(df["is_excluded"])
        df["region"] = region
        df["playlist"] = playlist
        df["actor_uid"] = region + "/" + playlist + "#" + df["global_actor"].astype(str)
        frames.append(df)
        metas.append(_read_json(fpath.parent / "metadata.json"))

    corpus = pd.concat(frames, ignore_index=True)

    # Canonical corpus-wide speaker id from the cross-playlist linking pass
    # (scripts/link_actors.py). Falls back to actor_uid when no linking was run.
    links_csv = config.METADATA_DIR / "speaker_links.csv"
    if links_csv.exists():
        links = pd.read_csv(links_csv)
        uid_to_global = dict(zip(links["actor_uid"], links["speaker_global"]))
        corpus["speaker_global"] = corpus["actor_uid"].map(uid_to_global)
        corpus["speaker_global"] = corpus["speaker_global"].fillna(corpus["actor_uid"])
    else:
        corpus["speaker_global"] = corpus["actor_uid"]

    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(config.CORPUS_MANIFEST, index=False)

    kept = corpus[(~corpus["is_excluded"]) & (corpus["tier"].isin(config.KEEP_TIERS))]
    per_region = []
    for region, g in kept.groupby("region"):
        per_region.append({
            "region": region,
            "playlists": int(g["playlist"].nunique()),
            "episodes": int(g["file_id"].nunique()),
            "speakers": int(g["speaker_global"].nunique()),
            "actors": int(g["actor_uid"].nunique()),
            "segments": int(len(g)),
            "minutes": round(g["duration"].sum() / 60.0, 2),
        })

    corpus_meta = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "n_resources": len(metas),
        "totals": {
            "playlists": int(kept["playlist"].nunique()),
            "episodes": int(kept["file_id"].nunique()),
            "speakers": int(kept["speaker_global"].nunique()),
            "actors": int(kept["actor_uid"].nunique()),
            "segments_kept": int(len(kept)),
            "segments_total": int(len(corpus)),
            "minutes_kept": round(kept["duration"].sum() / 60.0, 2),
        },
        "per_region": per_region,
        "temporal": temporal_coverage(metas),
        "resources": metas,
    }
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CORPUS_METADATA, "w", encoding="utf-8") as fh:
        json.dump(corpus_meta, fh, ensure_ascii=False, indent=2)

    return corpus
