"""Join recovered YouTube ids from the sidecars into the corpus manifests.

The ids live in the harvest sidecars (see phase2_harvest/apply_recovered_ids.py), but the
release promise in the paper's Release section is about the *manifest*: a reader gets
per-segment rows and has to be able to fetch the source audio from them. That needs the
video id on the row, so this adds `youtube_id` and `source_url` columns keyed on the
recording (the basename of `master_audio_path`).

Reads every sidecar, not just the recently recovered ones, so ids that were captured at
harvest time come along too.

Rows with no id -- short-form clips, and the handful of YouTube recordings whose source is
gone -- get an empty string, which is the honest value and keeps the column parseable.
# ponytail: keyed on the audio basename because that is the only field the manifests and
# the sidecars share; file_id is a corpus-internal hash the sidecars never saw.

Run:
    python -m libyan_did.phase4_validation.add_source_ids --dry-run
    python -m libyan_did.phase4_validation.add_source_ids
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

from libyan_did.shared import config

AUDIO_ROOT = Path(r"D:\Research\DID_Set\libyan")
MANIFESTS = ["splits_manifest.csv", "splits_manifest_v2.csv",
             "verified_manifest.csv", "corpus_manifest.csv"]
_SEP = re.compile(r"[\\/]")


def stem_of(path: str) -> str:
    """Basename without extension, tolerant of both path separators."""
    return os.path.splitext(_SEP.split(str(path))[-1])[0]


def sidecar_index() -> dict[str, tuple[str, str]]:
    idx: dict[str, tuple[str, str]] = {}
    for p in AUDIO_ROOT.rglob("*.json"):
        if p.name.endswith(".json.bak"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        yid = d.get("youtube_id")
        if yid and yid != "unknown":
            idx[p.stem] = (str(yid), str(d.get("source_url") or ""))
    return idx


def annotate(df: pd.DataFrame, idx: dict[str, tuple[str, str]]) -> tuple[pd.DataFrame, int]:
    stems = df["master_audio_path"].fillna("").map(stem_of)
    df = df.copy()
    df["youtube_id"] = [idx.get(s, ("", ""))[0] for s in stems]
    df["source_url"] = [idx.get(s, ("", ""))[1] for s in stems]
    return df, int((df.youtube_id != "").sum())


def main() -> None:
    dry = "--dry-run" in sys.argv
    idx = sidecar_index()
    print(f"sidecars carrying a real id: {len(idx)}\n")

    for name in MANIFESTS:
        path = config.MANIFEST_DIR / name
        if not path.exists():
            print(f"  - {name}: not present, skipped")
            continue
        df = pd.read_csv(path, low_memory=False)
        if "master_audio_path" not in df.columns:
            # The channel-disjoint split carries only file_id, so borrow the
            # file_id -> path mapping from the manifest that has both.
            if "file_id" not in df.columns:
                print(f"  - {name}: no join key, skipped")
                continue
            base = pd.read_csv(config.MANIFEST_DIR / "splits_manifest.csv", low_memory=False)
            path_by_id = dict(zip(base.file_id, base.master_audio_path))
            df = df.copy()
            df["master_audio_path"] = df.file_id.map(path_by_id)
            out, n = annotate(df, idx)
            drop_helper = True
        else:
            out, n = annotate(df, idx)
            drop_helper = False
        recordings = out["master_audio_path"].fillna("").map(stem_of)
        print(f"  - {name}: {n}/{len(out)} rows ({n/max(len(out),1)*100:.1f}%) "
              f"| {recordings[out.youtube_id != ''].nunique()}/{recordings.nunique()} recordings")
        if drop_helper:
            out = out.drop(columns=["master_audio_path"])
        if not dry:
            shutil.copy2(path, path.with_suffix(".csv.bak"))
            out.to_csv(path, index=False, encoding="utf-8")

    print("\n(dry run -- nothing written)" if dry else "\nwritten; .csv.bak backups alongside")


def _selfcheck() -> None:
    assert stem_of(r"D:\a\b\LIB_x_001.wav") == "LIB_x_001"
    assert stem_of("D:/a/b/LIB_y_002.wav") == "LIB_y_002"
    assert stem_of(r"C:\dir\ep_[ 3 ]_004.wav") == "ep_[ 3 ]_004"   # brackets survive
    df = pd.DataFrame({"master_audio_path": [r"D:\x\a.wav", r"D:\x\b.wav", None]})
    out, n = annotate(df, {"a": ("ID123456789", "http://u")})
    assert n == 1 and out.youtube_id.tolist() == ["ID123456789", "", ""]
    assert out.source_url.tolist() == ["http://u", "", ""]
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
