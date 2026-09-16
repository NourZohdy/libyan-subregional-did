"""Write verified youtube ids from recovered_ids.csv back into the harvest sidecars.

`recovered_ids.csv` is the working record; the sidecars are what the rest of the pipeline
(and the release manifest) actually read. Only rows confirmed against the local audio
(`confidence == high`) are applied.

Every touched sidecar is copied to `<name>.json.bak` first, so a bad run is undone with a
rename. Nothing else in the sidecar is modified.
# ponytail: writes only youtube_id/source_url/duration_sec; upload_date stays "unknown"
# because we never verified it -- backfill_ids.py can fill it later from the same ids.

Run:
    python -m libyan_did.phase2_harvest.apply_recovered_ids --dry-run
    python -m libyan_did.phase2_harvest.apply_recovered_ids
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from libyan_did.shared import config

RECOVERED_CSV = config.METADATA_DIR / "recovered_ids.csv"
AUDIO_ROOT = Path(r"D:\Research\DID_Set\libyan")


_INDEX: dict[str, Path] = {}


def sidecar_for(wav_name: str) -> Path | None:
    # Indexed by walking, not rglob: episode names contain "[" and "]", which glob reads
    # as a character class and then fails to find the file that plainly exists.
    if not _INDEX:
        for p in AUDIO_ROOT.rglob("*.json"):
            _INDEX.setdefault(p.stem, p)
    return _INDEX.get(Path(wav_name).stem)


def main() -> None:
    dry = "--dry-run" in sys.argv
    df = pd.read_csv(RECOVERED_CSV, dtype=str)
    good = df[df.confidence == "high"]
    print(f"{len(good)} verified rows to apply (dry-run={dry})")

    written = missing = skipped = 0
    for _, r in good.iterrows():
        sc = sidecar_for(str(r.file))
        if sc is None:
            print(f"  ! no sidecar for {r.file}")
            missing += 1
            continue
        data = json.loads(sc.read_text(encoding="utf-8"))
        # pandas hands back the float nan for empty cells; str() would write "nan".
        url = next((str(v) for v in (r.playlist_url, r.channel_url)
                    if pd.notna(v) and str(v).strip() not in ("", "nan")), "")
        if data.get("youtube_id") == r.youtube_id and data.get("source_url") == url:
            skipped += 1
            continue
        data["youtube_id"] = r.youtube_id
        data["duration_sec"] = float(r.duration_sec)
        data["source_url"] = url
        data["video_title"] = str(r.remote_title or data.get("video_title", ""))
        data["notes"] = (str(data.get("notes", "")) + " | id recovered by duration match").strip(" |")
        if not dry:
            shutil.copy2(sc, sc.with_suffix(".json.bak"))
            sc.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
        written += 1

    print(f"\nwritten={written}  already-correct={skipped}  sidecar-missing={missing}")
    if dry:
        print("(dry run -- nothing changed; drop --dry-run to apply)")


if __name__ == "__main__":
    main()
