"""Per-playlist pipeline-state index (``playlist_state.csv``).

The truth always lives on disk and is keyed by the ``playlist_id`` written into every
sidecar (``acquire.write_sidecar`` -> ``_playlist_id_from_url``):

    downloaded -> ``<root>/<dialect>/<show>/*.wav`` exists
    diarized   -> sibling ``*.rttm`` exists
    clustered  -> ``outputs/corpus/<region>/<show>/metadata.json`` exists

This file persists only what disk can't show — the **picked-but-not-downloaded** rows and
**cached remote counts / upload-date range** — and is overlaid with a fresh disk scan via
:func:`sync_from_disk`. It reuses the registry CSV idiom (pipe-delimited, every cell
sanitised) so Arabic show titles never corrupt a row.

Legacy shows whose sidecars predate ``playlist_id`` fall back to a ``name:<dialect>/<show>``
key (and ``notes="legacy:no-pid"``) — they only match the live YouTube list once the
Phase-C back-fill writes the real id into their sidecars.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from libyan_did.shared import config
from libyan_did.shared import harvest
from libyan_did.shared import registry

STATE_FILE = config.DISCOVERY_DIR / "playlist_state.csv"

COLS = ["playlist_id", "channel_id", "title", "region", "region_raw", "rel_path", "url", "stage",
        "n_videos", "n_downloaded", "n_diarized", "n_clustered", "n_actors",
        "upload_date_min", "upload_date_max", "last_action", "notes"]

STAGES = ["picked", "downloaded", "diarized", "clustered"]


def read(path=STATE_FILE) -> pd.DataFrame:
    return registry.read(path, cols=COLS)[COLS]


def write(df: pd.DataFrame, path=STATE_FILE) -> None:
    registry.write(df.reindex(columns=COLS, fill_value=""), path,
                   backup=Path(path).with_suffix(".csv.bak"))


def _today() -> str:
    return date.today().isoformat()


def _read_json(p) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a missing/garbled sidecar must not kill a scan
        return {}


# --------------------------------------------------------------------------- #
# Persisted-row mutation
# --------------------------------------------------------------------------- #
def upsert(rows: list[dict], path=STATE_FILE) -> pd.DataFrame:
    """Insert/merge rows by ``playlist_id`` (only non-empty values overwrite)."""
    df = read(path).set_index("playlist_id")
    for r in rows:
        pid = str(r.get("playlist_id") or "").strip()
        if not pid:
            continue
        if pid not in df.index:
            df.loc[pid] = {c: "" for c in COLS if c != "playlist_id"}
        for k, v in r.items():
            if k in COLS and k != "playlist_id" and str(v) != "":
                df.at[pid, k] = v
        df.at[pid, "last_action"] = _today()
    out = df.reset_index()
    write(out, path)
    return out


def set_stage(playlist_id: str, stage: str, *, n_actors=None, path=STATE_FILE) -> None:
    df = read(path).set_index("playlist_id")
    pid = str(playlist_id)
    if pid not in df.index:
        df.loc[pid] = {c: "" for c in COLS if c != "playlist_id"}
    df.at[pid, "stage"] = stage
    if n_actors is not None:
        df.at[pid, "n_actors"] = str(n_actors)
    df.at[pid, "last_action"] = _today()
    write(df.reset_index(), path)


# --------------------------------------------------------------------------- #
# Disk truth
# --------------------------------------------------------------------------- #
def _corpus_meta(region: str, show: str) -> dict | None:
    p = config.CORPUS_DIR / region / show / "metadata.json"
    return _read_json(p) if p.exists() else None


def disk_rows(root) -> list[dict]:
    """One state row per on-disk show folder, keyed by sidecar ``playlist_id``.

    Fail-soft: an unreadable/absent root yields ``[]`` so the index still loads.
    """
    try:
        resources = harvest.enumerate_resources(root)
    except Exception:  # noqa: BLE001
        return []

    out: list[dict] = []
    for r in resources:
        eps = r["episodes"]
        n_dl = len(eps)
        n_dia = sum(1 for e in eps if e.get("rttm_path"))
        pid = chan = ""
        title = r["playlist"]
        dates: list[str] = []
        for e in eps:
            if not e.get("json_path"):
                continue
            j = _read_json(e["json_path"])
            pid = pid or str(j.get("playlist_id") or "").strip()
            chan = chan or str(j.get("channel_id") or "").strip()
            title = j.get("playlist_name") or title
            ud = str(j.get("upload_date") or "").strip()
            if ud and ud.lower() != "unknown":
                dates.append(ud)
        legacy = not pid
        key = pid or f"name:{r['region_raw']}/{r['playlist']}"

        cmeta = _corpus_meta(r["region"], r["playlist"])
        if cmeta is not None:
            stage = "clustered"
        elif n_dl and n_dia >= n_dl:
            stage = "diarized"
        else:
            stage = "downloaded"

        out.append({
            "playlist_id": key, "channel_id": chan, "title": title,
            "region": r["region"], "region_raw": r["region_raw"], "rel_path": r["rel_path"],
            "stage": stage,
            "n_downloaded": str(n_dl), "n_diarized": str(n_dia),
            "n_clustered": str((cmeta or {}).get("n_segments_kept", "")) if cmeta else "",
            "n_actors": str((cmeta or {}).get("n_actors", "")) if cmeta else "",
            "upload_date_min": min(dates) if dates else "",
            "upload_date_max": max(dates) if dates else "",
            "notes": "legacy:no-pid" if legacy else "",
        })
    return out


def sync_from_disk(root, path=STATE_FILE) -> pd.DataFrame:
    """Overlay the on-disk truth onto the persisted rows; persist + return the merged view.

    Persisted rows survive (picked-but-not-downloaded + cached ``n_videos``/``url``); disk
    values overwrite only where non-empty, so they can't blank out remote-only fields.
    """
    df = read(path).set_index("playlist_id")
    rows = disk_rows(root)
    present = {d["playlist_id"] for d in rows}
    for d in rows:
        pid = d["playlist_id"]
        if pid not in df.index:
            df.loc[pid] = {c: "" for c in COLS if c != "playlist_id"}
        for k, v in d.items():
            if k in COLS and k != "playlist_id" and str(v) != "":
                df.at[pid, k] = v
    # Prune ghosts: a row that claims an on-disk stage (downloaded/diarized/clustered) but whose
    # folder is gone this scan was deleted from disk -> drop it so the UI stops showing it. Only
    # when the scan actually found something (rows != []), so a mistyped / temporarily-missing
    # root can't wipe the whole index. 'picked'-but-not-yet-downloaded rows have no folder and
    # are KEPT.
    if rows:
        ghosts = [pid for pid, r in df.iterrows()
                  if pid not in present
                  and str(r.get("stage", "")) in ("downloaded", "diarized", "clustered")]
        if ghosts:
            df = df.drop(index=ghosts)
    out = df.reset_index()
    write(out, path)
    return out


def summary(df: pd.DataFrame) -> dict:
    """Counts per stage over a state frame (for the sidebar / Home dashboard)."""
    s = df["stage"].fillna("")
    return {st: int((s == st).sum()) for st in STAGES}


def demo() -> None:
    """Self-check: sync keys by sidecar playlist_id and counts wav/rttm correctly."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "libyan"
        show = root / "tripoli" / "نص نص"
        show.mkdir(parents=True)
        # two episodes, one with a sibling rttm; both carry the SAME playlist_id in JSON
        for i, has_rttm in [(1, True), (2, False)]:
            base = show / f"LIB_tripoli_x_{i:03d}"
            base.with_suffix(".wav").write_bytes(b"RIFFx")
            base.with_suffix(".json").write_text(json.dumps(
                {"playlist_id": "PLtest", "channel_id": "UCabc",
                 "playlist_name": "نص نص", "upload_date": f"2023010{i}"}),
                encoding="utf-8")
            if has_rttm:
                base.with_suffix(".rttm").write_text("", encoding="utf-8")

        rows = disk_rows(root)
        assert len(rows) == 1, rows
        row = rows[0]
        assert row["playlist_id"] == "PLtest", row            # keyed by sidecar id, not folder name
        assert row["rel_path"] == "tripoli/نص نص", row        # on-disk folder for legacy-safe ops
        assert row["n_downloaded"] == "2" and row["n_diarized"] == "1", row
        assert row["stage"] == "downloaded", row              # not all diarized -> downloaded
        assert row["upload_date_min"] == "20230101" and row["upload_date_max"] == "20230102", row
        assert row["channel_id"] == "UCabc", row
    print("OK pstate.demo: disk scan keys by playlist_id; wav/rttm/date counts correct.")


if __name__ == "__main__":
    demo()
