"""Validate recovered YouTube ids before trusting them (camera-ready provenance).

An agent proposed `outputs/metadata/recovered_ids.csv`. Its own claim that a match is
"high confidence" is not evidence, so this re-checks the parts that can be checked
offline against our own audio:

  1. every id is a syntactically real 11-char YouTube id,
  2. no id is reused for two different local files (the classic failure: many local
     episodes collapsed onto one popular video),
  3. every row's duration_sec matches the WAV on disk -- i.e. the agent did not quietly
     alter our side of the comparison,
  4. the claimed remote duration is within tolerance of the real local duration.

What this canNOT check offline is whether the id actually points at that video; rows that
survive here still need the network spot-check (`--sample N` prints ids to open).
# ponytail: offline checks only. yt-dlp/API verification would need network + a key;
# the sample list is enough to catch a fabricating agent.

Run:
    python -m libyan_did.phase2_harvest.check_recovered_ids
    python -m libyan_did.phase2_harvest.check_recovered_ids --sample 10
"""

from __future__ import annotations

import contextlib
import re
import sys
import wave
from pathlib import Path

import pandas as pd

from libyan_did.shared import config

RECOVERED_CSV = config.METADATA_DIR / "recovered_ids.csv"
AUDIO_ROOT = Path(r"D:\Research\DID_Set\libyan")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
TOLERANCE_SEC = 20.0


_WAV_INDEX: dict[str, Path] = {}


def _index() -> dict[str, Path]:
    # Built by walking, not glob: episode names contain "[" and "]", which glob reads as
    # character classes and then silently fails to match the real file.
    if not _WAV_INDEX:
        for p in AUDIO_ROOT.rglob("*.wav"):
            _WAV_INDEX.setdefault(p.name, p)
    return _WAV_INDEX


def wav_duration(name: str) -> float | None:
    hit = _index().get(name)
    if hit is None:
        return None
    try:
        with contextlib.closing(wave.open(str(hit))) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


def check(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    claimed = df[df.youtube_id.notna() & (df.youtube_id.astype(str).str.strip() != "")]

    bad_id = claimed[~claimed.youtube_id.astype(str).str.match(ID_RE)]
    for _, r in bad_id.iterrows():
        problems.append(f"malformed id {r.youtube_id!r} for {r.file}")

    dupes = claimed[claimed.duplicated("youtube_id", keep=False)].sort_values("youtube_id")
    for yid, grp in dupes.groupby("youtube_id"):
        # Sharing an id is legitimate when the same video was harvested twice under
        # different filenames -- identical local durations prove it. Differing durations
        # mean two distinct episodes were collapsed onto one video, which is a real error.
        durs = {round(float(x), 1) for x in grp.duration_sec}
        if len(durs) > 1:
            problems.append(f"id {yid} claimed for {len(grp)} files of DIFFERENT lengths "
                            f"{sorted(durs)}: {list(grp.file)[:4]}")
        else:
            print(f"  note: {yid} is a duplicate harvest ({len(grp)} files, "
                  f"identical {durs.pop()}s) -- same source video")

    for _, r in df.iterrows():
        real = wav_duration(str(r.file))
        if real is None:
            problems.append(f"no local wav found for {r.file}")
            continue
        if abs(real - float(r.duration_sec)) > 1.0:
            problems.append(f"{r.file}: csv duration {r.duration_sec}s != wav {real:.1f}s")
        rd = pd.to_numeric(pd.Series([r.get("remote_duration_sec")]), errors="coerce").iloc[0]
        # A row already marked `rejected` is a match we threw out on the duration rule --
        # that is the rule working, not a problem to report again.
        if pd.notna(rd) and abs(rd - real) > TOLERANCE_SEC and r.get("confidence") != "rejected":
            problems.append(f"{r.file}: remote {rd:.0f}s vs local {real:.0f}s "
                            f"(delta {abs(rd-real):.0f}s > {TOLERANCE_SEC:.0f}s) "
                            f"but marked {r.get('confidence')}")
    return problems


def verify_online(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remote_duration_sec from YouTube and promote/demote `confidence`.

    The proposing agent could read titles and ids but not runtimes, so every match came
    back `low`. yt-dlp answers the one question that decides it, without downloading.
    """
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        for i, row in df.iterrows():
            yid = str(row.youtube_id).strip()
            if not yid:
                continue
            local = wav_duration(str(row.file))
            try:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={yid}", download=False)
            except Exception as exc:
                df.at[i, "note"] = f"lookup failed: {str(exc)[:80]}"
                df.at[i, "confidence"] = "none"
                continue
            remote = float(info.get("duration") or 0)
            df.at[i, "remote_title"] = str(info.get("title", ""))
            df.at[i, "remote_duration_sec"] = str(int(remote))
            if local is None:
                continue
            delta = abs(remote - local)
            df.at[i, "duration_delta_sec"] = str(round(delta, 1))
            df.at[i, "confidence"] = "high" if delta <= TOLERANCE_SEC else "rejected"
            print(f"  {'OK ' if delta <= TOLERANCE_SEC else 'BAD'} {yid} delta={delta:6.0f}s  "
                  f"{str(info.get('title',''))[:40]}")
    return df


def main() -> None:
    if not RECOVERED_CSV.exists():
        raise SystemExit(f"{RECOVERED_CSV} not found -- nothing to validate yet")

    if "--verify" in sys.argv:
        df = pd.read_csv(RECOVERED_CSV, dtype=str).fillna({"youtube_id": ""})
        df["duration_sec"] = pd.to_numeric(df.duration_sec, errors="coerce")
        df = verify_online(df)
        df.to_csv(RECOVERED_CSV, index=False, encoding="utf-8-sig")
        print(f"\nwrote {RECOVERED_CSV}")
        print(df.confidence.value_counts().to_string())
        return

    df = pd.read_csv(RECOVERED_CSV, dtype=str).fillna({"youtube_id": ""})
    df["duration_sec"] = pd.to_numeric(df.duration_sec, errors="coerce")

    print(f"rows: {len(df)}")
    if "confidence" in df:
        print(df.confidence.value_counts().to_string())

    problems = check(df)
    print(f"\n{len(problems)} problem(s)")
    for p in problems[:40]:
        print("  !", p)

    if "--sample" in sys.argv:
        n = int(sys.argv[sys.argv.index("--sample") + 1])
        hi = df[df.get("confidence", "") == "high"].head(n)
        print(f"\nspot-check these by hand (open and compare title + length):")
        for _, r in hi.iterrows():
            print(f"  https://youtu.be/{r.youtube_id}  {float(r.duration_sec)/60:5.1f} min  "
                  f"{str(r.title_from_filename)[:45]}")

    raise SystemExit(1 if problems else 0)


def _selfcheck() -> None:
    df = pd.DataFrame({
        "file": ["a.wav", "b.wav"],
        "title_from_filename": ["t1", "t2"],
        "duration_sec": [100.0, 200.0],
        "youtube_id": ["abcdefghijk", "abcdefghijk"],   # duplicate on purpose
        "remote_duration_sec": [100.0, 900.0],           # second one way off
        "confidence": ["high", "high"],
    })
    probs = check(df)
    assert any("DIFFERENT lengths" in p for p in probs), probs
    assert any("no local wav found" in p for p in probs), probs
    # same id + same duration = one video harvested twice, not an error
    same = df.assign(duration_sec=[100.0, 100.0])
    assert not any("DIFFERENT lengths" in p for p in check(same)), check(same)
    assert not ID_RE.match("short"), "id regex too loose"
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
