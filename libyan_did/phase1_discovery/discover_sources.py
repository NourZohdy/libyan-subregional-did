"""Bulk source discovery: run the Arabic keyword bank through yt-dlp search.

The LLM agent (discovery/DISCOVERY_AGENT.md) is the careful path, but for BULK collection
this runs each keyword through `yt-dlp ytsearch`, groups the hits into candidate CHANNELS,
dedups against the blocklist + existing candidates, and appends `pending` rows to
discovery/candidates.psv for human review in app/review_sources.py.

`region` is a TENTATIVE tag from the keyword that surfaced the channel (a city keyword →
its region; a generic "بودكاست ليبي" → Unknown). Confidence is low by design — the human
confirms/fixes region in the app. Metadata only: downloads NO audio.

Usage (from libyan_did_pipeline/):
    python scripts/discover_sources.py                       # full bank
    python scripts/discover_sources.py --per-search 20 --max-channels 200
    python scripts/discover_sources.py --region Fezzan       # one region's keywords
    python scripts/discover_sources.py --counts              # also fetch video_count (slower)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import pandas as pd  # noqa: E402

from libyan_did.shared import config
from libyan_did.shared import registry  # noqa: E402

DISC = config.DISCOVERY_DIR
MAIN = DISC / "candidates.psv"
BLOCKLIST = DISC / "harvested_blocklist.txt"
KEYWORDS_FILE = DISC / "keywords.json"
COLS = registry.COLS

# Region-tagged Arabic keyword bank (Fezzan-heavy). Generic terms -> Unknown (human re-tags).
# Mixes: city+format combos, identity/heritage terms, minority communities, #hashtags, and
# Latin/Franco spellings — hashtags and bare city names widen recall (the human prunes noise).
# This is the FALLBACK; the live bank lives in data/keywords.json (load_keywords below), so it
# can be edited without touching code and is shared with the DISCOVERY_AGENT.md spec.
_DEFAULT_KEYWORDS: dict[str, list[str]] = {
    "Fezzan": [
        # core city + format
        "بودكاست سبها", "حوار سبها", "سبها مقابلة", "سبها لقاء", "مايك الشارع سبها", "شباب سبها",
        "أوباري لقاء", "أوباري بودكاست", "أوباري مقابلة", "مرزق فزان", "مرزق لقاء",
        "غات ليبيا", "غات لقاء", "القطرون فزان", "القطرون لقاء", "براك الشاطئ", "براك الشاطئ لقاء",
        "وادي الشاطئ لقاء", "وادي الشاطئ بودكاست", "الجفرة هون ودان", "هون ودان لقاء", "سوكنة لقاء",
        "فزان لقاء", "الجنوب الليبي بودكاست", "صوت فزان", "صوت الجنوب", "ديوانية سبها",
        # smaller southern towns
        "أم الأرانب", "تمنهنت", "تساوة فزان", "إدري الشاطئ", "زويلة فزان", "تراغن", "غدوة فزان",
        # identity / heritage / communities
        "تراث فزان", "أهازيج فزان", "مجالس الجنوب", "فزاني", "سبهاوي",
        "تبو ليبيا", "التبو", "طوارق ليبيا أوباري", "تماشق", "أمازيغ الجنوب",
        # hashtags
        "#فزان", "#سبها", "#الجنوب_الليبي", "#أوباري", "#مرزق", "#غات", "#وادي_الشاطئ",
        "#براك_الشاطئ", "#القطرون", "#تبو", "#طوارق",
        # Latin / Franco
        "Fezzan podcast", "Sabha interview", "Ubari libya", "Murzuq", "Ghat libya",
    ],
    "Cyrenaica": [
        "بودكاست بنغازي", "حوار بنغازي", "بنغازي لقاء", "مايك الشارع بنغازي", "مقابلات شارع بنغازي",
        "شباب بنغازي بودكاست", "البيضاء ليبيا لقاء", "البيضاء بودكاست", "درنة بودكاست", "درنة لقاء",
        "طبرق لقاء", "طبرق بودكاست", "المرج ليبيا", "المرج لقاء", "اجدابيا لقاء", "اجدابيا بودكاست",
        "شحات لقاء", "سوسة درنة", "قمينس", "الأبيار ليبيا", "توكرة", "الكفرة لقاء", "جالو أوجلة",
        "الجبل الأخضر لقاء", "الجبل الأخضر بودكاست", "بودكاست برقة", "صوت برقة", "تراث برقة",
        "برقاوي", "بنغازية", "لهجة بنغازية",
        # hashtags
        "#بنغازي", "#برقة", "#درنة", "#طبرق", "#البيضاء", "#الجبل_الأخضر", "#المرج", "#اجدابيا",
        # Latin
        "Benghazi podcast", "Derna libya", "Tobruk", "Bayda libya",
    ],
    "Tripolitania": [
        "بودكاست طرابلس", "حوار طرابلس", "مايك الشارع طرابلس", "مقابلات شارع طرابلس", "شباب طرابلس بودكاست",
        "بودكاست مصراتة", "مصراتة لقاء", "الزاوية ليبيا لقاء", "الزاوية بودكاست", "صبراتة لقاء",
        "صرمان", "العجيلات ليبيا", "غريان لقاء", "غريان بودكاست", "نالوت يفرن أمازيغ", "نالوت لقاء",
        "يفرن لقاء", "جادو كاباو", "الزنتان لقاء", "الزنتان بودكاست", "ككلة", "زليتن بودكاست", "زليتن لقاء",
        "الخمس لبدة", "الخمس لقاء", "ترهونة لقاء", "ترهونة بودكاست", "بني وليد لقاء", "سرت لقاء",
        "تاجوراء", "جنزور ليبيا", "طرابلسي", "مصراتي", "صوت الغرب", "أمازيغ نفوسة",
        # hashtags
        "#طرابلس", "#مصراتة", "#الزاوية", "#غريان", "#الزنتان", "#نالوت", "#زليتن", "#الخمس",
        "#ترهونة", "#سرت", "#بني_وليد",
        # Latin
        "Tripoli podcast", "Misrata libya", "Zintan", "Sirte libya",
    ],
    "Unknown": [
        "بودكاست ليبي", "البودكاست الليبي", "بودكاست ليبيا", "حوار ليبي", "مقابلة ليبية", "لقاء ليبي",
        "مايك الشارع ليبيا", "مقابلات شارع ليبيا", "آراء الشارع الليبي", "حوار الشارع ليبيا",
        "يوتيوبر ليبي", "بودكاست بنات ليبيا", "برنامج حواري ليبي", "ديوانية ليبية", "مجلس ليبي",
        "قصة ليبية", "تجربة ليبية", "إذاعة محلية ليبيا",
        # hashtags
        "#ليبيا", "#بودكاست_ليبي", "#الشارع_الليبي",
        # Latin
        "Libyan podcast", "libya street interview", "libyan youtuber", "libya talk show",
    ],
}


def load_keywords(path: Path = KEYWORDS_FILE) -> dict[str, list[str]]:
    """Region -> keyword list, read from ``data/keywords.json`` (``_DEFAULT_KEYWORDS`` fallback).

    Accepts either ``{"regions": {...}}`` or a bare ``{region: [...]}`` mapping; any malformed
    or missing file falls back to the embedded default so the script never hard-fails.
    """
    try:
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        regions = data.get("regions", data) if isinstance(data, dict) else None
        bank = {r: list(kws) for r, kws in regions.items()
                if isinstance(kws, list)} if isinstance(regions, dict) else {}
        return bank or _DEFAULT_KEYWORDS
    except Exception:  # noqa: BLE001
        return _DEFAULT_KEYWORDS


KEYWORDS: dict[str, list[str]] = load_keywords()


def ytsearch(keyword: str, n: int) -> list[tuple[str, str, str]]:
    """(channel_id, channel_name, sample_title) for the top n search hits — metadata only."""
    cmd = ["yt-dlp", f"ytsearch{n}:{keyword}", "--flat-playlist", "--no-warnings",
           "--ignore-errors", "--print", "%(channel_id)s\t%(channel)s\t%(title)s"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=180)
    except Exception as exc:  # noqa: BLE001
        print(f"    ! search failed for {keyword!r} ({type(exc).__name__})")
        return []
    rows = []
    for line in (out.stdout or "").splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0].startswith("UC"):
            rows.append((p[0], p[1], p[2] if len(p) > 2 else ""))
    return rows


def video_count(channel_id: str) -> str:
    cmd = ["yt-dlp", f"https://www.youtube.com/channel/{channel_id}/videos",
           "--flat-playlist", "--no-warnings", "--playlist-end", "1", "--print", "%(playlist_count)s"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=120)
        v = (out.stdout or "").strip().splitlines()
        return v[0] if v and v[0].isdigit() else ""
    except Exception:  # noqa: BLE001
        return ""


def load_blocklist_names() -> set[str]:
    names: set[str] = set()
    if not BLOCKLIST.exists():
        return names
    for ln in BLOCKLIST.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or "|" not in ln:
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) >= 2:
            names.add(parts[1].lower())          # show_folder
        if len(parts) >= 4 and parts[3] != "unknown":
            for ch in parts[3].split(";"):
                names.add(ch.strip().lower())     # channel_or_author
    return names


def clean(s: str) -> str:
    return registry.sanitize(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-search", type=int, default=20, help="yt-dlp results per keyword")
    ap.add_argument("--min-hits", type=int, default=2,
                    help="keep only channels surfaced by >= this many keywords (precision filter)")
    ap.add_argument("--max-channels", type=int, default=200, help="cap new channels added")
    ap.add_argument("--region", default=None, help="only this region's keyword set")
    ap.add_argument("--counts", action="store_true", help="also fetch video_count per channel (slow)")
    args = ap.parse_args()

    existing = registry.read(MAIN)
    have_ids = set(existing["channel_id"].str.strip()) - {""}
    blocked = load_blocklist_names()
    banks = {args.region: KEYWORDS[args.region]} if args.region else KEYWORDS

    agg: dict[str, dict] = defaultdict(lambda: {"name": "", "regions": Counter(),
                                                "titles": [], "kws": [], "hits": 0})
    for region, kws in banks.items():
        for kw in kws:
            hits = ytsearch(kw, args.per_search)
            for cid, ch, title in hits:
                d = agg[cid]
                d["name"] = ch or d["name"]
                d["regions"][region] += 1
                d["hits"] += 1
                if title and len(d["titles"]) < 3:
                    d["titles"].append(title)
                if kw not in d["kws"]:
                    d["kws"].append(kw)
            print(f"  [{region}] {kw!r}: {len(hits)} hits, {len(agg)} channels so far")

    rows = []
    for cid, d in agg.items():
        if cid in have_ids or d["name"].lower() in blocked:
            continue
        region = d["regions"].most_common(1)[0][0]
        rows.append({
            "url": f"https://www.youtube.com/channel/{cid}",
            "url_type": "channel", "region": region, "source": clean(d["name"]),
            "channel_id": cid, "video_count": "", "playlist_count": "", "format_type": "",
            "lang_flag": "arabic", "confidence": "low",
            "evidence": clean(f"keyword hits: {', '.join(d['kws'][:4])}; sample: {d['titles'][0] if d['titles'] else ''}"),
            "discovery_path": clean(d["kws"][0] if d["kws"] else ""),
            "status": "pending", "notes": f"yt-dlp search; {d['hits']} hits",
            "_hits": d["hits"],
        })
    rows.sort(key=lambda r: r["_hits"], reverse=True)
    kept = [r for r in rows if r["_hits"] >= args.min_hits]
    print(f"\n[filter] {len(rows)} channels -> {len(kept)} with >= {args.min_hits} keyword hits")
    rows = kept[:args.max_channels]
    for r in rows:
        r.pop("_hits", None)

    if args.counts:
        print(f"\n[counts] fetching video_count for {len(rows)} channels ...")
        for r in rows:
            r["video_count"] = video_count(r["channel_id"])

    new_df = pd.DataFrame(rows, columns=COLS)
    out = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    out = out.drop_duplicates("channel_id", keep="first").fillna("")  # new rows lack review_note etc.
    if MAIN.exists() and not (DISC / "candidates.prediscover.psv").exists():
        (DISC / "candidates.prediscover.psv").write_text(
            MAIN.read_text(encoding="utf-8-sig"), encoding="utf-8")
    registry.write(out, MAIN)

    print(f"\n[discover] {len(agg)} channels surfaced; {len(new_df)} new added "
          f"(skipped {len(agg) - len(new_df)} dup/blocked/over-cap)")
    print(f"[discover] candidates.psv now {len(out)} rows. By region:")
    for reg, c in out["region"].value_counts().items():
        print(f"    {reg:>13}: {c}")


if __name__ == "__main__":
    main()
