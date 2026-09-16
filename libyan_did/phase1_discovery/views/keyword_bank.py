"""Keyword bank (window 0) — edit the region-tagged Arabic search keywords in the app.

Discovery (``discover_sources.py``) searches YouTube with one keyword bank per region and
tags each hit with that region. The bank lives in ``data/keywords.json`` (read by
``discover_sources.load_keywords``, with an embedded fallback). This page lets you grow /
prune it without hand-editing JSON: one keyword per line, per region. Region keys are FIXED
(``Fezzan / Cyrenaica / Tripolitania / Unknown``) — discovery looks them up by exact name.

Edits are written straight back to ``data/keywords.json``; the next ``discover_sources.py``
run picks them up. No audio, no network here — just the keyword list.
"""

from __future__ import annotations

import json

import streamlit as st

from libyan_did.shared import config

KEYWORDS_FILE = config.DISCOVERY_DIR / "keywords.json"
# Exact keys discovery expects (discover_sources.load_keywords / region_suggest). Do not rename.
REGION_KEYS = ["Tripolitania", "Cyrenaica", "Fezzan", "Unknown"]


def load_raw() -> dict:
    """The whole JSON doc (so we preserve ``_comment`` and any extra keys on save)."""
    try:
        d = json.loads(KEYWORDS_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {"regions": {}}
    except Exception:  # noqa: BLE001 — missing/garbled file -> start from empty, never crash
        return {"regions": {}}


def regions_of(doc: dict) -> dict:
    """The ``{region: [kw,...]}`` map, tolerating either ``{regions:{…}}`` or a bare map."""
    reg = doc.get("regions", doc) if isinstance(doc, dict) else {}
    return {r: list(v) for r, v in reg.items() if isinstance(v, list)}


def clean_lines(text: str) -> list[str]:
    """One keyword per non-blank line, trimmed, de-duplicated preserving order."""
    seen, out = set(), []
    for ln in text.splitlines():
        kw = ln.strip()
        if kw and kw not in seen:
            seen.add(kw)
            out.append(kw)
    return out


st.title("🔑 Keyword bank — region-tagged discovery search terms")

doc = load_raw()
banks = regions_of(doc)
extra_keys = [r for r in banks if r not in REGION_KEYS]

st.caption(f"Edited file: `{KEYWORDS_FILE}` · read by `discover_sources.py`. One keyword per line. "
           "Region names are fixed — discovery matches them exactly. Arabic, hashtags (#فزان) and "
           "Latin terms all work.")

tabs = st.tabs([f"{r}  ({len(banks.get(r, []))})" for r in REGION_KEYS])
edited: dict[str, str] = {}
for tab, r in zip(tabs, REGION_KEYS):
    with tab:
        edited[r] = st.text_area(f"{r} keywords", value="\n".join(banks.get(r, [])),
                                 height=420, key=f"kw_{r}",
                                 label_visibility="collapsed")
        st.caption(f"{len(clean_lines(edited[r]))} keyword(s) after cleanup "
                   "(blank lines + duplicates dropped on save).")

c1, c2 = st.columns([1, 4])
with c1:
    if st.button("💾 Save keyword bank", type="primary"):
        new_regions = {r: clean_lines(edited[r]) for r in REGION_KEYS}
        for r in extra_keys:                      # keep any non-standard region lists untouched
            new_regions[r] = banks[r]
        out = dict(doc) if isinstance(doc, dict) else {}
        out.pop("regions", None)
        # Preserve a leading _comment first, then regions, then any other top-level keys.
        ordered = {}
        if "_comment" in out:
            ordered["_comment"] = out.pop("_comment")
        ordered["regions"] = new_regions
        ordered.update(out)
        KEYWORDS_FILE.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
        total = sum(len(v) for v in new_regions.values())
        st.success(f"Saved {total} keyword(s) across {len(new_regions)} region(s) → {KEYWORDS_FILE.name}. "
                   "Re-run `discover_sources.py` to search with the new terms.")
with c2:
    if extra_keys:
        st.caption(f"Also preserving non-standard region key(s): {', '.join(extra_keys)} "
                   "(discovery ignores these unless added to its region list).")
    st.caption("Next: `python -m libyan_did.phase1_discovery.discover_sources` → vet them on "
               "the **Review channels** page.")
