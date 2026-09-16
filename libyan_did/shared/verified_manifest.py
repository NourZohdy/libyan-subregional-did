"""Freeze human validation onto every segment row → the verified manifest.

This is the single source of truth for turning the raw ``corpus_manifest.csv`` +
the human decisions (``speaker_verifications.csv``, ``merge_decisions.csv``) into
the segment-level ``verified_manifest.csv`` that all downstream steps (splits,
paper tables, ASR/DID training) consume.

The logic was previously inline in ``app/validate_speakers.py``; it lives here so
the Streamlit "Export" button and ``scripts/freeze_verified_manifest.py`` run the
*exact* same code. Decisions propagate within a human-merged speaker group, the
validator-resolved region overrides the raw region, and the dust floor on the
auto-linked identity (``speaker_global``) excludes insufficient-evidence speakers
by rule — overriding any human keep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRUE = {"true", "1", "1.0", "yes"}


def _s(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)


def build_verified_manifest(
    df: pd.DataFrame,
    vmap: dict,
    mdec: dict,
    *,
    regions,
    keep_decisions,
    min_segments: int,
    min_seconds: float,
) -> pd.DataFrame:
    """Return ``df`` with speaker_final / decision / dialect / floor / verified_keep.

    Parameters mirror the config constants so callers stay declarative:
    ``regions`` (valid region names), ``keep_decisions`` (decisions kept in the
    corpus), ``min_segments`` / ``min_seconds`` (the dust floor).
    """
    df = df.copy()
    if "speaker_global" not in df.columns:
        df["speaker_global"] = df["actor_uid"]

    # Human "same" pairs union the canonical speaker groups -> speaker_final.
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    uid_to_global = dict(zip(df["actor_uid"].astype(str), df["speaker_global"].astype(str)))
    for (a, b), d in mdec.items():
        if d.get("decision") == "same":
            ga, gb = uid_to_global.get(a), uid_to_global.get(b)
            if ga and gb:
                ra, rb = find(ga), find(gb)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    df["speaker_final"] = df["speaker_global"].astype(str).map(find)

    # Region resolved by the validator on "same" pairs (esp. cross-region ones)
    # applies to the WHOLE final speaker group.
    grp_region: dict[str, str] = {}
    for (a, b), d in mdec.items():
        rr = _s(d.get("resolved_region"))
        if d.get("decision") == "same" and rr in regions:
            ga = uid_to_global.get(a)
            if ga:
                grp_region[find(ga)] = rr

    df["speaker_decision"] = df["actor_uid"].map(
        lambda a: vmap.get(str(a), {}).get("decision", "unreviewed"))
    df["speaker_reason"] = df["actor_uid"].map(
        lambda a: _s(vmap.get(str(a), {}).get("reason", "")))

    # Actors never reviewed inherit a decision from a reviewed actor of the SAME
    # final speaker group (one human = one decision).
    grp_dec: dict[str, tuple[str, str]] = {}
    for uid, v in vmap.items():
        g = find(uid_to_global.get(uid, uid))
        if v.get("decision") not in (None, "", "unreviewed"):
            grp_dec.setdefault(g, (v["decision"], _s(v.get("reason"))))
    unrev = df["speaker_decision"] == "unreviewed"
    df.loc[unrev, "speaker_decision"] = df.loc[unrev, "speaker_final"].map(
        lambda g: grp_dec.get(g, ("unreviewed", ""))[0])
    df.loc[unrev, "speaker_reason"] = df.loc[unrev, "speaker_final"].map(
        lambda g: grp_dec.get(g, ("unreviewed", ""))[1])

    def final_dialect(r):
        if r["speaker_decision"] == "wrong_region" and r["speaker_reason"]:
            return r["speaker_reason"]
        return grp_region.get(r["speaker_final"], r["region"])

    df["final_dialect"] = df.apply(final_dialect, axis=1)

    # A merged speaker must have ONE region. A cross-region "same" merge with no
    # resolved_region would otherwise leave a speaker_final spanning two regions
    # (some rows fall back to their raw region), which breaks speaker-disjoint
    # splitting — the same person lands in train via one region and dev via the
    # other. Resolve each speaker_final to its duration-dominant dialect.
    dom = (df.groupby(["speaker_final", "final_dialect"])["duration"].sum()
           .reset_index().sort_values("duration", ascending=False)
           .drop_duplicates("speaker_final")
           .set_index("speaker_final")["final_dialect"])
    df["final_dialect"] = df["speaker_final"].map(dom)

    # Dust floor on the AUTO-LINKED identity (speaker_global): below it the
    # speaker is insufficient evidence by rule — overrides any human keep.
    active = (~df["is_excluded"].astype(str).str.strip().str.lower().isin(_TRUE)
              if "is_excluded" in df.columns else pd.Series(True, index=df.index))
    ev = (df[active].groupby("speaker_global")
          .agg(evidence_segments=("actor_uid", "size"),
               evidence_seconds=("duration", "sum")))
    df = df.merge(ev, on="speaker_global", how="left")
    df["evidence_segments"] = df["evidence_segments"].fillna(0).astype(int)
    df["evidence_seconds"] = df["evidence_seconds"].fillna(0.0)
    below = ((df["evidence_segments"] < min_segments)
             | (df["evidence_seconds"] < min_seconds))
    df["exclusion_rule"] = np.where(below, "insufficient_evidence", "")
    df["verified_keep"] = df["speaker_decision"].isin(keep_decisions) & ~below
    return df
