"""Phase-2 global clustering: content gates -> MAD audit -> per-speaker centroids ->
global AHC -> Kneedle tiering + leakage purge. Writes decisions back into the manifest.

Adds columns: global_actor, tier, exclude_reason, and updates is_excluded for rejects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from libyan_did.shared.config import (
    ACTOR_MERGE_AUTO_SIM,
    ACTOR_MERGE_REVIEW_SIM,
    ACTOR_MERGE_REVIEW_Z,
    AHC_MERGE_THRESHOLD,
    KNEEDLE_SENSITIVITY,
    MAD_MULTIPLIER,
    MERGE_REVIEW_MIN_SEGS,
    MIN_SPEECH_RATIO,
    MUSIC_REJECT_PROB,
    PANNS_SPEECH_KEEP_PROB,
    SNR_REJECT_FLOOR_DB,
)

MERGE_CANDIDATE_COLUMNS = ["actor_a", "actor_b", "similarity", "n_segments_a", "n_segments_b"]


# --------------------------------------------------------------------------- #
# Content gates (music / noise / silence) — run before clustering
# --------------------------------------------------------------------------- #
def apply_content_gates(
    df: pd.DataFrame,
    *,
    min_speech_ratio: float = MIN_SPEECH_RATIO,
    snr_floor_db: float = SNR_REJECT_FLOOR_DB,
    music_reject_prob: float = MUSIC_REJECT_PROB,
    speech_keep_prob: float = PANNS_SPEECH_KEEP_PROB,
) -> pd.DataFrame:
    """Exclude non-speech / unusable segments before they reach clustering.

    * ``music_prob >= music_reject_prob`` AND ``panns_speech_prob < speech_keep_prob``
      -> "music" (PANNs CNN14: jingles/singing, which Silero VAD scores as speech).
      The speech condition keeps talk over a background-music bed — TV shows talk
      over music constantly and that speech is valid DID data.
    * ``speech_ratio < min_speech_ratio`` -> "low_speech" (music, singing, applause,
      silence-padded turns the diarizer mislabeled as a speaker).
    * ``snr_db < snr_floor_db`` -> "low_snr" (unusable noise floor).

    Sets ``is_excluded`` and records the first matching ``exclude_reason``.
    """
    out = df.copy()
    if "exclude_reason" not in out.columns:
        out["exclude_reason"] = ""

    if "music_prob" in out.columns:
        bad = out["music_prob"].fillna(0.0) >= music_reject_prob
        if "panns_speech_prob" in out.columns:
            bad &= out["panns_speech_prob"].fillna(1.0) < speech_keep_prob
        out.loc[bad, "is_excluded"] = True
        out.loc[bad & (out["exclude_reason"] == ""), "exclude_reason"] = "music"

    if "speech_ratio" in out.columns:
        bad = out["speech_ratio"].fillna(1.0) < min_speech_ratio
        out.loc[bad, "is_excluded"] = True
        out.loc[bad & (out["exclude_reason"] == ""), "exclude_reason"] = "low_speech"

    if "snr_db" in out.columns:
        bad = out["snr_db"].fillna(99.0) < snr_floor_db
        out.loc[bad, "is_excluded"] = True
        out.loc[bad & (out["exclude_reason"] == ""), "exclude_reason"] = "low_snr"

    return out


# --------------------------------------------------------------------------- #
# Local MAD outlier audit (per file_id + speaker_id)
# --------------------------------------------------------------------------- #
def mad_outlier_mask(embeddings: np.ndarray, multiplier: float = MAD_MULTIPLIER) -> np.ndarray:
    """Return boolean mask of inliers by MAD on cosine distance to the median centroid.

    Leys et al. 2013: (x - median) / (1.4826 * MAD) <= multiplier (2.5 = moderate-
    conservative). One-sided on purpose: only segments unusually FAR from the speaker
    centroid are outliers (other speakers / noise). Segments unusually close are the
    speaker's most prototypical audio and must never be flagged.
    """
    if len(embeddings) <= 2:
        return np.ones(len(embeddings), dtype=bool)
    centroid = np.median(embeddings, axis=0)
    centroid /= (np.linalg.norm(centroid) or 1.0)
    dists = 1.0 - embeddings @ centroid
    med = np.median(dists)
    mad = np.median(np.abs(dists - med)) or 1e-9
    scores = (dists - med) / (1.4826 * mad)
    return scores <= multiplier


def audit_local_speakers(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Flag MAD outliers per (file_id, speaker_id) among rows not already excluded."""
    out = df.copy()
    if "exclude_reason" not in out.columns:
        out["exclude_reason"] = ""
    for _, idx in out.groupby(["file_id", "speaker_id"]).groups.items():
        active = [i for i in idx if not bool(out.at[i, "is_excluded"])]
        if len(active) <= 2:
            continue
        mask = mad_outlier_mask(embeddings[active])
        for i, keep in zip(active, mask):
            if not keep:
                out.at[i, "is_excluded"] = True
                if out.at[i, "exclude_reason"] == "":
                    out.at[i, "exclude_reason"] = "mad_outlier"
    return out


# --------------------------------------------------------------------------- #
# Per-local-speaker centroids -> global AHC
# --------------------------------------------------------------------------- #
def local_centroids(df: pd.DataFrame, embeddings: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
    """Mean embedding per (file_id, speaker_id) over inliers. Returns (C, keys)."""
    keys, cents = [], []
    active = df[~df["is_excluded"]]
    for key, idx in active.groupby(["file_id", "speaker_id"]).groups.items():
        idx = list(idx)
        c = embeddings[idx].mean(axis=0)
        c /= (np.linalg.norm(c) or 1.0)
        keys.append(key)
        cents.append(c)
    return np.array(cents), keys


def global_ahc(centroids: np.ndarray, threshold: float = AHC_MERGE_THRESHOLD) -> np.ndarray:
    """Agglomerative clustering on cosine distance; returns integer labels."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if len(centroids) == 1:
        return np.array([0])
    dist = pdist(centroids, metric="cosine")
    Z = linkage(dist, method="average")
    return fcluster(Z, t=threshold, criterion="distance") - 1


def assign_global_actors(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Cluster local speakers across episodes into global_actor IDs."""
    out = df.copy()
    out["global_actor"] = -1
    centroids, keys = local_centroids(out, embeddings)
    if len(centroids) == 0:
        return out
    labels = global_ahc(centroids)
    key_to_actor = {k: int(lbl) for k, lbl in zip(keys, labels)}
    for (file_id, speaker_id), actor in key_to_actor.items():
        mask = (out["file_id"] == file_id) & (out["speaker_id"] == speaker_id)
        out.loc[mask, "global_actor"] = actor
    return out


# --------------------------------------------------------------------------- #
# Post-AHC actor merge pass (fixes one speaker split across 2-3 actors)
# --------------------------------------------------------------------------- #
def _actor_centroids(df: pd.DataFrame, embeddings: np.ndarray) -> dict[int, np.ndarray]:
    """L2-normalized mean embedding per global_actor over non-excluded rows."""
    active = df[(~df["is_excluded"]) & (df["global_actor"] >= 0)]
    cents: dict[int, np.ndarray] = {}
    for actor, idx in active.groupby("global_actor").groups.items():
        c = embeddings[list(idx)].mean(axis=0)
        c /= (np.linalg.norm(c) or 1.0)
        cents[int(actor)] = c
    return cents


def merge_actor_clusters(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    auto_sim: float = ACTOR_MERGE_AUTO_SIM,
    review_sim: float = ACTOR_MERGE_REVIEW_SIM,
    review_z: float = ACTOR_MERGE_REVIEW_Z,
    min_segs: int = MERGE_REVIEW_MIN_SEGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge over-split actors after AHC; emit gray-zone pairs for human review.

    AHC's average-linkage criterion under-merges the same speaker across episodes
    (channel/session shift). This pass compares denoised *actor centroids*:

    * pairs with cosine similarity >= ``auto_sim``  -> merged automatically (union-find)
    * pairs in ``[review_floor, auto_sim)``         -> returned as merge candidates for
      the validation app's same/different screen (nothing is changed automatically)

    The review floor is ADAPTIVE: ``max(review_sim, median + review_z * 1.4826*MAD)``
    over this playlist's own pair-sim distribution — playlists whose shared studio
    channel inflates ALL pair sims would otherwise flood the queue with pairs that
    are normal background there (i.e. different people). Pairs involving an actor
    with fewer than ``min_segs`` kept segments are never queued.

    Actor IDs are re-compacted to 0..K-1 after merging. Returns
    ``(df, candidates)`` where candidates has MERGE_CANDIDATE_COLUMNS.
    """
    out = df.copy()
    empty = pd.DataFrame(columns=MERGE_CANDIDATE_COLUMNS)
    cents = _actor_centroids(out, embeddings)
    if len(cents) < 2:
        return out, empty

    # ---- auto-merge via union-find on centroid similarity ----
    ids = sorted(cents)
    C = np.stack([cents[a] for a in ids])
    sims = C @ C.T
    parent = {a: a for a in ids}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sims[i, j] >= auto_sim:
                ra, rb = find(ids[i]), find(ids[j])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    root_of = {a: find(a) for a in ids}
    new_id = {root: k for k, root in enumerate(sorted(set(root_of.values())))}
    relabel = {a: new_id[root_of[a]] for a in ids}
    out["global_actor"] = out["global_actor"].map(lambda a: relabel.get(a, a))

    # ---- gray zone among the (merged) actors -> human same/different queue ----
    cents2 = _actor_centroids(out, embeddings)
    ids2 = sorted(cents2)
    if len(ids2) < 2:
        return out, empty
    C2 = np.stack([cents2[a] for a in ids2])
    sims2 = C2 @ C2.T
    seg_counts = (out[(~out["is_excluded"]) & (out["global_actor"] >= 0)]
                  .groupby("global_actor").size())

    # Adaptive floor: only pairs that are robust-z outliers vs. THIS playlist's own
    # pair-sim background are worth human time (channel-normalized review band).
    triu = sims2[np.triu_indices(len(ids2), k=1)]
    review_floor = review_sim
    if triu.size >= 10:
        med = float(np.median(triu))
        mad = float(np.median(np.abs(triu - med))) * 1.4826
        review_floor = max(review_sim, med + review_z * mad)

    rows = []
    for i in range(len(ids2)):
        for j in range(i + 1, len(ids2)):
            s = float(sims2[i, j])
            if not (review_floor <= s < auto_sim):
                continue
            na = int(seg_counts.get(ids2[i], 0))
            nb = int(seg_counts.get(ids2[j], 0))
            if min(na, nb) < min_segs:
                continue  # dust actor: the decision can't meaningfully change the corpus
            rows.append({
                "actor_a": ids2[i], "actor_b": ids2[j], "similarity": round(s, 4),
                "n_segments_a": na, "n_segments_b": nb,
            })
    cand = pd.DataFrame(rows, columns=MERGE_CANDIDATE_COLUMNS)
    return out, cand.sort_values("similarity", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Kneedle Pareto tiering
# --------------------------------------------------------------------------- #
def tier_actors(df: pd.DataFrame) -> pd.DataFrame:
    """Assign Tier1/2/3 by descending segment mass using Kneedle elbows.

    PURELY DESCRIPTIVE: the tier is a per-speaker mass label (used by analysis tables
    and the later split-balancing step). It never excludes anything — rejection is the
    job of the content gates and human validation. Rows without a clustered actor
    (excluded before clustering) keep tier="reject" as an inert marker.
    """
    from kneed import KneeLocator

    out = df.copy()
    if "exclude_reason" not in out.columns:
        out["exclude_reason"] = ""
    out["tier"] = "reject"
    active = out[(~out["is_excluded"]) & (out["global_actor"] >= 0)]
    counts = active.groupby("global_actor").size().sort_values(ascending=False)
    if counts.empty:
        return out

    actors = counts.index.tolist()
    y = counts.values.astype(float)
    x = np.arange(1, len(y) + 1)

    # Two successive knees split the ranked actors into 3 tiers.
    boundaries = []
    if len(y) >= 3:
        kn = KneeLocator(x, y, curve="convex", direction="decreasing", S=KNEEDLE_SENSITIVITY)
        if kn.knee:
            boundaries.append(int(kn.knee))
    cut1 = boundaries[0] if boundaries else max(1, len(y) // 3)
    cut2 = min(len(y), cut1 + max(1, (len(y) - cut1) // 2))

    tier_of = {}
    for rank, actor in enumerate(actors, start=1):
        if rank <= cut1:
            tier_of[actor] = "Tier1"
        elif rank <= cut2:
            tier_of[actor] = "Tier2"
        else:
            tier_of[actor] = "Tier3"

    out.loc[out["global_actor"] >= 0, "tier"] = out["global_actor"].map(tier_of).fillna("reject")
    return out


def fill_missing_exclusion_reasons(
    df: pd.DataFrame,
    *,
    min_speech_ratio: float = MIN_SPEECH_RATIO,
    snr_floor_db: float = SNR_REJECT_FLOOR_DB,
    music_reject_prob: float = MUSIC_REJECT_PROB,
    speech_keep_prob: float = PANNS_SPEECH_KEEP_PROB,
) -> pd.DataFrame:
    """Backstop: guarantee every ``is_excluded`` row carries an ``exclude_reason``.

    Re-applies the ``apply_content_gates`` thresholds in the same priority order
    (music -> low_speech -> low_snr); any excluded row matching none was removed by the
    MAD audit, so it is labelled ``mad_outlier``. A rescored cache can carry
    ``is_excluded=True`` from a prior run whose reason column was never populated (the
    controls were first clustered before the music-scoring pass, so their gate reasons
    were dropped). Without this, the exclusion count and its reason breakdown disagree.
    Only blank reasons are touched; existing labels and ``is_excluded`` are unchanged.
    """
    out = df.copy()
    if "exclude_reason" not in out.columns:
        out["exclude_reason"] = ""
    excl = _as_bool(out["is_excluded"])  # robust for bool, object, and pyarrow-string dtypes
    blank = excl & (out["exclude_reason"].fillna("") == "")
    if not blank.any():
        return out

    def _num(col, fill):
        return pd.to_numeric(out.get(col), errors="coerce").fillna(fill) if col in out.columns else pd.Series(fill, index=out.index)

    music = blank & (_num("music_prob", 0.0) >= music_reject_prob) & (_num("panns_speech_prob", 1.0) < speech_keep_prob)
    out.loc[music, "exclude_reason"] = "music"
    blank &= out["exclude_reason"].fillna("") == ""
    low_speech = blank & (_num("speech_ratio", 1.0) < min_speech_ratio)
    out.loc[low_speech, "exclude_reason"] = "low_speech"
    blank &= out["exclude_reason"].fillna("") == ""
    low_snr = blank & (_num("snr_db", 99.0) < snr_floor_db)
    out.loc[low_snr, "exclude_reason"] = "low_snr"
    blank &= out["exclude_reason"].fillna("") == ""
    out.loc[blank, "exclude_reason"] = "mad_outlier"
    return out


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def run_clustering(df: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full Phase-2 clustering: content gates -> MAD audit -> global actors ->
    post-AHC merge pass -> descriptive tiering.

    Returns ``(manifest_df, merge_candidates_df)``; the candidates are gray-zone
    actor pairs for the validation app's same/different screen.
    """
    out = apply_content_gates(df)
    out = audit_local_speakers(out, embeddings)
    out = assign_global_actors(out, embeddings)
    out, merge_candidates = merge_actor_clusters(out, embeddings)
    out = tier_actors(out)
    out = fill_missing_exclusion_reasons(out)  # never leave an excluded seg unlabeled
    return out, merge_candidates
