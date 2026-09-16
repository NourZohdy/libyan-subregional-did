"""Cross-playlist speaker linking -> canonical corpus-wide speaker IDs.

Per-playlist clustering means the same host on two shows gets two ``actor_uid``s,
which (a) lets one human span train AND test in the "speaker-disjoint" splits and
(b) makes the validator review the same person twice. This pass compares every
actor centroid corpus-wide (a few-thousand-square cosine matrix — cheap):

* same-region pairs with sim >= ``LINK_AUTO_SIM``      -> linked automatically
* same-region pairs in ``[LINK_REVIEW_SIM, AUTO)``     -> human same/different queue
* CROSS-REGION pairs >= ``LINK_REVIEW_SIM``            -> ALWAYS queued, never auto-
  linked: a confident cross-region match means one of the two region labels is wrong,
  which only a human may decide.

No clips/folders move. Outputs (``config.METADATA_DIR``):
    speaker_links.csv     actor_uid -> speaker_global (canonical ID) + per-actor stats
    link_candidates.csv   gray-zone pairs for the validation app

``combine.py`` joins ``speaker_global`` onto the corpus manifest; the splits step
must group on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared import config

LINK_COLUMNS = ["actor_uid", "speaker_global", "region", "playlist",
                "n_segments", "minutes", "auto_linked_with"]
CANDIDATE_COLUMNS = ["actor_a", "actor_b", "similarity", "cross_region",
                     "region_a", "region_b", "n_segments_a", "n_segments_b",
                     "minutes_a", "minutes_b"]

_TRUE = {"true", "1", "1.0", "yes"}


def collect_actor_centroids(corpus_dir: Path | None = None) -> pd.DataFrame:
    """One row per actor_uid with its L2-normalized centroid and stats.

    Reads every ``corpus/<region>/<playlist>/{final_manifest.csv,embeddings.npy}``
    pair (row-aligned by construction).
    """
    corpus_dir = corpus_dir or config.CORPUS_DIR
    finals = sorted(corpus_dir.glob("*/*/final_manifest.csv"))
    if not finals:
        raise FileNotFoundError(f"no final_manifest.csv under {corpus_dir}")

    rows = []
    for fpath in finals:
        region = fpath.parent.parent.name
        playlist = fpath.parent.name
        emb_path = fpath.parent / "embeddings.npy"
        if not emb_path.exists():
            continue
        df = pd.read_csv(fpath)
        if len(df) == 0:
            continue
        emb = np.load(emb_path)
        if len(df) != len(emb):
            raise RuntimeError(f"{region}/{playlist}: manifest/embeddings row mismatch")
        df["is_excluded"] = df["is_excluded"].astype(str).str.strip().str.lower().isin(_TRUE)
        active = df[(~df["is_excluded"]) & (df["global_actor"] >= 0)]
        for actor, idx in active.groupby("global_actor").groups.items():
            idx = list(idx)
            c = emb[idx].mean(axis=0)
            c /= (np.linalg.norm(c) or 1.0)
            g = active.loc[idx]
            rows.append({
                "actor_uid": f"{region}/{playlist}#{int(actor)}",
                "region": region,
                "playlist": playlist,
                "n_segments": len(idx),
                "minutes": round(g["duration"].sum() / 60.0, 2),
                "centroid": c.astype(np.float32),
            })
    return pd.DataFrame(rows)


def link_actors(
    actors: pd.DataFrame,
    *,
    auto_sim: float = config.LINK_AUTO_SIM,
    review_sim: float = config.LINK_REVIEW_SIM,
    cross_region_review_sim: float = config.LINK_CROSS_REGION_REVIEW_SIM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union-find over actor centroids -> (links_df, candidates_df)."""
    n = len(actors)
    uids = actors["actor_uid"].tolist()
    regions = actors["region"].tolist()
    C = np.stack(actors["centroid"].to_list())
    sims = C @ C.T

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    auto_partner: dict[int, set[int]] = {}
    cand_rows = []
    iu, ju = np.triu_indices(n, k=1)
    interesting = sims[iu, ju] >= review_sim
    n_segs = actors["n_segments"].to_numpy()
    for i, j in zip(iu[interesting], ju[interesting]):
        s = float(sims[i, j])
        cross = regions[i] != regions[j]
        if cross and s < cross_region_review_sim:
            continue  # coincidental cross-region similarity: not worth review time
        if s < auto_sim and min(int(n_segs[i]), int(n_segs[j])) < config.MERGE_REVIEW_MIN_SEGS:
            continue  # dust actor: review decision can't meaningfully change the corpus
        if s >= auto_sim and not cross:
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
            auto_partner.setdefault(i, set()).add(j)
            auto_partner.setdefault(j, set()).add(i)
        else:
            cand_rows.append({
                "actor_a": uids[i], "actor_b": uids[j], "similarity": round(s, 4),
                "cross_region": cross,
                "region_a": regions[i], "region_b": regions[j],
                "n_segments_a": int(actors["n_segments"].iat[i]),
                "n_segments_b": int(actors["n_segments"].iat[j]),
                "minutes_a": float(actors["minutes"].iat[i]),
                "minutes_b": float(actors["minutes"].iat[j]),
            })

    # Canonical IDs: groups ordered by total minutes (biggest speakers first).
    roots = [find(i) for i in range(n)]
    group_minutes: dict[int, float] = {}
    for i, r in enumerate(roots):
        group_minutes[r] = group_minutes.get(r, 0.0) + float(actors["minutes"].iat[i])
    ordered = sorted(group_minutes, key=lambda r: -group_minutes[r])
    gid = {r: f"spk_{k:05d}" for k, r in enumerate(ordered)}

    links = pd.DataFrame({
        "actor_uid": uids,
        "speaker_global": [gid[r] for r in roots],
        "region": regions,
        "playlist": actors["playlist"].tolist(),
        "n_segments": actors["n_segments"].tolist(),
        "minutes": actors["minutes"].tolist(),
        "auto_linked_with": ["; ".join(sorted(uids[j] for j in auto_partner.get(i, ())))
                             for i in range(n)],
    }, columns=LINK_COLUMNS)
    cands = (pd.DataFrame(cand_rows, columns=CANDIDATE_COLUMNS)
             .sort_values("similarity", ascending=False).reset_index(drop=True))
    return links, cands


def run_linking(corpus_dir: Path | None = None) -> dict:
    """Full pass: collect centroids, link, write the two CSVs. Returns a summary."""
    actors = collect_actor_centroids(corpus_dir)
    links, cands = link_actors(actors)
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    links_path = config.METADATA_DIR / "speaker_links.csv"
    cands_path = config.METADATA_DIR / "link_candidates.csv"
    links.to_csv(links_path, index=False)
    cands.to_csv(cands_path, index=False)
    n_linked = int((links.groupby("speaker_global").size() > 1).sum())
    return {
        "n_actors": len(links),
        "n_speakers_global": int(links["speaker_global"].nunique()),
        "n_multi_actor_speakers": n_linked,
        "n_candidates": len(cands),
        "n_cross_region_candidates": int(cands["cross_region"].sum()) if len(cands) else 0,
        "links_path": str(links_path),
        "candidates_path": str(cands_path),
    }
