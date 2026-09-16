"""Per-resource (per-playlist) processing — the incremental unit of the corpus.

Each show folder under ``<root>/<dialect>/<playlist>/`` is processed independently:

    sibling RTTM (no diarization) -> timestamp manifest -> ECAPA embeddings ->
    within-playlist clustering/tiering -> clips written alongside the manifests at
    ``corpus/<region>/<playlist>/actor_<id>/`` -> per-resource final_manifest.csv +
    metadata.json (+ its own state lock) — one self-contained folder per playlist.

Because resources are independent, adding a new playlist, or a new episode to an
existing playlist, only reprocesses that one (small) playlist; the rest of the corpus
is untouched. ``src/combine.py`` then merges every resource into the corpus manifest,
and ``src/splits.py`` builds the train/dev/test splits — both as separate later steps.

Note: when a resource changes it is rebuilt in full (manifest + embeddings + clustering)
so the manifest and the row-aligned ``embeddings.npy`` can never drift. Playlists are
small, so this is cheap; the incremental win is at the *playlist* granularity.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared import config
from libyan_did.shared import cluster, embed, export, segment, state

_TRUE = {"true", "1", "1.0", "yes"}

# Chars illegal in a Windows path component. A channel key can be a free-text sidecar
# `channel_or_author` (e.g. "Sarmad Network | شبكة سرمد"), not a folder name, so it may
# contain these; folder-derived names never do, so sanitizing is a no-op for them.
_ILLEGAL_PATH_CHARS = '<>:"|?*'


def _safe_dir_component(name: str) -> str:
    """Filesystem-safe directory name for a channel/playlist key. Legal names pass through
    unchanged (preserving incremental state); names with illegal chars get them replaced and
    a short hash appended so distinct raw keys can't collide. The RAW key is still what the
    manifest stores as ``source`` (the split's channel key) — only the folder name is changed."""
    cleaned = "".join("_" if c in _ILLEGAL_PATH_CHARS else c for c in name).rstrip(" .")
    if cleaned == name:
        return name
    return f"{cleaned or 'x'}_{hashlib.sha1(name.encode('utf-8')).hexdigest()[:6]}"


def _as_bool(series: pd.Series) -> pd.Series:
    """Robustly coerce an is_excluded column (bool or round-tripped text) to bool."""
    return series.astype(str).str.strip().str.lower().isin(_TRUE)


def resource_paths(region: str, playlist: str) -> dict:
    rdir = config.CORPUS_DIR / region / _safe_dir_component(playlist)
    return {
        "dir": rdir,
        "segments": rdir / "master_segments.csv",
        "embeddings": rdir / "embeddings.npy",
        "final": rdir / "final_manifest.csv",
        "state": rdir / "state.json",
        "metadata": rdir / "metadata.json",
        "merge_candidates": rdir / "merge_candidates.csv",
    }


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def build_resource_metadata(resource: dict, final_df: pd.DataFrame, params_hash: str) -> dict:
    """Summarise one processed resource into a self-contained metadata record."""
    fdf = final_df.copy()
    fdf["is_excluded"] = _as_bool(fdf["is_excluded"])
    kept = fdf[(~fdf["is_excluded"]) & (fdf["tier"].isin(config.KEEP_TIERS))]

    reasons = {}
    if "exclude_reason" in fdf.columns:
        reasons = (fdf[fdf["is_excluded"]]["exclude_reason"].fillna("").replace("", "unknown")
                   .value_counts().to_dict())
    clean = int((kept["quality"] == "clean").sum()) if "quality" in kept.columns else 0
    noisy = int((kept["quality"] == "noisy").sum()) if "quality" in kept.columns else 0

    by_file = fdf.groupby("file_id")
    kept_by_file = kept.groupby("file_id")
    episodes_meta = []
    for e in resource["episodes"]:
        fid = e["file_id"]
        n_seg = int(len(by_file.get_group(fid))) if fid in by_file.groups else 0
        kmin = (round(kept_by_file.get_group(fid)["duration"].sum() / 60.0, 2)
                if fid in kept_by_file.groups else 0.0)
        j = _read_json(Path(e["json_path"])) if e.get("json_path") else {}
        episodes_meta.append({
            "file_id": fid, "basename": e["basename"], "has_rttm": bool(e.get("rttm_path")),
            "n_segments": n_seg, "kept_minutes": kmin,
            "youtube_id": j.get("youtube_id"), "video_title": j.get("video_title"),
            "channel_or_author": j.get("channel_or_author"), "source_name": j.get("source_name"),
            "upload_date": j.get("upload_date"), "domain": j.get("domain"),
        })

    return {
        "region": resource["region"], "region_raw": resource["region_raw"],
        "playlist": resource["playlist"], "macro": resource["macro"],
        "source_tree": resource["rel_path"],
        "n_episodes": len(resource["episodes"]),
        "n_segments_total": int(len(fdf)),
        "n_segments_kept": int(len(kept)),
        "n_actors": int(kept["global_actor"].nunique()) if len(kept) else 0,
        "kept_minutes": round(kept["duration"].sum() / 60.0, 2) if len(kept) else 0.0,
        "segment_minutes_total": round(fdf["duration"].sum() / 60.0, 2) if len(fdf) else 0.0,
        "tier_counts": kept["tier"].value_counts().to_dict() if len(kept) else {},
        "clean": clean, "noisy": noisy,
        "clean_pct": round(100 * clean / (clean + noisy), 1) if (clean + noisy) else 0.0,
        "exclude_reasons": reasons,
        "params_hash": params_hash,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "clips_dir": str(config.CORPUS_DIR / resource["region"] / resource["playlist"]),
        "episodes": episodes_meta,
    }


def process_resource(
    resource: dict,
    *,
    hf_token: str | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Process (or skip, if unchanged) one resource. Returns its metadata record with a
    ``status`` of ``processed`` / ``skipped`` / ``empty``."""
    region, playlist = resource["region"], resource["playlist"]
    p = resource_paths(region, playlist)
    p["dir"].mkdir(parents=True, exist_ok=True)

    all_ids = [e["file_id"] for e in resource["episodes"]]
    sig = config.params_signature()
    cur_hash = state.params_hash(sig)

    if force and p["state"].exists():
        p["state"].unlink()
    plan = state.plan_run(p["state"], sig, all_ids)
    need = force or plan["mode"] == "reset" or bool(plan["to_process"]) or not p["final"].exists()

    if not need:
        final_df = pd.read_csv(p["final"])
        meta = build_resource_metadata(resource, final_df, cur_hash)
        meta["status"] = "skipped"
        _write_json(p["metadata"], meta)
        if verbose:
            print(f"   = {region}/{playlist}: up to date "
                  f"({meta['n_segments_kept']} kept, {meta['n_actors']} actors)")
        return meta

    # ---- rebuild the whole resource (keeps manifest <-> embeddings row-aligned) ----
    rows: list[dict] = []
    _pipe = None  # pyannote pipeline, loaded once iff some episode is missing its RTTM
    for e in resource["episodes"]:
        try:
            wav, sr = segment.load_mono_waveform(e["audio_path"])
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"   ! audio load failed {e['basename']}: {exc}; hard-max split")
            wav, sr = None, config.SAMPLE_RATE
        rttm_path = e.get("rttm_path")
        if not rttm_path:
            if hf_token:
                from libyan_did.shared import diarize
                if _pipe is None:
                    _pipe = diarize._load_pipeline(hf_token)
                rttm_path = str(diarize.diarize_episode(
                    e["audio_path"], e["file_id"], pipeline=_pipe,
                    out_dir=Path(e["audio_path"]).parent))  # sibling .rttm -> persistent + idempotent
            else:
                if verbose:
                    print(f"   ! no RTTM for {e['basename']} and no --hf-token; skipped")
                continue
        rows.extend(segment.process_file_metadata(
            rttm_path, file_id=e["file_id"], macro_class=resource["macro"],
            sub_cat=region, playlist=playlist, source=playlist,
            master_audio_path=e["audio_path"], waveform=wav, sr=sr,
        ))

    seg_df = pd.DataFrame(rows, columns=segment.MANIFEST_COLUMNS)
    seg_df.to_csv(p["segments"], index=False)

    if len(seg_df) == 0:
        final_df = seg_df.assign(global_actor=-1, tier="reject", exclude_reason="empty")
        final_df.to_csv(p["final"], index=False)
        np.save(p["embeddings"], np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32))
        meta = build_resource_metadata(resource, final_df, cur_hash)
        meta["status"] = "empty"
        _write_json(p["metadata"], meta)
        state.save_state(p["state"], {"params_hash": cur_hash, "processed_file_ids": sorted(all_ids)})
        if verbose:
            print(f"   ! {region}/{playlist}: produced no segments")
        return meta

    embed.embed_manifest(manifest_csv=p["segments"], out_npy=p["embeddings"])
    return _cluster_and_finalize(resource, p, cur_hash, all_ids,
                                 status="processed", verbose=verbose)


def _cluster_and_finalize(
    resource: dict,
    p: dict,
    cur_hash: str,
    all_ids: list[str],
    *,
    status: str,
    verbose: bool,
) -> dict:
    """Shared tail: cluster the cached manifest+embeddings, export clips, write
    final manifest / merge candidates / metadata / state."""
    region, playlist = resource["region"], resource["playlist"]
    seg_df = pd.read_csv(p["segments"])
    emb = np.load(p["embeddings"])
    if len(seg_df) != len(emb):
        raise RuntimeError(
            f"{region}/{playlist}: manifest ({len(seg_df)}) and embeddings ({len(emb)}) "
            "row counts differ — rebuild the resource with --force")

    final_df, merge_candidates = cluster.run_clustering(seg_df, emb)
    final_df.to_csv(p["final"], index=False)
    merge_candidates.to_csv(p["merge_candidates"], index=False)

    clips = export.export_resource_clips(final_df, dest_dir=p["dir"])
    meta = build_resource_metadata(resource, final_df, cur_hash)
    meta["status"] = status
    meta["n_clips_written"] = len(clips)
    meta["n_merge_candidates"] = int(len(merge_candidates))
    _write_json(p["metadata"], meta)
    state.save_state(p["state"], {"params_hash": cur_hash, "processed_file_ids": sorted(all_ids)})
    if verbose:
        print(f"   + {region}/{playlist}: {meta['n_segments_kept']}/{meta['n_segments_total']} kept, "
              f"{meta['n_actors']} actors, {len(clips)} clips, clean {meta['clean_pct']}%, "
              f"{len(merge_candidates)} merge candidate(s)")
    return meta


def recluster_resource(resource: dict, *, verbose: bool = True) -> dict:
    """Re-run ONLY clustering + clip export on the cached manifest + embeddings.

    Used when clustering logic/thresholds change (these are not in params_signature,
    so the state lock would otherwise skip the resource). No audio is re-decoded and
    no embeddings are recomputed. Falls back to status="missing_inputs" when the
    cached files are absent (run process_resource for those).
    """
    region, playlist = resource["region"], resource["playlist"]
    p = resource_paths(region, playlist)
    if not (p["segments"].exists() and p["embeddings"].exists()):
        if verbose:
            print(f"   ! {region}/{playlist}: no cached segments/embeddings — needs a full run")
        return {"status": "missing_inputs", "region": region, "playlist": playlist}

    all_ids = [e["file_id"] for e in resource["episodes"]]
    cur_hash = state.params_hash(config.params_signature())
    # An empty resource (produced no segments) has a header-only master_segments.csv;
    # clustering it would KeyError. Nothing to recluster — its final_manifest is already empty.
    if len(pd.read_csv(p["segments"])) == 0:
        if verbose:
            print(f"   = {region}/{playlist}: no segments — recluster skipped")
        return {"status": "empty", "region": region, "playlist": playlist}
    return _cluster_and_finalize(resource, p, cur_hash, all_ids,
                                 status="reclustered", verbose=verbose)
