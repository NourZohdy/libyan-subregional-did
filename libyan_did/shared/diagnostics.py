"""Corpus-diversity diagnostics and paper-ready figures/tables (§4.4, §4.5, #3).

Table functions are pure pandas (testable here). Figure functions lazy-import
matplotlib/seaborn so importing this module never requires a plotting backend.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from libyan_did.shared.config import FIGURES_DIR, TABLES_DIR


def _actor_key(df: pd.DataFrame) -> str:
    """Prefer the post-validation merged identity, then the corpus-unique key;
    fall back for legacy single-run frames."""
    for col in ("speaker_final", "actor_uid", "global_actor", "speaker_id"):
        if col in df.columns:
            return col
    return "speaker_id"


def _dialect_key(df: pd.DataFrame) -> str:
    """Prefer the validator-resolved dialect (wrong_region fixes + merge regions),
    then the raw pipeline region."""
    for col in ("final_dialect", "sub_cat", "region"):
        if col in df.columns:
            return col
    return "sub_cat"


# --------------------------------------------------------------------------- #
# Tables (pure pandas)
# --------------------------------------------------------------------------- #
def per_dialect_summary(df: pd.DataFrame, *, dialect_col: str | None = None) -> pd.DataFrame:
    """Hours, distinct speakers, distinct channels (sources), segment counts per dialect."""
    dialect_col = dialect_col or _dialect_key(df)
    rows = []
    for dialect, g in df.groupby(dialect_col):
        rows.append({
            "dialect": dialect,
            "segments": len(g),
            "hours": round(g["duration"].sum() / 3600.0, 3),
            "distinct_speakers": g[_actor_key(g)].nunique(),
            "distinct_channels": g["source"].nunique(),
            "distinct_episodes": g["file_id"].nunique(),
        })
    return pd.DataFrame(rows).sort_values("dialect").reset_index(drop=True)


def per_speaker_distribution(df: pd.DataFrame, *, actor_col: str | None = None) -> pd.DataFrame:
    """Per-speaker segment-count distribution — flags anchor-dominated corpora (§4.5)."""
    key = actor_col or _actor_key(df)
    counts = df.groupby(key).agg(
        segments=("duration", "size"),
        hours=("duration", lambda s: round(s.sum() / 3600.0, 4)),
        dialect=(_dialect_key(df), "first"),
    ).reset_index().rename(columns={key: "speaker"})
    return counts.sort_values("segments", ascending=False).reset_index(drop=True)


def split_summary(df: pd.DataFrame, *, split_col: str = "split") -> pd.DataFrame:
    """Segments / speakers / hours per (split, dialect) — the paper's split table.

    Keyed on the verified corpus identities (``speaker_final`` / ``final_dialect``)
    when present, so speaker counts reflect human merges and corrected regions.
    """
    acol, dcol = _actor_key(df), _dialect_key(df)
    rows = []
    for (split, dialect), g in df.groupby([split_col, dcol]):
        rows.append({
            "split": split,
            "dialect": dialect,
            "segments": len(g),
            "speakers": g[acol].nunique(),
            "hours": round(g["duration"].sum() / 3600.0, 2),
            "channels": g["source"].nunique() if "source" in g.columns else 0,
        })
    return pd.DataFrame(rows).sort_values([split_col, "dialect"]).reset_index(drop=True)


def top_source_share(df: pd.DataFrame, *, dialect_col: str | None = None,
                     source_col: str = "source") -> pd.DataFrame:
    """Per class, the single channel with the largest duration share (honesty column, D9).

    ``share_pct`` is a fraction in [0, 1] — top channel hours over class total hours.
    """
    dcol = dialect_col or _dialect_key(df)
    rows = []
    for dialect, g in df.groupby(dcol):
        by_src = g.groupby(source_col)["duration"].sum()
        top_src = by_src.idxmax()
        top_hours = by_src.max() / 3600.0
        total_hours = by_src.sum() / 3600.0
        rows.append({
            "dialect": dialect,
            "top_source": top_src,
            "top_source_hours": round(top_hours, 3),
            "share_pct": round(top_hours / total_hours, 4) if total_hours else 0.0,
        })
    return pd.DataFrame(rows).sort_values("share_pct", ascending=False).reset_index(drop=True)


def control_provenance(df: pd.DataFrame, prov: pd.DataFrame, *,
                       class_col: str | None = None) -> pd.DataFrame:
    """ADI-17 / ADI-20 / self-mined episode counts per control class (D8).

    Controls-only: rows whose class label is a ``config.CONTROL_CODES`` code are kept,
    Libyan rows dropped. The class label is resolved via ``_dialect_key`` (A1 fix) — the
    combined 21-class frame's Libyan rows have ``final_dialect`` but may lack a ``class``
    column, so keying on a hard-coded ``"class"`` would raise KeyError. ``prov`` is the
    episode-level side-table (``file_id`` -> provenance); the join is on ``file_id``.
    """
    from libyan_did.shared.config import CONTROL_CODES

    col = class_col or _dialect_key(df)
    controls = df[df[col].isin(CONTROL_CODES)]
    ep = controls[[col, "file_id"]].drop_duplicates().merge(
        prov[["file_id", "provenance"]].drop_duplicates("file_id"), on="file_id", how="left")

    rows = []
    for cls, g in ep.groupby(col):
        adi17 = int((g["provenance"] == "ADI-17").sum())
        adi20 = int((g["provenance"] == "ADI-20").sum())
        self_mined = int((g["provenance"] == "self-mined").sum())
        total = g["file_id"].nunique()
        rows.append({
            "class": cls,
            "adi17_episodes": adi17,
            "adi20_episodes": adi20,
            "self_mined_episodes": self_mined,
            "catalog_pct": round((adi17 + adi20) / total, 4) if total else 0.0,
        })
    return pd.DataFrame(rows).sort_values("class").reset_index(drop=True)


def tier_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "tier" not in df.columns:
        return pd.DataFrame(columns=["tier", "segments", "speakers"])
    rows = []
    for tier, g in df.groupby("tier"):
        rows.append({
            "tier": tier,
            "segments": len(g),
            "speakers": g[_actor_key(g)].nunique(),
        })
    return pd.DataFrame(rows).sort_values("tier").reset_index(drop=True)


def quality_counts(df: pd.DataFrame) -> pd.DataFrame:
    """MASC-style clean/noisy segment counts per dialect, among the kept corpus."""
    if "quality" not in df.columns:
        return pd.DataFrame(columns=["dialect", "clean", "noisy", "clean_pct"])
    rows = []
    for dialect, g in df.groupby(_dialect_key(df)):
        clean = int((g["quality"] == "clean").sum())
        noisy = int((g["quality"] == "noisy").sum())
        tot = clean + noisy
        rows.append({
            "dialect": dialect, "clean": clean, "noisy": noisy,
            "clean_pct": round(100 * clean / tot, 1) if tot else 0.0,
        })
    return pd.DataFrame(rows).sort_values("dialect").reset_index(drop=True)


def write_tables(df: pd.DataFrame, out_dir: Path = TABLES_DIR) -> dict[str, Path]:
    """Write all summary CSVs; returns {name: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, table in {
        "per_dialect_summary": per_dialect_summary(df),
        "per_speaker_distribution": per_speaker_distribution(df),
        "tier_counts": tier_counts(df),
        "quality_counts": quality_counts(df),
    }.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written


# --------------------------------------------------------------------------- #
# Figures (lazy matplotlib/seaborn)
# --------------------------------------------------------------------------- #
def _new_ax(figsize=(8, 5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    return plt, fig, ax


def plot_duration_distribution(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    import seaborn as sns
    plt, fig, ax = _new_ax()
    sns.violinplot(data=df, x=_dialect_key(df), y="duration", ax=ax, cut=0)
    ax.set_title("Segment duration distribution by dialect")
    ax.set_xlabel("dialect"); ax.set_ylabel("duration (s)")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "duration_distribution.png"
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    return path


def plot_speakers_per_dialect(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    import seaborn as sns
    summary = per_dialect_summary(df)
    plt, fig, ax = _new_ax()
    sns.barplot(data=summary, x="dialect", y="distinct_speakers", ax=ax)
    ax.set_title("Distinct speakers per dialect")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "speakers_per_dialect.png"
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    return path


def plot_upload_coverage(temporal: dict, out_dir: Path = FIGURES_DIR):
    """Bar chart of episodes per upload YEAR — the corpus's temporal coverage.

    Takes the ``temporal`` dict from ``combine.temporal_coverage`` (episode-level upload
    dates aren't in the segment manifest). Returns the figure path, or ``None`` when no
    episode is dated yet (run ``backfill_ids`` first).
    """
    by_year = (temporal or {}).get("by_year") or {}
    if not by_year:
        return None
    plt, fig, ax = _new_ax()
    ax.bar(list(by_year.keys()), list(by_year.values()), color="#1f77b4")
    ax.set_title("Corpus coverage by video upload year")
    ax.set_xlabel("upload year"); ax.set_ylabel("episodes")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "upload_coverage.png"
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    return path


def write_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    return [
        plot_duration_distribution(df, out_dir),
        plot_speakers_per_dialect(df, out_dir),
    ]
