"""Generate paper-ready figures from the manifests + embeddings.

Reads final_manifest.csv (+ embeddings.npy, splits_manifest.csv, raw audio) and writes
every figure as BOTH 300-dpi PNG and vector PDF (for LaTeX) into outputs/figures/paper/.

Run (in the diarization env, from libyan_did_pipeline/):
    python scripts/make_paper_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from libyan_did.shared import config

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
OUT = config.FIGURES_DIR / "paper"
OUT.mkdir(parents=True, exist_ok=True)

FINAL = config.MANIFEST_DIR / "final_manifest.csv"
SPLITS = config.MANIFEST_DIR / "splits_manifest.csv"

PAL = {"Tier1": "#2ecc71", "Tier2": "#3498db", "Tier3": "#f39c12", "reject": "#e74c3c"}


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def raw_minutes() -> float:
    import soundfile as sf
    total = 0.0
    for p in sorted(config.RAW_AUDIO_DIR.glob("*.wav")):
        info = sf.info(str(p))
        total += info.frames / info.samplerate
    return total / 60.0


# --------------------------------------------------------------------------- #
def main():
    df = pd.read_csv(FINAL)
    emb = np.load(config.MANIFEST_DIR / "embeddings.npy")
    kept = df[~df["is_excluded"]].copy()
    print(f"final_manifest: {len(df)} segs | kept {len(kept)} | actors {kept['global_actor'].nunique()}")

    # ---------- F1: Acoustic funnel (minutes) ----------
    try:
        seg_min = df["duration"].sum() / 60.0
        after_speech = df[df["exclude_reason"] != "low_speech"]["duration"].sum() / 60.0
        after_snr = df[~df["exclude_reason"].isin(["low_speech", "low_snr"])]["duration"].sum() / 60.0
        kept_min = kept["duration"].sum() / 60.0
        try:
            raw = raw_minutes()
        except Exception:
            raw = float("nan")
        stages = ["Raw\naudio", "Single-spk\nsegments", "After music/\nnoise gate",
                  "After SNR\nfloor", "Kept\n(MAD-clean)"]
        vals = [raw, seg_min, after_speech, after_snr, kept_min]
        colors = ["#95a5a6", "#34495e", "#9b59b6", "#2980b9", "#2ecc71"]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(stages, vals, color=colors, edgecolor="black")
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} min",
                        ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Audio (minutes)")
        ax.set_title("Acoustic curation funnel")
        save(fig, "F1_acoustic_funnel")
    except Exception as e:
        print("  ! F1 failed:", e)

    # ---------- F2: Exclusion reasons ----------
    try:
        rc = df[df["is_excluded"]]["exclude_reason"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(x=rc.values, y=rc.index, ax=ax, color="#e74c3c", edgecolor="black")
        for i, v in enumerate(rc.values):
            ax.text(v, i, f" {v}", va="center", fontweight="bold")
        ax.set_xlabel("Segments removed"); ax.set_ylabel("")
        ax.set_title("Why segments were excluded")
        save(fig, "F2_exclusion_reasons")
    except Exception as e:
        print("  ! F2 failed:", e)

    # ---------- F3: Duration distribution ----------
    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(kept["duration"], bins=30, kde=True, color="#3498db", edgecolor="black", ax=ax)
        m = kept["duration"].mean()
        ax.axvline(m, color="red", ls="--", lw=2, label=f"mean {m:.2f}s")
        ax.set_xlabel("Segment duration (s)"); ax.set_ylabel("Count")
        ax.set_title("Kept-segment duration distribution"); ax.legend()
        save(fig, "F3_duration_distribution")
    except Exception as e:
        print("  ! F3 failed:", e)

    # ---------- F4: Speaker-mass Zipf/Pareto with tier lines ----------
    try:
        cnt = kept.groupby("global_actor").size().sort_values(ascending=False)
        tier_of = kept.groupby("global_actor")["tier"].agg(lambda s: s.mode().iat[0])
        ranks = np.arange(1, len(cnt) + 1)
        cols = [PAL.get(tier_of[a], "#7f8c8d") for a in cnt.index]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(ranks, cnt.values, c=cols, s=45, edgecolors="black", zorder=3)
        ax.plot(ranks, cnt.values, color="#2c3e50", lw=1, alpha=0.5, zorder=2)
        ax.set_yscale("log")
        # tier boundary vlines
        tiers_ranked = [tier_of[a] for a in cnt.index]
        for t, lbl in [("Tier1", "T1|T2"), ("Tier2", "T2|T3")]:
            idxs = [i for i, tt in enumerate(tiers_ranked) if tt == t]
            if idxs:
                ax.axvline(max(idxs) + 1.5, color="gray", ls=":", lw=1.5)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=PAL[t], label=t) for t in ["Tier1", "Tier2", "Tier3"]])
        ax.set_xlabel("Actor rank"); ax.set_ylabel("Segments (log)")
        ax.set_title("Speaker-mass distribution (Zipf) with Kneedle tiers")
        save(fig, "F4_speaker_mass_zipf")
    except Exception as e:
        print("  ! F4 failed:", e)

    # ---------- F5: Tier composition ----------
    try:
        g = kept.groupby("tier").agg(segments=("duration", "size"),
                                     speakers=("global_actor", "nunique")).reset_index()
        g = g[g["tier"].isin(["Tier1", "Tier2", "Tier3"])]
        gm = g.melt(id_vars="tier", value_vars=["segments", "speakers"])
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.barplot(data=gm, x="tier", y="value", hue="variable", ax=ax, edgecolor="black")
        for c in ax.containers:
            ax.bar_label(c, fontweight="bold")
        ax.set_yscale("log"); ax.set_xlabel(""); ax.set_ylabel("count (log)")
        ax.set_title("Tier composition")
        save(fig, "F5_tier_composition")
    except Exception as e:
        print("  ! F5 failed:", e)

    # ---------- F6: Audio by duration bucket ----------
    try:
        buckets = config.DURATION_BUCKETS
        rows = []
        for name, (lo, hi) in buckets.items():
            mins = kept[(kept["duration"] >= lo) & (kept["duration"] < hi)]["duration"].sum() / 60.0
            rows.append({"bucket": f"{name}\n{lo:.0f}-{hi:.0f}s", "minutes": mins})
        bdf = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.barplot(data=bdf, x="bucket", y="minutes", ax=ax, color="#16a085", edgecolor="black")
        ax.bar_label(ax.containers[0], fmt="%.1f", fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("Audio (min)")
        ax.set_title("Kept audio by duration bucket")
        save(fig, "F6_duration_buckets")
    except Exception as e:
        print("  ! F6 failed:", e)

    # ---------- F7: SNR distribution ----------
    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df["snr_db"].dropna(), bins=40, color="#8e44ad", edgecolor="black", ax=ax)
        ax.axvline(config.SNR_CLEAN_THRESHOLD_DB, color="green", ls="--", lw=2,
                   label=f"clean ≥ {config.SNR_CLEAN_THRESHOLD_DB:.0f} dB")
        ax.axvline(config.SNR_REJECT_FLOOR_DB, color="red", ls="--", lw=2,
                   label=f"reject < {config.SNR_REJECT_FLOOR_DB:.0f} dB")
        ax.set_xlabel("SNR proxy (dB)"); ax.set_ylabel("Count")
        ax.set_title("Per-segment SNR distribution"); ax.legend()
        save(fig, "F7_snr_distribution")
    except Exception as e:
        print("  ! F7 failed:", e)

    # ---------- F8: speech_ratio distribution + gate ----------
    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(data=df, x="speech_ratio", bins=40, hue=df["is_excluded"].map({True: "excluded", False: "kept"}),
                     multiple="stack", edgecolor="black", ax=ax, palette={"kept": "#2ecc71", "excluded": "#e74c3c"})
        ax.axvline(config.MIN_SPEECH_RATIO, color="black", ls="--", lw=2,
                   label=f"gate = {config.MIN_SPEECH_RATIO}")
        ax.set_xlabel("VAD speech ratio"); ax.set_ylabel("Count")
        ax.set_title("Speech-density distribution (music/noise pile up near 0)")
        ax.legend(title="")
        save(fig, "F8_speech_ratio_gate")
    except Exception as e:
        print("  ! F8 failed:", e)

    # ---------- F9: speech_ratio x SNR decision scatter ----------
    try:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for lab, sub, c in [("kept", df[~df["is_excluded"]], "#2ecc71"),
                            ("excluded", df[df["is_excluded"]], "#e74c3c")]:
            ax.scatter(sub["speech_ratio"], sub["snr_db"], s=14, alpha=0.5, c=c, label=lab, edgecolors="none")
        ax.axvline(config.MIN_SPEECH_RATIO, color="black", ls="--", lw=1.5)
        ax.axhline(config.SNR_CLEAN_THRESHOLD_DB, color="green", ls=":", lw=1.5)
        ax.axhline(config.SNR_REJECT_FLOOR_DB, color="red", ls=":", lw=1.5)
        ax.set_xlabel("VAD speech ratio"); ax.set_ylabel("SNR proxy (dB)")
        ax.set_title("Decision space: speech-density × SNR"); ax.legend()
        save(fig, "F9_speechratio_snr_scatter")
    except Exception as e:
        print("  ! F9 failed:", e)

    # ---------- F10: clean/noisy split ----------
    try:
        q = kept["quality"].value_counts()
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.barplot(x=q.index, y=q.values, ax=ax,
                    palette={"clean": "#2ecc71", "noisy": "#f39c12"}, edgecolor="black")
        ax.bar_label(ax.containers[0], fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("Segments")
        ax.set_title("Clean vs noisy (kept corpus)")
        save(fig, "F10_clean_noisy")
    except Exception as e:
        print("  ! F10 failed:", e)

    # ---------- F11: t-SNE of embeddings by actor ----------
    try:
        from sklearn.manifold import TSNE
        ek = emb[kept.index.values]
        ga = kept["global_actor"].values
        top = pd.Series(ga).value_counts().index[:8].tolist()
        n = len(ek)
        ts = TSNE(n_components=2, metric="cosine", init="pca",
                  perplexity=min(30, max(5, n // 20)), random_state=42)
        xy = ts.fit_transform(ek)
        fig, ax = plt.subplots(figsize=(8, 7))
        mask_other = ~np.isin(ga, top)
        ax.scatter(xy[mask_other, 0], xy[mask_other, 1], s=10, c="#cccccc", label="other", edgecolors="none")
        palette = sns.color_palette("tab10", len(top))
        for a, col in zip(top, palette):
            m = ga == a
            ax.scatter(xy[m, 0], xy[m, 1], s=20, color=col, label=f"actor {a}", edgecolors="none")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("t-SNE of ECAPA embeddings (top-8 actors)")
        ax.legend(markerscale=2, fontsize=8, ncol=2)
        save(fig, "F11_tsne_actors")
    except Exception as e:
        print("  ! F11 failed:", e)

    # ---------- F12 + F13: cross-episode centroid heatmap + dendrogram ----------
    try:
        from sklearn.metrics.pairwise import cosine_distances
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        cents, keys, tiers = [], [], []
        for (fid, spk), idx in kept.groupby(["file_id", "speaker_id"]).groups.items():
            idx = list(idx)
            c = emb[idx].mean(axis=0)
            c /= (np.linalg.norm(c) or 1.0)
            cents.append(c); keys.append((fid, spk))
            tiers.append(kept.loc[idx[0], "global_actor"])
        C = np.array(cents)
        order = np.argsort(tiers)
        D = cosine_distances(C[order])
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        sns.heatmap(D, cmap="viridis_r", vmin=0, vmax=0.8, square=True, cbar_kws={"label": "cosine distance"},
                    xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title("Cross-episode local-speaker distance matrix\n(block-diagonal = same actor across episodes)")
        save(fig, "F12_crossepisode_heatmap")

        if len(C) > 2:
            Z = linkage(squareform(cosine_distances(C), checks=False), method="average")
            fig, ax = plt.subplots(figsize=(11, 4.5))
            dendrogram(Z, ax=ax, color_threshold=config.AHC_MERGE_THRESHOLD, no_labels=True)
            ax.axhline(config.AHC_MERGE_THRESHOLD, color="red", ls="--",
                       label=f"merge threshold {config.AHC_MERGE_THRESHOLD}")
            ax.set_ylabel("cosine distance"); ax.set_title("Global AHC dendrogram (local speakers → actors)")
            ax.legend()
            save(fig, "F13_ahc_dendrogram")
    except Exception as e:
        print("  ! F12/F13 failed:", e)

    # ---------- F14: split composition ----------
    try:
        if SPLITS.exists():
            sp = pd.read_csv(SPLITS)
            g = sp.groupby("split").agg(segments=("duration", "size"),
                                        speakers=("global_actor", "nunique")).reset_index()
            gm = g.melt(id_vars="split", value_vars=["segments", "speakers"])
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            sns.barplot(data=gm, x="split", y="value", hue="variable", ax=ax, edgecolor="black",
                        order=["train", "dev", "test"])
            for c in ax.containers:
                ax.bar_label(c, fontweight="bold")
            ax.set_yscale("log"); ax.set_xlabel(""); ax.set_ylabel("count (log)")
            ax.set_title("Train/dev/test composition")
            save(fig, "F14_split_composition")
    except Exception as e:
        print("  ! F14 failed:", e)

    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
