"""Paper FINDING figures from the VERIFIED corpus (six figures, PNG + PDF).

Companion to make_paper_figures.py (which draws the acoustic/method figures on
the raw manifest). This script draws the corpus- and finding-level figures on
the post-validation data and writes them next to the others in
outputs/figures/paper/. Every number comes from cached CSVs — no model re-runs.

Figures (see docs / plan):
  C1 lid_baseline   off-the-shelf Arabic-DID Libyan recall + confusion (headline)
  C2 composition    hours + speakers per region
  C3 risk_validity  risk_score by human decision (triage AUC)
  C4 gender_region  male/female speakers per region
  C5 speaker_mass   Lorenz curve + Gini of speaker-hours
  C6 splits         train/dev/test hours per region

Run (diarization env, from libyan_did_pipeline/):
    python scripts/make_corpus_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from libyan_did.shared import config
from libyan_did.shared import diagnostics  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
OUT = config.FIGURES_DIR / "paper"
OUT.mkdir(parents=True, exist_ok=True)

VERIFIED = config.MANIFEST_DIR / "verified_manifest.csv"
SPLITS = config.MANIFEST_DIR / "splits_manifest.csv"
SCORES = config.METADATA_DIR / "actor_scores.csv"
VERIFS = config.GOLD_DIR / "speaker_verifications.csv"
_TRUE = {"true", "1", "1.0", "yes"}

REGIONS = ["Tripolitania", "Cyrenaica", "Fezzan"]
REGION_PAL = {"Tripolitania": "#2980b9", "Cyrenaica": "#27ae60", "Fezzan": "#e67e22"}
GENDER_PAL = {"male": "#3498db", "female": "#e74c3c"}


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {name}")


def _auc(reject: np.ndarray, keep: np.ndarray) -> float:
    """Rank AUC = P(risk(reject) > risk(keep)), ties counted as 0.5."""
    if not len(reject) or not len(keep):
        return float("nan")
    gt = np.greater.outer(reject, keep).sum()
    eq = np.equal.outer(reject, keep).sum()
    return (gt + 0.5 * eq) / (len(reject) * len(keep))


def main():
    vm = pd.read_csv(VERIFIED)
    vm["is_excluded"] = vm["is_excluded"].astype(str).str.strip().str.lower().isin(_TRUE)
    kept = vm[vm["verified_keep"].astype(bool) & (~vm["is_excluded"])].copy()
    sc = pd.read_csv(SCORES); sc["actor_uid"] = sc["actor_uid"].astype(str)
    kept_uids = set(kept["actor_uid"].astype(str))
    ksc = sc[sc["actor_uid"].isin(kept_uids)].copy()      # verified-Libyan actors w/ LID scores
    print(f"verified corpus: {len(kept):,} segs | {kept['duration'].sum()/3600:.1f} h | "
          f"{kept['speaker_final'].nunique()} speakers | scored actors {len(ksc)}")

    # ---------- C1: off-the-shelf DID baseline (headline) ----------
    try:
        sp = pd.read_csv(SPLITS)
        amap = ksc.set_index("actor_uid")
        # actor -> (top_dialect, region) for all verified, and the test subset
        ksc["is_lib"] = (ksc["top_dialect"] == "LIB")
        rec_all = {"Overall": ksc["is_lib"].mean()}
        for r in REGIONS:
            sub = ksc[ksc["region"] == r]
            rec_all[r] = sub["is_lib"].mean() if len(sub) else np.nan
        test_uids = set(sp[sp["split"] == "test"]["actor_uid"].astype(str))
        ktest = ksc[ksc["actor_uid"].isin(test_uids)]
        rec_test = ktest["is_lib"].mean() if len(ktest) else np.nan

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8),
                                       gridspec_kw={"width_ratios": [1.05, 1]})
        cats = ["Overall"] + REGIONS
        vals = [rec_all[c] * 100 for c in cats]
        cols = ["#34495e"] + [REGION_PAL[r] for r in REGIONS]
        bars = axL.bar(cats, vals, color=cols, edgecolor="black")
        for b, v in zip(bars, vals):
            axL.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
                     ha="center", va="bottom", fontweight="bold")
        axL.axhline(rec_test * 100, color="red", ls="--", lw=2,
                    label=f"held-out test split: {rec_test*100:.0f}%")
        axL.set_ylim(0, 100); axL.set_ylabel("Top-1 Libyan recall (%)")
        axL.set_title(f"Off-the-shelf Arabic DID recognises only\n"
                      f"{rec_all['Overall']*100:.0f}% of verified Libyan speakers as Libyan "
                      f"(n={len(ksc)})")
        axL.legend(loc="upper right")

        wrong = ksc[~ksc["is_lib"]]["top_dialect"].value_counts()
        axR.barh(wrong.index[::-1], wrong.values[::-1], color="#c0392b", edgecolor="black")
        for i, v in enumerate(wrong.values[::-1]):
            axR.text(v, i, f" {v}", va="center", fontweight="bold")
        axR.set_xlabel("Verified Libyan speakers mislabelled (count)")
        axR.set_title("When wrong, which dialect it predicts")
        save(fig, "C1_lid_baseline")
    except Exception as e:
        print("  ! C1 failed:", e)

    # ---------- C2: corpus composition (hours + speakers per region) ----------
    try:
        s = diagnostics.per_dialect_summary(kept)
        s = s.set_index("dialect").reindex(REGIONS).reset_index()
        x = np.arange(len(REGIONS)); w = 0.38
        fig, ax = plt.subplots(figsize=(8, 4.8))
        b1 = ax.bar(x - w / 2, s["hours"], w, label="hours", color="#16a085", edgecolor="black")
        ax.set_ylabel("Hours of speech"); ax.set_xticks(x); ax.set_xticklabels(REGIONS)
        ax.bar_label(b1, fmt="%.1f", fontweight="bold")
        ax2 = ax.twinx()
        b2 = ax2.bar(x + w / 2, s["distinct_speakers"], w, label="speakers",
                     color="#8e44ad", edgecolor="black")
        ax2.set_ylabel("Distinct speakers"); ax2.grid(False)
        ax2.bar_label(b2, fmt="%d", fontweight="bold")
        ax.set_title("Corpus composition by region")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")
        save(fig, "C2_composition")
    except Exception as e:
        print("  ! C2 failed:", e)

    # ---------- C3: risk_score by human decision (triage validity) ----------
    try:
        v = pd.read_csv(VERIFS).dropna(subset=["risk_score"])
        order = ["verified", "wrong_region", "unsure", "msa", "mixed_cluster",
                 "non_libyan", "bad_audio"]
        order = [d for d in order if d in set(v["decision"])]
        keep = v[v["decision"].isin(["verified", "wrong_region"])]["risk_score"].values
        rej = v[v["decision"].isin(["bad_audio", "non_libyan", "msa"])]["risk_score"].values
        auc = _auc(rej, keep)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        pal = {"verified": "#27ae60", "wrong_region": "#2ecc71", "unsure": "#95a5a6",
               "msa": "#f1c40f", "mixed_cluster": "#e67e22", "non_libyan": "#e74c3c",
               "bad_audio": "#c0392b"}
        sns.boxplot(data=v, x="decision", y="risk_score", order=order, ax=ax,
                    palette=pal, showfliers=False)
        sns.stripplot(data=v, x="decision", y="risk_score", order=order, ax=ax,
                      color="black", size=2, alpha=0.25)
        ax.set_xlabel(""); ax.set_ylabel("risk_score")
        ax.set_title(f"Automated risk score sorts rejects above keeps "
                     f"(AUC = {auc:.2f})")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        save(fig, "C3_risk_validity")
    except Exception as e:
        print("  ! C3 failed:", e)

    # ---------- C4: gender x region (per final speaker) ----------
    try:
        gmap = sc.set_index("actor_uid")["gender_majority"]
        kg = kept[["speaker_final", "actor_uid", "final_dialect", "duration"]].copy()
        kg["actor_uid"] = kg["actor_uid"].astype(str)
        kg["gender"] = kg["actor_uid"].map(gmap)
        kg = kg.dropna(subset=["gender"])
        # duration-dominant gender per final speaker
        dom = (kg.groupby(["speaker_final", "final_dialect", "gender"])["duration"].sum()
               .reset_index().sort_values("duration", ascending=False)
               .drop_duplicates("speaker_final"))
        tab = (dom.groupby(["final_dialect", "gender"])["speaker_final"].nunique()
               .unstack(fill_value=0).reindex(REGIONS))
        for g in ("male", "female"):
            if g not in tab.columns:
                tab[g] = 0
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bottom = np.zeros(len(tab))
        for g in ("male", "female"):
            ax.bar(tab.index, tab[g], bottom=bottom, label=g, color=GENDER_PAL[g],
                   edgecolor="black")
            for i, (val, bo) in enumerate(zip(tab[g], bottom)):
                if val:
                    ax.text(i, bo + val / 2, int(val), ha="center", va="center",
                            color="white", fontweight="bold")
            bottom += tab[g].values
        female_share = tab["female"].sum() / tab.sum().sum() * 100
        ax.set_ylabel("Distinct speakers")
        ax.set_title(f"Speaker gender by region "
                     f"({100-female_share:.0f}% male overall)")
        ax.legend(title="")
        save(fig, "C4_gender_region")
    except Exception as e:
        print("  ! C4 failed:", e)

    # ---------- C5: Lorenz curve + Gini of speaker-hours ----------
    try:
        sph = (kept.groupby("speaker_final")["duration"].sum() / 3600).sort_values().values
        n = len(sph); cum = np.cumsum(sph); cum = cum / cum[-1]
        lorenz_x = np.arange(1, n + 1) / n
        gini = (2 * np.arange(1, n + 1) - n - 1).dot(sph) / (n * sph.sum())
        fig, ax = plt.subplots(figsize=(6.2, 6))
        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.5, label="perfect equality")
        ax.plot(np.concatenate([[0], lorenz_x]), np.concatenate([[0], cum]),
                color="#2c3e50", lw=2.5, label=f"speakers (Gini = {gini:.2f})")
        ax.fill_between(np.concatenate([[0], lorenz_x]), np.concatenate([[0], cum]),
                        np.concatenate([[0], lorenz_x]), color="#2c3e50", alpha=0.12)
        ax.set_xlabel("Cumulative share of speakers (smallest → largest)")
        ax.set_ylabel("Cumulative share of speech hours")
        ax.set_title(f"Speaker-mass concentration ({n} speakers)\n"
                     f"top 10 speakers hold {sph[::-1][:10].sum()/sph.sum()*100:.0f}% of audio")
        ax.legend(loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        save(fig, "C5_speaker_mass")
    except Exception as e:
        print("  ! C5 failed:", e)

    # ---------- C6: train/dev/test hours per region ----------
    try:
        sp = pd.read_csv(SPLITS)
        ss = diagnostics.split_summary(sp)
        piv = ss.pivot(index="split", columns="dialect", values="hours").reindex(
            ["train", "dev", "test"])[REGIONS]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bottom = np.zeros(len(piv))
        for r in REGIONS:
            ax.bar(piv.index, piv[r], bottom=bottom, label=r, color=REGION_PAL[r],
                   edgecolor="black")
            for i, (val, bo) in enumerate(zip(piv[r], bottom)):
                if val and val > 1.5:
                    ax.text(i, bo + val / 2, f"{val:.1f}", ha="center", va="center",
                            color="white", fontweight="bold", fontsize=9)
            bottom += piv[r].values
        for i, tot in enumerate(bottom):
            ax.text(i, tot + 1, f"{tot:.1f} h", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Hours of speech")
        ax.set_title("Speaker-disjoint split composition (80/10/10 by duration)")
        ax.legend(title="region", loc="upper right")
        save(fig, "C6_splits")
    except Exception as e:
        print("  ! C6 failed:", e)

    # ---------- C7: sub-regional DID test confusion (if benchmark has run) ----------
    try:
        cmpath = config.TABLES_DIR / "did_confusion_test_lid_finetune.csv"
        if cmpath.exists():
            cm = pd.read_csv(cmpath, index_col=0)
            labels = [c.replace("pred_", "") for c in cm.columns]
            M = cm.to_numpy(dtype=float)
            rown = M / M.sum(axis=1, keepdims=True)              # recall per true row
            annot = np.array([[f"{rown[i, j]*100:.0f}%\n({int(M[i, j])})"
                               for j in range(len(labels))] for i in range(len(labels))])
            sub = ""
            rpath = config.TABLES_DIR / "did_results.csv"
            if rpath.exists():
                res = pd.read_csv(rpath)
                r = res[(res["model"] == "LID-ECAPA-FT (E6)") & (res["level"] == "segment")
                        & (res["split"] == "test")]
                if len(r):
                    sub = (f"\nsegment-level test: acc {float(r['accuracy'].iloc[0]):.2f}, "
                           f"macro-F1 {float(r['macro_f1'].iloc[0]):.2f}")
            fig, ax = plt.subplots(figsize=(5.8, 5.2))
            sns.heatmap(rown, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,
                        cbar_kws={"label": "row-normalized (recall)"},
                        xticklabels=labels, yticklabels=labels, ax=ax, square=True)
            ax.set_xlabel("predicted region"); ax.set_ylabel("true region")
            ax.set_title("Sub-regional DID confusion (LID-ECAPA fine-tuned)" + sub)
            save(fig, "C7_did_confusion")
    except Exception as e:
        print("  ! C7 failed:", e)

    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
