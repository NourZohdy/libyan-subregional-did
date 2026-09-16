"""21-class v2 corpus figures (F1-F7, PNG + PDF), drawn from the v2 manifests only.

Companion to make_corpus_figures.py (the 3-region Libyan figures). This module reads the
combined 21-class kept frame + splits_manifest_v2.csv and draws the composition / honesty /
split figures with 21 classes. Horizontal layouts throughout — 21 vertical labels are
unreadable, which is why a separate module exists. No model re-runs; every number traces to
a v2 manifest.

Figures (contracts/figures-and-tables.md §Figures):
  F1 composition        hours + distinct speakers per class (21 rows)
  F2 channels           distinct channels per class, ≤2-channel classes annotated
  F3 seg_length         segment-length histogram (3-10 s), Libyan vs Non-Libyan overlay
  F4 top_source_share   per-class top-channel duration share, macro-coloured
  F5 control_provenance ADI-17 / ADI-20 / self-mined episodes per control (needs T016)
  F6 splits             class × split hours grid (coverage heatmap)
  F7 gender_region      male/female Libyan speakers per region (controls excluded)

Run (diarization env, from repo root):
    python -m libyan_did.phase6_reporting.make_v2_corpus_figures
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from libyan_did.shared import config
from libyan_did.shared import diagnostics  # noqa: E402
from libyan_did.phase4_validation.make_corpus_tables import _combined_kept  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
OUT = config.FIGURES_DIR / "paper" / "V2"      # keep the 21-class v2 figures out of the v1 paper/ dir
OUT.mkdir(parents=True, exist_ok=True)

SPLITS_V2 = config.MANIFEST_DIR / "splits_manifest_v2.csv"
PROVENANCE = config.METADATA_DIR / "control_provenance.csv"
SCORES = config.METADATA_DIR / "actor_scores.csv"

REGIONS = list(config.REGIONS)                       # 3 Libyan region names
REGION_PAL = {"Tripolitania": "#2980b9", "Cyrenaica": "#27ae60", "Fezzan": "#e67e22"}
GENDER_PAL = {"male": "#3498db", "female": "#e74c3c"}
CONTROL_COLOR = "#95a5a6"                            # one muted colour for the 18 controls


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {name}")


def _is_libyan(label: str) -> bool:
    return label in REGIONS


def _class_order(per: pd.DataFrame) -> list[str]:
    """Controls by hours (ascending) then the 3 Libyan regions — so in a barh the
    Libyan classes land on top and controls sort by mass below them."""
    idx = per.set_index("dialect")
    ctrls = [c for c in idx.index if not _is_libyan(c)]
    ctrls = idx.loc[ctrls].sort_values("hours").index.tolist()
    libs = idx.loc[[r for r in REGIONS if r in idx.index]].sort_values("hours").index.tolist()
    return ctrls + libs


def _row_colors(order: list[str]) -> list[str]:
    return [REGION_PAL[c] if _is_libyan(c) else CONTROL_COLOR for c in order]


def main():
    kept = _combined_kept()
    per = diagnostics.per_dialect_summary(kept)
    order = _class_order(per)
    per_o = per.set_index("dialect").reindex(order)
    colors = _row_colors(order)
    n = len(order)
    print(f"combined corpus: {len(kept):,} segs | {kept['duration'].sum()/3600:.1f} h | {n} classes")

    # ---------- F1: composition (hours + distinct speakers per class) ----------
    try:
        fig, (axH, axS) = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
        y = np.arange(n)
        axH.barh(y, per_o["hours"], color=colors, edgecolor="black")
        axH.set_yticks(y); axH.set_yticklabels(order)
        axH.set_xlabel("Hours of speech"); axH.invert_xaxis()
        axH.yaxis.tick_right()
        for i, v in enumerate(per_o["hours"]):
            axH.text(v, i, f"{v:.0f} ", va="center", ha="right", fontsize=8)
        axS.barh(y, per_o["distinct_speakers"], color=colors, edgecolor="black")
        axS.set_xlabel("Distinct speakers")
        for i, v in enumerate(per_o["distinct_speakers"]):
            axS.text(v, i, f" {int(v)}", va="center", fontsize=8)
        fig.suptitle("Corpus composition by class (Libyan regions on top, 18 controls below)")
        save(fig, "F1_composition")
    except Exception as e:
        print("  ! F1 failed:", e)

    # ---------- F2: distinct channels per class ----------
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        y = np.arange(n)
        ax.barh(y, per_o["distinct_channels"], color=colors, edgecolor="black")
        ax.set_yticks(y); ax.set_yticklabels(order)
        ax.set_xlabel("Distinct channels (sources)")
        for i, (lbl, v) in enumerate(zip(order, per_o["distinct_channels"])):
            note = "  ⚠ thin (≤2)" if v <= 2 else ""
            ax.text(v, i, f" {int(v)}{note}", va="center", fontsize=8,
                    color="#c0392b" if v <= 2 else "black")
        ax.set_title("Channels per class — ≤2-channel classes flagged (thin control)")
        save(fig, "F2_channels")
    except Exception as e:
        print("  ! F2 failed:", e)

    # ---------- F3: segment-length histogram, Libyan vs Non-Libyan ----------
    try:
        m = kept.copy()
        m["macro"] = np.where(m[diagnostics._dialect_key(m)].isin(REGIONS), "Libyan", "Non-Libyan")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bins = np.linspace(3, 10, 29)
        for macro, col in (("Libyan", "#2980b9"), ("Non-Libyan", CONTROL_COLOR)):
            ax.hist(m[m["macro"] == macro]["duration"], bins=bins, alpha=0.6,
                    label=macro, color=col, edgecolor="white", density=True)
        ax.set_xlim(3, 10); ax.set_xlabel("Segment length (s)"); ax.set_ylabel("Density")
        ax.legend(title="")
        ax.set_title("Segment-length distribution (all mass within the 3–10 s gate)")
        save(fig, "F3_seg_length")
    except Exception as e:
        print("  ! F3 failed:", e)

    # ---------- F4: top-source share per class ----------
    try:
        tss = diagnostics.top_source_share(kept).set_index("dialect").reindex(order)
        fig, ax = plt.subplots(figsize=(8, 8))
        y = np.arange(n)
        ax.barh(y, tss["share_pct"] * 100, color=colors, edgecolor="black")
        ax.set_yticks(y); ax.set_yticklabels(order)
        ax.set_xlabel("Share of class hours from its single largest channel (%)")
        ax.axvline(60, color="#c0392b", ls="--", lw=1.5, label="60% concentration line")
        for i, v in enumerate(tss["share_pct"] * 100):
            ax.text(v, i, f" {v:.0f}%", va="center", fontsize=8)
        ax.legend(loc="lower right")
        ax.set_title("Single-channel concentration per class (higher = less diverse)")
        save(fig, "F4_top_source_share")
    except Exception as e:
        print("  ! F4 failed:", e)

    # ---------- F5: control provenance (needs T016 side-table) ----------
    try:
        if not PROVENANCE.exists():
            print("  ! F5 skipped: control_provenance.csv missing (T016 blocked)")
        else:
            prov = pd.read_csv(PROVENANCE)
            cp = diagnostics.control_provenance(kept, prov).set_index("class")
            cp = cp.sort_values("catalog_pct")
            fig, ax = plt.subplots(figsize=(8, 7))
            yy = np.arange(len(cp))
            left = np.zeros(len(cp))
            for col, color, lbl in (("adi17_episodes", "#2c3e50", "ADI-17"),
                                    ("adi20_episodes", "#16a085", "ADI-20"),
                                    ("self_mined_episodes", CONTROL_COLOR, "self-mined")):
                ax.barh(yy, cp[col], left=left, color=color, edgecolor="white", label=lbl)
                left += cp[col].to_numpy()
            ax.set_yticks(yy); ax.set_yticklabels(cp.index)
            ax.set_xlabel("Control episodes"); ax.legend(title="provenance", loc="lower right")
            ax.set_title("Control provenance: catalog-sourced vs self-mined episodes")
            save(fig, "F5_control_provenance")
    except Exception as e:
        print("  ! F5 failed:", e)

    # ---------- F6: class × split coverage heatmap (hours) ----------
    try:
        if not SPLITS_V2.exists():
            print("  ! F6 skipped: splits_manifest_v2.csv missing (run build_splits)")
        else:
            sp = pd.read_csv(SPLITS_V2)
            ss = diagnostics.split_summary(sp)
            piv = (ss.pivot(index="dialect", columns="split", values="hours")
                   .reindex(index=order[::-1]))          # Libyan on top of the heatmap
            for c in ("train", "dev", "test"):
                if c not in piv.columns:
                    piv[c] = np.nan
            piv = piv[["train", "dev", "test"]]
            fig, ax = plt.subplots(figsize=(6, 9))
            sns.heatmap(piv, annot=True, fmt=".1f", cmap="Blues",
                        cbar_kws={"label": "hours"}, linewidths=0.5, linecolor="white",
                        ax=ax, mask=piv.isna())
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_title("Split coverage (hours) — every class has a filled test cell;\n"
                         "thin classes show an empty dev cell")
            save(fig, "F6_splits")
    except Exception as e:
        print("  ! F6 failed:", e)

    # ---------- F7: gender × region (Libyan only; controls excluded) ----------
    try:
        sc = pd.read_csv(SCORES); sc["actor_uid"] = sc["actor_uid"].astype(str)
        gmap = sc.set_index("actor_uid")["gender_majority"]
        lib = kept[kept[diagnostics._dialect_key(kept)].isin(REGIONS)].copy()
        lib["actor_uid"] = lib["actor_uid"].astype(str)
        lib["gender"] = lib["actor_uid"].map(gmap)
        lib = lib.dropna(subset=["gender"])
        dom = (lib.groupby(["speaker_final", "final_dialect", "gender"])["duration"].sum()
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
            ax.bar(tab.index, tab[g], bottom=bottom, label=g, color=GENDER_PAL[g], edgecolor="black")
            for i, (val, bo) in enumerate(zip(tab[g], bottom)):
                if val:
                    ax.text(i, bo + val / 2, int(val), ha="center", va="center",
                            color="white", fontweight="bold")
            bottom += tab[g].values
        female_share = tab["female"].sum() / tab.sum().sum() * 100
        ax.set_ylabel("Distinct speakers")
        ax.set_title(f"Libyan speaker gender by region ({100-female_share:.0f}% male overall;\n"
                     "controls excluded — upper-bound clustering only)")
        ax.legend(title="")
        save(fig, "F7_gender_region")
    except Exception as e:
        print("  ! F7 failed:", e)

    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
