"""Regenerate Figure 4 (C1_lid_baseline) for the camera-ready (plan item P0-4).

Copied from the C1 block of make_corpus_figures.py with two changes:
  - reads the paper snapshot (splits_manifest.csv, 799 actor_uid) instead of
    verified_manifest.csv, which is post-expansion and no longer the paper;
  - the title says "playlist-level speaker clusters", not "speakers".
Numbers are unchanged (74% overall, 78% test, Fezzan 68%).

    python -m libyan_did.phase6_reporting.make_c1_camera_ready
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from libyan_did.shared import config

OUT = config.PROJECT_ROOT / "manuscripts" / "corpus_paper" / "figures"
REGIONS = ["Tripolitania", "Cyrenaica", "Fezzan"]
REGION_PAL = {"Tripolitania": "#2980b9", "Cyrenaica": "#27ae60", "Fezzan": "#e67e22"}


def main() -> None:
    sp = pd.read_csv(config.MANIFEST_DIR / "splits_manifest.csv", usecols=["actor_uid", "split"])
    sp["actor_uid"] = sp["actor_uid"].astype(str)
    sc = pd.read_csv(config.METADATA_DIR / "actor_scores.csv")
    sc["actor_uid"] = sc["actor_uid"].astype(str)
    ksc = sc[sc["actor_uid"].isin(set(sp["actor_uid"]))].copy()
    assert len(ksc) == 799, len(ksc)

    ksc["is_lib"] = ksc["top_dialect"] == "LIB"
    rec_all = {"Overall": ksc["is_lib"].mean()}
    for r in REGIONS:
        rec_all[r] = ksc.loc[ksc["region"] == r, "is_lib"].mean()
    test_uids = set(sp.loc[sp["split"] == "test", "actor_uid"])
    rec_test = ksc.loc[ksc["actor_uid"].isin(test_uids), "is_lib"].mean()

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
    axL.set_ylim(0, 100)
    axL.set_ylabel("Top-1 Libyan recall (%)")
    axL.set_title(f"Off-the-shelf Arabic DID recognises only\n"
                  f"{rec_all['Overall']*100:.0f}% of verified playlist-level speaker clusters "
                  f"as Libyan (n={len(ksc)})")
    axL.legend(loc="upper right")

    wrong = ksc[~ksc["is_lib"]]["top_dialect"].value_counts()
    axR.barh(wrong.index[::-1], wrong.values[::-1], color="#c0392b", edgecolor="black")
    for i, v in enumerate(wrong.values[::-1]):
        axR.text(v, i, f" {v}", va="center", fontweight="bold")
    axR.set_xlabel("Verified Libyan speaker clusters mislabelled (count)")
    axR.set_title("When wrong, which dialect it predicts")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"C1_lid_baseline.{ext}", bbox_inches="tight", dpi=300)
    print(f"OK C1_lid_baseline -> {OUT}  overall {rec_all['Overall']*100:.1f}% "
          f"test {rec_test*100:.1f}% " + " ".join(f"{r} {rec_all[r]*100:.0f}%" for r in REGIONS))


if __name__ == "__main__":
    main()
