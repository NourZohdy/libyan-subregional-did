"""Channel-clustered bootstrap CIs for the channel-disjoint E6 result (camera-ready P1-3).

The rebuttal to pVFL Q3 / 2RmL quoted channel-clustered intervals that no script in
the repo produced -- they were computed ad hoc during the rebuttal (method recorded in
manuscripts/corpus_paper/reviews/MASTER_REVIEW_AND_ANSWERS.md, B5/B6). This script is
that computation, made permanent.

The frozen prediction file did_lid_finetune_preds_test_chan.csv carries no channel key,
so the channel-disjoint split is rebuilt with the same assign_channels() the E6 run used
and joined POSITIONALLY: did_finetune_lid.py builds `te` with reset_index and loads it
with shuffle=False, so row i of the predictions is row i of the rebuilt test frame. The
join is checked, not assumed -- see check() below.

Never join frozen artifacts to current data on speaker_final: those ids are reassigned
whenever linking re-runs, which silently drops rows (that mistake once produced 13
Fezzan channels instead of 12).

Outputs:
  outputs/tables/did_channel_disjoint_ci.csv   per-class + macro F1 with CIs at 3 units
  outputs/tables/did_channel_counts.csv        test channels/speakers/segments per region

Run (from repo root):
    python -m libyan_did.phase5_did.did_channel_ci
    python -m libyan_did.phase5_did.did_channel_ci --check   # alignment assert only
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from libyan_did.shared import config
from libyan_did.phase5_did.did_channel_disjoint import assign_channels

REGIONS = ["Tripolitania", "Cyrenaica", "Fezzan"]
SEED, N_BOOT = 1337, 1000
TABLES = config.OUTPUT_DIR / "tables" if hasattr(config, "OUTPUT_DIR") else None
PREDS = "outputs/tables/did_lid_finetune_preds_test_chan.csv"
COLS = ["master_audio_path", "start_time", "end_time", "final_dialect",
        "split", "speaker_final", "playlist", "duration"]

# Values quoted to the reviewers and the Area Chair. If a run stops matching these,
# the paper's published numbers are wrong -- stop and tell the chairs, do not edit these.
QUOTED = {"Macro": (0.609, 0.486, 0.677), "Fezzan": (0.313, 0.051, 0.455)}


def build_test_frame() -> pd.DataFrame:
    """Rebuild the channel-disjoint test split exactly as did_finetune_lid.py did."""
    sp = pd.read_csv(config.MANIFEST_DIR / "splits_manifest.csv")
    sp = sp[COLS].dropna(subset=["master_audio_path", "final_dialect", "split"])
    assign = assign_channels(sp[["playlist", "final_dialect", "duration"]])
    sp = sp.dropna(subset=["playlist"]).copy()
    sp["split"] = sp["playlist"].map(assign)
    sp = sp.dropna(subset=["split"])
    return sp[sp["split"] == "test"].reset_index(drop=True)


def load_aligned() -> pd.DataFrame:
    """Frozen predictions with the channel key restored. Raises if alignment breaks."""
    preds = pd.read_csv(PREDS)
    te = build_test_frame()
    if len(te) != len(preds):
        raise SystemExit(f"row count {len(te)} != frozen {len(preds)} -- split rebuild drifted")
    for col, frozen in [("final_dialect", "true"), ("speaker_final", "speaker_final")]:
        m = (te[col].to_numpy() == preds[frozen].to_numpy()).mean()
        if m < 1.0:
            raise SystemExit(f"{col} sequence matches only {m:.2%} -- positional join invalid")
    out = preds.copy()
    out["playlist"] = te["playlist"].to_numpy()
    out["duration"] = te["duration"].to_numpy()
    return out


def f1s(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-class F1 and macro-F1 over integer-coded labels."""
    out = np.empty(len(REGIONS))
    for i in range(len(REGIONS)):
        tp = np.count_nonzero((t == i) & (p == i))
        fp = np.count_nonzero((t != i) & (p == i))
        fn = np.count_nonzero((t == i) & (p != i))
        denom = 2 * tp + fp + fn
        out[i] = 2 * tp / denom if denom else 0.0
    return out, float(out.mean())


def bootstrap(df: pd.DataFrame, yt: np.ndarray, yp: np.ndarray,
              unit: str | None) -> dict[str, np.ndarray]:
    """Percentile bootstrap. unit=None resamples segments, else whole groups of `unit`."""
    rng = np.random.default_rng(SEED)
    acc = {k: np.empty(N_BOOT) for k in REGIONS + ["Macro"]}
    blocks = None if unit is None else [df.groupby(unit).indices[k]
                                        for k in df.groupby(unit).indices]
    for k in range(N_BOOT):
        if blocks is None:
            idx = rng.integers(0, len(df), len(df))
        else:
            idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        pc, mac = f1s(yt[idx], yp[idx])
        for j, r in enumerate(REGIONS):
            acc[r][k] = pc[j]
        acc["Macro"][k] = mac
    return acc


def channel_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Both channel definitions. They differ: carrier sums to 50, dominant to 48.

    'carrier'  = channels holding >=1 speaker of that region (what the rebuttal wording
                 implies for Fezzan); sums above the true channel count because two
                 channels hold speakers of more than one region after validation.
    'dominant' = the region holding most of a channel's duration; this is the rule
                 assign_channels() uses, and the one that sums to the real 48.
    """
    dom = (df.groupby(["playlist", "true"])["duration"].sum().reset_index()
           .sort_values(["duration", "true"], ascending=[False, True])
           .groupby("playlist").tail(1).set_index("playlist")["true"])
    rows = []
    for r in REGIONS:
        g = df[df["true"] == r]
        rows.append({"region": r,
                     "channels_carrier": g["playlist"].nunique(),
                     "channels_dominant": int((dom == r).sum()),
                     "speakers": g["speaker_final"].nunique(),
                     "segments": len(g),
                     "hours": round(g["duration"].sum() / 3600, 2)})
    rows.append({"region": "Total",
                 "channels_carrier": sum(r["channels_carrier"] for r in rows),
                 "channels_dominant": df["playlist"].nunique(),
                 "speakers": df["speaker_final"].nunique(),
                 "segments": len(df),
                 "hours": round(df["duration"].sum() / 3600, 2)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the positional join and the quoted CIs, write nothing")
    args = ap.parse_args()

    df = load_aligned()
    code = {r: i for i, r in enumerate(REGIONS)}
    yt, yp = df["true"].map(code).to_numpy(), df["pred"].map(code).to_numpy()
    pc, mac = f1s(yt, yp)
    point = dict(zip(REGIONS, pc)) | {"Macro": mac}

    print(f"aligned: {len(df):,} test segments | {df['playlist'].nunique()} channels "
          f"| {df['speaker_final'].nunique()} speakers")

    rows = []
    for name, unit in [("iid_segment", None), ("speaker_clustered", "speaker_final"),
                       ("channel_clustered", "playlist")]:
        b = bootstrap(df, yt, yp, unit)
        print(f"\n--- {name} ---")
        for r in REGIONS + ["Macro"]:
            lo, hi = np.percentile(b[r], [2.5, 97.5])
            rows.append({"resampling_unit": name, "region": r, "f1": round(point[r], 3),
                         "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "n_boot": N_BOOT,
                         "seed": SEED})
            print(f"  {r:14s} F1 {point[r]:.3f}  [{lo:.3f}, {hi:.3f}]")

    ci = pd.DataFrame(rows)
    chan = ci[ci["resampling_unit"] == "channel_clustered"].set_index("region")
    bad = [r for r, (f, lo, hi) in QUOTED.items()
           if (round(chan.loc[r, "f1"], 3), chan.loc[r, "ci_lo"], chan.loc[r, "ci_hi"]) != (f, lo, hi)]
    if bad:
        raise SystemExit(f"\n!! {bad} no longer match the values quoted to the Area Chair. "
                         f"Do not silently update the paper -- investigate first.")
    print(f"\nOK: channel-clustered macro and Fezzan match the values quoted in rebuttal.")

    counts = channel_counts(df)
    print(f"\n{counts.to_string(index=False)}")

    if args.check:
        print("\n--check: nothing written.")
        return
    out = config.MANIFEST_DIR.parent / "tables"
    out.mkdir(parents=True, exist_ok=True)
    ci.to_csv(out / "did_channel_disjoint_ci.csv", index=False)
    counts.to_csv(out / "did_channel_counts.csv", index=False)
    print(f"\nwrote {out/'did_channel_disjoint_ci.csv'}, {out/'did_channel_counts.csv'}")


if __name__ == "__main__":
    main()
