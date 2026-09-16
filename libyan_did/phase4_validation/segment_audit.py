"""Freeze the 207-segment purity audit sample (camera-ready P1-2).

Promised to pVFL and 2RmL: a stratified sample of released TEST segments, hand-labelled for
MSA, code-switching, mixed-speaker audio, and wrong region.

Design frozen 2026-08-23 (see manuscripts/corpus_paper/CAMERA_READY_PLAN.md, P1-2):
  207 segments = 23 per cell x 3 regions x 3 duration bands
  caps: at most 3 segments per speaker and 8 per channel, counted across the whole sample
  duration bands are PER-REGION terciles, not global -- global cuts leave regional cells
  lopsided because the regions differ in clip length

Why the caps are 3/8 and not the 2/4 first drafted: the released test split holds only 34
Fezzan speakers, so a cap of 2 ceilings Fezzan at 68 segments while 3 bands x 23 needs 69.
At 2 per speaker the cap stops being a diversity control and becomes the sampler.

Deliberately SEPARATE from blind_study.py: that is annotator 2's speaker-level agreement
study, this is annotator 1's segment-level purity audit. Different annotator, different unit,
different output directory. Nothing here reads or writes outputs/blind_review/.

Run (from repo root):
    python -m libyan_did.phase4_validation.segment_audit sample --seed 20260823
    python -m libyan_did.phase4_validation.segment_audit verify
    python -m libyan_did.phase4_validation.segment_audit report
    streamlit run libyan_did/phase4_validation/segment_app.py --server.port 8503
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.phase4_validation.blind_study import sha256_file

REGIONS = ["Tripolitania", "Cyrenaica", "Fezzan"]
BANDS = ["short", "mid", "long"]
N_PER_CELL = 23
MAX_PER_SPEAKER = 3
MAX_PER_CHANNEL = 8
N_SAMPLE = N_PER_CELL * len(REGIONS) * len(BANDS)          # 207

MANIFEST_PATH = "outputs/manifests/splits_manifest.csv"
DEFAULT_OUT_DIR = "outputs/segment_audit"
KEY = ["file_id", "start_time", "end_time"]

# One binary field per property. They overlap -- a segment can be MSA *and* mixed-speaker --
# so this is four independent questions, never one multiple-choice.
DECISION_COLUMNS = (
    "item_id", "annotator", "is_msa", "is_code_switched", "is_mixed_speaker",
    "region_judgement", "indeterminate", "note", "timestamp",
)

ANNOTATOR = "NOUR"
REGION_JUDGEMENTS = (*REGIONS, "cannot_tell")
# Only these columns may reach the app: no path, no final_dialect, no playlist.
BLIND_COLUMNS = ["item_id", "serve_order", "master_audio_path", "start_time", "end_time"]


def decisions_path(out_dir=DEFAULT_OUT_DIR) -> Path:
    return Path(out_dir) / f"decisions_{ANNOTATOR.lower()}.csv"


def load_blind_queue(out_dir=DEFAULT_OUT_DIR) -> pd.DataFrame:
    """Frozen sample restricted to what the app needs to play a clip. The path is used to
    open the file and is never rendered -- the directory names carry the region."""
    df = pd.read_csv(Path(out_dir) / "sample_segments_frozen.csv", usecols=BLIND_COLUMNS)
    return df.sort_values("serve_order").reset_index(drop=True)


def completed_item_ids(out_dir=DEFAULT_OUT_DIR) -> set[str]:
    path = decisions_path(out_dir)
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["item_id"] for row in csv.DictReader(handle)}


def append_decision(out_dir, item_id, is_msa, is_code_switched, is_mixed_speaker,
                    region_judgement, indeterminate, note) -> None:
    """Append one row and fsync, same durability rule as blind_study.append_decision."""
    if region_judgement not in REGION_JUDGEMENTS:
        raise ValueError(f"region_judgement {region_judgement!r} is not allowed")
    if item_id in completed_item_ids(out_dir):
        raise ValueError(f"{item_id!r} is already complete")
    path = decisions_path(out_dir)
    creating = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if creating:
            writer.writerow(DECISION_COLUMNS)
        writer.writerow((item_id, ANNOTATOR, int(is_msa), int(is_code_switched),
                         int(is_mixed_speaker), region_judgement, int(indeterminate), note,
                         datetime.now().astimezone().isoformat()))
        handle.flush()
        os.fsync(handle.fileno())


def load_test_pool() -> pd.DataFrame:
    """Released test segments, with per-region duration bands attached."""
    df = pd.read_csv(MANIFEST_PATH)
    df = df[df["split"] == "test"].copy()
    df = df.dropna(subset=["master_audio_path", "final_dialect", "speaker_final", "playlist"])
    bad = set(df["final_dialect"].unique()) - set(REGIONS)
    if bad:
        raise SystemExit(f"unexpected labels in final_dialect: {bad}")

    parts, cuts = [], {}
    for region in REGIONS:
        g = df[df["final_dialect"] == region].copy()
        lo, hi = g["duration"].quantile([1 / 3, 2 / 3]).to_numpy()
        cuts[region] = [round(float(lo), 3), round(float(hi), 3)]
        g["band"] = pd.cut(g["duration"], [-np.inf, lo, hi, np.inf], labels=BANDS)
        parts.append(g)
    pool = pd.concat(parts, ignore_index=True)
    pool["cell"] = pool["final_dialect"] + ":" + pool["band"].astype(str)
    return pool, cuts


def draw(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """23 per cell, random within cell, respecting the per-speaker and per-channel caps.

    Caps are counted over the WHOLE sample, not per cell -- a speaker giving 3 segments to
    one band must not give 3 more to another. Cells are visited smallest-first so the tight
    ones (Fezzan) claim their quota before the caps are spent elsewhere.
    """
    rng = np.random.default_rng(seed)
    spk_used: dict[str, int] = {}
    chan_used: dict[str, int] = {}
    drawn = []

    cells = sorted(pool["cell"].unique(), key=lambda c: (pool["cell"] == c).sum())
    for cell in cells:
        cand = pool[pool["cell"] == cell]
        cand = cand.iloc[rng.permutation(len(cand))]
        taken = 0
        for _, row in cand.iterrows():
            if taken == N_PER_CELL:
                break
            s, c = row["speaker_final"], row["playlist"]
            if spk_used.get(s, 0) >= MAX_PER_SPEAKER or chan_used.get(c, 0) >= MAX_PER_CHANNEL:
                continue
            spk_used[s] = spk_used.get(s, 0) + 1
            chan_used[c] = chan_used.get(c, 0) + 1
            drawn.append(row)
            taken += 1
        if taken < N_PER_CELL:
            raise SystemExit(f"cell {cell}: only {taken} of {N_PER_CELL} drawable under caps")

    sample = pd.DataFrame(drawn).reset_index(drop=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample.insert(0, "serve_order", np.arange(1, len(sample) + 1))
    sample.insert(0, "item_id", [f"G{v:03d}" for v in sample["serve_order"]])
    return sample[["item_id", "serve_order", *KEY, "master_audio_path", "duration",
                   "speaker_final", "playlist", "final_dialect", "band", "cell"]]


def check(sample: pd.DataFrame) -> None:
    """The one runnable check: everything the design promises must hold on the frozen file."""
    assert len(sample) == N_SAMPLE, f"{len(sample)} rows, expected {N_SAMPLE}"
    assert not sample.duplicated(KEY).any(), "duplicate segments in sample"
    counts = sample["cell"].value_counts()
    assert len(counts) == 9 and (counts == N_PER_CELL).all(), f"uneven cells:\n{counts}"
    over_s = sample["speaker_final"].value_counts().max()
    over_c = sample["playlist"].value_counts().max()
    assert over_s <= MAX_PER_SPEAKER, f"a speaker gives {over_s} segments (cap {MAX_PER_SPEAKER})"
    assert over_c <= MAX_PER_CHANNEL, f"a channel gives {over_c} segments (cap {MAX_PER_CHANNEL})"
    assert (sample["serve_order"].to_numpy() == np.arange(1, N_SAMPLE + 1)).all()


def cmd_sample(args) -> int:
    out_dir = Path(args.out_dir)
    frozen = out_dir / "sample_segments_frozen.csv"
    if frozen.exists() and not args.force:
        raise SystemExit(f"{frozen} already exists. The sample is frozen -- refusing to "
                         f"redraw. Pass --force only if nobody has listened yet.")

    pool, cuts = load_test_pool()
    sample = draw(pool, args.seed)
    check(sample)

    out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(frozen, index=False)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_sample": N_SAMPLE,
        "design": f"{N_PER_CELL} per cell x {len(REGIONS)} regions x {len(BANDS)} duration bands",
        "caps": {"per_speaker": MAX_PER_SPEAKER, "per_channel": MAX_PER_CHANNEL,
                 "scope": "counted across the whole sample, not per cell"},
        "duration_band_cuts_seconds": cuts,
        "band_rule": "per-region terciles of segment duration",
        "source_manifest": {"path": MANIFEST_PATH, "sha256": sha256_file(MANIFEST_PATH)},
        "eligibility_rule": "released segments with split == test in the frozen manifest",
        "decision_columns": list(DECISION_COLUMNS),
        "annotator": "annotator 1 (NOUR), blind to the released label",
        "primary_endpoint": ("per-region rates of MSA, code-switching, mixed-speaker audio "
                             "and region disagreement; 'indeterminate' counts as its own "
                             "category and is reported, never silently dropped"),
        "reporting_rule": ("Wilson intervals are nominal -- up to 3 segments share a speaker. "
                           "Add one speaker-clustered bootstrap as a sensitivity interval. "
                           "Any pooled across-region rate must be population-weighted by true "
                           "test-split region sizes, since allocation is equal but the split "
                           "is not."),
        "sample_sha256": sha256_file(frozen),
    }
    (out_dir / "precommit_segments.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"drew {len(sample)} segments -> {frozen}")
    print(f"duration band cuts (s): {cuts}")
    print(f"\nper cell:\n{sample.groupby(['final_dialect', 'band'], observed=True).size().to_string()}")
    print(f"\ndistinct speakers {sample['speaker_final'].nunique()} "
          f"| distinct channels {sample['playlist'].nunique()} "
          f"| max per speaker {sample['speaker_final'].value_counts().max()} "
          f"| max per channel {sample['playlist'].value_counts().max()}")
    print(f"\nwrote {out_dir/'precommit_segments.json'}")
    return 0


def cmd_verify(args) -> int:
    out_dir = Path(args.out_dir)
    frozen = out_dir / "sample_segments_frozen.csv"
    record = json.loads((out_dir / "precommit_segments.json").read_text(encoding="utf-8"))
    sample = pd.read_csv(frozen)
    check(sample)
    digest = sha256_file(frozen)
    if digest != record["sample_sha256"]:
        raise SystemExit(f"sample file has changed since it was frozen!\n"
                         f"  precommit: {record['sample_sha256']}\n  now:       {digest}")
    print(f"OK: {len(sample)} segments, cells even, caps respected, hash matches precommit.")
    return 0


# Reported outcomes. Each is a binary property of a segment; wrong_region and cannot_tell are
# derived from region_judgement against the released label. Indeterminate rows count as their
# own category and are NOT excluded from the denominators of the others.
OUTCOMES = ("is_msa", "is_code_switched", "is_mixed_speaker", "wrong_region", "cannot_tell",
            "indeterminate")
BOOT_REPLICATES = 2000


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap(df: pd.DataFrame, col: str, seed: int, n_boot: int) -> tuple[float, float]:
    """Resample speakers with replacement, carrying every sampled segment of each speaker."""
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in df.groupby("speaker_final")]
    rates = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals = np.concatenate([groups[i] for i in pick])
        rates.append(vals.mean())
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def score(decisions: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    df = decisions.merge(sample[["item_id", "final_dialect", "band", "speaker_final"]],
                         on="item_id", how="inner")
    df["cannot_tell"] = (df["region_judgement"] == "cannot_tell").astype(int)
    df["wrong_region"] = ((df["region_judgement"] != "cannot_tell")
                          & (df["region_judgement"] != df["final_dialect"])).astype(int)
    return df


def cmd_report(args) -> int:
    out_dir = Path(args.out_dir)
    sample = pd.read_csv(out_dir / "sample_segments_frozen.csv")
    decisions = pd.read_csv(decisions_path(out_dir))
    missing = set(sample["item_id"]) - set(decisions["item_id"])
    if missing:
        print(f"WARNING: {len(missing)} of {len(sample)} segments not yet judged")
    df = score(decisions, sample)

    # Population weights: true test-split region sizes, not the equal 69/69/69 allocation.
    test = pd.read_csv(MANIFEST_PATH, usecols=["split", "final_dialect"])
    test = test[test["split"] == "test"]
    weights = test["final_dialect"].value_counts(normalize=True)

    rows = []
    for i, col in enumerate(OUTCOMES):
        for region, g in df.groupby("final_dialect"):
            k, n = int(g[col].sum()), len(g)
            lo, hi = wilson(k, n)
            blo, bhi = cluster_bootstrap(g, col, seed=args.seed + i, n_boot=args.boot)
            rows.append(dict(outcome=col, region=region, n=n, k=k, rate=k / n,
                             wilson_lo=lo, wilson_hi=hi, boot_lo=blo, boot_hi=bhi))
        per_region = {r["region"]: r["rate"] for r in rows if r["outcome"] == col}
        pooled = sum(weights[r] * per_region[r] for r in REGIONS)
        rows.append(dict(outcome=col, region="pooled_weighted", n=len(df),
                         k=int(df[col].sum()), rate=pooled,
                         wilson_lo=np.nan, wilson_hi=np.nan, boot_lo=np.nan, boot_hi=np.nan))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "purity_rates.csv", index=False)

    confusion = pd.crosstab(df["final_dialect"], df["region_judgement"])
    confusion.to_csv(out_dir / "region_confusion.csv")
    by_band = df.groupby("band")[list(OUTCOMES)].mean().loc[BANDS]
    by_band.to_csv(out_dir / "purity_by_band.csv")

    print(f"{len(df)} of {len(sample)} segments scored; weights {weights.round(3).to_dict()}")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
        print("\nregion confusion (released -> judged):\n" + confusion.to_string())
        print("\nby duration band:\n" + by_band.to_string())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="draw and freeze the 207-segment sample")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    s.add_argument("--force", action="store_true", help="redraw over an existing frozen sample")
    s.set_defaults(func=cmd_sample)

    v = sub.add_parser("verify", help="re-check the frozen sample against its precommit")
    v.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    v.set_defaults(func=cmd_verify)

    r = sub.add_parser("report", help="score decisions_nour.csv against the released labels")
    r.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    r.add_argument("--boot", type=int, default=BOOT_REPLICATES)
    r.add_argument("--seed", type=int, default=20260823)
    r.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
