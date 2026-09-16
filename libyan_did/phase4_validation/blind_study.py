"""Blind second-annotator review sampling and agreement reporting."""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = ("Tripolitania", "Cyrenaica", "Fezzan")
TERCILES = ("low", "mid", "high")
JUDGEMENTS = (
    "Tripolitania",
    "Cyrenaica",
    "Fezzan",
    "MSA",
    "Non-Libyan",
    "Mixed speakers",
    "Insufficient evidence",
    "Unplayable",
)
OTHER_JUDGEMENTS = (
    "MSA",
    "Non-Libyan",
    "Mixed speakers",
    "Insufficient evidence",
)
ROSTER = ("mansour", "test_dryrun")
EXCLUDED_ANNOTATOR = "NOUR"
N_PER_CELL = 10
N_SAMPLE = 90
CLIPS_PER_ITEM = 10
BOOT_REPLICATES = 2000
MANIFEST_PATH = "outputs/manifests/splits_manifest.csv"
RISK_PATH = "outputs/gold/speaker_verifications.csv"
DEFAULT_OUT_DIR = "outputs/blind_review"
DECISION_COLUMNS = (
    "item_id",
    "annotator",
    "judgement",
    "note",
    "timestamp",
    "n_clips_presented",
)

assert EXCLUDED_ANNOTATOR not in ROSTER


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def out_path(out_dir, name) -> Path:
    return Path(out_dir) / name


def decisions_path(out_dir, annotator) -> Path:
    return out_path(out_dir, f"decisions_{annotator}.csv")


def sample_path(out_dir=DEFAULT_OUT_DIR) -> Path:
    return out_path(out_dir, "sample_frozen.csv")


def extension_path(out_dir=DEFAULT_OUT_DIR) -> Path:
    """The optional second tranche. A separate file so sample_frozen.csv stays
    byte-identical and the sample_sha256 in precommit.json keeps verifying."""
    return out_path(out_dir, "sample_extension.csv")


def precommit_path(out_dir=DEFAULT_OUT_DIR) -> Path:
    return out_path(out_dir, "precommit.json")


def lock_path(out_dir=DEFAULT_OUT_DIR) -> Path:
    return out_path(out_dir, ".session.lock")


def ensure_out_dir(out_dir) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def load_precommit(out_dir=DEFAULT_OUT_DIR) -> dict:
    path = precommit_path(out_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run sample first")
    return json.loads(path.read_text(encoding="utf-8"))


def build_eligible_pool(
    manifest_path=MANIFEST_PATH, risk_path=RISK_PATH
) -> pd.DataFrame:
    # final_dialect is the released post-merge label and is single-valued per
    # speaker; the raw region column is multi-valued for cross-region merges.
    manifest = pd.read_csv(
        manifest_path, usecols=["speaker_final", "final_dialect", "actor_uid"]
    ).rename(columns={"final_dialect": "region"})
    region_counts = manifest.groupby("speaker_final")["region"].nunique()
    ambiguous = region_counts[region_counts != 1]
    if not ambiguous.empty:
        raise ValueError(
            "speakers resolve to more than one region: "
            + ", ".join(map(str, ambiguous.index[:5]))
        )

    speakers = (
        manifest.groupby("speaker_final", as_index=False)
        .agg(region=("region", "first"), n_segments=("region", "size"))
    )
    actor_risk = (
        pd.read_csv(risk_path, usecols=["actor_uid", "risk_score"])
        .groupby("actor_uid", as_index=False)["risk_score"]
        .max()
    )
    speaker_risk = (
        manifest[["speaker_final", "actor_uid"]]
        .drop_duplicates()
        .merge(actor_risk, on="actor_uid", how="left")
        .groupby("speaker_final", as_index=False)["risk_score"]
        .max()
    )
    pool = speakers.merge(speaker_risk, on="speaker_final", how="left")
    missing = pool.loc[pool["risk_score"].isna(), "speaker_final"]
    if not missing.empty:
        raise ValueError(
            "speakers resolve no risk score: " + ", ".join(map(str, missing[:5]))
        )
    return pool[["speaker_final", "region", "n_segments", "risk_score"]]


def assign_terciles(pool: pd.DataFrame) -> pd.DataFrame:
    result = pool.sort_values(
        ["region", "risk_score", "speaker_final"], kind="mergesort"
    ).reset_index(drop=True)
    ranks = result.groupby("region", sort=False)["risk_score"].rank(method="first")
    result["risk_tercile"] = ranks.groupby(result["region"], sort=False).transform(
        lambda values: pd.qcut(values, 3, labels=TERCILES)
    )
    result["risk_tercile"] = result["risk_tercile"].astype(str)
    result["cell"] = result["region"] + ":" + result["risk_tercile"]
    return result


def draw_sample(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drawn = []
    for region in REGIONS:
        for tercile in TERCILES:
            cell = f"{region}:{tercile}"
            candidates = pool[pool["cell"] == cell]
            if len(candidates) < N_PER_CELL:
                raise ValueError(
                    f"cell {cell} has {len(candidates)} speakers; needs {N_PER_CELL}"
                )
            drawn.append(candidates.iloc[rng.choice(len(candidates), N_PER_CELL, False)])
    sample = pd.concat(drawn, ignore_index=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    if sample["speaker_final"].nunique() != N_SAMPLE:
        raise ValueError("drawn sample does not contain 90 unique speakers")
    sample.insert(0, "serve_order", np.arange(1, N_SAMPLE + 1))
    sample.insert(0, "item_id", [f"S{value:02d}" for value in sample["serve_order"]])
    return sample[
        [
            "item_id",
            "serve_order",
            "speaker_final",
            "region",
            "risk_tercile",
            "cell",
            "risk_score",
            "n_segments",
        ]
    ]


def write_precommit(out_dir, seed: int, sample_df: pd.DataFrame) -> None:
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "source_manifest": {
            "path": MANIFEST_PATH,
            "sha256": sha256_file(MANIFEST_PATH),
        },
        "risk_source": {"path": RISK_PATH, "sha256": sha256_file(RISK_PATH)},
        "eligibility_rule": (
            "Released speakers in the frozen manifest. The released region label "
            "is the manifest's final_dialect column, not its raw region column: "
            "final_dialect is the post-merge decision and resolves to exactly one "
            "value for all 645 released speakers, whereas region is multi-valued "
            "for 5 cross-region merged speakers and differs from final_dialect for "
            "26. final_dialect is the reference the agreement is scored against. "
            "Risk is the maximum over each speaker's actor clusters; terciles are "
            "assigned within region."
        ),
        "sample_design": {
            "n": N_SAMPLE,
            "per_cell": N_PER_CELL,
            "regions": list(REGIONS),
            "terciles": list(TERCILES),
        },
        "evidence_rule": (
            "First presentation: 10 timeline-spread clips, preferring clean then "
            "longest within each slice; all segments when fewer than 10 are "
            "available. The annotator may then request further rounds of 10 "
            "clips drawn at random from the same speaker, without limit. "
            "n_clips_presented records every clip presented across all rounds, "
            "so items judged on extra evidence can be identified at analysis "
            "time; a value above 10 means the annotator asked for more."
        ),
        "judgement_options": list(JUDGEMENTS),
        "annotator_roster": list(ROSTER),
        "analysis_plan": (
            "Overall raw agreement and Cohen's kappa over a 4x4 table with Other "
            "pooling non-region judgements; 2000-replicate bootstrap stratified "
            "on 9 cells; per-region raw agreement and confusion without kappa; "
            "Unplayable excluded from the denominator and reported separately."
        ),
        "sample_sha256": sha256_file(sample_path(out_dir)),
    }
    # Carry any recorded extensions forward: this function rebuilds the record
    # from scratch, so without this a later rewrite would drop them silently.
    if precommit_path(out_dir).exists():
        existing = load_precommit(out_dir).get("extensions")
        if existing:
            record["extensions"] = existing
    precommit_path(out_dir).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )


def cmd_sample(args):
    ensure_out_dir(args.out_dir)
    frozen = sample_path(args.out_dir)
    if frozen.exists():
        print(f"Refusing to overwrite {frozen}; move it by hand first.", file=sys.stderr)
        return 1
    try:
        pool = assign_terciles(build_eligible_pool())
        sample = draw_sample(pool, args.seed)
        sample.to_csv(frozen, index=False)
        write_precommit(args.out_dir, args.seed, sample)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Sample error: {exc}", file=sys.stderr)
        return 1
    print(sample.groupby("cell", sort=True).size().to_string(name="n"))
    print(f"sample SHA-256: {sha256_file(frozen)}")
    return 0


def draw_extension(pool: pd.DataFrame, already: set, seed: int) -> pd.DataFrame:
    """Every released speaker not already in the frozen sample, shuffled.

    The balanced 90 keep their ids and serve order; these continue after them as
    E001, E002, ... so the tranche a decision belongs to is readable from the
    item_id alone. The shuffle is what keeps region and risk out of the serving
    order (FR-009); the prefix encodes tranche only, never provenance (FR-007).
    """
    rest = pool[~pool["speaker_final"].isin(already)].copy()
    if rest.empty:
        raise ValueError("no released speakers remain outside the frozen sample")
    rng = np.random.default_rng(seed)
    rest = rest.sort_values("speaker_final", kind="mergesort").reset_index(drop=True)
    rest = rest.iloc[rng.permutation(len(rest))].reset_index(drop=True)
    rest.insert(0, "serve_order", np.arange(N_SAMPLE + 1, N_SAMPLE + 1 + len(rest)))
    rest.insert(0, "item_id", [f"E{i:03d}" for i in range(1, len(rest) + 1)])
    return rest[
        [
            "item_id",
            "serve_order",
            "speaker_final",
            "region",
            "risk_tercile",
            "cell",
            "risk_score",
            "n_segments",
        ]
    ]


def cmd_extend(args):
    frozen = sample_path(args.out_dir)
    target = extension_path(args.out_dir)
    if not frozen.exists():
        print(f"{frozen} does not exist; run sample first.", file=sys.stderr)
        return 1
    if target.exists():
        print(f"Refusing to overwrite {target}; move it by hand first.", file=sys.stderr)
        return 1
    try:
        primary = pd.read_csv(frozen)
        pool = assign_terciles(build_eligible_pool())
        extension = draw_extension(pool, set(primary["speaker_final"]), args.seed)
        extension.to_csv(target, index=False)
        record_extension(args.out_dir, args.seed, extension)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Extend error: {exc}", file=sys.stderr)
        return 1
    print(extension.groupby("cell", sort=True).size().to_string(name="n"))
    print(f"{len(primary)} primary + {len(extension)} extension = "
          f"{len(primary) + len(extension)} items")
    print(f"extension SHA-256: {sha256_file(target)}")
    return 0


def record_extension(out_dir, seed: int, extension: pd.DataFrame) -> None:
    """Append the extension to precommit.json without touching what is frozen."""
    record = load_precommit(out_dir)
    record.setdefault("extensions", []).append(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "n": int(len(extension)),
            "file": extension_path(out_dir).name,
            "sha256": sha256_file(extension_path(out_dir)),
            "rule": (
                "Every released speaker outside the primary sample, shuffled at "
                "this seed and served after item 90 as E001 onward. Drawn before "
                "any listening began. The primary 90 stay the study's own floor "
                "-- no sample size was promised to reviewers -- and are reported "
                "on their own so the headline figure never depends on when the "
                "annotator stopped. Completed extension items are reported "
                "separately with their own denominator. Per-region rates come "
                "from the primary 90 only: the extension mirrors the corpus "
                "rather than being balanced by cell."
            ),
        }
    )
    precommit_path(out_dir).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )


def select_clips(
    manifest_df: pd.DataFrame, speaker_final, k: int = CLIPS_PER_ITEM
) -> list[dict]:
    group = manifest_df[
        manifest_df["speaker_final"].astype(str) == str(speaker_final)
    ].copy()
    group = group.sort_values(["file_id", "start_time"]).reset_index(drop=True)
    if group.empty:
        return []
    group["_clean"] = (
        (group["quality"] == "clean").astype(int)
        if "quality" in group.columns
        else 1
    )
    k = min(k, len(group))
    picked = []
    for index in range(k):
        low = int(index * len(group) / k)
        high = max(int((index + 1) * len(group) / k), low + 1)
        timeline_slice = group.iloc[low:high]
        timeline_slice = timeline_slice[
            ~timeline_slice.index.isin([row.name for row in picked])
        ]
        if timeline_slice.empty:
            continue
        picked.append(
            timeline_slice.sort_values(
                ["_clean", "duration"], ascending=[False, False]
            ).iloc[0]
        )
    return [
        {
            "path": row["master_audio_path"],
            "start": float(row["start_time"]),
            "end": float(row["end_time"]),
        }
        for row in picked
    ]


def sample_clips(
    manifest_df: pd.DataFrame,
    speaker_final,
    k: int = CLIPS_PER_ITEM,
    seed: int = 0,
    exclude: set[tuple[str, float]] | None = None,
) -> list[dict]:
    """k segments drawn at random from the speaker, for the reshuffle control.

    select_clips stays the frozen timeline-spread protocol shown first; this is
    the extra evidence the annotator can ask for. Returns the same three fields
    and no provenance.
    """
    group = manifest_df[
        manifest_df["speaker_final"].astype(str) == str(speaker_final)
    ].sort_values(["file_id", "start_time"]).reset_index(drop=True)
    if exclude:
        keys = list(
            zip(
                group["master_audio_path"].astype(str),
                group["start_time"].astype(float),
            )
        )
        group = group.loc[
            [key not in exclude for key in keys]
        ].reset_index(drop=True)
    if group.empty:
        return []
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(len(group), min(k, len(group)), replace=False))
    return [
        {
            "path": row["master_audio_path"],
            "start": float(row["start_time"]),
            "end": float(row["end_time"]),
        }
        for _, row in group.iloc[picked].iterrows()
    ]


def sample_clips_exclude_self_check() -> None:
    manifest = pd.DataFrame(
        {
            "speaker_final": ["speaker"] * 3,
            "file_id": ["file"] * 3,
            "start_time": [0.0, 1.0, 2.0],
            "end_time": [1.0, 2.0, 3.0],
            "master_audio_path": ["audio.wav"] * 3,
        }
    )
    excluded = {("audio.wav", 0.0)}
    clips = sample_clips(manifest, "speaker", k=10, seed=7, exclude=excluded)
    assert len(clips) == 2
    assert all((str(clip["path"]), clip["start"]) not in excluded for clip in clips)


def load_blind_queue(out_dir=DEFAULT_OUT_DIR) -> pd.DataFrame:
    columns = ["item_id", "serve_order", "speaker_final"]
    frames = [pd.read_csv(sample_path(out_dir), usecols=columns)]
    if extension_path(out_dir).exists():
        frames.append(pd.read_csv(extension_path(out_dir), usecols=columns))
    return pd.concat(frames, ignore_index=True).sort_values(
        "serve_order", ignore_index=True
    )[columns]


def load_reference(out_dir=DEFAULT_OUT_DIR) -> pd.DataFrame:
    """The frozen sample plus any extension, with the released label. Analysis
    side only — the session reaches the corpus through load_blind_queue."""
    frames = [pd.read_csv(sample_path(out_dir)).assign(tranche="primary")]
    if extension_path(out_dir).exists():
        frames.append(pd.read_csv(extension_path(out_dir)).assign(tranche="extension"))
    return pd.concat(frames, ignore_index=True)


def completed_item_ids(out_dir, annotator) -> set[str]:
    path = decisions_path(out_dir, annotator)
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["item_id"] for row in csv.DictReader(handle)}


def append_decision(
    out_dir,
    annotator,
    item_id,
    judgement,
    note,
    n_clips_presented,
) -> None:
    if annotator not in ROSTER or annotator == EXCLUDED_ANNOTATOR:
        raise ValueError(f"annotator check failed: {annotator!r} is not allowed")
    if judgement not in JUDGEMENTS:
        raise ValueError(f"judgement check failed: {judgement!r} is not allowed")
    if item_id in completed_item_ids(out_dir, annotator):
        raise ValueError(f"item_id check failed: {item_id!r} is already complete")

    ensure_out_dir(out_dir)
    path = decisions_path(out_dir, annotator)
    creating = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if creating:
            writer.writerow(DECISION_COLUMNS)
        writer.writerow(
            (
                item_id,
                annotator,
                judgement,
                note,
                datetime.now().astimezone().isoformat(),
                n_clips_presented,
            )
        )
        handle.flush()
        os.fsync(handle.fileno())


def score_decisions(
    sample_df: pd.DataFrame, decisions_df: pd.DataFrame
) -> pd.DataFrame:
    unknown = sorted(set(decisions_df["item_id"]) - set(sample_df["item_id"]))
    if unknown:
        raise ValueError("unknown item_id: " + ", ".join(map(str, unknown)))
    scored = decisions_df.merge(sample_df, on="item_id", how="inner")
    scored["agree"] = scored["judgement"] == scored["region"]
    scored["scored"] = scored["judgement"] != "Unplayable"
    scored["collapsed"] = scored["judgement"].where(
        scored["judgement"].isin(REGIONS), "Other"
    )
    return scored


def cohens_kappa(reference, collapsed) -> float:
    labels = (*REGIONS, "Other")
    table = pd.crosstab(
        pd.Series(list(reference), name="reference"),
        pd.Series(list(collapsed), name="collapsed"),
    ).reindex(index=labels, columns=labels, fill_value=0)
    total = table.to_numpy().sum()
    if total == 0:
        return float("nan")
    observed = np.trace(table.to_numpy()) / total
    expected = (
        table.sum(axis=1).to_numpy() @ table.sum(axis=0).to_numpy()
    ) / total**2
    return float("nan") if expected == 1 else float((observed - expected) / (1 - expected))


def bootstrap_ci(
    scored_df: pd.DataFrame, n_boot: int = BOOT_REPLICATES, seed: int = 0
) -> dict:
    empty = {
        "raw_lo": float("nan"),
        "raw_hi": float("nan"),
        "kappa_lo": float("nan"),
        "kappa_hi": float("nan"),
    }
    if scored_df.empty or n_boot <= 0:
        return empty
    rng = np.random.default_rng(seed)
    groups = [group for _, group in scored_df.groupby("cell", sort=False)]
    raw_values = np.empty(n_boot)
    kappa_values = np.empty(n_boot)
    for index in range(n_boot):
        replicate = pd.concat(
            [
                group.iloc[rng.integers(0, len(group), len(group))]
                for group in groups
            ],
            ignore_index=True,
        )
        raw_values[index] = replicate["agree"].mean()
        kappa_values[index] = cohens_kappa(
            replicate["region"], replicate["collapsed"]
        )
    finite_kappa = kappa_values[np.isfinite(kappa_values)]
    kappa_bounds = (
        np.percentile(finite_kappa, [2.5, 97.5])
        if len(finite_kappa)
        else (float("nan"), float("nan"))
    )
    raw_bounds = np.percentile(raw_values, [2.5, 97.5])
    return {
        "raw_lo": float(raw_bounds[0]),
        "raw_hi": float(raw_bounds[1]),
        "kappa_lo": float(kappa_bounds[0]),
        "kappa_hi": float(kappa_bounds[1]),
    }


def by_region_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        group = scored_df[scored_df["region"] == region]
        counts = group["judgement"].value_counts()
        rows.append(
            {
                "region": region,
                "n": len(group),
                "n_agree": int(group["agree"].sum()),
                "raw_agreement": group["agree"].mean(),
                **{judgement: int(counts.get(judgement, 0)) for judgement in JUDGEMENTS},
            }
        )
    return pd.DataFrame(rows)


def confusion_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    labels = (*REGIONS, "Other")
    table = pd.crosstab(scored_df["region"], scored_df["collapsed"])
    table = table.reindex(index=labels, columns=labels, fill_value=0)
    table.index.name = "released_label"
    table.columns.name = "judgement"
    return table


def summarise(scored: pd.DataFrame, label: str, n_boot: int) -> dict:
    active = scored[scored["scored"]]
    intervals = bootstrap_ci(active, n_boot=n_boot)
    return {
        "tranche": label,
        "n_scored": len(active),
        "n_unplayable": int((~scored["scored"]).sum()),
        "raw_agreement": active["agree"].mean() if len(active) else float("nan"),
        "raw_lo": intervals["raw_lo"],
        "raw_hi": intervals["raw_hi"],
        "kappa": cohens_kappa(active["region"], active["collapsed"])
        if len(active)
        else float("nan"),
        "kappa_lo": intervals["kappa_lo"],
        "kappa_hi": intervals["kappa_hi"],
    }


def cmd_report(args):
    try:
        precommit = load_precommit(args.out_dir)
        annotator = args.annotator or precommit["annotator_roster"][0]
        reference = load_reference(args.out_dir)
        decisions = pd.read_csv(decisions_path(args.out_dir, annotator))
        scored = score_decisions(reference, decisions)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Report error: {exc}", file=sys.stderr)
        return 1

    # The balanced 90 are always reported alone, so the headline figure never
    # depends on when the annotator stopped. How many extension items get judged
    # does, so that figure is secondary and carries its own denominator.
    primary = scored[scored["tranche"] == "primary"]
    rows = [summarise(primary, "primary", args.boot)]
    if (scored["tranche"] == "extension").any():
        rows.append(summarise(scored, "primary+extension", args.boot))
    pd.DataFrame(rows).to_csv(
        out_path(args.out_dir, "agreement_overall.csv"), index=False
    )

    active = primary[primary["scored"]]
    by_region_table(active).to_csv(
        out_path(args.out_dir, "agreement_by_region.csv"), index=False
    )
    confusion_table(active).to_csv(out_path(args.out_dir, "confusion.csv"))

    n_primary = int((reference["tranche"] == "primary").sum())
    for row in rows:
        total = n_primary if row["tranche"] == "primary" else len(reference)
        print(f"[{row['tranche']}] {row['n_scored']} of {total} scored, "
              f"{row['n_unplayable']} unplayable")
    done = set(decisions["item_id"])
    outstanding = n_primary - len(
        done & set(reference.loc[reference["tranche"] == "primary", "item_id"])
    )
    if outstanding:
        print(f"{outstanding} of the balanced 90 still outstanding")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample")
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    sample.set_defaults(func=cmd_sample)

    extend = subparsers.add_parser("extend")
    extend.add_argument("--seed", type=int, required=True)
    extend.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    extend.set_defaults(func=cmd_extend)

    report = subparsers.add_parser("report")
    report.add_argument("--annotator")
    report.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    report.add_argument("--boot", type=int, default=BOOT_REPLICATES)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
