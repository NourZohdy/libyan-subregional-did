"""Speaker-disjoint train/dev/test split exporter (Part I #1, §4.3).

Uses the global speaker IDs (``global_actor``) as the grouping key so no speaker
appears in more than one split — the single most important guard for DID, which
otherwise re-identifies speakers instead of learning dialect. Splits are balanced
per dialect (``sub_cat``). Whole speakers are assigned atomically.
"""

from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd

from libyan_did.shared.config import PER_SPEAKER_SEGMENT_CAP, SPLIT_BALANCE_BY, SPLIT_RATIOS, SPLIT_SEED


def assign_speaker_disjoint_splits(
    df: pd.DataFrame,
    *,
    actor_col: str = "global_actor",
    dialect_col: str = "sub_cat",
    ratios: dict[str, float] = SPLIT_RATIOS,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``split`` column in {train,dev,test}.

    Greedy per-dialect assignment: shuffle each dialect's speakers, then assign each
    whole speaker to the split currently most under its target segment quota. This
    keeps speakers disjoint while approximately honouring ``ratios`` by segment count
    within every dialect.
    """
    if actor_col not in df.columns:
        raise KeyError(f"{actor_col!r} not in dataframe; run clustering first.")

    rng = random.Random(seed)
    split_names = list(ratios.keys())
    out = df.copy()
    out["split"] = None

    for dialect, d_df in out.groupby(dialect_col):
        # segments per speaker in this dialect
        seg_counts = d_df.groupby(actor_col).size().to_dict()
        speakers = list(seg_counts.keys())
        rng.shuffle(speakers)

        total = sum(seg_counts.values())
        targets = {s: ratios[s] * total for s in split_names}
        assigned = {s: 0 for s in split_names}
        speaker_to_split: dict[str, str] = {}

        # Assign the largest speakers first for a tighter fit to targets.
        for spk in sorted(speakers, key=lambda s: -seg_counts[s]):
            # pick split with the largest remaining (target - assigned) headroom
            best = max(split_names, key=lambda s: targets[s] - assigned[s])
            speaker_to_split[spk] = best
            assigned[best] += seg_counts[spk]

        mask = out[dialect_col] == dialect
        out.loc[mask, "split"] = out.loc[mask, actor_col].map(speaker_to_split)

    return out


def verify_no_speaker_overlap(df: pd.DataFrame, *, actor_col: str = "global_actor") -> bool:
    """True iff every speaker appears in exactly one split. Raises on overlap."""
    by_split = defaultdict(set)
    for split, s_df in df.groupby("split"):
        by_split[split] = set(s_df[actor_col].unique())

    splits = list(by_split.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = by_split[splits[i]] & by_split[splits[j]]
            if overlap:
                raise AssertionError(
                    f"Speaker overlap between {splits[i]} and {splits[j]}: {sorted(overlap)[:5]}"
                )
    return True


def _region_col(df: pd.DataFrame) -> str:
    return "region" if "region" in df.columns else "sub_cat"


def _actor_col(df: pd.DataFrame) -> str:
    return "actor_uid" if "actor_uid" in df.columns else "global_actor"


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Segment & speaker counts per (split, region) for reporting."""
    rcol, acol = _region_col(df), _actor_col(df)
    rows = []
    for (split, region), g in df.groupby(["split", rcol]):
        rows.append({
            "split": split,
            "dialect": region,
            "segments": len(g),
            "speakers": g[acol].nunique(),
            "hours": round(g["duration"].sum() / 3600.0, 3),
        })
    return pd.DataFrame(rows).sort_values(["split", "dialect"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Step 3: balanced, capped, speaker-disjoint splits over the COMBINED corpus
# --------------------------------------------------------------------------- #
def cap_per_speaker(
    df: pd.DataFrame,
    *,
    actor_col: str = "actor_uid",
    cap: int | None = None,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Randomly down-sample any speaker with more than ``cap`` segments.

    Anchor speakers (recurring hosts) otherwise dominate the corpus and any split that
    contains them. Capping equalises speaker mass before assignment. ``cap=None`` is a
    no-op. The clip files on disk are untouched; only the split manifest is sub-sampled.
    """
    if not cap:
        return df
    rng = random.Random(seed)
    keep: list = []
    for _, g in df.groupby(actor_col):
        idx = list(g.index)
        keep.extend(rng.sample(idx, cap) if len(idx) > cap else idx)
    return df.loc[sorted(keep)].copy()


def build_balanced_splits(
    df: pd.DataFrame,
    *,
    actor_col: str | None = None,
    region_col: str | None = None,
    balance: str = SPLIT_BALANCE_BY,
    ratios: dict[str, float] = SPLIT_RATIOS,
    seed: int = SPLIT_SEED,
    cap: int | None = PER_SPEAKER_SEGMENT_CAP,
) -> pd.DataFrame:
    """Assign whole speakers to train/dev/test, disjoint and balanced **within each
    region**, optionally capping anchor speakers first.

    Largest-speaker-first greedy: each whole speaker goes to the split with the most
    remaining headroom toward its target mass, where mass is segment ``duration`` (or
    ``segments`` count). Keeping the assignment per-region means each dialect is split
    independently, so no region is missing from dev/test. Speakers stay disjoint because
    a speaker is assigned atomically.
    """
    region_col = region_col or _region_col(df)
    actor_col = actor_col or _actor_col(df)
    work = cap_per_speaker(df, actor_col=actor_col, cap=cap, seed=seed)

    rng = random.Random(seed)
    names = list(ratios.keys())
    out = work.copy()
    out["split"] = None

    for region, g in out.groupby(region_col):
        if balance == "duration":
            mass = g.groupby(actor_col)["duration"].sum().to_dict()
        else:
            mass = g.groupby(actor_col).size().astype(float).to_dict()
        total = sum(mass.values()) or 1.0
        targets = {s: ratios[s] * total for s in names}
        assigned = {s: 0.0 for s in names}

        speakers = list(mass.keys())
        rng.shuffle(speakers)  # break ties stochastically but reproducibly
        speaker_to_split: dict = {}
        for spk in sorted(speakers, key=lambda s: -mass[s]):
            best = max(names, key=lambda s: targets[s] - assigned[s])
            speaker_to_split[spk] = best
            assigned[best] += mass[spk]

        mask = out[region_col] == region
        out.loc[mask, "split"] = out.loc[mask, actor_col].map(speaker_to_split)

    return out


# --------------------------------------------------------------------------- #
# Channel-disjoint, coverage-first splits (21-class corpus v2, FR-006/FR-007)
# --------------------------------------------------------------------------- #
def _dialect_col(df: pd.DataFrame) -> str:
    """Prefer the validator-resolved dialect, then the raw pipeline region/class."""
    for col in ("final_dialect", "sub_cat", "region"):
        if col in df.columns:
            return col
    return "sub_cat"


def verify_no_channel_overlap(df: pd.DataFrame, *, source_col: str = "source") -> bool:
    """True iff every channel (``source``) appears in exactly one split. Raises on overlap.

    The primary leakage guard for corpus v2: a channel spanning two split sides would
    put the same recording source on both, defeating the source-disjoint guarantee.
    """
    by_split = defaultdict(set)
    for split, s_df in df.groupby("split"):
        by_split[split] = set(s_df[source_col].unique())

    splits = list(by_split.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = by_split[splits[i]] & by_split[splits[j]]
            if overlap:
                raise AssertionError(
                    f"Channel overlap between {splits[i]} and {splits[j]}: {sorted(overlap)[:5]}"
                )
    return True


def _cover_required(assign: dict, mass: dict, required: list[str], active: list[str]) -> None:
    """Repair pass: guarantee each split in ``required`` gets ≥1 channel, moving the
    smallest channel from the donor split with the most channels. In-place on ``assign``."""
    members: dict[str, list] = defaultdict(list)
    for ch, s in assign.items():
        members[s].append(ch)
    for req in required:
        if members[req]:
            continue
        donors = [s for s in active if s != req and len(members[s]) >= 2]
        if not donors:  # cannot cover without emptying another required split
            continue
        donor = max(donors, key=lambda s: len(members[s]))
        smallest = min(members[donor], key=lambda ch: mass[ch])
        assign[smallest] = req
        members[donor].remove(smallest)
        members[req].append(smallest)


def build_channel_disjoint_splits(
    df: pd.DataFrame,
    *,
    source_col: str = "source",
    dialect_col: str | None = None,
    ratios: dict[str, float] = SPLIT_RATIOS,
    seed: int = SPLIT_SEED,  # kept for API parity; assignment is deterministic
) -> pd.DataFrame:
    """Assign whole channels to train/dev/test, disjoint and **coverage-first** per class.

    The split atom is the whole channel (``source``), never the speaker, so no channel
    spans two sides (FR-006). Coverage-first (FR-007): every class gets ≥1 channel on
    ``test``; ``dev`` is populated only for classes with ≥3 channels (a 2-channel class is
    1 train + 1 test + 0 dev; a 1-channel class is a single test channel). No floor, no
    drop, no merge (D6).

    Assignment mirrors ``build_balanced_splits`` (largest-first, most-headroom greedy) but
    over channels, then a repair pass guarantees the coverage rule. Returns a copy of ``df``
    with a ``split`` column in {train, dev, test}.
    """
    dcol = dialect_col or _dialect_col(df)
    ordered_splits = list(ratios.keys())  # train, dev, test — deterministic tie order
    out = df.copy()

    # A channel is assigned to ONE split globally, so it never spans two sides even if its
    # segments carry more than one class label (e.g. a validator wrong_region fix). Each
    # channel's total duration is its mass; it is grouped under its DOMINANT class (the class
    # holding most of its duration) for the per-class coverage-first assignment.
    ch_mass = out.groupby(source_col)["duration"].sum().to_dict()
    dom = (out.groupby([source_col, dcol])["duration"].sum()
           .reset_index().sort_values("duration", ascending=False)
           .drop_duplicates(source_col).set_index(source_col)[dcol].to_dict())

    global_map: dict = {}
    for cls in sorted({str(c) for c in dom.values()}):
        chans = [c for c, d in dom.items() if str(d) == cls]
        mass = {c: ch_mass[c] for c in chans}
        n_ch = len(mass)
        if n_ch <= 1:
            active = ["test"]           # single channel -> test (coverage)
            required: list[str] = []
        elif n_ch == 2:
            active = ["train", "test"]  # 1 train + 1 test + 0 dev
            required = ["test"]
        else:
            active = list(ordered_splits)
            required = ["test", "dev"]

        total = sum(mass.values()) or 1.0
        targets = {s: ratios.get(s, 0.0) * total for s in active}
        assigned = {s: 0.0 for s in active}
        assign: dict = {}
        # largest-first, assign each channel to the active split with the most headroom
        for ch in sorted(mass, key=lambda c: (-mass[c], str(c))):
            best = max(active, key=lambda s: targets[s] - assigned[s])
            assign[ch] = best
            assigned[best] += mass[ch]

        _cover_required(assign, mass, required, active)

        if "dev" not in set(assign.values()):
            print(f"[channel-disjoint] {cls}: thin class ({n_ch} channels) -> empty dev (kept, not dropped)")
        global_map.update(assign)

    out["split"] = out[source_col].map(global_map)
    return out
