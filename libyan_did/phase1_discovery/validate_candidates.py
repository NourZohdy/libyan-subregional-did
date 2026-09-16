"""Integrity check for the discovery registry — exit non-zero if anything is wrong.

Run it before/after editing ``candidates.psv`` (or in CI) to catch the failure modes that
silently corrupt the registry:

  * **column-shifted rows** — an unescaped ``|`` in a title produces the wrong field count
    (the bug that once put ``expand:RTV LEBDA`` into the ``status`` column). Checked with a
    quote-aware ``csv`` parse, so a *properly quoted* pipe is NOT flagged.
  * **out-of-enum** ``status`` / ``region`` / ``url_type`` values.
  * **missing / duplicate ``channel_id``** (playlists from one channel legitimately share a
    channel_id; two *channel* rows must not).
  * **pipe / doubled-quote contamination** left in a cell (means the file was written by
    something other than ``shared.registry`` — robust readers tolerate it, naive ones won't).

Usage (from the repo root):
    python -m libyan_did.phase1_discovery.validate_candidates
    python -m libyan_did.phase1_discovery.validate_candidates --registry path/to.psv --strict
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from libyan_did.shared import config
from libyan_did.shared import registry


def _data_lines(path: Path) -> list[str]:
    """Non-blank, non-comment lines (matches how registry.read filters)."""
    return [ln for ln in path.read_text(encoding="utf-8-sig").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def validate(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []

    # 1) raw, quote-aware field-count check — catches genuinely shifted rows.
    lines = _data_lines(path)
    if len(lines) < 2:
        return ["registry is empty or header-only"], warns
    rows = list(csv.reader(lines, delimiter="|"))
    header = [c.strip() for c in rows[0]]
    ncol = len(header)
    for n, r in enumerate(rows[1:], start=1):   # n = 1-based data-row number
        if len(r) != ncol:
            snippet = " | ".join(r[:3])
            errors.append(f"data row {n}: {len(r)} fields, expected {ncol} "
                          f"(unescaped pipe / shifted row): {snippet}")

    # 2) semantic checks via the robust reader.
    df = registry.read(path)
    channel_ids: dict[str, int] = {}
    for idx, row in df.iterrows():
        st, rg, ut = row["status"].strip(), row["region"].strip(), row["url_type"].strip()
        url, cid = row["url"].strip(), row["channel_id"].strip()
        tag = (url[:58] + "…") if len(url) > 58 else (url or f"row {idx}")
        if not url:
            errors.append(f"row {idx}: empty url")
        if st and st not in registry.STATUS_OPTS:
            errors.append(f"{tag}: status '{st}' not in {registry.STATUS_OPTS}")
        if rg and rg not in registry.REGION_OPTS:
            errors.append(f"{tag}: region '{rg}' not in {registry.REGION_OPTS}")
        if ut and ut not in registry.URL_TYPE_OPTS:
            warns.append(f"{tag}: url_type '{ut}' not in {registry.URL_TYPE_OPTS}")
        if ut == "channel":
            if not cid:
                warns.append(f"{tag}: channel row has no channel_id")
            else:
                channel_ids[cid] = channel_ids.get(cid, 0) + 1
        for col in registry.COLS:
            v = str(row[col])
            if "|" in v:
                warns.append(f"{tag}: literal pipe in cell '{col}' (not registry-written)")
            if '""' in v:
                warns.append(f"{tag}: doubled-quote run in cell '{col}'")

    for cid, count in channel_ids.items():
        if count > 1:
            errors.append(f"channel_id {cid} appears on {count} distinct channel rows (duplicate)")

    return errors, warns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(config.DISCOVERY_DIR / "candidates.psv"))
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures too")
    args = ap.parse_args()

    path = Path(args.registry)
    if not path.exists():
        raise SystemExit(f"no registry at {path}")

    errors, warns = validate(path)
    for w in warns:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")
    n = len(registry.read(path))
    print(f"\n[validate] {path.name}: {n} rows · {len(errors)} error(s) · {len(warns)} warning(s)")
    if errors or (args.strict and warns):
        sys.exit(1)
    print("[validate] OK")


if __name__ == "__main__":
    main()
