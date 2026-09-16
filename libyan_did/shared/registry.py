"""Single source of truth for the discovery registry (``candidates.psv``).

Every Phase-1 tool reads and writes the candidate registry through this module so the
parse, the schema, and the sanitizer never drift (there used to be five near-identical
``read_psv`` copies). The registry is **pipe-delimited** because Arabic titles routinely
contain commas.

Robustness, the reason this module exists:

* **Reading** uses a single-char ``sep="|"`` that *honours* the quotes ``pandas`` writes.
  The old readers used a regex separator ``sep=r"\\s*\\|\\s*"`` which silently disabled
  quote handling, so a playlist whose title contained a literal ``|`` (e.g. ``"| FULL HD"``)
  shifted every later column right and corrupted the row. ``skipinitialspace`` + a per-cell
  strip keep the LLM agent's spaced-pipe ``parts/*.psv`` parsing too.
* **Writing** runs every cell through :func:`sanitize` first, so a stray pipe / newline /
  runaway quote-run can never be written into the registry in the first place.

Only *full* ``#`` comment lines are dropped (the example header); we must never use pandas
``comment="#"``, which would truncate any cell containing a ``#hashtag``.
"""

from __future__ import annotations

import io
import re
import shutil
from pathlib import Path

import pandas as pd

# Canonical schema — the contract shared with Phase 2's harvester registry loader.
COLS = ["url", "url_type", "region", "source", "channel_id", "video_count",
        "playlist_count", "format_type", "lang_flag", "confidence", "evidence",
        "discovery_path", "status", "notes", "review_note"]

REGION_OPTS = ["Tripolitania", "Cyrenaica", "Fezzan", "Mixed", "Non-Libyan", "Unknown"]

# State machine (channels AND playlists share it):
#   pending -> approved | rejected
# Window 1 approves Libyan CHANNELS (a gate to window 2); window 2 writes the picked
# PLAYLISTS as approved rows — those approved playlists are the harvest targets.
STATUS_OPTS = ["pending", "approved", "rejected"]

URL_TYPE_OPTS = ["channel", "playlist", "video"]

_QUOTE_RUN = re.compile(r'"{2,}')


def sanitize(cell) -> str:
    """Make any value safe to live in a single pipe-delimited cell.

    Drops the delimiter and line breaks, and collapses the runaway doubled-quote runs that
    a previous broken write/read cycle baked into some ``evidence`` cells.
    """
    s = "" if cell is None else str(cell)
    s = s.replace("|", "/").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = _QUOTE_RUN.sub('"', s)
    return s.strip()


def read(path, *, cols: list[str] | None = None) -> pd.DataFrame:
    """Read a pipe-delimited registry into an all-string DataFrame.

    Honours quotes, tolerates spaced pipes, fills any missing canonical columns, and keeps
    extra columns (e.g. ``review_note``) after the canonical ones. Returns an empty frame
    (with the schema) if the file is absent or header-only.
    """
    cols = list(cols) if cols is not None else list(COLS)
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=cols)
    lines = [ln for ln in path.read_text(encoding="utf-8-sig").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str,
                     keep_default_na=False, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:                       # strip spaced-pipe whitespace from every cell
        df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras].fillna("")


def write(df: pd.DataFrame, path, *, backup=None) -> None:
    """Write ``df`` as a pipe-delimited registry, sanitising every cell first.

    If ``backup`` is given and points at a path that does not yet exist, a one-time copy of
    the current file is made there before overwriting (idempotent across checkpoint writes).
    """
    path = Path(path)
    if backup is not None:
        backup = Path(backup)
        if path.exists() and not backup.exists():
            shutil.copyfile(path, backup)
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(sanitize)
    out.to_csv(path, sep="|", index=False, encoding="utf-8")
