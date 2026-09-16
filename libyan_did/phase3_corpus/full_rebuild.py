"""One-shot full rebuild: music scoring -> recluster -> linking -> actor scoring ->
combine. Each stage is incremental/resumable, so re-running after an interruption
continues where it stopped.

Usage:
    python -m libyan_did.phase3_corpus.full_rebuild --root "D:\\Research\\DID_Set\\libyan"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

# Each stage is the phase-3 module, run as `python -m libyan_did.phase3_corpus.<stage>`
# so it imports cleanly from the installed package regardless of working directory.
PKG = "libyan_did.phase3_corpus"


def run_stage(name: str, module: str, extra: list[str] | None = None) -> None:
    print(f"\n{'=' * 70}\nSTAGE: {name}\n{'=' * 70}", flush=True)
    t0 = time.time()
    res = subprocess.run([sys.executable, "-m", module, *(extra or [])])
    dt = (time.time() - t0) / 60.0
    if res.returncode != 0:
        print(f"!! stage '{name}' FAILED after {dt:.1f} min (exit {res.returncode})")
        sys.exit(res.returncode)
    print(f"-- stage '{name}' done in {dt:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dialect parent, e.g. D:\\...\\libyan")
    args = ap.parse_args()

    t0 = time.time()
    run_stage("1/5 music scoring (PANNs)", f"{PKG}.score_segments", ["--root", args.root])
    run_stage("2/5 recluster", f"{PKG}.run_resources", ["--root", args.root, "--recluster-only"])
    run_stage("3/5 speaker linking", f"{PKG}.link_actors")
    run_stage("4/5 actor triage scoring", f"{PKG}.score_actors")
    run_stage("5/5 combine corpus", f"{PKG}.combine_corpus")
    print(f"\nFULL REBUILD COMPLETE in {(time.time() - t0) / 60.0:.1f} min")
    print("Next: python -m streamlit run libyan_did/phase4_validation/validate_app.py")


if __name__ == "__main__":
    main()
