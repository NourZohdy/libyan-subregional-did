"""E6 (LID-ECAPA fine-tuned) accuracy by test-segment duration bucket (camera-ready P1-5).

Promised to pVFL: "we will add the robustness analysis for E6". did_extra.py bucketed E2/E4
only. This re-uses the saved E6 test predictions instead of re-running the model.

did_lid_finetune_preds_test.csv has no segment key; its rows are in the order of
splits_manifest.csv[split == "test"] (did_finetune_lid.py line ~289). That alignment is
verified row by row on speaker_final and true label before any number is computed.

    python -m libyan_did.phase5_did.did_e6_duration
"""

import pandas as pd

from libyan_did.shared import config

TABLES = config.TABLES_DIR
EDGES = [3.0, 5.0, 7.0, 10.01]           # same buckets as did_extra.py
LABELS = ["3-5s", "5-7s", "7-10s"]
MODEL = "LID-ECAPA-FT (E6)"


def main() -> None:
    te = pd.read_csv(config.MANIFEST_DIR / "splits_manifest.csv",
                     usecols=["split", "speaker_final", "final_dialect", "duration"])
    te = te[te["split"] == "test"].reset_index(drop=True)
    pr = pd.read_csv(TABLES / "did_lid_finetune_preds_test.csv")
    assert len(te) == len(pr), (len(te), len(pr))
    assert (te["speaker_final"].to_numpy() == pr["speaker_final"].to_numpy()).all()
    assert (te["final_dialect"].to_numpy() == pr["true"].to_numpy()).all()

    bucket = pd.cut(te["duration"], bins=EDGES, labels=LABELS, right=False)
    correct = pr["pred"] == pr["true"]
    rows = [{"model": MODEL, "bucket": lab, "n": int((bucket == lab).sum()),
             "accuracy": round(float(correct[bucket == lab].mean()), 4)} for lab in LABELS]
    new = pd.DataFrame(rows)

    path = TABLES / "did_duration_buckets.csv"
    old = pd.read_csv(path)
    out = pd.concat([old[old["model"] != MODEL], new], ignore_index=True)
    out.to_csv(path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
