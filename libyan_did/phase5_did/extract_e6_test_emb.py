"""Extract the fine-tuned E6 encoder's 256-d embeddings for the TEST segments.

E6 (did_finetune_lid.py) fine-tunes the VoxLingua107 ECAPA end to end on the 3-way
task. Its penultimate embedding (before the linear head) is the representation the
model actually learns. We run the saved checkpoint over the held-out TEST audio and
cache emb + (region, speaker) so the t-SNE plot can be rebuilt without GPU.

Held-out test only: if these embeddings separate by region, it is generalization,
not memorization of training speakers.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/extract_e6_test_emb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from libyan_did.shared import config
from libyan_did.phase5_did.did_finetune import SegDataset, set_seed  # noqa: E402
from libyan_did.phase5_did.did_finetune_lid import build_model, CollateLID  # noqa: E402

SPLITS = config.MANIFEST_DIR / "splits_manifest.csv"
CKPT = config.MANIFEST_DIR / "did_finetune_lid_speaker_best.pt"
OUT_EMB = config.MANIFEST_DIR / "e6_test_emb.npy"
OUT_IDX = config.MANIFEST_DIR / "e6_test_index.csv"
REGIONS = list(config.REGIONS)
MAX_SECONDS = 10.0          # E6's training/eval window
BATCH = 16


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")

    sp = pd.read_csv(SPLITS)
    te = sp[sp["split"] == "test"].reset_index(drop=True)
    te = te.dropna(subset=["master_audio_path", "final_dialect"]).reset_index(drop=True)
    print(f"test segments: {len(te):,}")

    # rebuild the exact model and load fine-tuned weights (SpecAug is param-free)
    model = build_model(device, specaug_freq=12, specaug_time=40, specaug_n_time=2)
    sd = torch.load(CKPT, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    ds = SegDataset(te, MAX_SECONDS)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0,
                    collate_fn=CollateLID())

    embs, spks = [], []
    done = 0
    with torch.no_grad():
        for wav, wl, y, spk in dl:
            wav, wl = wav.to(device), wl.to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda"),
                                dtype=torch.bfloat16):
                # replicate forward up to the embedding (no head, no specaug in eval)
                with torch.autocast(device_type=device, enabled=False):
                    feats = model.compute_features(wav.float())
                    feats = model.mean_var_norm(feats, wl)
                emb = model.embedding_model(feats, wl).squeeze(1)        # (B, 256)
            embs.append(emb.float().cpu().numpy())
            spks.extend(spk)
            done += len(spk)
            if done % (BATCH * 50) == 0 or done == len(te):
                print(f"  {done}/{len(te)}")

    X = np.concatenate(embs, axis=0)
    assert X.shape == (len(te), 256), f"emb shape {X.shape}"
    np.save(OUT_EMB, X)
    idx = pd.DataFrame({"speaker_final": spks,
                        "region": te["final_dialect"].to_numpy()})
    idx.to_csv(OUT_IDX, index=False)
    print(f"saved -> {OUT_EMB.name} {X.shape} and {OUT_IDX.name}")


if __name__ == "__main__":
    main()
