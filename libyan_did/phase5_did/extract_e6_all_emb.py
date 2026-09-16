"""Extract the fine-tuned E6 encoder's 256-d embeddings for ALL segments (645 speakers).

Same as extract_e6_test_emb.py but over the whole splits_manifest (train+dev+test),
so the t-SNE can show all 645 speakers. Caches emb + (speaker, region, split) for a
GPU-free plot.

Run (diarization env, from libyan_did_pipeline/):
    python scripts/extract_e6_all_emb.py
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
OUT_EMB = config.MANIFEST_DIR / "e6_all_emb.npy"
OUT_IDX = config.MANIFEST_DIR / "e6_all_index.csv"
REGIONS = list(config.REGIONS)
MAX_SECONDS = 10.0
BATCH = 16


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")

    sp = pd.read_csv(SPLITS)
    df = sp.dropna(subset=["master_audio_path", "final_dialect"]).reset_index(drop=True)
    print(f"all segments: {len(df):,} | speakers: {df['speaker_final'].nunique()}")

    model = build_model(device, specaug_freq=12, specaug_time=40, specaug_n_time=2)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    ds = SegDataset(df, MAX_SECONDS)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0,
                    collate_fn=CollateLID())

    embs, spks = [], []
    done = 0
    with torch.no_grad():
        for wav, wl, y, spk in dl:
            wav, wl = wav.to(device), wl.to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda"),
                                dtype=torch.bfloat16):
                with torch.autocast(device_type=device, enabled=False):
                    feats = model.compute_features(wav.float())
                    feats = model.mean_var_norm(feats, wl)
                emb = model.embedding_model(feats, wl).squeeze(1)
            embs.append(emb.float().cpu().numpy())
            spks.extend(spk)
            done += len(spk)
            if done % (BATCH * 100) == 0 or done == len(df):
                print(f"  {done}/{len(df)}")

    X = np.concatenate(embs, axis=0)
    assert X.shape == (len(df), 256), f"emb shape {X.shape}"
    np.save(OUT_EMB, X)
    pd.DataFrame({"speaker_final": spks,
                 "region": df["final_dialect"].to_numpy(),
                 "split": df["split"].to_numpy()}).to_csv(OUT_IDX, index=False)
    print(f"saved -> {OUT_EMB.name} {X.shape} and {OUT_IDX.name}")


if __name__ == "__main__":
    main()
