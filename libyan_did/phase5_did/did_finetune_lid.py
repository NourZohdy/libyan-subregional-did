"""E6: end-to-end fine-tuned VoxLingua107 ECAPA-LID for 3-way sub-regional DID.

E5 (did_lid_probe.py) froze the VoxLingua107 ECAPA-LID encoder and trained a
logistic probe on its 256-d embedding. E6 fine-tunes the ECAPA-TDNN encoder
together with a fresh linear 3-way head, end to end. This is the supervisor's
model and the paper's new headline.

Two fixes for the Fezzan minority class are built in:
  * class-weighted cross-entropy (--class-weight sqrt, the tempered default;
    inverse / effective / none also available),
  * SpecAugment (time+freq masking) on the Fbank features in training only.

The split regime is selectable. --split-scheme speaker uses the published
speaker-disjoint split from splits_manifest.csv; --split-scheme channel rebuilds
a channel-disjoint split with did_channel_disjoint.assign_channels (whole channels
to one split) and retrains from scratch, so we report E6 under both regimes.

Built for 8 GB VRAM (RTX 5060). ECAPA-TDNN is small (~22M params), so a real batch
fits with bf16 autocast and no gradient checkpointing. Audio is read by random
access (soundfile start/stop) per segment from master_audio_path -- no feature
cache. The model is loaded with the COPY fetch strategy (Windows symlink needs
admin); see did_lid_probe.load_lid_mods.

Outputs (idempotent; model name carries a [chan] suffix on the channel run):
  - appends "LID-ECAPA-FT (E6)" segment+speaker rows to outputs/tables/did_results.csv
  - outputs/tables/did_confusion_test_lid_finetune[_chan].csv
  - appends rows to did_perclass_test.csv and did_bootstrap_ci.csv
  - appends per-class rows to did_channel_disjoint.csv (split_type matches the scheme)
    so make_channel_disjoint_fig.py (C9) can plot E6
  - outputs/tables/did_lid_finetune_preds_test[_chan].csv (speaker_final, true, pred)

Run (diarization env, from libyan_did_pipeline/):
    python scripts/did_finetune_lid.py --smoke                     # plumbing check
    python scripts/did_finetune_lid.py --split-scheme speaker      # full run (GPU)
    python scripts/did_finetune_lid.py --split-scheme channel      # full run (GPU)
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

# Reduce CUDA fragmentation OOMs on 8 GB VRAM (must precede torch CUDA init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd

from libyan_did.shared import config

# Reuse backbone-agnostic helpers from the XLS-R fine-tune and the LID probe.
from libyan_did.phase5_did.did_finetune import (SegDataset, boot_ci, upsert, macro_f1, accuracy,  # noqa: E402
                          set_seed)
from libyan_did.phase5_did.did_lid_probe import load_lid_mods  # noqa: E402
from libyan_did.phase5_did.did_channel_disjoint import assign_channels  # noqa: E402

SPLITS = config.MANIFEST_DIR / "splits_manifest.csv"
TABLES = config.TABLES_DIR
SEED = 1337
REGIONS = list(config.REGIONS)
R2I = {r: i for i, r in enumerate(REGIONS)}
I2R = {i: r for r, i in R2I.items()}
SR = config.SAMPLE_RATE
LID_DIM = 256


def _autocast(device):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class CollateLID:
    """Pad a batch of waveforms to (B, Tmax) and the relative lengths speechbrain wants."""

    def __call__(self, batch):
        import torch
        wavs = [b[0] for b in batch]
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)
        spk = [b[2] for b in batch]
        tmax = max(w.shape[0] for w in wavs)
        padded = np.zeros((len(wavs), tmax), dtype=np.float32)
        rel = np.empty(len(wavs), dtype=np.float32)
        for j, w in enumerate(wavs):
            padded[j, : w.shape[0]] = w
            rel[j] = w.shape[0] / tmax
        return torch.from_numpy(padded), torch.from_numpy(rel), y, spk


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(device, specaug_freq, specaug_time, specaug_n_time):
    import torch.nn as nn
    import torchaudio

    compute_features, mean_var_norm, embedding_model = load_lid_mods(device)
    embedding_model.requires_grad_(True)                   # EncoderClassifier loads it frozen for inference

    class SpecAug(nn.Module):
        """SpecAugment on Fbank feats (B, time, freq); train-mode only via caller."""

        def __init__(self):
            super().__init__()
            self.fmask = torchaudio.transforms.FrequencyMasking(specaug_freq)
            self.tmasks = nn.ModuleList(
                [torchaudio.transforms.TimeMasking(specaug_time) for _ in range(specaug_n_time)])

        def forward(self, feats):
            x = feats.transpose(1, 2)                     # (B, freq, time)
            x = self.fmask(x)
            for m in self.tmasks:
                x = m(x)
            return x.transpose(1, 2)

    class LIDClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.compute_features = compute_features      # Fbank, no params
            self.mean_var_norm = mean_var_norm            # sentence norm, no params
            self.embedding_model = embedding_model        # ECAPA-TDNN, trainable
            self.specaug = SpecAug()
            self.head = nn.Linear(LID_DIM, len(REGIONS))

        def forward(self, wav, wav_lens):
            import torch
            # Fbank in fp32 (STFT is unstable under autocast); embedding under ambient autocast.
            with torch.autocast(device_type=wav.device.type, enabled=False):
                feats = self.compute_features(wav.float())
                feats = self.mean_var_norm(feats, wav_lens)
                if self.training:
                    feats = self.specaug(feats)
            emb = self.embedding_model(feats, wav_lens).squeeze(1)   # (B, 256)
            return self.head(emb)

    model = LIDClassifier().to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params {n_train/1e6:.1f}M / {n_all/1e6:.1f}M")
    return model


# --------------------------------------------------------------------------- #
# Class weights
# --------------------------------------------------------------------------- #
def class_weights(y_ids: np.ndarray, scheme: str) -> np.ndarray | None:
    counts = np.bincount(y_ids, minlength=len(REGIONS)).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    if scheme == "none":
        return None
    if scheme == "inverse":
        w = counts.sum() / (len(REGIONS) * counts)
    elif scheme == "sqrt":
        w = np.sqrt(counts.sum() / counts)
    elif scheme == "effective":                           # Cui et al. 2019
        beta = 0.999
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(scheme)
    return w / w.mean()                                   # keep loss scale comparable


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
def evaluate(model, loader, device):
    import torch
    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
    yt, yp, sk = [], [], []
    with torch.no_grad():
        for wav, wl, y, spk in loader:
            wav, wl = wav.to(device), wl.to(device)
            with _autocast(device):
                logits = model(wav, wl)
            yp.append(logits.float().argmax(-1).cpu().numpy())
            yt.append(y.numpy()); sk.extend(spk)
    model.train()
    return np.concatenate(yt), np.concatenate(yp), np.array(sk)


def speaker_scores(yt, yp, spk):
    d = pd.DataFrame({"spk": spk, "true": yt, "pred": yp})
    gg = d.groupby("spk").agg(true=("true", "first"),
                              pred=("pred", lambda x: x.value_counts().idxmax()))
    return (accuracy(gg["true"].to_numpy(), gg["pred"].to_numpy()),
            macro_f1(gg["true"].to_numpy(), gg["pred"].to_numpy()))


def perclass(cm) -> list[dict]:
    out, ps, rs, fs = [], [], [], []
    for i, r in enumerate(REGIONS):
        tp = cm[i, i]
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append({"region": r, "precision": round(prec, 3),
                    "recall": round(rec, 3), "f1": round(f1, 3)})
        ps.append(prec); rs.append(rec); fs.append(f1)
    out.append({"region": "Macro", "precision": round(np.mean(ps), 3),
                "recall": round(np.mean(rs), 3), "f1": round(np.mean(fs), 3)})
    return out


def upsert2(path: Path, new_df: pd.DataFrame, keys: list[str]) -> None:
    if path.exists():
        old = pd.read_csv(path)
        if set(keys).issubset(old.columns):
            mask = old.set_index(keys).index.isin(new_df.set_index(keys).index)
            old = old[~mask]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--split-scheme", choices=["speaker", "channel"], default="speaker")
    ap.add_argument("--class-weight", choices=["none", "sqrt", "inverse", "effective"],
                    default="sqrt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=250, help="optimizer steps between dev evals")
    ap.add_argument("--patience", type=int, default=8, help="dev evals without improvement -> stop")
    ap.add_argument("--enc-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--cap-per-speaker", type=int, default=None,
                    help="cap train segments per speaker_final; dev/test untouched")
    ap.add_argument("--specaug-freq", type=int, default=12)
    ap.add_argument("--specaug-time", type=int, default=40)
    ap.add_argument("--specaug-n-time", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA; this will be extremely slow.")

    scheme = args.split_scheme
    suffix = "" if scheme == "speaker" else " [chan]"
    model_name = f"LID-ECAPA-FT (E6){suffix}"
    split_type = "speaker_disjoint" if scheme == "speaker" else "channel_disjoint"
    tag = "" if scheme == "speaker" else "_chan"
    best_ckpt = config.MANIFEST_DIR / f"did_finetune_lid_{scheme}_best.pt"

    sp = pd.read_csv(SPLITS)
    cols = ["master_audio_path", "start_time", "end_time", "final_dialect",
            "split", "speaker_final", "playlist", "duration"]
    sp = sp[cols].dropna(subset=["master_audio_path", "final_dialect", "split"])
    bad = set(sp["final_dialect"].unique()) - set(REGIONS)
    assert not bad, f"unexpected labels in final_dialect: {bad}"

    if scheme == "channel":                               # rebuild a channel-disjoint split
        assign = assign_channels(sp[["playlist", "final_dialect", "duration"]])
        sp = sp.dropna(subset=["playlist"]).copy()
        sp["split"] = sp["playlist"].map(assign)
        sp = sp.dropna(subset=["split"])
        # no channel may cross splits
        for a, b in [("train", "test"), ("train", "dev"), ("dev", "test")]:
            ca = set(sp.loc[sp["split"] == a, "playlist"])
            cb = set(sp.loc[sp["split"] == b, "playlist"])
            assert not (ca & cb), f"channel leak {a}/{b}!"
        print("channel-disjoint split: disjointness asserts passed")

    if args.smoke:
        sp = (sp.groupby("split", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), 120), random_state=SEED)))
        args.epochs, args.eval_every, args.patience = 1, 10, 99

    tr = sp[sp["split"] == "train"].reset_index(drop=True)
    dv = sp[sp["split"] == "dev"].reset_index(drop=True)
    te = sp[sp["split"] == "test"].reset_index(drop=True)
    if args.cap_per_speaker:
        rng = np.random.default_rng(SEED)
        keep = []
        for _, g in tr.groupby("speaker_final"):
            idx = g.index.to_numpy()
            if len(idx) > args.cap_per_speaker:
                idx = rng.choice(idx, args.cap_per_speaker, replace=False)
            keep.extend(idx.tolist())
        before = len(tr)
        tr = tr.loc[sorted(keep)].reset_index(drop=True)
        print(f"per-speaker cap {args.cap_per_speaker}: train {before:,} -> {len(tr):,} segments")

    print(f"scheme={scheme} | train {len(tr):,} | dev {len(dv):,} | test {len(te):,} segments")
    for nm, d in [("train", tr), ("dev", dv), ("test", te)]:
        print(f"  {nm} region segs:", d["final_dialect"].value_counts().reindex(REGIONS).to_dict())

    collate = CollateLID()
    g = torch.Generator().manual_seed(SEED)
    workers = 0 if args.smoke else args.workers
    dl_tr = DataLoader(SegDataset(tr, args.max_seconds), batch_size=args.batch, shuffle=True,
                       num_workers=workers, collate_fn=collate, generator=g,
                       pin_memory=(device == "cuda"), drop_last=True,
                       persistent_workers=workers > 0)
    dl_dv = DataLoader(SegDataset(dv, args.max_seconds), batch_size=max(8, args.batch),
                       shuffle=False, num_workers=workers, collate_fn=collate,
                       pin_memory=(device == "cuda"), persistent_workers=workers > 0)
    dl_te = DataLoader(SegDataset(te, args.max_seconds), batch_size=max(8, args.batch),
                       shuffle=False, num_workers=workers, collate_fn=collate,
                       pin_memory=(device == "cuda"), persistent_workers=workers > 0)

    model = build_model(device, args.specaug_freq, args.specaug_time, args.specaug_n_time)
    enc_params = [p for p in model.embedding_model.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.enc_lr},
         {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)

    w = class_weights(tr["final_dialect"].map(R2I).to_numpy(), args.class_weight)
    w_t = None if w is None else torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w_t)
    print(f"class-weight ({args.class_weight}): "
          f"{ {r: round(float(x), 3) for r, x in zip(REGIONS, (w if w is not None else [1,1,1]))} }")

    steps_per_epoch = max(1, len(dl_tr) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_dev, best_step, stale, gstep = -1.0, -1, 0, 0
    t0 = time.time()
    stop = False

    def run_dev_eval():
        nonlocal best_dev, best_step, stale
        yt, yp, _ = evaluate(model, dl_dv, device)
        f1, acc = macro_f1(yt, yp), accuracy(yt, yp)
        improved = f1 > best_dev + 1e-4
        if improved:
            best_dev, best_step, stale = f1, gstep, 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            stale += 1
        print(f"  [dev @ step {gstep}] macroF1 {f1:.4f} acc {acc:.4f} "
              f"(best {best_dev:.4f}){'  <- best' if improved else ''}  {time.time()-t0:.0f}s")

    print(f"\ntraining: batch {args.batch} x accum {args.accum} | enc_lr {args.enc_lr} "
          f"head_lr {args.head_lr} | warmup {warmup}/{total_steps} | bf16")
    model.train()
    for epoch in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for it, (wav, wl, y, _) in enumerate(dl_tr):
            wav, wl, y = wav.to(device), wl.to(device), y.to(device)
            with _autocast(device):
                logits = model(wav, wl)
                loss = loss_fn(logits, y) / args.accum
            loss.backward()
            running += loss.item() * args.accum
            if (it + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % 50 == 0:
                    print(f"  epoch {epoch} step {gstep} loss {running/(50*args.accum):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s")
                    running = 0.0
                if gstep % args.eval_every == 0:
                    run_dev_eval()
                    if stale >= args.patience:
                        print(f"early stop: {stale} dev evals without improvement")
                        stop = True
                        break
        if stop:
            break
        run_dev_eval()
        if stale >= args.patience:
            print(f"early stop after epoch {epoch}")
            break

    if best_step < 0:
        run_dev_eval()
    print(f"\nbest dev macroF1 {best_dev:.4f} @ step {best_step}; loading best checkpoint")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # ---- final dev + test ----
    yt_dv, yp_dv, spk_dv = evaluate(model, dl_dv, device)
    yt_te, yp_te, spk_te = evaluate(model, dl_te, device)
    dev_acc, dev_f1 = accuracy(yt_dv, yp_dv), macro_f1(yt_dv, yp_dv)
    test_acc, test_f1 = accuracy(yt_te, yp_te), macro_f1(yt_te, yp_te)
    sdev_acc, sdev_f1 = speaker_scores(yt_dv, yp_dv, spk_dv)
    stest_acc, stest_f1 = speaker_scores(yt_te, yp_te, spk_te)
    print(f"\nSEGMENT  dev macroF1 {dev_f1:.4f} acc {dev_acc:.4f} | "
          f"test macroF1 {test_f1:.4f} acc {test_acc:.4f}")
    print(f"SPEAKER  dev macroF1 {sdev_f1:.4f} | test macroF1 {stest_f1:.4f} acc {stest_acc:.4f}")

    if args.smoke:
        print("\nsmoke test OK (no CSVs written). Rerun without --smoke for the real run.")
        return

    # ---- write outputs ----
    TABLES.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame([
        {"model": model_name, "level": "segment", "split": "dev",
         "accuracy": round(dev_acc, 4), "macro_f1": round(dev_f1, 4)},
        {"model": model_name, "level": "segment", "split": "test",
         "accuracy": round(test_acc, 4), "macro_f1": round(test_f1, 4)},
        {"model": model_name, "level": "speaker", "split": "dev",
         "accuracy": round(sdev_acc, 4), "macro_f1": round(sdev_f1, 4)},
        {"model": model_name, "level": "speaker", "split": "test",
         "accuracy": round(stest_acc, 4), "macro_f1": round(stest_f1, 4)},
    ])
    upsert(TABLES / "did_results.csv", res)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yt_te, yp_te, labels=list(range(len(REGIONS))))
    pd.DataFrame(cm, index=[f"true_{r}" for r in REGIONS],
                 columns=[f"pred_{r}" for r in REGIONS]).to_csv(
        TABLES / f"did_confusion_test_lid_finetune{tag}.csv")

    pcs = perclass(cm)
    upsert(TABLES / "did_perclass_test.csv",
           pd.DataFrame([{"model": model_name, **d} for d in pcs]))
    # feed the channel-disjoint figure/table (E2/E4 live here too)
    upsert2(TABLES / "did_channel_disjoint.csv",
            pd.DataFrame([{"split_type": split_type, "model": "LID-ECAPA-FT (E6)", **d}
                          for d in pcs]),
            keys=["split_type", "model", "region"])

    lo, hi = boot_ci(yt_te, yp_te)
    upsert(TABLES / "did_bootstrap_ci.csv", pd.DataFrame([
        {"model": model_name, "macro_f1": round(test_f1, 4),
         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}]))

    pd.DataFrame({"speaker_final": spk_te,
                  "true": [I2R[i] for i in yt_te],
                  "pred": [I2R[i] for i in yp_te]}).to_csv(
        TABLES / f"did_lid_finetune_preds_test{tag}.csv", index=False)

    print(f"\n==== per-class (test, segment) [{model_name}] ====")
    print(pd.DataFrame(pcs).to_string(index=False))
    print(f"\nbootstrap 95% CI (test segment macroF1): {test_f1:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"\nwrote: did_results.csv, did_confusion_test_lid_finetune{tag}.csv, "
          f"did_perclass_test.csv, did_channel_disjoint.csv, did_bootstrap_ci.csv, "
          f"did_lid_finetune_preds_test{tag}.csv")


if __name__ == "__main__":
    main()
