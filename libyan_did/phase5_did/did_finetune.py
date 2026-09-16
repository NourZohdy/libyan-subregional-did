"""E7: end-to-end fine-tuned XLS-R-300m for 3-way sub-regional DID.

E3 froze the encoder and trained a logistic probe on top. E7 fine-tunes the full
facebook/wav2vec2-xls-r-300m encoder together with a linear 3-way head, end to
end. The CNN feature extractor stays frozen; the 24 transformer layers and the
head are trained. We train on the train split, keep the checkpoint with the best
dev macro-F1, and report test.

E7 mirrors the E6 (LID-ECAPA) fine-tuning recipe so the two are a clean BACKBONE
comparison: the same class-weighted cross-entropy (--class-weight sqrt) for Fezzan,
SpecAugment (wav2vec2's built-in time/feature masking, the XLS-R analogue of E6's
Fbank SpecAugment), the same splits, and the same metrics. --split-scheme selects
speaker- or channel-disjoint training (channel reuses did_channel_disjoint.assign_channels).

Built for 8 GB VRAM (RTX 5060): batch 1 + gradient accumulation, gradient
checkpointing, bf16 autocast, segments capped at --max-seconds. XLS-R-300m is ~14x
ECAPA's size, so this is SLOW (hours). If it OOMs, rerun with --unfreeze-top 12
(or 6) to train only the top transformer layers and freeze the rest.

Audio is read by random access (soundfile start/stop) per segment from
master_audio_path, so there are no full-file loads and no feature cache.

Outputs (idempotent; model name carries a [chan] suffix on the channel run):
  - appends "XLS-R-FT (E7)" segment+speaker rows to outputs/tables/did_results.csv
  - outputs/tables/did_confusion_test_finetune[_chan].csv
  - appends rows to did_perclass_test.csv and did_bootstrap_ci.csv
  - outputs/tables/did_finetune_preds_test[_chan].csv (speaker_final, true, pred)

Run (diarization env, from libyan_did_pipeline/):
    python scripts/did_finetune.py --smoke                  # tiny shapes/plumbing check
    python scripts/did_finetune.py --split-scheme speaker   # full run (GPU, hours)
    python scripts/did_finetune.py --split-scheme channel   # full run (GPU, hours)
    python scripts/did_finetune.py --unfreeze-top 12        # OOM / speed fallback
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Reduce CUDA fragmentation OOMs on 8 GB VRAM (must precede torch CUDA init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from libyan_did.shared import config
from libyan_did.phase5_did.did_channel_disjoint import assign_channels  # noqa: E402

SPLITS = config.MANIFEST_DIR / "splits_manifest.csv"
TABLES = config.TABLES_DIR
SEED = 1337
REGIONS = list(config.REGIONS)
R2I = {r: i for i, r in enumerate(REGIONS)}
SR = config.SAMPLE_RATE
MODEL_ID = "facebook/wav2vec2-xls-r-300m"


def set_seed(s: int = SEED) -> None:
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class SegDataset:
    """Random-access segment reader -> (waveform float32 @16 kHz, label_id, spk).

    Picklable for DataLoader workers (holds only numpy arrays + an sr cache);
    soundfile/torch are imported inside the methods, never stored on self.
    """

    def __init__(self, df: pd.DataFrame, max_seconds: float):
        self.paths = df["master_audio_path"].to_numpy()
        self.start = df["start_time"].to_numpy()
        self.end = df["end_time"].to_numpy()
        self.y = df["final_dialect"].map(R2I).to_numpy()
        self.spk = df["speaker_final"].astype(str).to_numpy()
        self.max_len = int(max_seconds * SR)
        self._sr: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def _file_sr(self, path: str) -> int:
        sr = self._sr.get(path)
        if sr is None:
            import soundfile as sf
            sr = sf.info(path).samplerate
            self._sr[path] = sr
        return sr

    def __getitem__(self, i: int):
        import soundfile as sf
        import torch
        import torchaudio
        path = self.paths[i]
        fsr = self._file_sr(path)
        a, b = int(self.start[i] * fsr), int(self.end[i] * fsr)
        seg, sr = sf.read(path, start=a, stop=b, dtype="float32", always_2d=True)
        mono = seg.mean(axis=1)
        if sr != SR:
            mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, SR).numpy()
        mono = mono[: self.max_len]
        if mono.shape[0] < 640:                       # too short for the conv stack
            mono = np.pad(mono, (0, 640 - mono.shape[0]))
        return mono.astype(np.float32), int(self.y[i]), str(self.spk[i])


class Collate:
    """Pad a batch of waveforms via the model's feature extractor."""

    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def __call__(self, batch):
        import torch
        wavs = [b[0] for b in batch]
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)
        spk = [b[2] for b in batch]
        inp = self.fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        return inp.input_values, inp.get("attention_mask"), y, spk


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(unfreeze_top: int | None, specaug: bool = True, grad_ckpt: bool = True):
    import torch.nn as nn
    from transformers import AutoModel

    enc = AutoModel.from_pretrained(MODEL_ID)
    # SpecAugment (built into wav2vec2): masks feature-encoder output during training,
    # the XLS-R analogue of E6's Fbank SpecAugment. Matches the fine-tuning recipe.
    enc.config.apply_spec_augment = specaug
    if specaug:
        enc.config.mask_time_prob = 0.075
        enc.config.mask_time_length = 10
        enc.config.mask_feature_prob = 0.012
        enc.config.mask_feature_length = 64
    enc.freeze_feature_encoder()                      # CNN frozen always
    if unfreeze_top is not None:
        for p in enc.parameters():
            p.requires_grad = False
        for layer in enc.encoder.layers[-unfreeze_top:]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(enc.encoder, "layer_norm"):
            for p in enc.encoder.layer_norm.parameters():
                p.requires_grad = True
    if grad_ckpt:
        enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    class DID(nn.Module):
        def __init__(self, encoder, hidden, n_classes):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, input_values, attention_mask):
            import torch
            out = self.encoder(input_values, attention_mask=attention_mask).last_hidden_state
            if attention_mask is not None:
                olen = self.encoder._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)).long()
                m = (torch.arange(out.size(1), device=out.device)[None, :]
                     < olen[:, None]).unsqueeze(-1).to(out.dtype)
                pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                pooled = out.mean(1)
            return self.head(pooled)

    model = DID(enc, enc.config.hidden_size, len(REGIONS))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params {n_train/1e6:.1f}M / {n_all/1e6:.1f}M")
    return model


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=list(range(len(REGIONS))), average="macro"))


def accuracy(y_true, y_pred) -> float:
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y_true, y_pred))


def class_weights(y_ids: np.ndarray, scheme: str) -> np.ndarray | None:
    """Train-split class weights, mean-normalized. Mirrors did_finetune_lid (E6)."""
    counts = np.bincount(y_ids, minlength=len(REGIONS)).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    if scheme == "none":
        return None
    if scheme == "inverse":
        w = counts.sum() / (len(REGIONS) * counts)
    elif scheme == "sqrt":
        w = np.sqrt(counts.sum() / counts)
    elif scheme == "effective":                       # Cui et al. 2019
        beta = 0.999
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(scheme)
    return w / w.mean()


@np.errstate(all="ignore")
def evaluate(model, loader, device):
    """Return (y_true, y_pred, spk) over a loader; eval mode, no grad."""
    import torch
    model.eval()
    was_ckpt = getattr(model.encoder, "is_gradient_checkpointing", False)
    if was_ckpt:
        model.encoder.gradient_checkpointing_disable()
    if device == "cuda":
        torch.cuda.empty_cache()
    yt, yp, sk = [], [], []
    with torch.no_grad():
        for input_values, attn, y, spk in loader:
            input_values = input_values.to(device)
            attn = attn.to(device) if attn is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_values, attn)
            yp.append(logits.float().argmax(-1).cpu().numpy())
            yt.append(y.numpy())
            sk.extend(spk)
    if was_ckpt:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return np.concatenate(yt), np.concatenate(yp), np.array(sk)


def boot_ci(y_true, y_pred, n=1000, seed=SEED):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    N = len(y_true)
    vals = np.empty(n)
    labels = list(range(len(REGIONS)))
    for k in range(n):
        idx = rng.integers(0, N, N)
        vals[k] = f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def upsert(path: Path, new_df: pd.DataFrame, key: str = "model") -> None:
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old[key].isin(new_df[key].unique())]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoFeatureExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=400, help="optimizer steps between dev evals")
    ap.add_argument("--patience", type=int, default=6, help="dev evals without improvement -> stop")
    ap.add_argument("--enc-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--cap-per-speaker", type=int, default=None,
                    help="cap train segments per speaker_final (decorrelates dominant "
                         "hosts and bounds compute); dev/test untouched")
    ap.add_argument("--unfreeze-top", type=int, default=None)
    ap.add_argument("--split-scheme", choices=["speaker", "channel"], default="speaker")
    ap.add_argument("--class-weight", choices=["none", "sqrt", "inverse", "effective"],
                    default="sqrt")
    ap.add_argument("--no-specaug", action="store_true", help="disable SpecAugment")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (faster; needs more VRAM)")
    ap.add_argument("--eval-batch", type=int, default=8, help="batch for dev/test eval (no grad)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=4, help="DataLoader prefetch_factor per worker")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after N optimizer steps and report timing (benchmark; writes nothing)")
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="optimizer steps between crash-safe '_last' checkpoints (model+optimizer+counters)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the '_last' checkpoint if present (survives sleep/crash mid-run)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA; this will be extremely slow.")

    scheme = args.split_scheme
    suffix = "" if scheme == "speaker" else " [chan]"
    model_name = f"XLS-R-FT (E7){suffix}"
    tag = "" if scheme == "speaker" else "_chan"
    best_ckpt = config.MANIFEST_DIR / f"did_finetune_xlsr_{scheme}_best.pt"
    last_ckpt = config.MANIFEST_DIR / f"did_finetune_xlsr_{scheme}_last.pt"

    sp = pd.read_csv(SPLITS)
    cols = ["master_audio_path", "start_time", "end_time", "final_dialect",
            "split", "speaker_final", "playlist", "duration"]
    sp = sp[cols].dropna(subset=["master_audio_path", "final_dialect", "split"])
    bad = set(sp["final_dialect"].unique()) - set(REGIONS)
    assert not bad, f"unexpected labels in final_dialect: {bad}"
    if scheme == "channel":                           # rebuild a channel-disjoint split
        assign = assign_channels(sp[["playlist", "final_dialect", "duration"]])
        sp = sp.dropna(subset=["playlist"]).copy()
        sp["split"] = sp["playlist"].map(assign)
        sp = sp.dropna(subset=["split"])
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

    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    collate = Collate(fe)
    g = torch.Generator().manual_seed(SEED)
    workers = 0 if args.smoke else args.workers
    extra = {"prefetch_factor": args.prefetch} if workers > 0 else {}
    dl_tr = DataLoader(SegDataset(tr, args.max_seconds), batch_size=args.batch, shuffle=True,
                       num_workers=workers, collate_fn=collate, generator=g,
                       pin_memory=(device == "cuda"), drop_last=True,
                       persistent_workers=workers > 0, **extra)
    dl_dv = DataLoader(SegDataset(dv, args.max_seconds), batch_size=args.eval_batch,
                       shuffle=False, num_workers=workers, collate_fn=collate,
                       pin_memory=(device == "cuda"), persistent_workers=workers > 0, **extra)
    dl_te = DataLoader(SegDataset(te, args.max_seconds), batch_size=args.eval_batch,
                       shuffle=False, num_workers=workers, collate_fn=collate,
                       pin_memory=(device == "cuda"), persistent_workers=workers > 0, **extra)

    model = build_model(args.unfreeze_top, specaug=not args.no_specaug,
                        grad_ckpt=not args.no_grad_ckpt).to(device)
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.enc_lr},
         {"params": head_params, "lr": args.head_lr}], weight_decay=0.01,
        fused=(device == "cuda"))
    w = class_weights(tr["final_dialect"].map(R2I).to_numpy(), args.class_weight)
    w_t = None if w is None else torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w_t)
    print(f"class-weight ({args.class_weight}): "
          f"{ {r: round(float(x), 3) for r, x in zip(REGIONS, (w if w is not None else [1, 1, 1]))} } | "
          f"specaug {not args.no_specaug}")

    best_dev = -1.0
    best_step = -1
    stale = 0
    gstep = 0
    start_epoch = 0
    skip_iters = 0                                    # train iters to fast-forward on resume
    t0 = time.time()
    stop = False

    def save_last(epoch: int, epoch_iter: int) -> None:
        """Atomic crash-safe checkpoint: weights + optimizer + counters."""
        tmp = last_ckpt.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "gstep": gstep, "epoch": epoch, "epoch_iter": epoch_iter,
                    "best_dev": best_dev, "best_step": best_step, "stale": stale},
                   tmp)
        os.replace(tmp, last_ckpt)

    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gstep, best_dev = ck["gstep"], ck["best_dev"]
        best_step, stale = ck["best_step"], ck["stale"]
        start_epoch, skip_iters = ck["epoch"], ck["epoch_iter"]
        print(f"RESUMED from {last_ckpt.name}: epoch {start_epoch}, "
              f"skip {skip_iters} iters, gstep {gstep}, best_dev {best_dev:.4f}")

    def run_dev_eval():
        nonlocal best_dev, best_step, stale
        yt, yp, _ = evaluate(model, dl_dv, device)
        f1 = macro_f1(yt, yp)
        acc = accuracy(yt, yp)
        improved = f1 > best_dev + 1e-4
        tag2 = ""
        if improved:
            best_dev, best_step, stale = f1, gstep, 0
            torch.save(model.state_dict(), best_ckpt)
            tag2 = "  <- best (saved)"
        else:
            stale += 1
        print(f"  [dev @ step {gstep}] macroF1 {f1:.4f} acc {acc:.4f} "
              f"(best {best_dev:.4f}){tag2}  {time.time()-t0:.0f}s")
        return improved

    notes = "bf16" + ("" if args.no_grad_ckpt else " + grad-ckpt") + (" + fused" if device == "cuda" else "")
    print(f"\ntraining: batch {args.batch} x accum {args.accum} = eff {args.batch*args.accum} | "
          f"enc_lr {args.enc_lr} head_lr {args.head_lr} | {notes}")
    model.train()
    for epoch in range(args.epochs):
        if epoch < start_epoch:                       # already completed before resume
            continue
        g.manual_seed(SEED + epoch)                   # deterministic per-epoch shuffle -> resumable
        ff = skip_iters if epoch == start_epoch else 0
        if ff:
            print(f"  fast-forwarding {ff} iters into epoch {epoch} (resume)")
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for it, (input_values, attn, y, _) in enumerate(dl_tr):
            if it < ff:                               # skip already-trained iters on resume
                continue
            input_values = input_values.to(device)
            attn = attn.to(device) if attn is not None else None
            y = y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_values, attn)
                loss = loss_fn(logits, y) / args.accum
            loss.backward()
            running += loss.item() * args.accum
            if (it + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if args.max_steps and gstep >= args.max_steps:
                    dt = time.time() - t0
                    print(f"benchmark: {gstep} opt-steps in {dt:.0f}s = {dt/gstep:.2f}s/step "
                          f"(eff batch {args.batch*args.accum})")
                    stop = True
                    break
                if gstep % 50 == 0:
                    print(f"  epoch {epoch} step {gstep} loss {running/ (50*args.accum):.4f} "
                          f"{time.time()-t0:.0f}s")
                    running = 0.0
                if not args.smoke and gstep % args.ckpt_every == 0:
                    save_last(epoch, it + 1)          # crash-safe; survives sleep mid-run
                if gstep % args.eval_every == 0:
                    run_dev_eval()
                    if stale >= args.patience:
                        print(f"early stop: {stale} dev evals without improvement")
                        stop = True
                        break
        if stop:
            break
        run_dev_eval()                                # end-of-epoch eval
        if not args.smoke:
            save_last(epoch + 1, 0)                   # clean boundary: resume starts next epoch
        if stale >= args.patience:
            print(f"early stop after epoch {epoch}")
            break

    if args.max_steps:
        print("benchmark done (no checkpoint/eval/CSV written).")
        return

    if best_step < 0:                                 # never evaluated (tiny run)
        run_dev_eval()
    print(f"\nbest dev macroF1 {best_dev:.4f} @ step {best_step}; loading best checkpoint")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # ---- final dev + test predictions ----
    yt_dv, yp_dv, _ = evaluate(model, dl_dv, device)
    yt_te, yp_te, spk_te = evaluate(model, dl_te, device)
    dev_acc, dev_f1 = accuracy(yt_dv, yp_dv), macro_f1(yt_dv, yp_dv)
    test_acc, test_f1 = accuracy(yt_te, yp_te), macro_f1(yt_te, yp_te)
    print(f"\nSEGMENT  dev macroF1 {dev_f1:.4f} acc {dev_acc:.4f} | "
          f"test macroF1 {test_f1:.4f} acc {test_acc:.4f}")

    # speaker-level: majority vote per speaker_final
    def speaker_scores(yt, yp, spk):
        d = pd.DataFrame({"spk": spk, "true": yt, "pred": yp})
        gg = d.groupby("spk").agg(true=("true", "first"),
                                  pred=("pred", lambda x: x.value_counts().idxmax()))
        return (accuracy(gg["true"].to_numpy(), gg["pred"].to_numpy()),
                macro_f1(gg["true"].to_numpy(), gg["pred"].to_numpy()))
    sdev_acc, sdev_f1 = speaker_scores(yt_dv, yp_dv, _)
    stest_acc, stest_f1 = speaker_scores(yt_te, yp_te, spk_te)
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
        TABLES / f"did_confusion_test_finetune{tag}.csv")

    pc = []
    ps, rs, fs = [], [], []
    for i, r in enumerate(REGIONS):
        tp = cm[i, i]
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        pc.append({"model": model_name, "region": r, "precision": round(prec, 3),
                   "recall": round(rec, 3), "f1": round(f1, 3)})
        ps.append(prec); rs.append(rec); fs.append(f1)
    pc.append({"model": model_name, "region": "Macro", "precision": round(np.mean(ps), 3),
               "recall": round(np.mean(rs), 3), "f1": round(np.mean(fs), 3)})
    upsert(TABLES / "did_perclass_test.csv", pd.DataFrame(pc))

    lo, hi = boot_ci(yt_te, yp_te)
    upsert(TABLES / "did_bootstrap_ci.csv", pd.DataFrame([
        {"model": model_name, "macro_f1": round(test_f1, 4),
         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}]))

    inv = {i: r for r, i in R2I.items()}
    pd.DataFrame({"speaker_final": spk_te,
                  "true": [inv[i] for i in yt_te],
                  "pred": [inv[i] for i in yp_te]}).to_csv(
        TABLES / f"did_finetune_preds_test{tag}.csv", index=False)

    print(f"\n==== per-class (test, segment) [{model_name}] ====")
    print(pd.DataFrame(pc).to_string(index=False))
    print(f"\nbootstrap 95% CI (test segment macroF1): {test_f1:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"\nwrote: did_results.csv, did_confusion_test_finetune{tag}.csv, "
          f"did_perclass_test.csv, did_bootstrap_ci.csv, did_finetune_preds_test{tag}.csv")


if __name__ == "__main__":
    main()
