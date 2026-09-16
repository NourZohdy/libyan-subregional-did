"""Age + gender inference (audeering wav2vec2) — the TEACHER model.

Wraps ``config.AGE_GENDER_MODEL`` (audeering/wav2vec2-large-robust-24-ft-age-gender,
Wagner et al. 2023): one forward pass returns age (0-1, x100 years) and a
(female, male, child) softmax. The checkpoint has custom heads, so the model
class from the audeering model card is reproduced here.

Used twice in the pipeline:
* scripts/build_gender_flags.py — labels a ~2k-segment sample once; a logistic
  probe on the cached ECAPA embeddings then scales gender to ALL segments free.
* post-validation speaker enrichment (age buckets + gender per verified
  speaker) reuses this exact wrapper.
"""

from __future__ import annotations

import numpy as np

from libyan_did.shared import config

GENDER_LABELS = ("female", "male", "child")


def _build():
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class ModelHead(nn.Module):
        def __init__(self, cfg, num_labels):
            super().__init__()
            self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
            self.dropout = nn.Dropout(cfg.final_dropout)
            self.out_proj = nn.Linear(cfg.hidden_size, num_labels)

        def forward(self, features):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            return self.out_proj(self.dropout(x))

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.config = cfg
            self.wav2vec2 = Wav2Vec2Model(cfg)
            self.age = ModelHead(cfg, 1)
            self.gender = ModelHead(cfg, 3)
            # model-card code says init_weights(); newer transformers need
            # post_init() to set up the tied-weights bookkeeping
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(config.AGE_GENDER_MODEL)
    model = AgeGenderModel.from_pretrained(config.AGE_GENDER_MODEL).to(device).eval()
    return processor, model, device


class AgeGenderScorer:
    """Lazy-loaded scorer; ``predict`` takes 16 kHz float32 mono slices."""

    def __init__(self) -> None:
        self._loaded = None

    def predict(self, slices: list[np.ndarray], batch_size: int = 8) -> list[dict]:
        """One dict per slice: age_years + female/male/child probs + label."""
        import torch

        if self._loaded is None:
            self._loaded = _build()
        processor, model, device = self._loaded

        out: list[dict] = []
        for i in range(0, len(slices), batch_size):
            batch = [np.asarray(s, dtype=np.float32) for s in slices[i:i + batch_size]]
            inputs = processor(batch, sampling_rate=config.SAMPLE_RATE,
                               return_tensors="pt", padding=True)
            with torch.no_grad():
                age, gender = model(inputs.input_values.to(device))
            age = age.squeeze(-1).cpu().numpy()
            gender = gender.cpu().numpy()
            for a, g in zip(np.atleast_1d(age), np.atleast_2d(gender)):
                out.append({
                    "age_years": float(a) * 100.0,
                    "female": float(g[0]), "male": float(g[1]), "child": float(g[2]),
                    "gender": GENDER_LABELS[int(np.argmax(g))],
                })
        return out
