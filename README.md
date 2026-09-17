# Sub-Regional Libyan Arabic Speech Corpus and DID Benchmark — release v1.0

Release package for the paper

> Nour A. Zohdy and Mansour Essgaer. 2026. *Beyond Monolithic Libyan Arabic: A Sub-Regional
> Speech Corpus and Dialect Identification Benchmark.* Proceedings of ArabicNLP 2026.

DOI of this package: https://doi.org/10.5281/zenodo.22802659
Code repository: https://github.com/NourZohdy/libyan-subregional-did
Hugging Face dataset: https://huggingface.co/datasets/NourZohdy/libyan-subregional-did

The corpus is Libyan Arabic speech from public YouTube and TikTok media, labeled with one of
three regions: **Tripolitania** (west), **Cyrenaica** (east), **Fezzan** (south). It has
133.7 hours of single-speaker segments (3–10 s) from 645 speakers across 161 channels
(57 YouTube channels or playlists, 104 TikTok accounts), with speaker-disjoint train/dev/test
splits and a second, channel-disjoint split.

**We do not redistribute audio.** The package gives you the segment manifest, the source
identifiers to fetch the audio yourself (a YouTube video ID for 717 of the 739 YouTube
recordings; a public account handle plus clip caption for every TikTok clip), and cached embeddings from two pretrained
models so the paper's probe benchmarks can be reproduced without any audio.

## Files

| File | What it is |
|---|---|
| `segments.csv` | 78,411 rows, one per released segment. Columns below. |
| `data/speaker_disjoint/{train,dev,test}.csv`, `data/channel_disjoint/{train,dev,test}.csv` | The same rows as `segments.csv`, cut by the `split` column (paper's main speaker-disjoint benchmark) or the `channel_split` column, so the dataset viewer shows train/dev/test. Use their `segment_id` column, not their local row number, to index the `.npy` files. |
| `channels.csv` | 161 rows, one per channel: platform, majority region, URL or handle where known, channel split, size. |
| `ecapa_spkrec_192.npy` | float32 `[78411, 192]`. Speaker-recognition ECAPA-TDNN embedding (`speechbrain/spkrec-ecapa-voxceleb`) of each segment. Row *i* is `segments.csv` row *i*. Model E2 in the paper. |
| `lid_ecapa_256.npy` | float32 `[78411, 256]`. VoxLingua107 language-ID ECAPA embedding (`speechbrain/lang-id-voxlingua107-ecapa`) of each segment. Row *i* is `segments.csv` row *i*. Model E5 in the paper. |
| `pipeline_v1.0.zip` | The `libyan_did` Python package: discovery, harvesting, diarization and segmentation, quality gates, clustering and linking, validation tooling, split generation, and the benchmark scripts (E0–E6). |
| `SHA256SUMS` | Checksums of every file above. |
| `LICENSE-DATA`, `LICENSE-CODE` | CC BY 4.0 for the manifests and embeddings; MIT for the code. |

Both embedding caches come from **pretrained models only**. No embeddings from the
fine-tuned model (E6) are released, so no label information enters the released features.

## `segments.csv` columns

| Column | Meaning |
|---|---|
| `segment_id` | 0-based row index; also the row index into both `.npy` files. |
| `file_id` | Recording (one video or clip). 1,281 distinct values. |
| `channel_id` | Source channel: a YouTube channel or playlist name, or `tiktok_<handle>`. 161 distinct values. |
| `platform` | `youtube` or `tiktok`. |
| `youtube_id` | YouTube video ID (YouTube rows). Empty for 22 of the 739 YouTube recordings (1,913 segments) whose ID could not be recovered. |
| `source_url` | Channel or playlist URL where we recorded one (14 of the 57 YouTube channels). Locate YouTube audio by `youtube_id`, not by this column. |
| `tiktok_handle` | Public TikTok account handle (TikTok rows). |
| `local_filename` | Our local recording filename for TikTok clips; it contains the clip's public caption and lets you match the clip on the account page. |
| `start_time`, `end_time`, `duration` | Segment boundaries in seconds within the recording, after our diarization and pause-based cutting. |
| `region` | Released label: `Tripolitania`, `Cyrenaica`, or `Fezzan`. This is the label after human review. |
| `provenance_region` | The channel's region before review. Differs from `region` for the 31 relabelled speakers (6,285 segments). |
| `speaker_id` | Anonymous speaker identifier (`spk_00000` …), constant across recordings and channels for the same speaker after cross-episode linking. 645 distinct values. |
| `split` | Speaker-disjoint split: `train`, `dev`, `test` (80/10/10 by duration). A speaker appears in exactly one split. This is the paper's main benchmark split. |
| `channel_split` | Channel-disjoint split: `train`, `dev`, `test`. A channel appears in exactly one split. Used for the robustness test in Section 5.7 of the paper. |
| `snr_db`, `speech_ratio`, `quality` | Signal-to-noise estimate, voice-activity speech ratio, and the resulting quality tag (`clean` = SNR ≥ 15 dB). |

## Rebuilding the audio

YouTube: download the video by `youtube_id` (for example with `yt-dlp`), convert to
16 kHz mono WAV, and cut `[start_time, end_time]`. TikTok: open the account
`https://www.tiktok.com/@<tiktok_handle>`, find the clip whose caption matches
`local_filename`, and cut the same interval. Segment boundaries are relative to the full
recording as published. The pipeline in `pipeline_v1.0.zip` contains the harvesting and
segmentation code we used.

## Reproducing the probe benchmarks

```python
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

seg = pd.read_csv("segments.csv")
X = np.load("lid_ecapa_256.npy")            # or ecapa_spkrec_192.npy
X /= np.linalg.norm(X, axis=1, keepdims=True)
tr, te = seg.split == "train", seg.split == "test"
clf = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], seg.region[tr])
print(f1_score(seg.region[te], clf.predict(X[te]), average="macro"))   # ≈ 0.66 for LID-ECAPA (E5)
```

Use `seg.channel_split` instead of `seg.split` for the channel-disjoint setting.

## Labels and their limits

Labels start from the channel's known region (provenance) and were reviewed by one
annotator per speaker cluster; 95.5% of kept speakers kept their provenance region. A second,
blind review by a co-author agreed with 81.4% of released labels (κ = 0.72). A segment-level
audit found that about 17% of test segments (33% for Fezzan) are Modern Standard Arabic rather
than dialect. Treat the labels as reviewed provenance labels, not gold perceptual labels. See
the paper's Section 3.3 and Limitations.

## Licenses and ethics

Manifests and embeddings: **CC BY 4.0** (`LICENSE-DATA`). Code: **MIT** (`LICENSE-CODE`).
These licenses do not cover the underlying YouTube or TikTok audio, which stays under its
publishers' rights and the platforms' terms.

Speakers are identified only by anonymous IDs. For TikTok, the public account handle is
included because it is the only way to locate the clip; for most TikTok accounts, which
contain a single speaker, the handle identifies that speaker. If you are a speaker or
publisher and want your entries removed, email nour.tammous@idp.sebhau.edu.ly and we will
remove them in the next version of this record.

## Citation

```bibtex
@inproceedings{zohdy2026libyan,
  title     = {Beyond Monolithic Libyan Arabic: A Sub-Regional Speech Corpus and Dialect Identification Benchmark},
  author    = {Zohdy, Nour A. and Essgaer, Mansour},
  booktitle = {Proceedings of ArabicNLP 2026},
  year      = {2026}
}
```

Data and code: Zohdy, N. A. and Essgaer, M. (2026). *Sub-Regional Libyan Arabic Speech
Corpus and Dialect Identification Benchmark: manifests, cached embeddings, and pipeline*
(v1.0). Zenodo. https://doi.org/10.5281/zenodo.22802659
