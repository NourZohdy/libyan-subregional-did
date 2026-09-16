"""Single source of truth for every pipeline constant.

Each value is either (a) a published default (cite the tool/model) or (b) tuned on
a held-out dev set and reported. Nothing is left as "empirically chosen". The paper's
methods section can be generated directly from the comments below.

See plan section 9 (Constants) for the full justification table.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# config.py lives at libyan_did/shared/config.py, so the repo root is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "libyan_did"
OUTPUTS = PROJECT_ROOT / "outputs"
MANUSCRIPTS_DIR = PROJECT_ROOT / "manuscripts"
CORPUS_PAPER_DIR = MANUSCRIPTS_DIR / "corpus_paper"

# Code-side locations (live inside the package, not the data store).
ASSETS_DIR = PACKAGE_ROOT / "shared" / "assets"              # bundled geojson etc.
DISCOVERY_DIR = PACKAGE_ROOT / "phase1_discovery" / "data"   # candidates.psv, blocklist, â€¦

RAW_AUDIO_DIR = OUTPUTS / "raw_audio"   # retained full episodes â€” NEVER auto-deleted
RTTM_DIR = OUTPUTS / "rttm"             # tiny diarization text files
MANIFEST_DIR = OUTPUTS / "manifests"    # master_segments.csv, embeddings.npy, state, splits
CLIPS_DIR = OUTPUTS / "clips"           # final exported clips (Tier/<global_actor>/)
REJECTS_DIR = OUTPUTS / "rejects"       # foreign guests / commercials
FIGURES_DIR = OUTPUTS / "figures"       # paper-ready .png @300 dpi
TABLES_DIR = OUTPUTS / "tables"         # paper-ready .csv summary tables
GOLD_DIR = OUTPUTS / "gold"             # human-validation GOLD subset: clips + sheet
CORPUS_DIR = OUTPUTS / "corpus"         # per-playlist self-contained unit: manifests + actor clips
METADATA_DIR = OUTPUTS / "metadata"     # combined corpus metadata

MASTER_SEGMENTS_CSV = MANIFEST_DIR / "master_segments.csv"
EMBEDDINGS_NPY = MANIFEST_DIR / "embeddings.npy"
STATE_JSON = MANIFEST_DIR / "segmentation_state.json"
CORPUS_MANIFEST = MANIFEST_DIR / "corpus_manifest.csv"     # all resources combined (final step)
CORPUS_METADATA = METADATA_DIR / "corpus_metadata.json"

# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
SAMPLE_RATE = 16000  # ECAPA/VoxCeleb standard; formants <8 kHz captured.
CHANNELS = 1         # mono

# --------------------------------------------------------------------------- #
# Silero VAD  (defaults from github.com/snakers4/silero-vad)
# --------------------------------------------------------------------------- #
VAD_THRESHOLD = 0.5                 # Silero default ("recommended for most datasets").
VAD_MIN_SPEECH_DURATION_MS = 250    # Silero default.
VAD_MIN_SILENCE_DURATION_MS = 250   # >Silero default 100; bridges micro-pauses (stated rationale).
VAD_SPEECH_PAD_MS = 30              # Silero default speech_pad_ms (the "fricative padding").

# --------------------------------------------------------------------------- #
# pyannote diarization (3.x / community-1 model-card defaults)
# --------------------------------------------------------------------------- #
PYANNOTE_PIPELINE = "pyannote/speaker-diarization-community-1"
PYANNOTE_MIN_DURATION_OFF = 0.08
# clustering threshold left at pipeline default unless a dev-set sweep justifies otherwise.

# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
# Floor raised 1.5 -> 3.0 s to match ADI-20 (discards <3 s) for benchmark comparability.
MIN_DURATION = 3.0
MAX_DURATION = 10.0          # vs ADI-20 30 s / MASC 20 s; shorter is fine for DID.
GAP_TOLERANCE = 0.5          # merge same-speaker turns separated by <= this (seconds).
SPLIT_WINDOW_S = 0.2         # find_best_split_time search half-window (energy-valley).

# Short-duration eval bucket (reported separately, ADI-17/20 short/med/long).
SHORT_BUCKET_RANGE = (1.5, 3.0)
DURATION_BUCKETS = {"short": (3.0, 5.0), "medium": (5.0, 8.0), "long": (8.0, 10.0)}

# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_DIM = 192          # ECAPA-TDNN (Desplanques et al. 2020), L2-normalized, cosine.

# --------------------------------------------------------------------------- #
# Clustering / purge  (tune + report on dev set; values below are starting points)
# --------------------------------------------------------------------------- #
MAD_MULTIPLIER = 2.5         # Leys et al. 2013 moderate-conservative (was 2.0 "poorly conservative").
AHC_MERGE_THRESHOLD = 0.30   # cosine; tune on labeled dev subset (optimize DER/NMI) and report.
GLOBAL_PURGE_LIMIT = 0.6     # tune + report on dev set.
MIN_SPEECH_RATIO = 0.75      # VAD-density QA threshold; tune + report.

# Kneedle (Satopaa et al. 2011) for Pareto tiering â€” no arbitrary cutoff.
KNEEDLE_SENSITIVITY = 1.0

# --------------------------------------------------------------------------- #
# Post-AHC actor merge pass (fixes one speaker split across 2-3 actors)
# Calibrated 2026-06-10 on the full corpus actor-centroid similarity distribution
# (scripts/calibrate_merge.py over 2,458 actors): bands sized so auto-merge sits in
# the same-voice tail (within-playlist p99.7+) and the human queue stays ~1h of work.
# --------------------------------------------------------------------------- #
ACTOR_MERGE_AUTO_SIM = 0.70    # within-playlist auto-merge (verification-grade sim)
ACTOR_MERGE_REVIEW_SIM = 0.60  # absolute floor of the same/different review band
# The review band is ADAPTIVE per playlist: drama-series playlists share one studio
# channel, which inflates ALL pairwise sims (a 0.65 pair is "normal background" there,
# i.e. almost surely two different people). A pair is queued only if it is BOTH above
# the absolute floor AND a >=Z robust-z outlier vs. its own playlist's pair-sim
# distribution (Leys et al. 2013 again). Calibrated 2026-06-10: cuts the queue from
# ~600+ pairs in the 13 largest playlists to ~80 statistically meaningful ones.
ACTOR_MERGE_REVIEW_Z = 4.0
# Pairs where either actor has fewer segments than this are never queued ("dust"
# actors: the merge decision cannot meaningfully change the corpus).
MERGE_REVIEW_MIN_SEGS = 3

# Cross-playlist speaker linking (canonical speaker id; folders untouched).
LINK_AUTO_SIM = 0.72           # ~185 same-region cross-show pairs auto-linked
LINK_REVIEW_SIM = 0.65         # [0.65, 0.72) -> ~320 review pairs
# Cross-REGION pairs imply a label conflict, so they are NEVER auto-linked; only
# high-confidence matches are queued (~90 pairs).
LINK_CROSS_REGION_REVIEW_SIM = 0.70

# --------------------------------------------------------------------------- #
# Dust floor (insufficient-evidence rule; adopted 2026-06-12)
# Calibrated on 711 human decisions: speakers with <5 segments were kept only
# 53-75% of the time vs 81-92% above, and each sub-floor review yielded ~8 s of
# corpus audio. Measured on the AUTO-LINKED aggregate (speaker_global, i.e.
# after auto-merge/auto-link, before human merge confirmation): a small actor
# auto-linked to a big speaker of the same person is NOT dust. Speakers below
# the floor are excluded from the corpus, the review queue, and the human merge
# queue. The rule is absolute: it overrides human keep decisions recorded on
# sub-floor speakers (decisions stay on file). Corpus cost: ~2.3% of hours.
# --------------------------------------------------------------------------- #
DUST_MIN_SEGMENTS = 5     # evidence floor: total segments per linked identity ...
DUST_MIN_SECONDS = 45.0   # ... OR total speech seconds; below either -> excluded

# --------------------------------------------------------------------------- #
# Content scoring models (run once per segment, written into the manifest)
# --------------------------------------------------------------------------- #
# PANNs CNN14 AudioSet tagger (Kong et al. 2020): music/singing detector.
# Silero VAD hears singing as speech, so the speech_ratio gate misses theme songs.
# Gate (verified on Ù†Øµ_Ù†Øµ, 2026-06-10): a segment is "music" only when the music
# evidence is high AND CNN14's Speech prob is low â€” TV shows constantly talk over a
# music bed (music_prob>=0.9 with speech_prob ~0.9), and that speech must be KEPT.
MUSIC_REJECT_PROB = 0.60       # max(Music,Singing) >= this ...
PANNS_SPEECH_KEEP_PROB = 0.50  # ... AND Speech < this -> excluded (reason: music)
PANNS_SAMPLE_RATE = 32000      # CNN14 was trained at 32 kHz; slices are resampled up.

# Per-ACTOR triage scores (ranking only â€” never auto-reject):
HUBERT_DIALECT_MODEL = "IbrahimAmin/hubert-arabic-spoken-dialect-classifier"
LID_MODEL = "speechbrain/lang-id-voxlingua107-ecapa"
ACTOR_SCORE_MAX_SEGMENTS = 10  # longest clean segments sampled per actor for scoring

# Gender-mix cluster-purity flag (ranking only â€” never auto-reject).
# audeering wav2vec2 labels GENDER_PROBE_SAMPLES sampled segments once (GPU);
# a logistic probe trained on those labels over the cached ECAPA embeddings then
# predicts gender for ALL segments for free. Per actor, gender_mix = fraction of
# minority-gender segments; a high value on a multi-segment actor means the
# cluster likely contains two people -> risk_score = max(base, 2*gender_mix).
AGE_GENDER_MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"
GENDER_PROBE_SAMPLES = 2000    # segments labeled by the teacher model
GENDER_PROBE_MIN_MARGIN = 0.30 # teacher |P(f)-P(m)| below this -> not a label
GENDER_MIX_MIN_SEGMENTS = 5    # flag only actors with enough segments to mean anything

# --------------------------------------------------------------------------- #
# Quality / labeling (two-tier GOLD/SILVER; Â§4c)
# --------------------------------------------------------------------------- #
SNR_CLEAN_THRESHOLD_DB = 15.0   # >= clean, else noisy (MASC-style clean/noisy flag).
SNR_REJECT_FLOOR_DB = 3.0       # hard floor: below this SNR proxy a segment is unusable -> rejected.
GOLD_SAMPLES_PER_DIALECT = 100  # segments sampled for human validation.
PER_SPEAKER_SEGMENT_CAP = None  # set an int to cap dominant speakers (Â§4.5).

# --------------------------------------------------------------------------- #
# Splits (speaker-disjoint; Â§4.3 / Part I #1)
# --------------------------------------------------------------------------- #
SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
SPLIT_SEED = 1337
# Balanced-split controls (separate step 3, run on the COMBINED corpus).
SPLIT_BALANCE_BY = "duration"   # balance split sizes by "duration" (seconds) or "segments".

# Libyan sub-dialect regions (Tripolitania=West, Cyrenaica=East, Fezzan=South).
REGIONS = ("Tripolitania", "Cyrenaica", "Fezzan")

# On-disk dialect folder names are kept verbatim; this maps them to the canonical region
# used for ALL output artifacts (clip tree, manifests, metadata). Case-insensitive; '-'
# and '_' are treated the same. Add new folder spellings here, never rename the folders.
REGION_ALIASES = {
    "benghazi": "Cyrenaica", "cyrenaica": "Cyrenaica", "east": "Cyrenaica", "eastern": "Cyrenaica",
    "tripoli": "Tripolitania", "tripolitania": "Tripolitania", "west": "Tripolitania", "western": "Tripolitania",
    "southern": "Fezzan", "south": "Fezzan", "fezzan": "Fezzan", "sabha": "Fezzan",
}

# 18 non-Libyan control classes (FR-001). macro = "Non-Libyan" comes for free from the
# non_libyan/ root folder via harvest._normalize_macro. These aliases make
# harvest.canonical_region() return the CODE (e.g. "EGY") instead of a title-cased folder name.
# (ADI's own Libyan class "LIB" is excluded per D8 â€” it is not a control.)
CONTROL_CODES = ("ALG", "EGY", "IRA", "JOR", "KSA", "KUW", "LEB", "MAU", "MOR",
                 "MSA", "OMA", "PAL", "QAT", "SUD", "SYR", "TUN", "UAE", "YEM")
# Map each on-disk control folder name (lowercased) -> its canonical CODE. On-disk folders
# are exactly the uppercased codes, so the lowercase->code map is sufficient (no extra spellings).
REGION_ALIASES.update({code.lower(): code for code in CONTROL_CODES})

# Canonical region -> on-disk dialect FOLDER name used when HARVESTING new sources into the
# source tree (src/acquire.py). Mirrors the original folders (benghazi/southern/tripoli) so new
# shows land beside the first harvest; enumerate_resources canonicalizes them back via
# REGION_ALIASES, so the output corpus always keys on the canonical REGIONS regardless.
REGION_TO_DIALECT_FOLDER = {
    "Tripolitania": "tripoli",
    "Cyrenaica": "benghazi",
    "Fezzan": "southern",
    "Non-Libyan": "non_libyan",
}

# Tier labels kept in the final corpus; everything else -> reject list.
KEEP_TIERS = ("Tier1", "Tier2", "Tier3")

# Toggle: materialize final clips to disk, or keep manifest+timestamps only.
# False -> per-playlist clustering writes only final_manifest.csv (timestamps); no .wav are
# cut. Everything downstream (linking, combine, scoring, both validation apps) slices audio
# on the fly from master_audio_path + start/end, so no physical clips are needed.
MATERIALIZE_CLIPS = False

# Local-folder ingestion (alternative to YouTube harvest). Set to a folder laid out as
#   <macro_class>/<sub_cat>/<playlist>/*.wav
# (e.g. libyan/cyrenaica/al_katiba/ep1.wav) to segment already-downloaded WAVs in place
# instead of downloading from playlists.py. Leave None to harvest from YouTube.
LOCAL_AUDIO_ROOT = None


def all_dirs() -> list[Path]:
    """Output directories the pipeline expects to exist."""
    return [
        RAW_AUDIO_DIR, RTTM_DIR, MANIFEST_DIR,
        REJECTS_DIR, FIGURES_DIR, TABLES_DIR, GOLD_DIR,
        CORPUS_DIR, METADATA_DIR,
    ]  # CLIPS_DIR intentionally omitted: the new pipeline writes clips inside CORPUS_DIR.


def ensure_dirs() -> None:
    for d in all_dirs():
        d.mkdir(parents=True, exist_ok=True)


def params_signature() -> dict:
    """The constants that, if changed, must trigger a hard reset (see src/state.py)."""
    return {
        "SAMPLE_RATE": SAMPLE_RATE,
        "VAD_THRESHOLD": VAD_THRESHOLD,
        "VAD_MIN_SPEECH_DURATION_MS": VAD_MIN_SPEECH_DURATION_MS,
        "VAD_MIN_SILENCE_DURATION_MS": VAD_MIN_SILENCE_DURATION_MS,
        "VAD_SPEECH_PAD_MS": VAD_SPEECH_PAD_MS,
        "PYANNOTE_PIPELINE": PYANNOTE_PIPELINE,
        "PYANNOTE_MIN_DURATION_OFF": PYANNOTE_MIN_DURATION_OFF,
        "MIN_DURATION": MIN_DURATION,
        "MAX_DURATION": MAX_DURATION,
        "GAP_TOLERANCE": GAP_TOLERANCE,
        "SPLIT_WINDOW_S": SPLIT_WINDOW_S,
        "ECAPA_SOURCE": ECAPA_SOURCE,
    }
