"""Libyan Arabic sub-national dialect-identification corpus pipeline.

Organized by pipeline phase:
    shared/            config + the reusable library (segment, embed, cluster, …)
    phase1_discovery/  find & vet YouTube sources -> approved candidates
    phase2_harvest/    download raw audio + diarize for approved sources
    phase3_corpus/     Step 1 (per-resource) + Step 2 (corpus-wide) build
    phase4_validation/ human validation -> freeze -> speaker-disjoint splits
    phase5_did/        the DID benchmark (E0-E7) + robustness/significance
    phase6_reporting/  figure & map drivers for the paper
"""
