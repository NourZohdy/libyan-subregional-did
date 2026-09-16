"""Libyan DID discovery & corpus workbench — Streamlit launcher.

One app, five pages (sidebar, in pipeline order):

    🔑 Keyword bank        edit the region-tagged Arabic discovery search terms
    🎬 Review channels     approve / reject the discovered YouTube channels
    ➕ Add from channel    download + diarize an approved channel's shows
    📂 My corpus           everything on disk; diarize / cluster your shows
    🔎 Validate clustering check one playlist's speaker clustering by ear

The page scripts live in ``views/``; ``st.navigation`` (not the magic ``pages/`` folder) defines
their order and titles, so this file is the single place that controls the sidebar.

Run:
    python -m streamlit run libyan_did/phase1_discovery/review_app.py
"""

from __future__ import annotations

import streamlit as st

from libyan_did.phase1_discovery import jobs

st.set_page_config(page_title="Libyan DID workbench", layout="wide")

# Live progress for any background download/diarize/cluster job — rendered globally so it follows
# you across every page and keeps refreshing while the work runs on its thread.
jobs.progress_panel()

st.navigation([
    st.Page("views/keyword_bank.py",        title="Keyword bank",        icon="🔑"),
    st.Page("views/review_channels.py",     title="Review channels",     icon="🎬"),
    st.Page("views/add_from_channel.py",    title="Add from channel",    icon="➕"),
    st.Page("views/my_corpus.py",           title="My corpus",           icon="📂"),
    st.Page("views/validate_clustering.py", title="Validate clustering", icon="🔎"),
]).run()
