"""Background-job runner so long steps (download / diarize / cluster) run OFF the Streamlit
script thread: the UI stays responsive (switch pages or browser tabs freely) and shows a live
progress bar, and the work keeps going regardless of where you navigate.

ponytail: one in-process dict + a daemon thread per job; no queue, no persistence — restarting
the Streamlit server forgets a running job. Fine for this single-user local app. The worker
thread NEVER calls ``st`` (no script context); it only mutates its job dict, which the main
thread reads to draw the bar.
"""

from __future__ import annotations

import threading
import uuid

import streamlit as st

_JOBS: dict[str, dict] = {}


def submit(kind: str, target) -> str:
    """Run ``target(report)`` in a daemon thread and return a job id.

    ``report(done, total=None, msg=None)`` updates the bar; ``target``'s return value (a short
    summary string) is shown on completion.
    """
    jid = uuid.uuid4().hex[:8]
    job = {"id": jid, "kind": kind, "status": "running",
           "done": 0, "total": 0, "msg": "starting…", "result": None, "error": None}
    _JOBS[jid] = job

    def report(done, total=None, msg=None):
        job["done"] = int(done)
        if total is not None:
            job["total"] = int(total)
        if msg is not None:
            job["msg"] = str(msg)

    def run():
        try:
            job["result"] = target(report)
            job["status"] = "done"
        except Exception as exc:  # noqa: BLE001 — surface in the panel, never crash the thread
            job["status"], job["error"] = "error", f"{type(exc).__name__}: {exc}"

    threading.Thread(target=run, daemon=True).start()
    return jid


def busy() -> bool:
    """True while this session's job is still running (so action buttons can disable)."""
    j = _JOBS.get(st.session_state.get("job"))
    return bool(j and j["status"] == "running")


@st.fragment(run_every="1s")
def _job_fragment() -> None:
    jid = st.session_state.get("job")
    job = _JOBS.get(jid) if jid else None
    if not job:
        return
    if job["status"] == "running":
        total, done = job["total"], job["done"]
        frac = min(done / total, 1.0) if total else 0.0
        st.progress(frac, text=f"⏳ {job['kind']}: {job['msg']}"
                    + (f"  ·  {done}/{total}" if total else ""))
        return
    # terminal — refresh disk caches once so tables reflect the new files, then show the result
    if st.session_state.get("_job_refreshed") != jid:
        st.cache_data.clear()
        st.session_state["_job_refreshed"] = jid
    if job["status"] == "error":
        st.error(f"❌ {job['kind']} failed: {job['error']}")
    else:
        st.success(f"✅ {job['kind']} finished — {job.get('result') or job['msg']}")
    if st.button("Dismiss", key="job_dismiss"):
        _JOBS.pop(jid, None)
        st.session_state.pop("job", None)
        st.session_state.pop("_job_refreshed", None)
        st.rerun()


def progress_panel() -> None:
    """Draw this session's active job (if any). Mount once, globally (the launcher) so it follows
    you across pages. Only mounts the 1-second auto-refresh while a job exists."""
    if st.session_state.get("job"):
        _job_fragment()


def start(kind: str, target) -> None:
    """Submit a job and show its panel on the next run (the standard call-site one-liner)."""
    st.session_state["job"] = submit(kind, target)
    st.session_state.pop("_job_refreshed", None)
    st.rerun()
