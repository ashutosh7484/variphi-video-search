"""
Streamlit UI for the Variphi Video Search Engine.

Run with:
    streamlit run ui.py

Features
--------
* Index a new video (file path or directory)
* Natural language search with temporal filter
* Visual results grid with thumbnails + timestamps
* Confidence bar chart
* Export results as JSON/CSV
* Benchmark panel
"""

import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG, INDEX_DIR, RESULTS_DIR

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Variphi Video Search",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4fc3f7, #9575cd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.sub-header { color: #78909c; font-size: 0.95rem; margin-top: 0; }
.result-card {
    background: #1a1a2e;
    border: 1px solid #2d2d50;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: #4fc3f7; }
.timestamp-badge {
    background: #0d47a1;
    color: #90caf9;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
}
.confidence-high  { color: #69f0ae; }
.confidence-med   { color: #ffcc02; }
.confidence-low   { color: #ff7043; }
.metric-box {
    background: #111827;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.metric-val  { font-size: 1.8rem; font-weight: 700; color: #4fc3f7; }
.metric-label{ font-size: 0.8rem; color: #78909c; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────

if "searcher" not in st.session_state:
    st.session_state.searcher      = None
if "searcher_index" not in st.session_state:
    st.session_state.searcher_index = None
if "last_results" not in st.session_state:
    st.session_state.last_results   = []
if "last_query" not in st.session_state:
    st.session_state.last_query     = ""
if "latency_ms" not in st.session_state:
    st.session_state.latency_ms     = 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _available_indices():
    return [f.stem for f in INDEX_DIR.glob("*.faiss")]


def _img_to_b64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _confidence_class(conf: float) -> str:
    if conf >= 60: return "confidence-high"
    if conf >= 35: return "confidence-med"
    return "confidence-low"


def _load_searcher(index_name: str):
    if st.session_state.searcher_index != index_name:
        from src.searcher import load_searcher
        with st.spinner(f"Loading index '{index_name}' …"):
            st.session_state.searcher       = load_searcher(index_name)
            st.session_state.searcher_index = index_name


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️  Settings")

    st.markdown("**Index management**")
    indices = _available_indices()
    selected_index = st.selectbox(
        "Active index",
        options=["— none —"] + indices,
        index=0,
    )

    st.divider()

    with st.expander("➕  Index a new video", expanded=not indices):
        new_video_path = st.text_input("Video path or directory")
        new_index_name = st.text_input("Index name (optional)")
        strategy       = st.radio("Sampling", ["adaptive", "uniform", "scene"], horizontal=True)

        if st.button("🚀  Start Indexing", use_container_width=True):
            if not new_video_path:
                st.error("Enter a video path first.")
            else:
                from src.indexer import index_video, index_directory
                p = Path(new_video_path)
                name = new_index_name or p.stem
                CFG.sampling.strategy = strategy
                try:
                    with st.spinner("Indexing … this may take a few minutes."):
                        t0 = time.perf_counter()
                        if p.is_dir():
                            index_directory(p, name)
                        else:
                            index_video(p, name)
                        elapsed = time.perf_counter() - t0
                    st.success(f"✓ Indexed '{name}' in {elapsed:.1f} s")
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing error: {e}")

    st.divider()

    st.markdown("**Search config**")
    top_k = st.slider("Top-K results", 1, 30, CFG.search.return_k)
    enable_rerank = st.checkbox("Re-ranking", value=CFG.search.enable_rerank)
    CFG.search.enable_rerank = enable_rerank

    st.divider()
    st.markdown("**Temporal filter** *(optional)*")
    use_time_filter = st.checkbox("Apply time window")
    time_from_str   = st.text_input("From  (HH:MM:SS)", "00:00:00", disabled=not use_time_filter)
    time_to_str     = st.text_input("To    (HH:MM:SS)", "99:59:59", disabled=not use_time_filter)


# ── Main panel ─────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🎬 Variphi Video Search</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Natural language querying over video archives</p>', unsafe_allow_html=True)
st.divider()

# Query bar
col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "Search query",
        placeholder='e.g. "person carrying a bag near the entrance" or "red vehicle after 18:00"',
        label_visibility="collapsed",
    )
with col_btn:
    search_btn = st.button("Search 🔍", use_container_width=True, type="primary")


# ── Search logic ───────────────────────────────────────────────────────────────

if search_btn and query:
    if selected_index == "— none —":
        st.warning("Select or build an index first (sidebar).")
    else:
        _load_searcher(selected_index)
        searcher = st.session_state.searcher

        from src.utils import hms_to_seconds
        time_start = hms_to_seconds(time_from_str) if use_time_filter else None
        time_end   = hms_to_seconds(time_to_str)   if use_time_filter else None

        with st.spinner("Searching …"):
            t0 = time.perf_counter()
            results = searcher.search(
                query      = query,
                top_k      = top_k,
                time_start = time_start,
                time_end   = time_end,
            )
            latency = (time.perf_counter() - t0) * 1000

        st.session_state.last_results = results
        st.session_state.last_query   = query
        st.session_state.latency_ms   = latency

        # Save results
        from src.utils import save_results_json, save_results_csv
        save_results_json(results, query)
        save_results_csv(results, query)


# ── Results display ────────────────────────────────────────────────────────────

results = st.session_state.last_results
query   = st.session_state.last_query

if results:
    # Metrics bar
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{len(results)}</div><div class="metric-label">Results</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{st.session_state.latency_ms:.0f} ms</div><div class="metric-label">Query latency</div></div>', unsafe_allow_html=True)
    with m3:
        best_conf = results[0]["confidence"] if results else 0
        st.markdown(f'<div class="metric-box"><div class="metric-val">{best_conf:.1f}%</div><div class="metric-label">Top confidence</div></div>', unsafe_allow_html=True)
    with m4:
        n_videos = len({r.get("video_file","") for r in results})
        st.markdown(f'<div class="metric-box"><div class="metric-val">{n_videos}</div><div class="metric-label">Videos hit</div></div>', unsafe_allow_html=True)

    st.markdown(f"\n**Query:** *{query}*")
    st.divider()

    # Results grid: 3 columns
    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3]:
            b64 = _img_to_b64(r.get("frame_path", ""))
            conf = r["confidence"]
            cc   = _confidence_class(conf)
            img_html = (
                f'<img src="data:image/jpeg;base64,{b64}" '
                f'style="width:100%;border-radius:8px;margin-bottom:8px;">'
                if b64 else
                '<div style="background:#1a1a2e;height:110px;border-radius:8px;margin-bottom:8px;'
                'display:flex;align-items:center;justify-content:center;color:#555">No thumbnail</div>'
            )
            st.markdown(f"""
<div class="result-card">
  {img_html}
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span class="timestamp-badge">⏱ {r['timestamp_hms']}</span>
    <span class="{cc}" style="font-weight:700">{conf:.1f}%</span>
  </div>
  <div style="font-size:0.78rem;color:#78909c;margin-top:6px">
    {Path(r.get('video_file','?')).name} · rank #{r['rank']}
  </div>
</div>
""", unsafe_allow_html=True)

    # Export
    st.divider()
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(results, indent=2),
            file_name="results.json",
            mime="application/json",
        )
    with col_dl2:
        import csv, io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["rank","timestamp_hms","score","confidence","video_file","frame_path"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in ["rank","timestamp_hms","score","confidence","video_file","frame_path"]} for r in results)
        st.download_button(
            "⬇ Download CSV",
            data=buf.getvalue(),
            file_name="results.csv",
            mime="text/csv",
        )

elif search_btn:
    st.info("No results found. Try a different query or check your index.")

else:
    # Welcome state
    st.markdown("""
    ### How to use

    1. **Index a video** using the sidebar → *➕ Index a new video*
    2. **Select** the index from the dropdown
    3. **Type a query** — spatial, object, or scene-based — and hit Search

    #### Example queries
    - `person near the entrance carrying a bag`
    - `red vehicle parked in zone 3`
    - `two people talking after 18:00`
    - `anything unusual in the corridor`
    """)
