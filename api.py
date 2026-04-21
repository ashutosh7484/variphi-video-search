"""
FastAPI REST API for the Variphi Video Search Engine.

Endpoints
---------
GET  /              → Health check
POST /index         → Trigger offline indexing (video file or directory)
GET  /indices       → List available indices
POST /search        → Natural language search
GET  /thumbnail/{frame_path:path} → Serve a thumbnail image
GET  /results       → List saved results files
GET  /benchmark     → Return last benchmark stats
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG, INDEX_DIR, RESULTS_DIR, THUMB_DIR
from src.utils import get_logger, save_results_json, save_results_csv, render_html_results
from src.searcher import VideoSearcher, load_searcher

log = get_logger("api")

app = FastAPI(
    title="Variphi Video Search Engine",
    description="Natural language querying over video archives using CLIP embeddings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# In-memory state
_searcher_cache: Dict[str, VideoSearcher] = {}
_last_benchmark: Dict[str, Any] = {}
_indexing_status: Dict[str, str] = {}


# ── Request / Response models ──────────────────────────────────────────────────

class IndexRequest(BaseModel):
    path: str                    = Field(..., description="Path to video file or directory")
    index_name: Optional[str]    = Field(None, description="Name for the index (default: filename)")
    strategy: str                = Field("adaptive", description="Sampling strategy: uniform|scene|adaptive")


class SearchRequest(BaseModel):
    query: str                   = Field(..., description="Natural language query")
    index_name: str              = Field(..., description="Name of the index to search")
    top_k: int                   = Field(10, ge=1, le=50)
    time_start: Optional[float]  = Field(None, description="Start time filter (seconds)")
    time_end: Optional[float]    = Field(None, description="End time filter (seconds)")
    save_results: bool           = Field(True, description="Persist results to disk")
    render_html: bool            = Field(False, description="Generate HTML results page")


class SearchResult(BaseModel):
    rank:          int
    timestamp_hms: str
    pts_sec:       float
    score:         float
    confidence:    float
    video_file:    str
    frame_path:    str
    thumbnail_url: str


class SearchResponse(BaseModel):
    query:          str
    index_name:     str
    latency_ms:     float
    results:        List[SearchResult]
    json_path:      Optional[str]
    html_path:      Optional[str]


# ── Background indexing task ────────────────────────────────────────────────────

def _run_indexing(path: str, index_name: str, strategy: str):
    """Runs in a thread pool via BackgroundTasks."""
    from src.indexer import index_video, index_directory

    _indexing_status[index_name] = "running"
    try:
        p = Path(path)
        # Temporarily override sampling strategy
        original_strategy = CFG.sampling.strategy
        CFG.sampling.strategy = strategy

        t0 = time.perf_counter()
        if p.is_dir():
            index, meta, idx_path = index_directory(p, index_name)
        else:
            index, meta, idx_path = index_video(p, index_name)

        elapsed = time.perf_counter() - t0
        mem_mb  = psutil.Process().memory_info().rss / 1024 / 1024

        _last_benchmark.update({
            "index_name":    index_name,
            "total_frames":  index.ntotal,
            "indexing_sec":  round(elapsed, 2),
            "frames_per_sec": round(index.ntotal / max(elapsed, 1e-6), 1),
            "peak_mem_mb":   round(mem_mb, 1),
        })
        _indexing_status[index_name] = "done"
        CFG.sampling.strategy = original_strategy

        # Evict cached searcher so it's reloaded fresh
        _searcher_cache.pop(index_name, None)

        log.info("Background indexing of '%s' complete in %.1f s", index_name, elapsed)

    except Exception as exc:
        _indexing_status[index_name] = f"error: {exc}"
        log.exception("Indexing failed for '%s'", index_name)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    ui_path = static_dir / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Variphi Video Search API — <a href='/docs'>docs</a></h2>")


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/index")
async def index_video_endpoint(req: IndexRequest, background_tasks: BackgroundTasks):
    """Kick off async indexing. Returns immediately; poll /index/status/{name}."""
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    index_name = req.index_name or p.stem
    if _indexing_status.get(index_name) == "running":
        return {"status": "already_running", "index_name": index_name}

    _indexing_status[index_name] = "queued"
    background_tasks.add_task(_run_indexing, req.path, index_name, req.strategy)
    return {"status": "queued", "index_name": index_name, "message": f"Poll /index/status/{index_name}"}


@app.get("/index/status/{index_name}")
async def index_status(index_name: str):
    status = _indexing_status.get(index_name, "not_started")
    idx_path = INDEX_DIR / f"{index_name}.faiss"
    return {
        "index_name": index_name,
        "status":     status,
        "exists":     idx_path.exists(),
        "size_mb":    round(idx_path.stat().st_size / 1024 / 1024, 2) if idx_path.exists() else 0,
    }


@app.get("/indices")
async def list_indices():
    """List all built indices."""
    indices = []
    for f in INDEX_DIR.glob("*.faiss"):
        meta_f = INDEX_DIR / f"{f.stem}.meta.json"
        n_frames = 0
        if meta_f.exists():
            meta = json.loads(meta_f.read_text())
            n_frames = len(meta)
        indices.append({
            "name":      f.stem,
            "size_mb":   round(f.stat().st_size / 1024 / 1024, 2),
            "n_frames":  n_frames,
            "modified":  f.stat().st_mtime,
        })
    return {"indices": sorted(indices, key=lambda x: x["modified"], reverse=True)}


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    """Natural language video search."""
    # Lazy-load or reuse cached searcher
    if req.index_name not in _searcher_cache:
        idx_path = INDEX_DIR / f"{req.index_name}.faiss"
        if not idx_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Index '{req.index_name}' not found. Run /index first.",
            )
        log.info("Loading searcher for index '%s' …", req.index_name)
        _searcher_cache[req.index_name] = load_searcher(req.index_name)

    searcher = _searcher_cache[req.index_name]
    t0       = time.perf_counter()

    results  = searcher.search(
        query      = req.query,
        top_k      = req.top_k,
        time_start = req.time_start,
        time_end   = req.time_end,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Attach thumbnail API URLs
    search_results = []
    for r in results:
        fp  = r.get("frame_path", "")
        url = f"/thumbnail/{fp}" if fp else ""
        search_results.append(SearchResult(thumbnail_url=url, **{
            k: r[k] for k in ["rank","timestamp_hms","pts_sec","score","confidence","video_file","frame_path"]
        }))

    # Persist
    json_path = html_path = None
    if req.save_results:
        jp = save_results_json(results, req.query)
        save_results_csv(results, req.query)
        json_path = str(jp)

    if req.render_html:
        hp = render_html_results(results, req.query)
        html_path = str(hp)

    return SearchResponse(
        query       = req.query,
        index_name  = req.index_name,
        latency_ms  = round(latency_ms, 2),
        results     = search_results,
        json_path   = json_path,
        html_path   = html_path,
    )


@app.get("/thumbnail/{frame_path:path}")
async def serve_thumbnail(frame_path: str):
    """Serve a thumbnail image by its absolute path."""
    p = Path(frame_path)
    if not p.exists() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/results")
async def list_results():
    """List available result files."""
    files = []
    for ext in ("*.json", "*.csv", "*.html"):
        for f in RESULTS_DIR.glob(ext):
            files.append({"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1), "path": str(f)})
    return {"results": sorted(files, key=lambda x: x["name"], reverse=True)}


@app.get("/benchmark")
async def benchmark():
    """Return last indexing benchmark stats."""
    mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
    return {**_last_benchmark, "current_mem_mb": round(mem_mb, 1)}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
