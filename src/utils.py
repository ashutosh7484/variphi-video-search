"""
Shared utilities: structured logging, benchmarking, result I/O, time parsing.
"""

import csv
import json
import logging
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import psutil

from config import CFG, LOG_DIR


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(CFG.log.level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(CFG.log.file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = get_logger("utils")


# ── Benchmarking ──────────────────────────────────────────────────────────────

class Benchmark:
    """Lightweight profiler — tracks wall-clock and peak RSS memory."""

    def __init__(self, label: str):
        self.label = label
        self._start_wall: float = 0.0
        self._start_mem:  int   = 0
        self.elapsed_ms:  float = 0.0
        self.delta_mb:    float = 0.0

    def __enter__(self):
        proc = psutil.Process()
        self._start_mem  = proc.memory_info().rss
        self._start_wall = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start_wall) * 1000
        proc = psutil.Process()
        self.delta_mb = (proc.memory_info().rss - self._start_mem) / 1024 / 1024
        log.info(
            "[bench] %s → %.1f ms  |  Δmem %.1f MB",
            self.label, self.elapsed_ms, self.delta_mb,
        )


@contextmanager
def timer(label: str) -> Generator[Benchmark, None, None]:
    b = Benchmark(label)
    with b:
        yield b


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def seconds_to_hms(seconds: float) -> str:
    """12345.6  →  '03:25:45'"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_seconds(hms: str) -> float:
    """'03:25:45'  →  12345.0"""
    parts = [float(p) for p in hms.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


# ── Temporal filter parsing ────────────────────────────────────────────────────

_TIME_PATTERN = re.compile(
    r"(?:after|from|between|before|until|up to)\s+"
    r"(\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}\s*[aApP][mM])",
    re.IGNORECASE,
)

_RANGE_PATTERN = re.compile(
    r"between\s+(\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}\s*[aApP][mM])"
    r"\s+and\s+(\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}\s*[aApP][mM])",
    re.IGNORECASE,
)

_AMPM_PATTERN = re.compile(r"(\d{1,2})\s*([aApP][mM])")


def _parse_time_token(token: str) -> float:
    """Parse '18:00', '6pm', '18:00:00' → seconds since midnight."""
    m = _AMPM_PATTERN.match(token.strip())
    if m:
        h = int(m.group(1))
        ampm = m.group(2).lower()
        if ampm == "pm" and h != 12:
            h += 12
        if ampm == "am" and h == 12:
            h = 0
        return h * 3600
    return hms_to_seconds(token.strip())


def extract_time_filter(query: str) -> Optional[Dict[str, float]]:
    """
    Returns {"start": float, "end": float} in seconds, or None.

    Examples handled:
      "after 18:00"           → {start: 64800, end: inf}
      "between 6pm and 8pm"   → {start: 64800, end: 72000}
      "before 10:30"          → {start: 0,     end: 37800}
    """
    rng = _RANGE_PATTERN.search(query)
    if rng:
        return {
            "start": _parse_time_token(rng.group(1)),
            "end":   _parse_time_token(rng.group(2)),
        }

    for match in _TIME_PATTERN.finditer(query):
        keyword = match.group(0).lower()
        ts      = _parse_time_token(match.group(1))
        if any(k in keyword for k in ("after", "from")):
            return {"start": ts, "end": float("inf")}
        if any(k in keyword for k in ("before", "until", "up to")):
            return {"start": 0.0, "end": ts}

    return None


def clean_query(query: str) -> str:
    """Strip temporal filter phrases from query so they don't confuse CLIP."""
    q = _RANGE_PATTERN.sub("", query)
    q = _TIME_PATTERN.sub("", q)
    return re.sub(r"\s{2,}", " ", q).strip()


# ── Result I/O ────────────────────────────────────────────────────────────────

def save_results_json(
    results: List[Dict[str, Any]],
    query: str,
    path: Optional[Path] = None,
) -> Path:
    """Append a query + results record to results.json (newline-delimited JSON)."""
    from config import RESULTS_DIR
    path = path or RESULTS_DIR / "results.json"
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "query":     query,
        "results":   results,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    log.info("Results saved → %s", path)
    return path


def save_results_csv(
    results: List[Dict[str, Any]],
    query: str,
    path: Optional[Path] = None,
) -> Path:
    from config import RESULTS_DIR
    path = path or RESULTS_DIR / "results.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "rank", "timestamp_hms", "score", "video_file", "frame_path"],
        )
        if write_header:
            writer.writeheader()
        for i, r in enumerate(results, 1):
            writer.writerow({
                "query":         query,
                "rank":          i,
                "timestamp_hms": r.get("timestamp_hms", ""),
                "score":         round(r.get("score", 0.0), 4),
                "video_file":    r.get("video_file", ""),
                "frame_path":    r.get("frame_path", ""),
            })
    log.info("Results CSV saved → %s", path)
    return path


def render_html_results(
    results: List[Dict[str, Any]],
    query: str,
    path: Optional[Path] = None,
) -> Path:
    """Write a self-contained HTML results page with embedded thumbnails."""
    from config import RESULTS_DIR
    import base64

    path = path or RESULTS_DIR / f"results_{int(time.time())}.html"

    rows = []
    for i, r in enumerate(results, 1):
        thumb_b64 = ""
        fp = r.get("frame_path", "")
        if fp and Path(fp).exists():
            with open(fp, "rb") as img_f:
                thumb_b64 = base64.b64encode(img_f.read()).decode()

        img_tag = (
            f'<img src="data:image/jpeg;base64,{thumb_b64}" '
            f'style="width:200px;border-radius:6px;">'
            if thumb_b64 else '<div style="width:200px;height:112px;background:#222;border-radius:6px;"></div>'
        )
        rows.append(f"""
        <tr>
          <td style="padding:8px;color:#aaa">{i}</td>
          <td style="padding:8px">{img_tag}</td>
          <td style="padding:8px;font-family:monospace;font-size:1.1em">{r.get('timestamp_hms','')}</td>
          <td style="padding:8px;color:#4fc3f7">{r.get('score',0):.4f}</td>
          <td style="padding:8px;color:#aaa;font-size:0.85em">{Path(r.get('video_file','')).name}</td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Video Search Results</title>
<style>
  body {{background:#0d0d0d;color:#eee;font-family:'Segoe UI',sans-serif;padding:32px}}
  h1 {{color:#4fc3f7;margin-bottom:4px}}
  .query {{color:#aaa;margin-bottom:24px;font-style:italic}}
  table {{border-collapse:collapse;width:100%}}
  tr:nth-child(even) {{background:#111}}
  th {{background:#1a1a2e;padding:10px;text-align:left;color:#4fc3f7}}
</style>
</head>
<body>
  <h1>🎬 Video Search Results</h1>
  <div class="query">Query: "{query}"</div>
  <table>
    <tr><th>#</th><th>Thumbnail</th><th>Timestamp</th><th>Score</th><th>Video</th></tr>
    {''.join(rows)}
  </table>
</body>
</html>"""

    path.write_text(html)
    log.info("HTML results page → %s", path)
    return path
