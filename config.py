"""
Central configuration for the Variphi Video Search Engine.
All tuneable hyperparameters live here — change them once, they propagate everywhere.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
INDEX_DIR      = BASE_DIR / "indices"
THUMB_DIR      = BASE_DIR / "thumbnails"
RESULTS_DIR    = BASE_DIR / "results"
LOG_DIR        = BASE_DIR / "logs"

for _d in (INDEX_DIR, THUMB_DIR, RESULTS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Model ─────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # CLIP backbone — swap to "ViT-L-14" for higher quality at cost of speed
    clip_model_name: str   = "ViT-B-32"
    clip_pretrained: str   = "openai"           # or "laion2b_s34b_b79k" for SigLIP-like
    device: str            = "auto"             # "auto" → cuda if available, else cpu
    use_fp16: bool         = True               # half-precision for ~2× speedup on GPU
    batch_size: int        = 64                 # frames per forward pass
    embedding_dim: int     = 512                # ViT-B-32 output dim


# ── Frame Sampling ─────────────────────────────────────────────────────────────
@dataclass
class SamplingConfig:
    strategy: str          = "adaptive"         # "uniform" | "scene" | "adaptive"
    uniform_fps: float     = 1.0                # frames per second for uniform mode
    scene_threshold: float = 0.35               # cosine distance threshold for scene cuts
    min_scene_gap_sec: float = 1.0              # ignore cuts < this many seconds apart
    max_frames_per_min: int = 120               # safety cap to control memory
    thumbnail_width: int   = 320
    thumbnail_height: int  = 180


# ── Vector Index ───────────────────────────────────────────────────────────────
@dataclass
class IndexConfig:
    # "flat" = exact search (small archives), "ivf" = ANN (large archives)
    index_type: str        = "flat"
    ivf_nlist: int         = 256                # number of Voronoi cells for IVF
    ivf_nprobe: int        = 32                 # cells to visit at query time
    metric: str            = "cosine"           # "cosine" | "l2"


# ── Retrieval ─────────────────────────────────────────────────────────────────
@dataclass
class SearchConfig:
    top_k: int             = 20                 # initial ANN candidates
    return_k: int          = 10                 # results returned to user
    # Temporal context: blend frame embedding with ±window neighbours
    temporal_window: int   = 3                  # frames on each side
    temporal_alpha: float  = 0.3                # weight of neighbours vs centre
    # Re-ranking
    enable_rerank: bool    = True
    rerank_top_n: int      = 50                 # candidates passed to re-ranker
    # Temporal filter (parsed from query or explicit args)
    time_filter_enabled: bool = True


# ── Logging ───────────────────────────────────────────────────────────────────
@dataclass
class LogConfig:
    level: str             = "INFO"
    file: Path             = LOG_DIR / "videosearch.log"


# ── Master config (import this everywhere) ────────────────────────────────────
@dataclass
class Config:
    model:    ModelConfig    = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    index:    IndexConfig    = field(default_factory=IndexConfig)
    search:   SearchConfig   = field(default_factory=SearchConfig)
    log:      LogConfig      = field(default_factory=LogConfig)


CFG = Config()
