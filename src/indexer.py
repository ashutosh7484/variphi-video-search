"""
Indexing Pipeline
=================
Responsibilities:
  1. Frame sampling  (adaptive scene-change + uniform fallback)
  2. Thumbnail extraction
  3. CLIP embedding generation (batched, FP16-aware)
  4. Temporal context blending
  5. FAISS index construction and persistence
  6. Metadata sidecar (frame_id → video, pts, timestamp)

Design decisions
----------------
* **Adaptive sampling**: compute per-frame histogram distance; emit a keyframe when
  the distance exceeds `scene_threshold` OR at least every `1/uniform_fps` seconds.
  This balances coverage and index size without any external scene-detection library.

* **CLIP ViT-B-32**: 512-d embeddings, well-supported, fast on CPU and GPU.
  Swappable to ViT-L-14 via config for higher recall.

* **Temporal context blending**: each frame's stored embedding is a weighted average
  of ±`temporal_window` neighbours.  This encodes short-term temporal context so
  a query like "person sitting down" can match transitional frames.

* **FAISS IndexFlatIP** (exact cosine) for archives up to ~200k frames;
  **IndexIVFFlat** with nlist=256 for larger archives.  The flat index fits
  200k×512×4B = ~410 MB RAM — acceptable for the stated use-case.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import open_clip
import faiss
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CFG, INDEX_DIR, THUMB_DIR
from src.utils import get_logger, seconds_to_hms, timer, Benchmark

log = get_logger("indexer")


# ── Model loading ─────────────────────────────────────────────────────────────

def _resolve_device() -> torch.device:
    if CFG.model.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(CFG.model.device)


def load_clip_model() -> Tuple[torch.nn.Module, object, torch.device]:
    """Load CLIP model + preprocessor, applying FP16 if requested."""
    device = _resolve_device()
    log.info("Loading CLIP %s (%s) on %s …", CFG.model.clip_model_name, CFG.model.clip_pretrained, device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        CFG.model.clip_model_name,
        pretrained=CFG.model.clip_pretrained,
    )
    model = model.eval().to(device)
    if CFG.model.use_fp16 and device.type == "cuda":
        model = model.half()
        log.info("FP16 enabled.")
    return model, preprocess, device


# ── Frame sampling ────────────────────────────────────────────────────────────

def _frame_histogram_distance(prev: np.ndarray, curr: np.ndarray) -> float:
    """
    Normalised L1 distance between HSV histograms of two BGR frames.
    Returns 0 (identical) … 1 (completely different).
    """
    def hist(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h.flatten()

    return float(cv2.compareHist(hist(prev), hist(curr), cv2.HISTCMP_BHATTACHARYYA))


def sample_frames(
    video_path: Path,
    strategy: str = CFG.sampling.strategy,
) -> List[Dict]:
    """
    Returns a list of dicts:
      { "frame_idx": int, "pts_sec": float, "timestamp_hms": str, "frame": np.ndarray }

    Strategies
    ----------
    uniform   — one frame every 1/uniform_fps seconds
    scene     — emit only on scene cuts
    adaptive  — scene cuts OR every 1/uniform_fps seconds, whichever comes first
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    log.info(
        "Video: %s | %.1f s | %.1f fps | %d frames",
        video_path.name, duration_sec, fps, total_frames,
    )

    step_frames   = max(1, int(fps / CFG.sampling.uniform_fps))
    min_gap       = int(CFG.sampling.min_scene_gap_sec * fps)
    samples       = []
    prev_frame    = None
    last_emit_idx = -min_gap
    frame_idx     = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pbar = tqdm(total=total_frames, desc=f"Sampling {video_path.name}", unit="fr")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pts_sec = frame_idx / fps
        emit    = False

        if strategy == "uniform":
            emit = (frame_idx % step_frames == 0)

        elif strategy == "scene":
            if prev_frame is not None and (frame_idx - last_emit_idx) >= min_gap:
                dist = _frame_histogram_distance(prev_frame, frame)
                emit = (dist >= CFG.sampling.scene_threshold)
            elif frame_idx == 0:
                emit = True

        elif strategy == "adaptive":
            if frame_idx == 0:
                emit = True
            else:
                uniform_due = (frame_idx % step_frames == 0)
                if prev_frame is not None and (frame_idx - last_emit_idx) >= min_gap:
                    dist = _frame_histogram_distance(prev_frame, frame)
                    scene_cut = (dist >= CFG.sampling.scene_threshold)
                else:
                    scene_cut = False
                emit = uniform_due or scene_cut

        if emit:
            samples.append({
                "frame_idx":     frame_idx,
                "pts_sec":       pts_sec,
                "timestamp_hms": seconds_to_hms(pts_sec),
                "frame":         frame.copy(),
            })
            last_emit_idx = frame_idx

        prev_frame = frame
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    log.info(
        "Sampled %d frames from %d total (%.1f%%)",
        len(samples), total_frames, 100 * len(samples) / max(total_frames, 1),
    )
    return samples


# ── Thumbnail extraction ──────────────────────────────────────────────────────

def save_thumbnails(
    samples: List[Dict],
    video_path: Path,
    thumb_dir: Path = THUMB_DIR,
) -> List[Path]:
    """Resize and save JPEG thumbnails; return list of paths."""
    video_stem = video_path.stem
    vid_thumb_dir = thumb_dir / video_stem
    vid_thumb_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for s in samples:
        name = f"{video_stem}_f{s['frame_idx']:07d}.jpg"
        out  = vid_thumb_dir / name
        small = cv2.resize(
            s["frame"],
            (CFG.sampling.thumbnail_width, CFG.sampling.thumbnail_height),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        paths.append(out)

    return paths


# ── CLIP embedding ────────────────────────────────────────────────────────────

def embed_frames(
    samples: List[Dict],
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
) -> np.ndarray:
    """
    Batch-encode frames through CLIP vision encoder.
    Returns float32 numpy array of shape (N, D).
    """
    batch_size = CFG.model.batch_size
    all_embeds = []

    for start in tqdm(
        range(0, len(samples), batch_size),
        desc="Embedding frames",
        unit="batch",
    ):
        batch_frames = samples[start : start + batch_size]
        images = []
        for s in batch_frames:
            pil = Image.fromarray(cv2.cvtColor(s["frame"], cv2.COLOR_BGR2RGB))
            images.append(preprocess(pil))

        pixel_values = torch.stack(images).to(device)
        if CFG.model.use_fp16 and device.type == "cuda":
            pixel_values = pixel_values.half()

        with torch.no_grad():
            feats = model.encode_image(pixel_values)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalise

        all_embeds.append(feats.cpu().float().numpy())

    return np.concatenate(all_embeds, axis=0)  # (N, D)


# ── Temporal context blending ─────────────────────────────────────────────────

def apply_temporal_blending(embeddings: np.ndarray) -> np.ndarray:
    """
    Replace each embedding with a weighted average of itself and ±window neighbours.
    Centre weight: (1 - alpha), each neighbour: alpha / (2*window).
    After blending, re-normalise to unit sphere.
    """
    if CFG.search.temporal_window == 0:
        return embeddings

    N, D  = embeddings.shape
    W     = CFG.search.temporal_window
    alpha = CFG.search.temporal_alpha
    blended = embeddings.copy() * (1.0 - alpha)

    for offset in range(-W, W + 1):
        if offset == 0:
            continue
        weight   = alpha / (2 * W)
        src_idx  = np.clip(np.arange(N) + offset, 0, N - 1)
        blended += embeddings[src_idx] * weight

    # Re-normalise
    norms = np.linalg.norm(blended, axis=1, keepdims=True)
    blended /= np.maximum(norms, 1e-8)
    return blended


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index appropriate for the dataset size.
    ≤ 200k frames → IndexFlatIP (exact, cosine via normalised vectors)
     > 200k frames → IndexIVFFlat (ANN, faster but approximate)
    """
    D = embeddings.shape[1]
    N = embeddings.shape[0]

    if CFG.index.index_type == "flat" or N <= 200_000:
        log.info("Building IndexFlatIP (exact cosine, N=%d)", N)
        index = faiss.IndexFlatIP(D)
    else:
        log.info("Building IndexIVFFlat (ANN, nlist=%d, N=%d)", CFG.index.ivf_nlist, N)
        quantiser = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantiser, D, CFG.index.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = CFG.index.ivf_nprobe

    index.add(embeddings)
    log.info("FAISS index built: %d vectors, %d dims", index.ntotal, D)
    return index


def save_index(
    index: faiss.Index,
    metadata: List[Dict],
    index_name: str,
    index_dir: Path = INDEX_DIR,
) -> Path:
    idx_path  = index_dir / f"{index_name}.faiss"
    meta_path = index_dir / f"{index_name}.meta.json"

    faiss.write_index(index, str(idx_path))
    # Strip the raw frame ndarray from metadata before serialising
    clean_meta = [
        {k: v for k, v in m.items() if k != "frame"}
        for m in metadata
    ]
    meta_path.write_text(json.dumps(clean_meta, indent=2))

    log.info("Index saved → %s", idx_path)
    log.info("Metadata saved → %s", meta_path)
    return idx_path


def load_index(index_name: str, index_dir: Path = INDEX_DIR):
    idx_path  = index_dir / f"{index_name}.faiss"
    meta_path = index_dir / f"{index_name}.meta.json"

    if not idx_path.exists():
        raise FileNotFoundError(f"No index found at {idx_path}")

    index    = faiss.read_index(str(idx_path))
    metadata = json.loads(meta_path.read_text())

    if CFG.index.index_type == "ivf" and hasattr(index, "nprobe"):
        index.nprobe = CFG.index.ivf_nprobe

    log.info("Loaded index '%s' — %d vectors", index_name, index.ntotal)
    return index, metadata


# ── High-level entry point ────────────────────────────────────────────────────

def index_video(
    video_path: Path,
    index_name: Optional[str] = None,
    model=None,
    preprocess=None,
    device=None,
) -> Tuple[faiss.Index, List[Dict], Path]:
    """
    Full pipeline for one video:
      sample → thumbnail → embed → temporal blend → faiss → persist

    Returns (faiss_index, metadata, index_path)
    """
    video_path = Path(video_path)
    index_name = index_name or video_path.stem

    if model is None:
        model, preprocess, device = load_clip_model()

    t0 = time.perf_counter()

    # 1. Frame sampling
    with timer("frame_sampling") as bm_sample:
        samples = sample_frames(video_path)

    n_frames   = len(samples)
    duration   = samples[-1]["pts_sec"] if samples else 0.0
    sample_fps = n_frames / max(duration, 1e-6)
    log.info("Sampling rate: %.2f frames/sec of video", sample_fps)

    # 2. Thumbnails
    with timer("thumbnail_save"):
        thumb_paths = save_thumbnails(samples, video_path)
    for s, tp in zip(samples, thumb_paths):
        s["frame_path"] = str(tp)
        s["video_file"] = str(video_path)

    # 3. Embed
    with timer("embedding") as bm_embed:
        embeddings = embed_frames(samples, model, preprocess, device)

    embed_fps = n_frames / max(bm_embed.elapsed_ms / 1000, 1e-6)
    log.info("Embedding throughput: %.1f frames/sec", embed_fps)

    # 4. Temporal blending
    with timer("temporal_blend"):
        embeddings = apply_temporal_blending(embeddings)

    # 5. Build FAISS index
    with timer("faiss_build"):
        index = build_faiss_index(embeddings)

    # 6. Persist
    with timer("persist"):
        idx_path = save_index(index, samples, index_name)

    total_sec = time.perf_counter() - t0
    log.info(
        "Indexing complete in %.1f s | frames=%d | throughput=%.1f fr/s",
        total_sec, n_frames, n_frames / total_sec,
    )

    return index, samples, idx_path


def index_directory(
    dir_path: Path,
    index_name: str = "archive",
) -> Tuple[faiss.Index, List[Dict], Path]:
    """
    Index all video files in a directory into a single merged FAISS index.
    Supported extensions: mp4, avi, mov, mkv, webm, m4v
    """
    EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    videos = [p for p in Path(dir_path).iterdir() if p.suffix.lower() in EXTS]
    if not videos:
        raise ValueError(f"No supported video files found in {dir_path}")

    log.info("Found %d video(s) to index.", len(videos))
    model, preprocess, device = load_clip_model()

    all_embeddings = []
    all_metadata   = []

    for vp in sorted(videos):
        log.info("── Indexing %s ──", vp.name)
        _, samples, _ = index_video(vp, model=model, preprocess=preprocess, device=device)
        # Reload embeddings from index (avoid keeping all in RAM at once)
        # Actually we need to keep them to merge — load from disk
        tmp_name  = vp.stem
        tmp_index, tmp_meta = load_index(tmp_name)
        D         = tmp_index.d
        vecs      = np.zeros((tmp_index.ntotal, D), dtype=np.float32)
        tmp_index.reconstruct_n(0, tmp_index.ntotal, vecs)
        all_embeddings.append(vecs)
        all_metadata.extend(tmp_meta)

    merged_embeddings = np.concatenate(all_embeddings, axis=0)
    merged_index      = build_faiss_index(merged_embeddings)
    idx_path          = save_index(merged_index, all_metadata, index_name)

    return merged_index, all_metadata, idx_path
