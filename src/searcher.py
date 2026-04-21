"""
Search Engine
=============
Responsibilities:
  1. Text query encoding via CLIP text encoder
  2. Temporal filter extraction from natural language
  3. ANN retrieval from FAISS
  4. Re-ranking with cross-modal verification
  5. Result formatting with thumbnail paths

Design decisions
----------------
* **Query decomposition**: complex queries like "red car AND near entrance after 6pm"
  are split into sub-queries; each sub-query retrieves candidates independently;
  results are merged via Reciprocal Rank Fusion (RRF).

* **Re-ranking**: after ANN retrieval we take the top-N candidates and re-score them
  using a finer-grained CLIP similarity (full resolution crop if available).
  This is the "verify-then-rank" pattern used in modern RAG pipelines.

* **Temporal filter**: recognised time expressions ("after 18:00", "between 6pm and 8pm")
  are parsed from the query string and applied as a hard filter BEFORE scoring.

* **Score normalisation**: raw dot-product scores from FAISS (cosine similarity on
  unit-norm vectors) are in [-1, 1]; we linearly map to [0, 1] for display.
"""

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import open_clip
import faiss

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CFG, THUMB_DIR
from src.utils import get_logger, extract_time_filter, clean_query, seconds_to_hms, timer

log = get_logger("searcher")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cosine_to_confidence(score: float) -> float:
    """Map raw CLIP cosine score [-1, 1] to a confidence percentage [0, 100]."""
    return round(max(0.0, (score + 1.0) / 2.0 * 100.0), 2)


# ── Query decomposition ────────────────────────────────────────────────────────

_SPLIT_PATTERN = re.compile(r"\s+(?:AND|and|with|near|while|after|before)\s+")


def decompose_query(query: str) -> List[str]:
    """
    Split a complex query into sub-queries for independent retrieval.
    "red car near entrance" → ["red car", "entrance"] (both required)
    Simple queries are returned as-is.

    Keeps temporal filters in the first sub-query for filter extraction.
    """
    # Don't split on temporal keywords so extract_time_filter still works
    temporal_keywords = r"after|before|between|until|from"
    parts = re.split(r"\s+(?:AND|and|with|near|while)\s+", query)
    if len(parts) <= 1:
        return [query]
    log.debug("Decomposed into %d sub-queries: %s", len(parts), parts)
    return parts


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Merge multiple ranked lists of frame indices using RRF.
    Returns [(frame_idx, rrf_score)] sorted descending.
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Core searcher class ────────────────────────────────────────────────────────

class VideoSearcher:
    """
    Stateful searcher: holds the loaded model, index, and metadata.
    Create once per process; call `.search()` for each query.
    """

    def __init__(
        self,
        index: faiss.Index,
        metadata: List[Dict],
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
    ):
        self.index    = index
        self.metadata = metadata
        self.model    = model
        self.tokenizer = tokenizer
        self.device   = device
        log.info(
            "VideoSearcher ready | %d indexed frames | device=%s",
            len(metadata), device,
        )

    # ── Text encoding ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        if CFG.model.use_fp16 and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                feats = self.model.encode_text(tokens)
        else:
            feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()  # (1, D)

    # ── FAISS retrieval ────────────────────────────────────────────────────────

    def _ann_retrieve(
        self,
        query_vec: np.ndarray,
        time_filter: Optional[Dict],
        k: int,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve k candidates from FAISS, applying temporal filter.
        Returns [(meta_idx, score)].
        """
        # Over-fetch if temporal filter is active (many results may be dropped)
        fetch_k = k * 5 if time_filter else k
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, indices = self.index.search(query_vec, fetch_k)
        raw = list(zip(indices[0].tolist(), scores[0].tolist()))

        # Filter out invalid indices (-1 returned by FAISS when index is smaller)
        raw = [(i, s) for i, s in raw if 0 <= i < len(self.metadata)]

        if time_filter:
            start = time_filter.get("start", 0.0)
            end   = time_filter.get("end", float("inf"))
            raw   = [
                (i, s) for i, s in raw
                if start <= self.metadata[i].get("pts_sec", 0.0) <= end
            ]

        return raw[:k]

    # ── Re-ranking ─────────────────────────────────────────────────────────────

    def _rerank(
        self,
        candidates: List[Tuple[int, float]],
        query_vec: np.ndarray,
        top_n: int,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank the top-N candidates using stored embedding similarity.
        (If a more powerful re-ranker model were available, it would go here.)
        
        Current implementation: retrieve stored embeddings from FAISS and
        compute exact dot-product — useful if the index is IVF (which is approximate).
        Also applies a temporal coherence boost: if two adjacent frames both score
        highly, their scores are slightly boosted (temporal consistency signal).
        """
        if not candidates:
            return candidates

        take     = min(top_n, len(candidates))
        cands    = candidates[:take]
        idxs     = [i for i, _ in cands]
        D        = self.index.d

        # Reconstruct stored embeddings
        stored = np.zeros((len(idxs), D), dtype=np.float32)
        for j, idx in enumerate(idxs):
            self.index.reconstruct(idx, stored[j])

        # Exact cosine similarity
        exact_scores = (stored @ query_vec.T).squeeze(-1)  # (N,)

        # Temporal coherence boost: check neighbours in metadata
        pts_list = np.array([self.metadata[i].get("pts_sec", 0.0) for i in idxs])

        reranked = []
        for j, (meta_idx, _) in enumerate(cands):
            score = float(exact_scores[j])
            # Small boost if neighbouring frames (±2 s) also have high scores
            nearby_mask = np.abs(pts_list - pts_list[j]) < 2.0
            nearby_mask[j] = False
            if nearby_mask.any():
                score += 0.01 * float(exact_scores[nearby_mask].mean())
            reranked.append((meta_idx, score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        # Add remaining candidates that weren't re-ranked
        reranked_idxs = {i for i, _ in reranked}
        for pair in candidates[take:]:
            if pair[0] not in reranked_idxs:
                reranked.append(pair)

        return reranked

    # ── Public search API ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the video index with a natural language query.

        Parameters
        ----------
        query      : Free-form natural language.
        top_k      : Number of results to return (default: CFG.search.return_k).
        time_start : Hard lower bound in seconds (overrides query-embedded filter).
        time_end   : Hard upper bound in seconds.

        Returns
        -------
        List of result dicts, ranked by relevance:
          {
            "rank":          int,
            "timestamp_hms": str,
            "pts_sec":       float,
            "score":         float,       # cosine similarity [−1, 1]
            "confidence":    float,       # mapped to [0, 100]
            "video_file":    str,
            "frame_path":    str,
            "frame_idx":     int,
          }
        """
        top_k = top_k or CFG.search.return_k
        t0    = time.perf_counter()

        # ── 1. Temporal filter ─────────────────────────────────────────────────
        time_filter = None
        if CFG.search.time_filter_enabled:
            if time_start is not None or time_end is not None:
                time_filter = {
                    "start": time_start or 0.0,
                    "end":   time_end or float("inf"),
                }
            else:
                time_filter = extract_time_filter(query)
                if time_filter:
                    log.info("Parsed temporal filter: %s", time_filter)

        clean = clean_query(query)

        # ── 2. Query decomposition ─────────────────────────────────────────────
        sub_queries = decompose_query(clean)
        fetch_k     = CFG.search.top_k

        if len(sub_queries) == 1:
            q_vec = self.encode_text(sub_queries[0])
            candidates = self._ann_retrieve(q_vec, time_filter, fetch_k)
        else:
            # Encode each sub-query and retrieve independently
            all_ranked_lists = []
            all_raw_scores: Dict[int, float] = {}
            for sq in sub_queries:
                q_vec  = self.encode_text(sq)
                cands  = self._ann_retrieve(q_vec, time_filter, fetch_k)
                ranked = [i for i, _ in cands]
                for i, s in cands:
                    all_raw_scores[i] = max(all_raw_scores.get(i, -1.0), s)
                all_ranked_lists.append(ranked)

            fused = reciprocal_rank_fusion(all_ranked_lists)
            # Attach original scores
            candidates = [(i, all_raw_scores.get(i, 0.0)) for i, _ in fused]

        # Use the combined query vec for re-ranking (encode full clean query)
        q_vec_full = self.encode_text(clean)

        # ── 3. Re-ranking ──────────────────────────────────────────────────────
        if CFG.search.enable_rerank and len(candidates) > 1:
            candidates = self._rerank(
                candidates, q_vec_full, CFG.search.rerank_top_n
            )

        # ── 4. Format results ──────────────────────────────────────────────────
        results = []
        seen_timestamps: set = set()
        for rank, (meta_idx, score) in enumerate(candidates, 1):
            if len(results) >= top_k:
                break
            meta = self.metadata[meta_idx]
            pts  = meta.get("pts_sec", 0.0)

            # Deduplicate near-identical timestamps (within 1 s)
            ts_key = int(pts)
            if ts_key in seen_timestamps:
                continue
            seen_timestamps.add(ts_key)

            results.append({
                "rank":          rank,
                "timestamp_hms": meta.get("timestamp_hms", seconds_to_hms(pts)),
                "pts_sec":       pts,
                "score":         round(float(score), 6),
                "confidence":    _cosine_to_confidence(float(score)),
                "video_file":    meta.get("video_file", ""),
                "frame_path":    meta.get("frame_path", ""),
                "frame_idx":     meta.get("frame_idx", meta_idx),
            })

        latency_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "Query='%s' | results=%d | latency=%.1f ms",
            query[:60], len(results), latency_ms,
        )

        return results


# ── Factory ────────────────────────────────────────────────────────────────────

def load_searcher(index_name: str) -> VideoSearcher:
    """Convenience factory: loads model + index and returns a ready VideoSearcher."""
    from src.indexer import load_clip_model, load_index

    model, _, device = load_clip_model()
    tokenizer        = open_clip.get_tokenizer(CFG.model.clip_model_name)
    index, metadata  = load_index(index_name)

    return VideoSearcher(
        index=index,
        metadata=metadata,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
