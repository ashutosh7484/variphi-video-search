"""
Evaluation Protocol
===================
Computes standard IR metrics over a ground-truth file.

Ground-truth format (JSON):
[
  {
    "query": "person near the entrance",
    "timestamps_sec": [12.0, 78.5, 134.2],
    "notes": "optional human annotation"
  },
  ...
]

Metrics reported
----------------
* Precision@K   for K in {1, 3, 5, 10}
* Recall@K      for K in {1, 3, 5, 10}
* MRR           (Mean Reciprocal Rank)
* Hit Rate@K

Usage
-----
python evaluate.py --index my_index --ground-truth eval/ground_truth.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.utils import get_logger

log = get_logger("evaluate")

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"


def is_hit(result_pts: float, gt_timestamps: List[float], tolerance: float) -> bool:
    return any(abs(result_pts - gt) <= tolerance for gt in gt_timestamps)


def evaluate(
    index_name: str,
    ground_truth_path: Path,
    tolerance_sec: float = 5.0,
    ks: List[int] = [1, 3, 5, 10],
    verbose: bool = True,
) -> Dict:
    from src.searcher import load_searcher

    with open(ground_truth_path) as f:
        gt_data: List[Dict] = json.load(f)

    searcher = load_searcher(index_name)
    max_k    = max(ks)

    all_p_at_k   = {k: [] for k in ks}
    all_r_at_k   = {k: [] for k in ks}
    rr_scores    = []
    hit_at_k     = {k: [] for k in ks}
    query_times  = []

    for entry in gt_data:
        query      = entry["query"]
        gt_pts     = entry.get("timestamps_sec", [])
        n_relevant = len(gt_pts)

        if verbose:
            print(f"  {CYAN}Q:{RESET} {query[:60]:60s}", end="", flush=True)

        t0      = time.perf_counter()
        results = searcher.search(query, top_k=max_k)
        qtime   = (time.perf_counter() - t0) * 1000
        query_times.append(qtime)

        rr = 0.0
        for rank_i, r in enumerate(results, 1):
            if is_hit(r["pts_sec"], gt_pts, tolerance_sec):
                if rr == 0.0:
                    rr = 1.0 / rank_i
                break
        rr_scores.append(rr)

        for k in ks:
            top_k_res = results[:k]
            hits = sum(1 for r in top_k_res if is_hit(r["pts_sec"], gt_pts, tolerance_sec))
            p = hits / k
            r_k = hits / max(n_relevant, 1)
            all_p_at_k[k].append(p)
            all_r_at_k[k].append(r_k)
            hit_at_k[k].append(1 if hits > 0 else 0)

        if verbose:
            print(f"  RR={rr:.2f}  {qtime:.0f}ms")

    results_dict = {}
    if verbose:
        print(f"\n  {BOLD}{'Metric':22s}  {'Value':>10}{RESET}")
        print("  " + "─" * 36)

    for k in ks:
        p = np.mean(all_p_at_k[k])
        r = np.mean(all_r_at_k[k])
        h = np.mean(hit_at_k[k])
        results_dict[f"Precision@{k}"] = round(float(p), 4)
        results_dict[f"Recall@{k}"]    = round(float(r), 4)
        results_dict[f"HitRate@{k}"]   = round(float(h), 4)
        if verbose:
            print(f"  {'Precision@'+str(k):22s}  {p:>10.4f}")
            print(f"  {'Recall@'+str(k):22s}  {r:>10.4f}")
            print(f"  {'HitRate@'+str(k):22s}  {h:>10.4f}")

    mrr = float(np.mean(rr_scores))
    results_dict["MRR"] = round(mrr, 4)
    results_dict["mean_query_latency_ms"] = round(float(np.mean(query_times)), 2)

    if verbose:
        print(f"  {'MRR':22s}  {mrr:>10.4f}")
        print(f"  {'Mean latency (ms)':22s}  {results_dict['mean_query_latency_ms']:>10.2f}")
        print()

    return results_dict


def generate_sample_ground_truth(output_path: Path):
    """Write a sample ground-truth JSON for reference."""
    sample = [
        {
            "query": "person walking through the corridor",
            "timestamps_sec": [12.0, 145.5],
            "notes": "Two separate instances"
        },
        {
            "query": "empty room with no people",
            "timestamps_sec": [300.0, 420.0, 600.0],
            "notes": ""
        },
        {
            "query": "two people talking near a table",
            "timestamps_sec": [78.0],
            "notes": ""
        },
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sample, indent=2))
    print(f"Sample ground-truth written → {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video Search Evaluation")
    parser.add_argument("--index",        required=True, help="Index name")
    parser.add_argument("--ground-truth", required=True, help="Path to GT JSON file")
    parser.add_argument("--tolerance",    type=float, default=5.0, help="Seconds tolerance")
    parser.add_argument("--gen-sample",   action="store_true", help="Generate a sample GT file")
    parser.add_argument("--output",       default="eval/ground_truth.json")
    args = parser.parse_args()

    if args.gen_sample:
        generate_sample_ground_truth(Path(args.output))
    else:
        metrics = evaluate(
            index_name        = args.index,
            ground_truth_path = Path(args.ground_truth),
            tolerance_sec     = args.tolerance,
        )
        out_path = Path("results") / "evaluation_metrics.json"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics saved → {out_path}")
