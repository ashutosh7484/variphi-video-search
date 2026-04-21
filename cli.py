#!/usr/bin/env python3
"""
Command-Line Interface
======================

Usage examples
--------------
# Index a single video
python cli.py index --input ./videos/cam01.mp4 --name cam01

# Index an entire directory
python cli.py index --input ./videos/ --name archive

# Search
python cli.py search --index cam01 --query "person near the door" --top-k 5

# Search with time filter (explicit)
python cli.py search --index cam01 --query "red vehicle" --from 18:00 --to 20:00

# Benchmark indexing
python cli.py benchmark --index cam01

# Start the API server
python cli.py serve --port 8000
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG
from src.utils import get_logger, hms_to_seconds, save_results_json, save_results_csv, render_html_results

log = get_logger("cli")


# ── ANSI colours ──────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
GREY    = "\033[90m"


def _bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _print_result(r: dict, i: int):
    conf = r.get("confidence", 0.0)
    colour = GREEN if conf >= 60 else YELLOW if conf >= 35 else RED
    print(f"\n  {BOLD}#{i}{RESET}  {CYAN}{r['timestamp_hms']}{RESET}  "
          f"{colour}{_bar(conf)}{RESET}  {conf:.1f}%")
    print(f"      Video : {GREY}{Path(r.get('video_file','?')).name}{RESET}")
    print(f"      Frame : {GREY}{r.get('frame_path','?')}{RESET}")
    print(f"      Score : {r.get('score',0):.4f}")


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_index(args: argparse.Namespace):
    from src.indexer import index_video, index_directory

    p = Path(args.input)
    if not p.exists():
        print(f"{RED}Error: path not found → {p}{RESET}")
        sys.exit(1)

    # Override sampling if specified
    if args.strategy:
        CFG.sampling.strategy = args.strategy

    index_name = args.name or p.stem
    t0 = time.perf_counter()
    print(f"\n{BOLD}Indexing{RESET} {CYAN}{p}{RESET} → index={BOLD}{index_name}{RESET}\n")

    if p.is_dir():
        index, meta, idx_path = index_directory(p, index_name)
    else:
        index, meta, idx_path = index_video(p, index_name)

    elapsed = time.perf_counter() - t0
    n       = index.ntotal
    fps_tput = n / max(elapsed, 1e-6)

    print(f"\n{GREEN}✓ Done!{RESET}")
    print(f"  Frames indexed : {n}")
    print(f"  Index path     : {idx_path}")
    print(f"  Elapsed        : {elapsed:.1f} s")
    print(f"  Throughput     : {fps_tput:.1f} frames/sec\n")


def cmd_search(args: argparse.Namespace):
    from src.searcher import load_searcher

    if not args.index:
        print(f"{RED}Error: --index is required for search.{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}Loading index{RESET} '{CYAN}{args.index}{RESET}' …")
    searcher = load_searcher(args.index)

    time_start: Optional[float] = None
    time_end:   Optional[float] = None
    if args.time_from:
        time_start = hms_to_seconds(args.time_from)
    if args.time_to:
        time_end = hms_to_seconds(args.time_to)

    print(f"\n{BOLD}Query:{RESET} \"{CYAN}{args.query}{RESET}\"")
    if time_start or time_end:
        print(f"  Time filter: [{args.time_from or '00:00:00'} → {args.time_to or 'end'}]")

    t0      = time.perf_counter()
    results = searcher.search(
        query      = args.query,
        top_k      = args.top_k,
        time_start = time_start,
        time_end   = time_end,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  {GREEN}{len(results)} result(s){RESET}  |  latency {YELLOW}{latency_ms:.1f} ms{RESET}")
    print("  " + "─" * 60)
    for r in results:
        _print_result(r, r["rank"])

    # Persist
    if results:
        jp = save_results_json(results, args.query)
        cp = save_results_csv(results, args.query)
        print(f"\n  Saved → {jp}")
        print(f"  Saved → {cp}")

    if args.html:
        hp = render_html_results(results, args.query)
        print(f"  HTML  → {hp}")

    print()


def cmd_benchmark(args: argparse.Namespace):
    """Quick benchmark: re-run a set of test queries and report latency stats."""
    from src.searcher import load_searcher
    import statistics

    if not args.index:
        print(f"{RED}Error: --index required.{RESET}")
        sys.exit(1)

    test_queries = [
        "person walking",
        "red vehicle",
        "empty corridor",
        "two people talking",
        "bright outdoor scene",
    ]

    print(f"\n{BOLD}Benchmark — index '{args.index}'{RESET}\n")
    searcher = load_searcher(args.index)
    latencies = []

    for q in test_queries:
        t0 = time.perf_counter()
        r  = searcher.search(q, top_k=10)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        print(f"  {GREY}{q:40s}{RESET}  {ms:6.1f} ms  →  {len(r)} results")

    print(f"\n  {BOLD}Mean latency:{RESET}   {statistics.mean(latencies):.1f} ms")
    print(f"  {BOLD}Median latency:{RESET} {statistics.median(latencies):.1f} ms")
    print(f"  {BOLD}p95 latency:{RESET}    {sorted(latencies)[int(0.95*len(latencies))]:.1f} ms\n")


def cmd_serve(args: argparse.Namespace):
    import uvicorn
    print(f"\n{BOLD}Starting API server{RESET} on http://0.0.0.0:{args.port}\n")
    uvicorn.run("api:app", host="0.0.0.0", port=args.port, reload=False)


def cmd_evaluate(args: argparse.Namespace):
    """Run evaluation protocol: test queries with ground-truth timestamps."""
    from src.searcher import load_searcher

    if not args.index or not args.ground_truth:
        print(f"{RED}Error: --index and --ground-truth required.{RESET}")
        sys.exit(1)

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"{RED}Ground-truth file not found: {gt_path}{RESET}")
        sys.exit(1)

    # Ground-truth format: [{"query": str, "timestamps_sec": [float, ...]}]
    with open(gt_path) as f:
        ground_truth = json.load(f)

    searcher = load_searcher(args.index)
    tolerance_sec = args.tolerance or 5.0
    ks = [1, 3, 5, 10]

    print(f"\n{BOLD}Evaluation — index '{args.index}'{RESET}")
    print(f"  Ground-truth entries : {len(ground_truth)}")
    print(f"  Timestamp tolerance  : ±{tolerance_sec} s\n")

    p_at_k  = {k: [] for k in ks}
    rr_list = []

    for entry in ground_truth:
        query   = entry["query"]
        gt_pts  = set(entry.get("timestamps_sec", []))
        results = searcher.search(query, top_k=max(ks))

        rr = 0.0
        for k in ks:
            top_k_res = results[:k]
            hits = sum(
                1 for r in top_k_res
                if any(abs(r["pts_sec"] - gts) <= tolerance_sec for gts in gt_pts)
            )
            p_at_k[k].append(hits / k)

            if rr == 0.0:
                for rank_i, r in enumerate(top_k_res, 1):
                    if any(abs(r["pts_sec"] - gts) <= tolerance_sec for gts in gt_pts):
                        rr = 1.0 / rank_i
                        break
        rr_list.append(rr)

    print(f"  {'Metric':20s}  {'Value':>8}")
    print("  " + "─" * 32)
    for k in ks:
        avg = sum(p_at_k[k]) / max(len(p_at_k[k]), 1)
        print(f"  {'Precision@'+str(k):20s}  {avg:>8.4f}")
    mrr = sum(rr_list) / max(len(rr_list), 1)
    print(f"  {'MRR':20s}  {mrr:>8.4f}\n")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="videosearch",
        description="Variphi Intelligent Video Search Engine",
    )
    sub = parser.add_subparsers(dest="command")

    # index
    p_idx = sub.add_parser("index", help="Index a video file or directory")
    p_idx.add_argument("--input",    required=True, help="Video file or directory path")
    p_idx.add_argument("--name",     help="Index name (default: filename)")
    p_idx.add_argument("--strategy", choices=["uniform", "scene", "adaptive"],
                       help="Frame sampling strategy")

    # search
    p_srch = sub.add_parser("search", help="Search with a natural language query")
    p_srch.add_argument("--index",     required=True, help="Index name")
    p_srch.add_argument("--query",     required=True, help="Natural language query")
    p_srch.add_argument("--top-k",     type=int, default=10)
    p_srch.add_argument("--from",      dest="time_from", help="Start time e.g. 18:00")
    p_srch.add_argument("--to",        dest="time_to",   help="End time e.g. 20:00")
    p_srch.add_argument("--html",      action="store_true", help="Render HTML results")

    # benchmark
    p_bm = sub.add_parser("benchmark", help="Benchmark query latency")
    p_bm.add_argument("--index", required=True)

    # serve
    p_srv = sub.add_parser("serve", help="Start the FastAPI server")
    p_srv.add_argument("--port", type=int, default=8000)

    # evaluate
    p_ev = sub.add_parser("evaluate", help="Run evaluation protocol")
    p_ev.add_argument("--index",        required=True)
    p_ev.add_argument("--ground-truth", required=True, help="JSON ground-truth file")
    p_ev.add_argument("--tolerance",    type=float, default=5.0,
                      help="Timestamp tolerance in seconds")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
