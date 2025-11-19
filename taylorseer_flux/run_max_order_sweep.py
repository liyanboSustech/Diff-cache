#!/usr/bin/env python3
"""
Sweep TaylorSeer max_order values and record cache statistics.

This script runs `run_flux.sh` multiple times while overriding the
`TAYLORSEER_MAX_ORDER` environment variable so that each run uses a different
Taylor expansion order. After every run it copies the emitted cache statistics
to a dedicated folder and stores a summary of the Taylor-specific cache size.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TaylorSeer with multiple max_order values and capture cache stats."
    )
    parser.add_argument(
        "--min-order",
        type=int,
        default=1,
        help="Minimum max_order to test (inclusive). Ignored when --orders is set.",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=6,
        help="Maximum max_order to test (inclusive). Ignored when --orders is set.",
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="*",
        help="Optional explicit list of max_order values to test.",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="run_flux.sh",
        help="Relative path to the driver script to execute for each run.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory (relative to repo root) where cache stats are written.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="results/max_order_sweep_summary.json",
        help="Path (relative to repo root) of the aggregated sweep summary JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose per-order summary already exists.",
    )
    return parser.parse_args()


def ensure_script_path(script_arg: str) -> Path:
    script_path = Path(script_arg)
    if not script_path.is_absolute():
        script_path = (REPO_ROOT / script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Driver script not found: {script_path}")
    return script_path


def extract_rank_from_name(name: str) -> Optional[int]:
    match = re.search(r"rank(\d+)", name)
    return int(match.group(1)) if match else None


def _find_latest(results_dir: Path, patterns: List[str]) -> Optional[Path]:
    """Return the most recently modified file matching one of the patterns."""
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(results_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _variant_from_aggregated(filename: str) -> Optional[str]:
    match = re.match(r"cache_stats_all_ranks_(.+)\.json", filename)
    return match.group(1) if match else None


def load_cache_stats(results_dir: Path, max_order: Optional[int] = None) -> tuple[List[Dict], Dict[str, Any]]:
    """Load cache stats emitted by the latest TaylorSeer run."""
    aggregated_patterns: List[str] = []
    if max_order is not None:
        aggregated_patterns.append(f"cache_stats_all_ranks_maxorder{max_order}_fresh*.json")
    aggregated_patterns.append("cache_stats_all_ranks*.json")

    aggregated_path = _find_latest(results_dir, aggregated_patterns)
    if aggregated_path and aggregated_path.exists():
        with aggregated_path.open("r") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = [data]
        variant_suffix = _variant_from_aggregated(aggregated_path.name)
        per_rank_paths: List[Path] = []
        if variant_suffix:
            per_rank_paths = sorted(results_dir.glob(f"cache_stats_rank*_{variant_suffix}.json"))
        return data, {"aggregated_path": aggregated_path, "per_rank_paths": per_rank_paths}

    per_rank_patterns: List[str] = []
    if max_order is not None:
        per_rank_patterns.append(f"cache_stats_rank*_maxorder{max_order}_fresh*.json")
    per_rank_patterns.append("cache_stats_rank*.json")

    per_rank_paths: List[Path] = []
    for pattern in per_rank_patterns:
        matches = sorted(results_dir.glob(pattern))
        if matches:
            per_rank_paths = matches
            break

    per_rank = []
    for path in per_rank_paths:
        with path.open("r") as fh:
            stats = json.load(fh)
        per_rank.append({"rank": extract_rank_from_name(path.name), "stats": stats})

    if not per_rank:
        raise FileNotFoundError(
            f"No cache stats found under {results_dir}. Make sure run_flux.sh completed successfully."
        )
    return per_rank, {"aggregated_path": None, "per_rank_paths": per_rank_paths}


def summarize_stats(entries: List[Dict]) -> Dict:
    """Compute aggregate statistics from the cache stats payload."""
    per_rank = []
    taylor_values = []
    baseline_values = []
    peak_values = []
    fresh_values = []

    for idx, entry in enumerate(entries):
        stats = entry.get("stats", entry)
        rank = entry.get("rank", stats.get("rank", idx))
        taylor_cache = stats.get("taylor_cache_size_mb", stats.get("taylor_cache_increase"))
        baseline_cache = stats.get("baseline_cache_size")
        peak_cache = stats.get("peak_cache_size")
        fresh_threshold = stats.get("fresh_threshold")

        if taylor_cache is not None:
            taylor_values.append(taylor_cache)
        if baseline_cache is not None:
            baseline_values.append(baseline_cache)
        if peak_cache is not None:
            peak_values.append(peak_cache)
        if fresh_threshold is not None:
            fresh_values.append(fresh_threshold)

        per_rank.append(
            {
                "rank": rank,
                "taylor_cache_size_mb": taylor_cache,
                "baseline_cache_size_mb": baseline_cache,
                "peak_cache_size_mb": peak_cache,
                "current_cache_size_mb": stats.get("current_cache_size"),
                "max_order": stats.get("max_order"),
                "fresh_threshold": fresh_threshold,
            }
        )

    def safe_mean(values: List[float]) -> Optional[float]:
        usable = [v for v in values if v is not None]
        if not usable:
            return None
        return sum(usable) / len(usable)

    return {
        "per_rank": per_rank,
        "mean_taylor_cache_size_mb": safe_mean(taylor_values),
        "max_taylor_cache_size_mb": max(taylor_values) if taylor_values else None,
        "mean_baseline_cache_size_mb": safe_mean(baseline_values),
        "mean_peak_cache_size_mb": safe_mean(peak_values),
        "fresh_threshold": fresh_values[0] if fresh_values else None,
    }


def copy_stats(destination: Path, sources: Dict[str, Any]):
    destination.mkdir(parents=True, exist_ok=True)
    aggregated_path: Optional[Path] = sources.get("aggregated_path")
    if aggregated_path and aggregated_path.exists():
        shutil.copy2(aggregated_path, destination / aggregated_path.name)

    for path in sources.get("per_rank_paths", []):
        shutil.copy2(path, destination / path.name)


def run_single_order(
    max_order: int,
    driver_script: Path,
    results_dir: Path,
    per_order_dir: Path,
    dry_run: bool,
) -> Dict:
    env = os.environ.copy()
    env["TAYLORSEER_MAX_ORDER"] = str(max_order)
    label = f"max_order_{max_order}"
    env["TAYLORSEER_SWEEP_LABEL"] = label
    cmd = ["bash", str(driver_script)]

    print(f"\n=== Running TaylorSeer with max_order={max_order} ===")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("Dry-run mode enabled; skipping execution.")
        return {
            "max_order": max_order,
            "skipped": True,
            "timestamp": time.time(),
        }

    start = time.time()
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    elapsed = time.time() - start

    stats_entries, stats_sources = load_cache_stats(results_dir, max_order=max_order)
    summary = summarize_stats(stats_entries)
    summary.update({"max_order": max_order, "timestamp": time.time(), "elapsed_sec": elapsed})

    copy_stats(per_order_dir, stats_sources)
    with (per_order_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    mean_cache = summary.get("mean_taylor_cache_size_mb")
    baseline_cache = summary.get("mean_baseline_cache_size_mb")
    print(
        f"[max_order={max_order}] Taylor cache ≈ {mean_cache:.2f} MB, "
        f"baseline ≈ {baseline_cache:.2f} MB, elapsed {elapsed:.1f}s"
        if mean_cache is not None and baseline_cache is not None
        else f"[max_order={max_order}] Completed in {elapsed:.1f}s"
    )
    return summary


def main():
    args = parse_args()
    driver_script = ensure_script_path(args.script)
    results_dir = (REPO_ROOT / args.results_dir).resolve()
    summary_path = (REPO_ROOT / args.summary).resolve()
    run_root = results_dir / "max_order_runs"

    if args.orders:
        orders = args.orders
    else:
        if args.max_order < args.min_order:
            raise ValueError("--max-order must be >= --min-order")
        orders = list(range(args.min_order, args.max_order + 1))

    sweep_summary = []
    if args.skip_existing and summary_path.exists():
        with summary_path.open("r") as fh:
            sweep_summary = json.load(fh)

    processed_orders = {entry.get("max_order") for entry in sweep_summary}

    for order in orders:
        per_order_dir = run_root / f"max_order_{order}"
        if args.skip_existing and order in processed_orders and per_order_dir.exists():
            print(f"Skipping max_order={order} (existing summary found).")
            continue
        summary = run_single_order(order, driver_script, results_dir, per_order_dir, args.dry_run)
        sweep_summary = [entry for entry in sweep_summary if entry.get("max_order") != order]
        sweep_summary.append(summary)

    sweep_summary.sort(key=lambda item: item.get("max_order", sys.maxsize))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as fh:
        json.dump(sweep_summary, fh, indent=2)

    if sweep_summary:
        print("\n=== Sweep summary ===")
        for entry in sweep_summary:
            if entry.get("skipped"):
                print(f"max_order={entry['max_order']}: skipped (dry-run)")
                continue
            mean_cache = entry.get("mean_taylor_cache_size_mb")
            baseline = entry.get("mean_baseline_cache_size_mb")
            elapsed = entry.get("elapsed_sec")
            fresh_value = entry.get("fresh_threshold")
            print(
                f"max_order={entry['max_order']} (fresh={fresh_value}) -> Taylor cache≈{mean_cache:.2f} MB, "
                f"baseline≈{baseline:.2f} MB, elapsed {elapsed:.1f}s"
                if mean_cache is not None and baseline is not None
                else f"max_order={entry['max_order']} (fresh={fresh_value}) -> elapsed {elapsed:.1f}s"
            )
    else:
        print("No runs were executed.")


if __name__ == "__main__":
    main()
