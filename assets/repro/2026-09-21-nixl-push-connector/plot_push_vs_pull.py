#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regenerate the push-vs-pull TTFT/TPOT figures from benchmark results.

Reads the JSON files written by ``vllm bench serve`` through
``disagg_kv_push_pull_benchmark.sh`` (named
``<mode>_qps<q>_in<in>_out<out>_iter<i>.json``), averages the iterations at
each point, and draws one panel per QPS.

Usage:
    python plot_push_vs_pull.py --results-dir ./results --metric ttft
    python plot_push_vs_pull.py --results-dir ./results --metric tpot
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

NAME_RE = re.compile(
    r"^(?P<mode>pull|push)_qps(?P<qps>[\d.]+)_in(?P<inlen>\d+)"
    r"_out(?P<outlen>\d+)_iter(?P<iter>\d+)\.json$"
)

MODE_STYLE = {
    "pull": {"color": "#d62728", "marker": "o", "label": "Pull mode"},
    "push": {"color": "#2ca02c", "marker": "o", "label": "Push mode"},
}


def collect(results_dir: Path, metric_key: str) -> dict:
    """mode -> qps -> input_len -> mean metric across iterations."""
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(results_dir.glob("*.json")):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        data = json.loads(path.read_text())
        value = data.get(metric_key)
        if value is None:
            continue
        acc[m["mode"]][float(m["qps"])][int(m["inlen"])].append(value)

    return {
        mode: {
            qps: {n: sum(v) / len(v) for n, v in sorted(by_len.items())}
            for qps, by_len in sorted(by_qps.items())
        }
        for mode, by_qps in acc.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--metric", choices=("ttft", "tpot"), default="ttft")
    ap.add_argument("--stat", choices=("mean", "median"), default="mean")
    ap.add_argument("--model", default="Qwen3-32B")
    ap.add_argument("--platform", default="AWS p5en instance cluster with AWS EFA")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    metric_key = f"{args.stat}_{args.metric}_ms"
    series = collect(args.results_dir, metric_key)
    if not series:
        raise SystemExit(f"no results with '{metric_key}' found in {args.results_dir}")

    qps_values = sorted({q for by_qps in series.values() for q in by_qps})
    ncols = 2
    nrows = (len(qps_values) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), squeeze=False)

    for idx, qps in enumerate(qps_values):
        ax = axes[idx // ncols][idx % ncols]
        for mode, by_qps in series.items():
            points = by_qps.get(qps)
            if not points:
                continue
            style = MODE_STYLE.get(mode, {})
            ax.plot(
                list(points),
                list(points.values()),
                linewidth=2.5,
                markersize=8,
                **style,
            )
        ax.set_title(f"QPS = {qps:g}", fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({n for p in series.values() for n in p.get(qps, {})}))
        ax.set_xticklabels(
            [
                f"{n // 1024}k"
                for n in sorted({n for p in series.values() for n in p.get(qps, {})})
            ]
        )
        ax.set_xlabel("Input prompt length (tokens)")
        ax.set_ylabel(f"{args.metric.upper()} (ms)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    for idx in range(len(qps_values), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        f"Push vs Pull KV Transfer: {args.metric.upper()} vs Input Length across QPS",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.945,
        f"Model: {args.model}   |   Platform: {args.platform}",
        ha="center",
        style="italic",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = args.output or Path(f"{args.metric}-push-vs-pull.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
