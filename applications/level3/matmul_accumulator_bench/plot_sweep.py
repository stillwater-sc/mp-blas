#!/usr/bin/env python3
"""Plot run_sweep.sh CSVs as the SVG charts in docs/level3-accumulator-benchmark.md.

Usage:
    plot_sweep.py --n128 N128.csv --n1024 DOUBLE.csv [FLOAT.csv ...] --out DIR [--machine NAME]

Writes four charts into DIR:
    level3-acc-throughput.svg    single P-core throughput per type and config (n = 128)
    level3-acc-quire-scaling.svg quire speedup vs threads per placement (n = 128)
    level3-acc-native-scaling.svg  plain / fma speedup vs threads, unpinned vs P-cores (n = 1024)
    level3-acc-size.svg          single-thread throughput at n = 128 vs n = 1024
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONFIGS = ["default", "1 plain", "2 fma", "3 quire"]
COLORS = {"default": "#7f7f7f", "1 plain": "#1f77b4", "2 fma": "#ff7f0e", "3 quire": "#2ca02c"}
TYPES = ["double", "float", "posit32"]
TYPE_LABEL = {"double": "double", "float": "float", "posit32": "posit<32,2>"}
PLACEMENT_LABEL = {
    "all": "unpinned",
    "pcore": "P-cores, 1 thread/core",
    "pcore2": "P-cores, both SMT threads",
    "ecore": "E-cores",
}
PLACEMENT_STYLE = {"all": "-o", "pcore": "-s", "pcore2": "--^", "ecore": ":v"}


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["min_ms"] in ("", "skipped"):
                    continue
                r["threads"] = int(r["threads"])
                r["n"] = int(r["n"])
                r["min_ms"] = float(r["min_ms"])
                r["mmac_per_s"] = float(r["mmac_per_s"])
                rows.append(r)
    return rows


def series(rows, **match):
    """{threads: row} for the rows matching every key=value."""
    out = {}
    for r in rows:
        if all(r[k] == v for k, v in match.items()):
            out[r["threads"]] = r
    return dict(sorted(out.items()))


def throughput_bars(rows, path, machine):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.2
    for i, cfg in enumerate(CONFIGS):
        xs, ys = [], []
        for t, ty in enumerate(TYPES):
            s = series(rows, placement="pcore", type=ty, config=cfg)
            if 1 in s:
                xs.append(t + (i - 1.5) * width)
                ys.append(s[1]["mmac_per_s"])
        bars = ax.bar(xs, ys, width, label=cfg, color=COLORS[cfg])
        for b, y in zip(bars, ys):
            ax.annotate(f"{y:.3g}", (b.get_x() + b.get_width() / 2, y), ha="center", va="bottom",
                        fontsize=7, xytext=(0, 2), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xticks(range(len(TYPES)), [TYPE_LABEL[t] for t in TYPES])
    ax.set_ylabel("million multiply-adds / s (log)")
    ax.set_title(f"C = A B, n = 128, one P-core thread\n{machine}")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.grid(axis="y", which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def speedup_axes(ax, max_threads):
    ax.plot([1, max_threads], [1, max_threads], color="#cccccc", lw=1, zorder=0, label="linear")
    ax.set_xticks([1, 2, 4, 8, 12, 16, 20])
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup over 1 thread")
    ax.grid(alpha=0.3)


def quire_scaling(rows, path, machine):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, ty in zip(axes, TYPES):
        speedup_axes(ax, 20)
        for pl in ["all", "pcore", "pcore2", "ecore"]:
            s = series(rows, placement=pl, type=ty, config="3 quire")
            if not s:
                continue
            base = s[min(s)]["min_ms"]
            ax.plot(list(s), [base / r["min_ms"] for r in s.values()], PLACEMENT_STYLE[pl],
                    label=PLACEMENT_LABEL[pl], ms=4)
        ax.set_title(f"quire<{TYPE_LABEL[ty]}>")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Quire GEMM thread scaling, n = 128 -- {machine}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def native_scaling(rows, path, machine):
    types = [t for t in TYPES if any(r["type"] == t for r in rows)]
    fig, axes = plt.subplots(1, len(types), figsize=(6 * len(types), 4), sharey=True, squeeze=False)
    for ax, ty in zip(axes[0], types):
        speedup_axes(ax, 20)
        for cfg in ["1 plain", "2 fma"]:
            for pl in ["all", "pcore2"]:
                s = series(rows, placement=pl, type=ty, config=cfg)
                if not s:
                    continue
                base = s[min(s)]["min_ms"]
                ax.plot(list(s), [base / r["min_ms"] for r in s.values()], PLACEMENT_STYLE[pl],
                        color=COLORS[cfg], label=f"{cfg}, {PLACEMENT_LABEL[pl]}", ms=4)
        ax.set_title(TYPE_LABEL[ty])
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Accumulator GEMM thread scaling, n = 1024 -- {machine}\n"
                 "unpinned stalls past 8 threads when E-cores take equal row chunks", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def size_bars(rows128, rows1024, path, machine):
    types = [t for t in TYPES if any(r["type"] == t for r in rows1024)]
    cfgs = ["default", "1 plain", "2 fma"]
    fig, axes = plt.subplots(1, len(types), figsize=(5 * len(types), 4), sharey=True, squeeze=False)
    for ax, ty in zip(axes[0], types):
        for i, (rows, n) in enumerate([(rows128, 128), (rows1024, 1024)]):
            ys = []
            for cfg in cfgs:
                s = series(rows, placement="pcore", type=ty, config=cfg)
                ys.append(s[1]["mmac_per_s"] if 1 in s else 0)
            xs = [c + (i - 0.5) * 0.38 for c in range(len(cfgs))]
            bars = ax.bar(xs, ys, 0.38, label=f"n = {n}", color=["#9ecae1", "#08519c"][i])
            for b, y in zip(bars, ys):
                ax.annotate(f"{y:.0f}", (b.get_x() + b.get_width() / 2, y), ha="center", va="bottom",
                            fontsize=7, xytext=(0, 2), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xticks(range(len(cfgs)), cfgs)
        ax.set_title(TYPE_LABEL[ty])
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("million multiply-adds / s (log)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"One P-core thread: the accumulator kernel falls off at n = 1024 -- {machine}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n128", nargs="+", required=True)
    ap.add_argument("--n1024", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--machine", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9})
    r128, r1024 = load(a.n128), load(a.n1024)
    throughput_bars(r128, out / "level3-acc-throughput.svg", a.machine)
    quire_scaling(r128, out / "level3-acc-quire-scaling.svg", a.machine)
    native_scaling(r1024, out / "level3-acc-native-scaling.svg", a.machine)
    size_bars(r128, r1024, out / "level3-acc-size.svg", a.machine)
    for f in sorted(out.glob("level3-acc-*.svg")):
        print(f)


if __name__ == "__main__":
    main()
