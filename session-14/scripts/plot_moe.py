"""
plot_moe.py -- dense run vs MoE run, from the two metrics CSVs.

    python plot_moe.py metrics_zero3_<id>.csv metrics_moe_<id>.csv plots/
"""

import os
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    col = lambda k: [float(r[k]) if r.get(k) not in ("", None) else None for r in rows]
    return {k: col(k) for k in ("step", "loss", "val_loss", "tok_per_s", "peak_mb",
                                "load_min", "load_max")}


def _xy(m, key):
    pts = [(s, v) for s, v in zip(m["step"], m[key]) if v is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def compare(dense_csv, moe_csv, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    dn, me = load(dense_csv), load(moe_csv)
    runs = [(dn, "dense", "#888"), (me, "MoE (4 experts, top-2)", "#d9480f")]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for m, name, c in runs:
        ax[0, 0].plot(*_xy(m, "loss"), color=c, label=name)
        ax[0, 1].plot(*_xy(m, "val_loss"), color=c, marker="o", ms=3, label=name)
        ax[1, 0].plot(*_xy(m, "tok_per_s"), color=c, label=name)
    ax[0, 0].set_title("train loss")
    ax[0, 1].set_title("val loss")
    ax[1, 0].set_title("throughput (tokens/s)")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("step"); a.legend()

    # MoE router health: share of routed slots, as a multiple of a uniform split
    ax[1, 1].plot(*_xy(me, "load_max"), label="busiest expert")
    ax[1, 1].plot(*_xy(me, "load_min"), label="idlest expert")
    ax[1, 1].axhline(1.0, ls=":", color="k", lw=0.8)
    ax[1, 1].set_title("MoE expert load (x uniform share)")
    ax[1, 1].set_xlabel("step"); ax[1, 1].legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "dense_vs_moe.png")
    fig.savefig(path, dpi=120)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    compare(*sys.argv[1:4])
