"""
plot.py -- read a training metrics CSV and render plots.

The training loop (rank 0 only) appends one row per logged step to a CSV with
columns: step, loss, val_loss, tok_per_s, lr, mem_mb  (val_loss/mem_mb optional).
Then:  python -m distributed.plot metrics.csv out/

Kaggle: just call plot_metrics("metrics.csv") in a cell.
"""

import os
import csv
import sys

import matplotlib
matplotlib.use("Agg")                      # headless: save files, no display needed
import matplotlib.pyplot as plt


def load_metrics(path):
    """CSV -> dict of column-name -> list (floats; blanks become None)."""
    cols = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v) if v not in ("", None) else None)
    return cols


def _series(cols, key):
    """Return (xs, ys) dropping rows where this column is blank."""
    xs, ys = [], []
    for st, v in zip(cols.get("step", []), cols.get(key, [])):
        if v is not None:
            xs.append(st)
            ys.append(v)
    return xs, ys


def plot_metrics(path="metrics.csv", out_dir="plots"):
    """Render loss, throughput, and LR curves from a metrics CSV."""
    cols = load_metrics(path)
    os.makedirs(out_dir, exist_ok=True)

    panels = [
        ("loss",      "training loss",   "loss",         "log"),
        ("val_loss",  "val loss",        "loss",         "log"),
        ("tok_per_s", "throughput",      "tokens / sec", "linear"),
        ("mem_mb",    "GPU memory",      "MB",           "linear"),
        ("lr",        "learning rate",   "lr",           "linear"),
    ]
    present = [p for p in panels if any(v is not None for v in cols.get(p[0], []))]
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, (key, title, ylabel, yscale) in zip(axes[0], present):
        xs, ys = _series(cols, key)
        ax.plot(xs, ys, lw=1.6)
        ax.set(title=title, xlabel="step", ylabel=ylabel)
        ax.set_yscale(yscale)
        ax.grid(alpha=.3)

    fig.tight_layout()
    out = os.path.join(out_dir, "training.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")

    # quick text summary
    _, loss = _series(cols, "loss")
    _, tps = _series(cols, "tok_per_s")
    if loss:
        print(f"loss: {loss[0]:.4f} -> {loss[-1]:.4f}  (min {min(loss):.4f})")
    if tps:
        print(f"throughput: mean {sum(tps) / len(tps):.0f} tok/s  (peak {max(tps):.0f})")
    return out


def compare(paths, labels, out_dir="plots"):
    """Overlay loss curves from several runs (e.g. 1-GPU vs 2-GPU)."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for path, label in zip(paths, labels):
        xs, ys = _series(load_metrics(path), "loss")
        ax.plot(xs, ys, lw=1.6, label=label)
    ax.set(title="loss comparison", xlabel="step", ylabel="loss")
    ax.set_yscale("log")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(out_dir, "compare.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")
    return out


def plot_holdings(path="holdings.csv", out_dir="plots"):
    """Bar chart of per-GPU memory holdings: params / grads / optimizer / ...

    holdings.csv has columns: component, mb  (written by the training script).
    This is the memory DDP replicates on every GPU -- exactly what ZeRO shards.
    """
    comps, mbs = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            comps.append(row["component"])
            mbs.append(float(row["mb"]))
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(comps, mbs, color=["#6ea8fe", "#9d7bff", "#ffb454", "#43d19e"][:len(comps)])
    ax.bar_label(bars, fmt="%.0f MB")
    ax.set(title=f"per-GPU holdings (total {sum(mbs):.0f} MB)", ylabel="MB")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    out = os.path.join(out_dir, "holdings.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "metrics.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "plots"
    plot_metrics(csv_path, out_dir)
