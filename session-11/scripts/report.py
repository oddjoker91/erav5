"""report.py -- reads results/*.jsonl, writes results/summary.json and every plot."""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import (best_lr, bias_mult, is_edge, lr_table, smooth, steps_until, table_min,
                      warmup_settle, width_report)
from run import RESULTS, loss_at, rows_of

C = ["#2563EB", "#D97706", "#0F766E", "#9333EA"]      # fixed order; every series also labelled
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10, "lines.linewidth": 2, "figure.dpi": 130})


def load(job):
    p = f"{RESULTS}/{job}.jsonl"
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/{name}.png", bbox_inches="tight")
    plt.close(fig)


def stats(by):
    v = np.array(list(by.values()), float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=len(v))


def plot_bias():
    """Needs no GPU: the closed form, plus five real gradients repeated for 20 steps."""
    g = ([.5, .4, .6, .45, .55] * 4)
    m = v = 0.0
    on, off = [], []
    for t, x in enumerate(g, 1):
        m, v = .9 * m + .1 * x, .999 * v + .001 * x * x
        on.append((m / (1 - .9 ** t)) / math.sqrt(v / (1 - .999 ** t)))
        off.append(m / math.sqrt(v))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    t = np.arange(1, 21)
    ax[0].plot(t, on, "-o", ms=4, color=C[0], label="corrected")
    ax[0].plot(t, off, "-o", ms=4, color=C[1], label="no correction")
    ax[0].axhline(1, color="#666", lw=1, ls="--")
    ax[0].set(xlabel="step", ylabel="step size / lr", title="First 20 steps: the gap is at its widest", ylim=(0, 7.2), xticks=[1, 5, 10, 15, 20])
    ax[0].legend(frameon=False)
    x = np.unique(np.logspace(0, 4, 300).astype(int))
    ax[1].plot(x, [bias_mult(i) for i in x], color=C[1], label="no correction / corrected")
    ax[1].axhline(1.05, color="#666", lw=1, ls="--")
    s = steps_until(1.05)
    ax[1].axvline(s, color="#666", lw=1, ls=":")
    ax[1].annotate(f"within 5% from step {s:,}", (s, 1.05), (90, 3.6), arrowprops=dict(arrowstyle="-", color="#666"))
    ax[1].set(xscale="log", xlabel="step (log scale)", ylabel="step multiplier", title="Same ratio over 10,000 steps", ylim=(0, 7.2))
    ax[1].legend(frameon=False)
    save(fig, "bias_correction")
    return dict(peak_step=max(range(1, 5000), key=bias_mult), peak=max(bias_mult(i) for i in range(1, 5000)),
                at_20=bias_mult(20), at_300=bias_mult(300),
                within={str(k): steps_until(k) for k in (1.5, 1.1, 1.05, 1.01)})


def part2():
    runs = load("part2")
    if not runs:
        return None
    short = [r for r in runs if r["spec"]["total"] == 300]
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, (bias, name) in enumerate([(True, "corrected"), (False, "no correction")]):
        rows = [r for r in short if r["spec"]["bias"] == bias]
        sel = lr_table(rows_of(rows), loss_at(300))
        rep = lr_table(rows_of(rows), loss_at(300, "report"))
        best = best_lr(sel)[0]
        untuned = min(sel, key=lambda x: abs(x - 3e-4))
        out[name] = dict(best_lr=best, best=stats(rep[best]), lr_near_3e4=untuned, at_3e4=stats(rep[untuned]), edge=is_edge(sel))
        lrs = sorted(rep)
        ax[1].plot(lrs, [np.mean(list(rep[x].values())) for x in lrs], "-o", ms=4, color=C[i], label=name)
        moved = [r["moves"] for r in rows if r["spec"]["lr"] == untuned]
        ax[0].plot(range(1, 21), np.mean(moved, axis=0), "-o", ms=4, color=C[i], label=name)
        out[name]["moves"] = np.mean(moved, axis=0).tolist()
    ax[0].set(xlabel="step", ylabel="median |update| / lr", title="Real model, every weight")
    ax[1].set(xscale="log", xlabel="learning rate", ylabel="val loss at step 300", title="Tuned separately")
    for a in ax:
        a.legend(frameon=False)
    save(fig, "bias_real_model")
    long = [r for r in runs if r["spec"]["total"] == 3000]
    gaps = {}
    for step in range(250, 3001, 250):
        d = [next(r for r in long if r["spec"]["bias"] is False and r["spec"]["seed"] == s)["report"].get(str(step), float("nan"))
             - next(r for r in long if r["spec"]["bias"] is True and r["spec"]["seed"] == s)["report"].get(str(step), float("nan"))
             for s in sorted({r["spec"]["seed"] for r in long})]
        gaps[str(step)] = [float(x) for x in d]
    out["long_gap_uncorrected_minus_corrected"] = gaps
    return out


def part3():
    runs = load("part3")
    if not runs:
        return None
    pick = lambda w: [r["ratios"] for r in runs if r["spec"]["warmup"] == w]
    on, off = pick(50), pick(0)
    gm = lambda rs, k: np.exp(np.mean([np.log(np.maximum(r[k], 1e-30)) for r in rs], axis=0))
    rows = {}
    for k in on[0]:
        a, b = gm(on, k), gm(off, k)
        rows[k] = dict(settle=warmup_settle(a, b, 50), peak_step=int(np.argmax(smooth(a))),
                       peak=float(smooth(a).max()), peak_no_warmup=float(smooth(b)[:50].max()),
                       start_no_warmup=float(b[0]), end=float(smooth(a)[-1]))
    settled = [v["settle"] for v in rows.values() if v["settle"] is not None]
    groups = dict(embeddings=("emb",), attention=("attn",), feed_forward=("ffn",), head_and_norms=("head", ".g"))
    def group(k):
        if "emb" in k: return "embeddings"
        if ".attn." in k: return "attention"
        if ".ffn." in k: return "feed_forward"
        return "head_and_norms"
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for a, name in zip(ax, groups):
        for k in on[0]:
            if group(k) == name:
                a.plot(smooth(gm(off, k)), color=C[1], lw=1, alpha=.8)
                a.plot(smooth(gm(on, k)), color=C[0], lw=1, alpha=.8)
        a.axvline(50, color="#666", ls=":", lw=1)
        a.set(yscale="log", xlabel="step", title=name.replace("_", " "))
    ax[0].plot([], [], color=C[0], label="warmup 50")
    ax[0].plot([], [], color=C[1], label="no warmup")
    ax[0].legend(frameon=False)
    ax[0].set_ylabel("||update|| / ||weight||")
    save(fig, "update_ratio")
    return dict(layers=rows, n=len(rows), n_settled=len(settled),
                median_settle=float(np.median(settled)) if settled else None,
                last_settle=int(max(settled)) if settled else None,
                median_peak=float(np.median([v["peak_step"] for v in rows.values()])))


def part4():
    runs = load("part4")
    if not runs:
        return None
    sets = {"cosine_300": lambda s: s["schedule"] == "cosine" and s["total"] == 300,
            "wsd_300": lambda s: s["schedule"] == "wsd",
            "cosine_200": lambda s: s["schedule"] == "cosine" and s["total"] == 200}
    out, tables = {}, {}
    for name, f in sets.items():
        rows = [r for r in runs if f(r["spec"])]
        sel = lr_table(rows_of(rows), loss_at(200))
        rep = lr_table(rows_of(rows), loss_at(200, "report"))
        best = best_lr(sel)[0]
        tables[name] = rep
        out[name] = dict(best_lr=best, edge=is_edge(sel), at_200=stats(rep[best]), per_seed=rep[best],
                         at_3e4_ish=stats(rep[min(rep, key=lambda x: abs(x - 2.5e-4))]))
        if name != "cosine_200":
            c = lr_table(rows_of(rows), loss_at(230, "report"))
            out[name]["after_30_step_cooldown"] = stats(c[best])
            out[name]["cooldown_per_seed"] = c[best]
    seeds = sorted(out["wsd_300"]["per_seed"])
    d = [out["wsd_300"]["per_seed"][s] - out["cosine_300"]["per_seed"][s] for s in seeds]
    out["paired_wsd_minus_cosine_at_200"] = dict(per_seed=d, mean=float(np.mean(d)))
    d = [out["wsd_300"]["cooldown_per_seed"][s] - out["cosine_300"]["cooldown_per_seed"][s] for s in seeds]
    out["paired_wsd_minus_cosine_after_cooldown"] = dict(per_seed=d, mean=float(np.mean(d)))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, (name, t) in enumerate(tables.items()):
        lrs = sorted(t)
        y = [np.mean(list(t[x].values())) for x in lrs]
        ax.plot(lrs, y, "-o", ms=4, color=C[i], label=name.replace("_", " "))
        ax.plot([out[name]["best_lr"]], [np.mean(list(t[out[name]["best_lr"]].values()))], "*", ms=14, color=C[i])
    ax.set(xscale="log", xlabel="peak learning rate", ylabel="val loss at step 200", title="Stopped at step 200, star = tuned choice")
    ax.legend(frameon=False)
    save(fig, "schedules")
    return out


def tables_for(job, steps):
    rows = [r for r in load(job) if r["spec"]["total"] == steps]
    return lr_table(rows_of(rows), loss_at(steps, "report")) if rows else None


def part5():
    tables = {w: tables_for(f"w{w}", 300) for w in (256, 512, 1024)}
    tables = {w: t for w, t in tables.items() if t}
    if len(tables) < 2:
        return None
    out = width_report(tables)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (w, t) in enumerate(sorted(tables.items())):
        lrs = sorted(t)
        y = [np.mean(list(t[x].values())) if np.isfinite(list(t[x].values())).all() else np.nan for x in lrs]
        lr_star = out["minima"][i]
        ax.plot(lrs, y, "-o", ms=4, color=C[i], label=f"width {w}: min at {lr_star:.2e}")
        ax.axvline(lr_star, color=C[i], ls=":", lw=1.5)
    ax.set(xscale="log", xlabel="peak learning rate", ylabel="val loss at step 300", title="Dotted lines mark each minimum")
    ax.legend(frameon=False)
    save(fig, "lr_vs_width")
    h = tables_for("horizon", 600)
    if h:
        out["horizon_256_600_steps"] = table_min(h)
    return out


if __name__ == "__main__":
    summary = dict(bias_closed_form=plot_bias(), part2=part2(), part3=part3(), part4=part4(), part5=part5())
    json.dump(summary, open(f"{RESULTS}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1)[:12000])
    print(sorted(glob.glob(f"{RESULTS}/*.png")))
