"""analysis.py -- schedules and the rules that turn runs into answers. numpy only."""
import math

import numpy as np

B1, B2 = 0.9, 0.999


def lr_factor(s, schedule, total, warmup, decay_frac=0.1, cooldown=None):
    """Fraction of peak LR at 0-indexed step s. cooldown=(start, length) ramps to 0 from start."""
    if cooldown is not None and s >= cooldown[0]:
        base = lr_factor(cooldown[0] - 1, schedule, total, warmup, decay_frac)
        return base * max(0.0, 1 - (s - cooldown[0] + 1) / cooldown[1])
    if s < warmup:
        return (s + 1) / warmup
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))
    if schedule == "wsd":
        start = int(total * (1 - decay_frac))
        return 1.0 if s < start else max(0.0, (total - s) / (total - start))
    raise ValueError(schedule)


def bias_mult(t, b1=B1, b2=B2):
    """Uncorrected Adam step / corrected Adam step at step t. Ignores eps; gradients cancel."""
    return (1 - b1 ** t) / math.sqrt(1 - b2 ** t)


def steps_until(threshold, b1=B1, b2=B2, limit=100_000):
    """Last moment the multiplier is above threshold, +1: the difference is gone from here on."""
    peak = max(range(1, 5000), key=lambda t: bias_mult(t, b1, b2))
    return next(t for t in range(peak, limit) if bias_mult(t, b1, b2) <= threshold)


def smooth(x, w=11):
    """Rolling median, trailing window so a value only uses the past."""
    x = np.asarray(x, float)
    return np.array([np.median(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


def warmup_settle(with_w, without_w, warmup, tol=1.1, hold=20):
    """First step >= warmup where the two ratios stay within tol for hold steps, else None."""
    a, b = smooth(with_w), smooth(without_w)
    close = np.abs(np.log(np.maximum(a, 1e-30) / np.maximum(b, 1e-30))) <= math.log(tol)
    for s in range(warmup, len(close) - hold + 1):
        if close[s:s + hold].all():
            return s
    return None


def lr_table(rows, value):
    """rows: dicts with lr, seed and value(row) -> {lr: {seed: loss or nan}}."""
    t = {}
    for r in rows:
        t.setdefault(r["lr"], {})[r["seed"]] = value(r)
    return dict(sorted(t.items()))


def best_lr(table):
    """Lowest mean loss; a diverged seed disqualifies that LR."""
    means = {}
    for lr, by in table.items():
        v = np.array(list(by.values()), float)
        means[lr] = float(v.mean()) if np.isfinite(v).all() else float("inf")
    return min(means, key=means.get), means


def is_edge(table):
    lrs = sorted(table)
    return best_lr(table)[0] in (lrs[0], lrs[-1])


def parabola_min(lrs, losses):
    """Minimum of a parabola through up to 5 points around the best LR, on log2(lr)."""
    lrs, losses = np.asarray(lrs, float), np.asarray(losses, float)
    keep = np.isfinite(losses)
    lrs, losses = lrs[keep], losses[keep]
    o = np.argsort(lrs)
    lrs, losses = lrs[o], losses[o]
    i = int(np.argmin(losses))
    lo, hi = max(0, i - 2), min(len(lrs), i + 3)
    x, y = np.log2(lrs[lo:hi]), losses[lo:hi]
    if len(x) < 3:
        return float(lrs[i])
    a, b, _ = np.polyfit(x, y, 2)
    if a <= 0:
        return float(lrs[i])
    v = -b / (2 * a)
    return float(2 ** np.clip(v, x[0], x[-1]))


def table_min(table, seeds=None):
    """LR at the minimum, averaging over seeds. A seed may be repeated (bootstrap)."""
    lrs, ys = [], []
    for lr, by in table.items():
        use = seeds if seeds is not None else list(by)
        v = [by[s] for s in use if s in by]
        if v:
            lrs.append(lr)
            ys.append(np.mean(v))
    return parabola_min(lrs, ys)


def fit_power(widths, lrs):
    """Fit lr = A * width^slope. Returns slope, log2A."""
    return tuple(np.polyfit(np.log2(widths), np.log2(lrs), 1))


def predict(widths, lrs, target):
    slope, c = fit_power(widths, lrs)
    return float(2 ** (slope * math.log2(target) + c))


def width_report(tables, target=4096, n_boot=2000, seed=0):
    """tables: {width: lr table}. Returns minima, slope, forecast, interval and a backtest."""
    ws = sorted(tables)
    mins = [table_min(tables[w]) for w in ws]
    slope, _ = fit_power(ws, mins)
    rng = np.random.default_rng(seed)
    boots, slopes = [], []
    for _ in range(n_boot):
        m = []
        for w in ws:
            all_seeds = sorted({s for by in tables[w].values() for s in by})
            m.append(table_min(tables[w], list(rng.choice(all_seeds, len(all_seeds)))))
        slopes.append(fit_power(ws, m)[0])
        boots.append(predict(ws, m, target))
    out = dict(widths=ws, minima=mins, slope=float(slope),
               forecast=predict(ws, mins, target),
               forecast_80=[float(x) for x in np.percentile(boots, [10, 90])],
               slope_80=[float(x) for x in np.percentile(slopes, [10, 90])],
               one_over_width=mins[0] * ws[0] / target)
    if len(ws) >= 3:
        back = predict(ws[:-1], mins[:-1], ws[-1])
        out.update(backtest_pred=back, backtest_true=mins[-1],
                   backtest_error_x=max(back / mins[-1], mins[-1] / back))
    return out
