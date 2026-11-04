"""run.py -- the experiments. Every finished run is one line in results/<job>.jsonl,
so a restart skips it.    python run.py <job> [--gpu N]"""
import argparse
import json
import os
import sys

from analysis import best_lr, is_edge, lr_table

SEEDS = [0, 1, 2]
RESULTS = "results"
LADDER = [1.25e-4 * 2 ** k for k in range(9)]        # 1.25e-4 ... 3.2e-2


class Store:
    def __init__(self, job, runner):
        os.makedirs(RESULTS, exist_ok=True)
        self.path, self.runner = f"{RESULTS}/{job}.jsonl", runner
        self.done = {}
        if os.path.exists(self.path):
            for line in open(self.path):
                r = json.loads(line)
                self.done[self.key(r["spec"])] = r

    @staticmethod
    def key(spec):
        return json.dumps(spec, sort_keys=True)

    def get(self, **spec):
        k = self.key(spec)
        if k not in self.done:
            r = self.runner(spec)
            self.done[k] = r
            with open(self.path, "a") as f:
                f.write(json.dumps(r) + "\n")
            last = r["train"][-1] if r["train"] else None
            print({a: b for a, b in spec.items() if a in ("D", "lr", "schedule", "seed", "bias", "stop")},
                  "train", last, "DIVERGED" if r["diverged"] else "", flush=True)
        return self.done[k]


def rows_of(runs):
    return [dict(lr=r["spec"]["lr"], seed=r["spec"]["seed"], r=r) for r in runs]


def loss_at(step, split="select"):
    return lambda x: float("nan") if x["r"]["diverged"] else x["r"][split].get(str(step), float("nan"))


def tune(store, grid, step, seeds=SEEDS, **spec):
    """Run a grid, and widen it until the best LR has worse neighbours on both sides."""
    grid = sorted(grid)
    for _ in range(4):
        runs = [store.get(lr=lr, seed=s, **spec) for lr in grid for s in seeds]
        t = lr_table(rows_of(runs), loss_at(step))
        if not is_edge(t):
            return best_lr(t)[0], grid
        grid = sorted(grid + ([grid[0] / 2] if best_lr(t)[0] == grid[0] else [grid[-1] * 2]))
    print("WARNING: still at the grid edge", spec, flush=True)
    return best_lr(t)[0], grid


def part2(store):
    base = dict(schedule="constant", total=300, warmup=0, stop=300, eval_at=[50, 100, 150, 200, 250, 300])
    for bias in (True, False):
        tune(store, [1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3], 300, bias=bias, moves=20, **base)
    for bias in (True, False):                       # long enough to reach the closed-form answer
        for seed in (0, 1):
            store.get(lr=3e-4, seed=seed, bias=bias, schedule="constant", total=3000, warmup=0, stop=3000,
                      eval_at=list(range(250, 3001, 250)))


def part3(store):
    for warmup in (50, 0):
        for seed in (0, 1):
            store.get(lr=3e-4, seed=seed, schedule="constant", total=500, warmup=warmup, stop=500, ratios=True)


def part4(store):
    grid = [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3]
    tune(store, grid, 200, schedule="cosine", total=300, warmup=30, stop=230,
         cooldown=[200, 30], eval_at=[100, 150, 200, 230])
    tune(store, grid, 200, schedule="wsd", total=300, warmup=30, stop=230,
         cooldown=[200, 30], eval_at=[100, 150, 200, 230])
    tune(store, grid, 200, schedule="cosine", total=200, warmup=30, stop=200, eval_at=[100, 150, 200])


def width(store, D, steps=300, seeds_full=SEEDS):
    base = dict(D=D, schedule="cosine", total=steps, warmup=30, stop=steps, eval_at=[steps])
    coarse = [store.get(lr=lr, seed=s, **base) for lr in LADDER for s in (0, 1)]
    best = best_lr(lr_table(rows_of(coarse), loss_at(steps)))[0]
    for lr in [best / 2, best / 2 ** 0.5, best, best * 2 ** 0.5, best * 2]:
        for s in seeds_full:
            store.get(lr=lr, seed=s, **base)


JOBS = dict(part2=part2, part3=part3, part4=part4,
            w256=lambda st: width(st, 256), w512=lambda st: width(st, 512), w1024=lambda st: width(st, 1024),
            horizon=lambda st: width(st, 256, steps=600, seeds_full=[0, 1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job", choices=[*JOBS, "smoke"])
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()
    import torch
    from engine import Data, run
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    data = Data(dev)
    if a.job == "smoke":
        for spec in [dict(lr=3e-4, seed=0, bias=False, schedule="constant", total=6, warmup=0, stop=6,
                          eval_at=[6], moves=3),
                     dict(lr=3e-4, seed=0, schedule="wsd", total=6, warmup=2, stop=6, cooldown=[4, 2],
                          eval_at=[4, 6], ratios=True),
                     dict(D=1024, lr=3e-4, seed=0, schedule="cosine", total=3, warmup=1, stop=3, eval_at=[3])]:
            r = run(spec, data)
            print("ok", spec["schedule"], r["train"], r["select"], flush=True)
        sys.exit(0)
    JOBS[a.job](Store(a.job, lambda spec: run(spec, data)))


if __name__ == "__main__":
    main()
