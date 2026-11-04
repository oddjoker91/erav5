"""engine.py -- one training run, described by a dict, with everything the parts need to log."""
import math
import os

import numpy as np
import torch

from analysis import lr_factor
from data_pipeline import ShardStream
from transformer import GPT, Tokenizer, loss_fn, make_masks

SHARD_ROOT = "/kaggle/input/datasets/gona26/tinystories-data-540m/shards"
B, T, TRAIN_TOKENS, VAL_TOKENS, RULER_BATCHES = 32, 256, 30_000_000, 2_000_000, 16


class Cfg:
    """Session 10's model: 2 layers, 8 heads. D is the width we sweep."""
    def __init__(self, D=256, H=8, L=2):
        self.V, self.T, self.D, self.H, self.L = Tokenizer().V, T, D, H, L
        self.HD, self.DFF = D // H, D * 4


class ManualAdam(torch.optim.Optimizer):
    """AdamW written out, so bias correction can be switched off."""
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, bias_correct=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.bias_correct = bias_correct

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                st["t"] += 1
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                mh, vh = st["m"], st["v"]
                if self.bias_correct:
                    mh, vh = mh / (1 - b1 ** st["t"]), vh / (1 - b2 ** st["t"])
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.addcdiv_(mh, vh.sqrt().add_(g["eps"]), value=-g["lr"])


class Data:
    """Train blocks plus two fixed val sets: `select` picks an LR, `report` scores it."""
    def __init__(self, dev, root=SHARD_ROOT):
        self.dev = dev
        self.train = ShardStream(f"{root}/train", B=B, T=T, seed=0, max_tokens=TRAIN_TOKENS)
        if not os.path.isdir(f"{root}/val"):
            raise SystemExit(f"no val shards in {root}: a ruler from train would flatter every run")
        val = ShardStream(f"{root}/val", B=B, T=T, seed=0, max_tokens=VAL_TOKENS)
        order = np.random.default_rng(123).permutation(val.n_blocks)
        need = 2 * RULER_BATCHES * B
        if need > len(order):
            raise SystemExit(f"val has {len(order)} blocks, need {need}")
        chunks = [val._batch(order[i:i + B]) for i in range(0, need, B)]
        tensors = [self.tensors(x, d) for x, d in chunks]
        self.select, self.report = tensors[:RULER_BATCHES], tensors[RULER_BATCHES:]

    def tensors(self, x, d):
        return (torch.from_numpy(np.ascontiguousarray(x)).to(self.dev),
                torch.from_numpy(np.ascontiguousarray(d)).to(self.dev))

    def batches(self, seed):
        """Same seed, same batches: paired runs differ only by what we changed."""
        rng = np.random.default_rng(seed)
        while True:
            order = rng.permutation(self.train.n_blocks)
            for i in range(0, len(order) - B + 1, B):
                yield self.tensors(*self.train._batch(order[i:i + B]))


@torch.no_grad()
def evaluate(model, ruler):
    model.eval()
    total, count = 0.0, 0
    for tokens, doc in ruler:
        attn, mask = make_masks(tokens, doc)
        loss, n = loss_fn(model(tokens, attn), tokens, mask)
        total, count = total + loss.item() * n, count + n
    model.train()
    return total / max(1, count)


def run(spec, data):
    """spec keys: D lr schedule total warmup stop seed bias eval_at cooldown ratios moves."""
    torch.manual_seed(spec["seed"])
    c = Cfg(D=spec.get("D", 256))
    model = GPT(c).to(data.dev)
    if "bias" in spec:
        opt = ManualAdam(model.parameters(), lr=spec["lr"], bias_correct=spec["bias"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"])
    params = dict(model.named_parameters())
    cooldown = tuple(spec["cooldown"]) if spec.get("cooldown") else None
    stream, ceiling = data.batches(spec["seed"]), 2 * math.log(c.V)
    out = dict(spec=spec, select={}, report={}, train=[], diverged=False,
               ratios={k: [] for k in params} if spec.get("ratios") else None,
               moves=[] if spec.get("moves") else None)

    for s in range(spec["stop"]):
        f = lr_factor(s, spec["schedule"], spec["total"], spec["warmup"], cooldown=cooldown)
        for g in opt.param_groups:
            g["lr"] = spec["lr"] * f
        tokens, doc = next(stream)
        attn, mask = make_masks(tokens, doc)
        loss, _ = loss_fn(model(tokens, attn), tokens, mask)
        if not math.isfinite(loss.item()) or (s > 10 and loss.item() > ceiling):
            out["diverged"] = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        track = out["ratios"] is not None or (out["moves"] is not None and s < spec["moves"])
        before = {k: p.detach().clone() for k, p in params.items()} if track else None
        opt.step()
        out["train"].append(round(loss.item(), 4))
        if out["ratios"] is not None:
            for k, p in params.items():
                out["ratios"][k].append(((p.detach() - before[k]).norm() / before[k].norm()).item())
        if out["moves"] is not None and s < spec["moves"]:
            d = torch.cat([(p.detach() - before[k]).abs().flatten() for k, p in params.items()])
            out["moves"].append((d.median() / max(g["lr"], 1e-30)).item())
        if s + 1 in spec.get("eval_at", []):
            out["select"][str(s + 1)] = evaluate(model, data.select)
            out["report"][str(s + 1)] = evaluate(model, data.report)
    del model, opt
    if data.dev != "cpu":
        torch.cuda.empty_cache()
    return out
