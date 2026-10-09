"""
moe_pipeline.py -- the Mixture-of-Experts version of transformer.GPT.

Only ONE thing changes vs the dense model: inside every Block, the single SwiGLU
FFN becomes E SwiGLU experts plus a small router that sends each token to its
top-k experts.

    dense Block:  x + attn(n1(x));  x + ffn(n2(x))
    MoE   Block:  x + attn(n1(x));  x + sum_{e in topk(x)} w_e(x) * expert_e(n2(x))

Attention, norms, embeddings and the head are reused unchanged from transformer.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Attention, SwiGLU, RMSNorm


class MoE(nn.Module):
    """Top-k routed SwiGLU experts (dropless: no token is ever dropped)."""

    def __init__(self, c, E=4, k=2):
        super().__init__()
        self.E, self.k = E, k
        self.router = nn.Linear(c.D, E, bias=False)
        self.experts = nn.ModuleList([SwiGLU(c) for _ in range(E)])
        # filled every forward; read by the training loop
        self.aux = None        # load-balancing loss (scalar tensor)
        self.load = None       # fraction of routed slots per expert, shape (E,)

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)                                   # (N,D), N=B*T
        # router in fp32 -> probabilities over experts
        probs = self.router(xf).float().softmax(-1)             # (N,E)
        topw, topi = probs.topk(self.k, dim=-1)                 # (N,k) each
        topw = topw / topw.sum(-1, keepdim=True)                # renormalise -> sums to 1

        out = torch.zeros_like(xf)
        for e, expert in enumerate(self.experts):
            # which (token, slot) pairs chose expert e
            tok, slot = (topi == e).nonzero(as_tuple=True)
            # ALWAYS call the expert, even with 0 tokens: keeps every expert in the
            # autograd graph so FSDP gets a (zero) grad for it on every rank
            y = expert(xf[tok])                                 # (n_e,D)
            out.index_add_(0, tok, y * topw[tok, slot, None].to(y.dtype))

        # Switch-style load balancing: E * sum_e f_e * P_e  (=1 when perfectly uniform)
        # f_e = share of routed slots that went to e (no grad), P_e = mean router prob
        f = F.one_hot(topi, self.E).float().sum((0, 1)) / (xf.size(0) * self.k)
        P = probs.mean(0)
        self.aux = self.E * (f * P).sum()
        self.load = f.detach()
        return out.view(B, T, D)


class MoEBlock(nn.Module):
    def __init__(self, c, E=4, k=2):
        super().__init__()
        self.c = c
        self.attn = Attention(c)
        self.n1 = RMSNorm(c.D)
        self.n2 = RMSNorm(c.D)
        self.ffn = MoE(c, E, k)

    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.ffn(self.n2(x))
        return x


class MoEGPT(nn.Module):
    """Same as transformer.GPT, but every Block's FFN is an MoE."""

    def __init__(self, c, E=4, k=2):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.V, c.D)
        self.pos_emb = nn.Embedding(c.T, c.D)
        self.blocks = nn.ModuleList([MoEBlock(c, E, k) for _ in range(c.L)])
        self.ns = RMSNorm(c.D)
        self.head = nn.Linear(c.D, c.V, bias=False)

    def forward(self, tokens, mask):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ns(x)
        return self.head(x)


def moe_layers(model):
    return [m for m in model.modules() if isinstance(m, MoE)]


def aux_loss(model):
    """Mean load-balancing loss over all MoE layers of the last forward."""
    layers = moe_layers(model)
    return sum(m.aux for m in layers) / len(layers)


def expert_load(model):
    """(L,E) tensor: per-layer share of routed slots each expert got."""
    return torch.stack([m.load for m in moe_layers(model)])


def param_counts(model, E, k):
    """(total, active-per-token) parameter counts for an MoEGPT."""
    total = sum(p.numel() for p in model.parameters())
    expert = sum(p.numel() for m in moe_layers(model) for p in m.experts.parameters())
    active = total - expert + expert * k // E
    return total, active
