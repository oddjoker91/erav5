
"""
reversible_pipeline.py -- reversible transformer (memory-cheap depth).

Idea: treat each block as an ODE step on TWO coupled streams (x1, x2). Because
the update is an invertible coupling, the backward pass can RECONSTRUCT every
block's input from its output instead of storing activations. Peak activation
memory becomes ~ONE block instead of L blocks -> you can push a bigger batch.

Reuses Attention / SwiGLU / RMSNorm from learn_pipeline so the parameter count
stays identical to the baseline GPT (same ~20M model, just trained reversibly).

Variants (which integrator couples the two streams):
    euler     additive coupling, 2 fn evals/block (F, G)   -- fastest
    midpoint  symmetric Verlet coupling, 3 fn evals (G,F,G) -- 2nd order, stabler

Both are EXACTLY reversible (reconstruction is algebraic, not a solve).
"""

import torch
import torch.nn as nn

from data_pipeline import Config
from learn_pipeline import Attention, SwiGLU, RMSNorm


class ReversibleBlock(nn.Module):
    """One invertible block over two streams.

    F = RMSNorm -> Attention   (mixes across time)
    G = RMSNorm -> SwiGLU      (mixes across channels)
    """

    def __init__(self, c, variant="euler"):
        super().__init__()
        self.variant = variant
        self.n1 = RMSNorm(c.D)
        self.attn = Attention(c)
        self.n2 = RMSNorm(c.D)
        self.ffn = SwiGLU(c)

    def F(self, x, mask):
        return self.attn(self.n1(x), mask)

    def G(self, x):
        return self.ffn(self.n2(x))

    def forward(self, x1, x2, mask):
        if self.variant == "euler":
            y1 = x1 + self.F(x2, mask)
            y2 = x2 + self.G(y1)
        else:  # midpoint (symmetric Verlet): half G, full F, half G
            h = x2 + 0.5 * self.G(x1)
            y1 = x1 + self.F(h, mask)
            y2 = h + 0.5 * self.G(y1)
        return y1, y2

    def inverse(self, y1, y2, mask):
        """Exactly undo forward(): reconstruct (x1, x2) from (y1, y2)."""
        if self.variant == "euler":
            x2 = y2 - self.G(y1)
            x1 = y1 - self.F(x2, mask)
        else:  # midpoint
            h = y2 - 0.5 * self.G(y1)
            x1 = y1 - self.F(h, mask)
            x2 = h - 0.5 * self.G(x1)
        return x1, x2

    def backward_pass(self, y1, y2, dy1, dy2, mask):
        """Reconstruct this block's input, recompute its forward WITH grad, and
        push (dy1, dy2) back into input grads + accumulate parameter grads."""
        with torch.no_grad():
            x1, x2 = self.inverse(y1, y2, mask)
        x1 = x1.detach().requires_grad_(True)
        x2 = x2.detach().requires_grad_(True)
        with torch.enable_grad():
            z1, z2 = self.forward(x1, x2, mask)
            torch.autograd.backward((z1, z2), (dy1, dy2))
        return x1.detach(), x2.detach(), x1.grad, x2.grad


class _ReversibleFunction(torch.autograd.Function):
    """Runs the whole stack with NO stored activations; rebuilds them in backward."""

    @staticmethod
    def forward(ctx, x, mask, blocks):
        with torch.no_grad():
            x1 = x2 = x
            for blk in blocks:
                x1, x2 = blk.forward(x1, x2, mask)
        ctx.blocks = blocks
        ctx.mask = mask
        ctx.save_for_backward(x1, x2)          # only the FINAL states, not per-layer
        return x1 + x2

    @staticmethod
    def backward(ctx, dout):
        y1, y2 = ctx.saved_tensors
        dy1 = dy2 = dout                        # output was y1 + y2
        for blk in reversed(ctx.blocks):
            y1, y2, dy1, dy2 = blk.backward_pass(y1, y2, dy1, dy2, ctx.mask)
        return dy1 + dy2, None, None           # x was fed to both streams


class ReversibleGPT(nn.Module):
    """Same shape/params as learn_pipeline.GPT, but the block stack is reversible."""

    def __init__(self, c, variant="euler"):
        super().__init__()
        self.c = c
        self.variant = variant
        self.tok_emb = nn.Embedding(c.V, c.D)
        self.pos_emb = nn.Embedding(c.T, c.D)
        self.blocks = nn.ModuleList([ReversibleBlock(c, variant) for _ in range(c.L)])
        self.ns = RMSNorm(c.D)
        self.head = nn.Linear(c.D, c.V, bias=False)

    def forward(self, tokens, mask):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        x = _ReversibleFunction.apply(x, mask, self.blocks)
        x = self.ns(x)
        return self.head(x)


if __name__ == "__main__":
    # quick reversibility self-check: inverse() must undo forward() to ~fp32 eps
    cfg = Config()
    cfg.D, cfg.H, cfg.L, cfg.T = 384, 8, 7, 256
    cfg.HD, cfg.DFF = cfg.D // cfg.H, cfg.D * 4
    for variant in ("euler", "midpoint"):
        blk = ReversibleBlock(cfg, variant)
        x1 = torch.randn(2, cfg.T, cfg.D)
        x2 = torch.randn(2, cfg.T, cfg.D)
        mask = torch.ones(2, 1, cfg.T, cfg.T, dtype=torch.bool).tril(0)
        y1, y2 = blk.forward(x1, x2, mask)
        r1, r2 = blk.inverse(y1, y2, mask)
        err = (r1 - x1).abs().max().item() + (r2 - x2).abs().max().item()
        print(f"{variant:9s} reconstruction error = {err:.2e}")
