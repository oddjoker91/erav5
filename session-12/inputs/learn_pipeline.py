"""
learn_pipeline.py -- PIPELINE B: token shards -> batches -> probabilities.

All DATA handling (tokenizer, shard writing, shard streaming) lives in
data_pipeline.py. This file starts from the streamed batches and turns them
into logits -> probabilities -> loss (the learning half).

    python data_pipeline.py     # once: raw -> token shards
    python learn_pipeline.py    # every run: shards -> batches (-> training)
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline import Config, get_streams, Tokenizer

def make_masks(tokens,doc_id):
    TOK = Tokenizer()
    B,T = tokens.shape
    # no peeking into future tokens
    casual = torch.tril(torch.ones(T,T, dtype=torch.bool, device=tokens.device)) # (T,T)
    # only attend to tokens within the same document
    same = doc_id[:,:,None] == doc_id[:,None,:] # (B,T,1) vs (B,1,T) -> (B,T,T)
    # combine causal and same-document masks
    attn = (casual & same).unsqueeze(1) # (B,1,T,T)
    # target tokens (shifted by one)
    tgt = tokens[:,1:]
    # identify positions where the document changes
    join = doc_id[:,1:] != doc_id[:,:-1] # (B,T-1) vs (B,T-1) -> comparing adjacent doc_ids
    # positions where the document changes (to avoid computing loss across document boundaries)
    loss = (tgt != TOK.PAD) & (~join)
    return attn, loss

class RMSNorm(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.d = d
        self.g = nn.Parameter(torch.ones(d))
    def forward(self,x):
        # RMSNorm forward pass
        ms = x.pow(2).mean(-1, keepdim=True) # (B,T,D) -> (B,T,1)
        return x*torch.rsqrt(ms + 1e-6)*self.g

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.D, c.D, bias=False)
        self.k = nn.Linear(c.D, c.D, bias=False)
        self.v = nn.Linear(c.D, c.D, bias=False)
        self.out = nn.Linear(c.D, c.D, bias=False)
    def forward(self,x,mask):
        c  = self.c
        B,T,_ = x.shape
        sp = lambda t: t.view(B,T,c.H, c.HD).transpose(1,2)
        # get q,k,v projections for multi-head attention
        q,k,v = sp(self.q(x)), sp(self.k(x)), sp(self.v(x)) # (B,c.H,T,c.HD)
        # multiply queries and keys to get attention scores
        s = (q @ k.transpose(-2,-1)) / (c.HD ** 0.5) # (B,c.H,T,c.HD) @ (B,c.H,c.HD,T) -> (B,c.H,T,T)
        # apply the attention mask to prevent attending to future tokens or tokens from different documents
        s = s.masked_fill(~mask, float('-inf'))
        # softmax to obtain attention weights
        w = torch.nan_to_num(s.softmax(-1))
        # value vectors are combined with attention weights to produce the context vectors
        ctx = (w @ v).transpose(1,2).reshape(B,T,c.D) 
        return self.out(ctx)

class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Linear(c.D, c.DFF, bias=False)
        self.up = nn.Linear(c.D, c.DFF, bias=False)
        self.bottom = nn.Linear(c.DFF, c.D, bias=False)
    def forward(self,x):
        return self.bottom(self.up(x) * F.silu(self.gate(x)))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.attn = Attention(c)
        self.n1 = RMSNorm(c.D)
        self.n2 = RMSNorm(c.D)
        self.ffn = SwiGLU(c)
    def forward(self,x,mask):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.ffn(self.n2(x))
        return x


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.V, c.D)
        self.pos_emb = nn.Embedding(c.T, c.D)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.L)])
        self.ns = RMSNorm(c.D)
        self.head = nn.Linear(c.D, c.V, bias=False)

    def forward(self,tokens, mask):
        B,T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x,mask)
        x = self.ns(x)
        logits = self.head(x)
        return logits

def loss_fn(logits, tokens, loss_mask):
    tgt = tokens[:, 1:].reshape(-1) # (B, T-1) -> (B*(T-1))
    lg = logits[:,:-1].reshape(-1,logits.size(-1)) # (B, T-1, V) -> (B*(T-1), V)
    nll = F.cross_entropy(lg, tgt, reduction='none')
    m = loss_mask.reshape(-1).float()
    loss = (nll * m).sum() / m.sum().clamp(min=1)
    return loss, int(m.sum())

def lr_at(s, WARMUP, STEPS):
    if s < WARMUP:
        return s / WARMUP
    t = (s - WARMUP) / max(1, STEPS - WARMUP)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * t))


# @torch.no_grad()
# def evaluate(model, val_stream, max_batches=20):
#     """Token-weighted mean loss over up to max_batches of the val stream."""
#     model.eval()
#     tot, ntok = 0.0, 0
#     for i, (x, d) in enumerate(val_stream()):        # fresh finite pass
#         if i >= max_batches:
#             break
#         tokens, doc_id = torch.from_numpy(x), torch.from_numpy(d)
#         attn, loss_mask = make_masks(tokens, doc_id)
#         loss, n = loss_fn(model(tokens, attn), tokens, loss_mask)
#         tot += loss.item() * n
#         ntok += n
#     model.train()
#     return tot / max(1, ntok)


# def _trace_shapes(model, tokens, doc_id):
#     """One-time forward pass that prints the output shape of every submodule."""
#     attn, loss_mask = make_masks(tokens, doc_id)
#     print("  inputs:")
#     print(f"    {'tokens':<26} {tuple(tokens.shape)}")
#     print(f"    {'doc_id':<26} {tuple(doc_id.shape)}")
#     print(f"    {'attn mask':<26} {tuple(attn.shape)}")
#     print(f"    {'loss mask':<26} {tuple(loss_mask.shape)}")
#     print("  module outputs:")

#     handles = []
#     def mk(name):
#         def hook(mod, inp, out):
#             if isinstance(out, torch.Tensor):
#                 print(f"    {name:<26} {tuple(out.shape)}")
#         return hook
#     for name, mod in model.named_modules():
#         if name:                                     # skip the root module
#             handles.append(mod.register_forward_hook(mk(name)))
#     with torch.no_grad():
#         logits = model(tokens, attn)
#     for h in handles:
#         h.remove()
#     print(f"    {'logits (head)':<26} {tuple(logits.shape)}")


# def train(STEPS=500, WARMUP=50, lr=3e-4, micro_B=4, accum_steps=4,
#           eval_every=100, log_every=1, max_shards=None, seed=0, debug=True):
#     cfg = Config()
#     torch.manual_seed(seed)

#     # STAGE 2: stream (tokens, doc_id) batches from the saved token shards.
#     train_stream, val_stream = get_streams(cfg, micro_B=micro_B, seed=seed,
#                                            max_shards=max_shards)
#     model = GPT(cfg)
#     opt = torch.optim.AdamW(model.parameters(), lr=lr)
#     print(f"eff. batch {micro_B * accum_steps}  |  ln(V) {math.log(cfg.V):.3f}")

#     if debug:                                        # one-time shape trace
#         x, d = next(train_stream)
#         print("shape trace:")
#         _trace_shapes(model, torch.from_numpy(x), torch.from_numpy(d))

#     seen = 0                                         # cumulative tokens trained on
#     for s in range(STEPS):
#         t0 = time.time()
#         cur_lr = lr * lr_at(s, WARMUP, STEPS)
#         for g in opt.param_groups:                   # LR warmup + cosine decay
#             g['lr'] = cur_lr
#         opt.zero_grad(set_to_none=True)

#         run_loss, step_tok = 0.0, 0
#         for _ in range(accum_steps):                 # accumulate micro-batches
#             x, d = next(train_stream)
#             tokens, doc_id = torch.from_numpy(x), torch.from_numpy(d)
#             attn, loss_mask = make_masks(tokens, doc_id)
#             loss, n = loss_fn(model(tokens, attn), tokens, loss_mask)
#             (loss / accum_steps).backward()          # grads add up; no step yet
#             run_loss, step_tok = run_loss + loss.item() * n, step_tok + n
#         gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         opt.step()                                   # one update per effective batch
#         seen += step_tok

#         if s % log_every == 0:
#             dt = time.time() - t0
#             avg = run_loss / max(1, step_tok)        # token-weighted mean over the step
#             print(f"step {s:4d} | lr {cur_lr:.2e} | loss {avg:.4f} | "
#                   f"gnorm {float(gnorm):.2f} | tok {step_tok} | seen {seen} | "
#                   f"{step_tok / max(dt, 1e-9):.0f} tok/s")
#         if s % eval_every == 0:
#             print(f"           val {evaluate(model, val_stream):.4f}")
#     return model


if __name__ == "__main__":
    # Requires token shards on disk first:  python data_pipeline.py
    # train(STEPS=500, eval_every=50)
    pass

