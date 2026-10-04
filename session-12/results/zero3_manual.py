#
# Manual ZeRO-3 -- what FSDP does, written out by hand (TEACHING reference).
# Simplifications vs real FSDP: shards only the transformer Blocks (embeddings/head
# stay replicated + DDP-synced), no comm/compute overlap, no prefetch. The point is
# to SEE the mechanism: shard params -> gather full weights per Block just-in-time ->
# run -> free; reduce_scatter grads in backward.

import os
import sys
import csv
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.func import functional_call          # run a module with externally-supplied params

BASE = "/kaggle/input/datasets/gona26/tinystories-slm"
DATA = f"{BASE}/tinystories_data"
sys.path.insert(0, BASE)
from data_pipeline import Config, get_streams
from learn_pipeline import GPT, Block, RMSNorm, make_masks, loss_fn, lr_at


def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            device_id=torch.device(f"cuda:{local_rank}"))
    return dist.get_rank(), dist.get_world_size(), local_rank


def measure_holdings(model, opt):
    # actual bytes from live tensors: shards + replicated root params (skip emptied)
    mb = lambda ts: sum(t.numel() * t.element_size() for t in ts) / 1e6
    params_mb = mb([p for p in model.parameters() if p.numel() > 0])
    grads_mb = mb([p.grad for p in model.parameters() if p.grad is not None])
    optim_mb = mb([t for st in opt.state.values() for t in st.values() if torch.is_tensor(t)])
    return params_mb, grads_mb, optim_mb


# ---- the heart of ZeRO-3: gather a sharded param in forward, scatter its grad in backward ----
class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, world_size, numel, shape):
        ctx.world_size = world_size
        ctx.shard_numel = shard.numel()
        # FORWARD = all_gather: pull every rank's slice -> rebuild the FULL param
        buffers = [torch.empty_like(shard) for _ in range(world_size)]
        dist.all_gather(buffers, shard.contiguous())
        full = torch.cat(buffers)[:numel].view(shape)     # drop padding, restore shape
        return full

    @staticmethod
    def backward(ctx, grad_full):
        ws = ctx.world_size
        # BACKWARD = reduce_scatter: sum grads across ranks, each rank KEEPS ONLY ITS SLICE
        grad_flat = grad_full.reshape(-1)
        pad = ctx.shard_numel * ws - grad_flat.numel()
        if pad:
            grad_flat = torch.cat([grad_flat, grad_flat.new_zeros(pad)])
        chunks = list(grad_flat.chunk(ws))
        out = torch.empty_like(chunks[0])
        dist.reduce_scatter(out, chunks)                  # out = sum over ranks of chunks[rank]
        out /= ws                                         # average (data-parallel mean)
        return out, None, None, None                      # grads only for `shard`


# ---- wraps ONE module: stores only this rank's shard of each weight, gathers on forward ----
class ShardedModule(nn.Module):
    def __init__(self, module, world_size, rank, device):
        super().__init__()
        self.mod = module
        self.ws, self.rank = world_size, rank
        # META -> real: materialize + init THIS module only (one Block at a time -> low peak,
        # the full model is never assembled on one GPU)
        module.to_empty(device=device, recurse=True)
        for sub in module.modules():
            with torch.no_grad():
                if isinstance(sub, RMSNorm):
                    sub.g.fill_(1.0)
                elif hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()
        # make the full weights identical across ranks so shards are consistent slices
        for p in module.parameters():
            dist.broadcast(p.data, src=0)
        # keep ONLY my 1/world_size chunk of each weight, then FREE the full weight
        self.meta = {}                                    # orig name -> (key, numel, shape)
        self.shards = nn.ParameterDict()
        for name, p in list(module.named_parameters()):
            flat = p.detach().reshape(-1)
            pad = (-flat.numel()) % world_size            # pad so it splits evenly
            if pad:
                flat = torch.cat([flat, flat.new_zeros(pad)])
            key = name.replace(".", "__")
            self.shards[key] = nn.Parameter(flat.chunk(world_size)[rank].clone())
            self.meta[name] = (key, p.numel(), tuple(p.shape))
        with torch.no_grad():
            for p in module.parameters():
                p.set_(torch.empty(0, device=device))     # drop the full weight

    def forward(self, *args, **kwargs):
        # gather every weight just-in-time -> a dict of FULL params
        full = {name: AllGather.apply(self.shards[key], self.ws, numel, shape)
                for name, (key, numel, shape) in self.meta.items()}
        # run the original module using those gathered params (nothing stored on self.mod)
        out = functional_call(self.mod, full, args, kwargs)
        return out                                        # `full` frees when it goes out of scope


def main():
    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    cfg = Config()
    micro_B = 32
    train_stream, _ = get_streams(cfg, shard_dir=f"{DATA}/shards",
                                  micro_B=micro_B, seed=1234,
                                  rank=rank, world_size=world_size)

    STEPS, WARMUP, base_lr, LOG_EVERY = 1500, 100, 3e-4, 50
    torch.manual_seed(0)
    # META init: shapes only, ZERO bytes -> the full model is never built on one GPU
    with torch.device("meta"):
        model = GPT(cfg)

    # shard each Block: each ShardedModule materializes+inits+shards itself, one at a
    # time -> peak memory ~ one Block, never the whole model
    model.blocks = nn.ModuleList(
        [ShardedModule(blk, world_size, rank, device) for blk in model.blocks]
    )
    # materialize the still-replicated root modules (embeddings / final norm / head)
    for m in (model.tok_emb, model.pos_emb, model.ns, model.head):
        m.to_empty(device=device, recurse=True)
        for sub in m.modules():
            with torch.no_grad():
                if isinstance(sub, RMSNorm):
                    sub.g.fill_(1.0)
                elif hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()
    # root params = replicated (not inside a ShardedModule's .shards); sync across ranks
    root_params = [p for n, p in model.named_parameters()
                   if ".shards." not in n and p.numel() > 0]
    for p in root_params:
        dist.broadcast(p.data, src=0)
    # optimize the shards + the root params (skip the emptied originals)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.numel() > 0], lr=base_lr)

    if rank == 0:
        logf = open("/kaggle/working/metrics_zero3_manual.csv", "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["step", "loss", "tok_per_s", "lr", "mem_mb"])
    tok_per_step = cfg.T * micro_B * world_size

    for s in range(1, STEPS + 1):
        t0 = time.time()
        cur_lr = base_lr * lr_at(s, WARMUP, STEPS)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, d = next(train_stream)
        tokens = torch.from_numpy(x).to(device)
        doc_id = torch.from_numpy(d).to(device)
        attn, loss_mask = make_masks(tokens, doc_id)
        # forward: each ShardedModule all_gathers its Block, runs it, frees it
        loss, n = loss_fn(model(tokens, attn), tokens, loss_mask)
        opt.zero_grad(set_to_none=True)
        # backward: Block grads are reduce_scattered automatically by AllGather.backward
        loss.backward()
        # root (replicated) params still need a manual DDP-style all_reduce
        for p in root_params:
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world_size
        opt.step()
        dt = time.time() - t0

        if s % LOG_EVERY == 0:
            stats = torch.tensor([loss.item() * n, float(n)], device=device)
            dist.all_reduce(stats)
            avg_loss = (stats[0] / stats[1]).item()
            if rank == 0:
                mem = torch.cuda.max_memory_allocated() / 1e6
                tps = tok_per_step / max(dt, 1e-9)
                writer.writerow([s, f"{avg_loss:.4f}", f"{tps:.0f}",
                                 f"{cur_lr:.2e}", f"{mem:.0f}"])
                logf.flush()
                print(f"step {s:4d} | loss {avg_loss:.4f} | {tps:.0f} tok/s | "
                      f"mem {mem:.0f}MB", flush=True)

    if rank == 0:
        logf.close()
    # per-rank holdings (sharded Blocks + replicated root): compare vs DDP / ZeRO-1 / FSDP
    params_mb, grads_mb, optim_mb = measure_holdings(model, opt)
    print(f"rank {rank} holdings: params {params_mb:.0f}MB grads {grads_mb:.0f}MB "
          f"optim {optim_mb:.0f}MB total {params_mb+grads_mb+optim_mb:.0f}MB", flush=True)
    if rank == 0:
        with open("/kaggle/working/holdings_zero3_manual.csv", "w", newline="") as hf:
            w = csv.writer(hf)
            w.writerow(["component", "mb"])
            w.writerow(["params", f"{params_mb:.2f}"])
            w.writerow(["grads", f"{grads_mb:.2f}"])
            w.writerow(["optimizer", f"{optim_mb:.2f}"])

    dist.barrier()
    if rank == 0:
        print("done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
