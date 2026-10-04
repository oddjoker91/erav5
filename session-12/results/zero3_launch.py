
# read env vars torchrun injects
import os
# csv metrics logging + step timing
import csv
import time
# add the Kaggle dataset folder to the import path
import sys
# for torch.cuda.set_device
import torch
# collectives: init/all_reduce/broadcast
import torch.distributed as dist
# ZeRO-3 engine: shards params + grads + optimizer, gathers layer-by-layer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# dataset root holds the loose .py files; add it to the import path
SRC = "/kaggle/input/datasets/gona26/tinystories-slm"
# shards + bpe live under this subfolder
DATA = f"{SRC}/tinystories_data"
sys.path.insert(0, SRC)
# model/vocab sizes + the data stream factory
from data_pipeline import Config, get_streams
# model + a Block (the unit we shard) + RMSNorm (needs custom init) + helpers
from learn_pipeline import GPT, Block, RMSNorm, make_masks, loss_fn, lr_at


def setup():
    # CPU: read this process's GPU index that torchrun injected (str -> int)
    local_rank = int(os.environ["LOCAL_RANK"])
    # GPU: pin this process to that one GPU (must happen before init)
    torch.cuda.set_device(local_rank)
    # GPU<->GPU: join the process group; device_id tells NCCL our GPU
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    # CPU: return identity read back from the group
    return dist.get_rank(), dist.get_world_size(), local_rank


def print0(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def model_info(model, cfg):
    P = sum(p.numel() for p in model.parameters())    # numel works on meta tensors too
    emb = 2 * cfg.V * cfg.D
    return (f"model: {P/1e6:.2f}M params | L={cfg.L} D={cfg.D} H={cfg.H} "
            f"T={cfg.T} V={cfg.V} | embed+head ~{emb/1e6:.1f}M ({100*emb/P:.0f}%)")


def measure_holdings(model, opt):
    # actual bytes from live tensors; under FSDP these are the LOCAL shards (1/N)
    mb = lambda ts: sum(t.numel() * t.element_size() for t in ts) / 1e6
    params_mb = mb(list(model.parameters()))
    grads_mb = mb([p.grad for p in model.parameters() if p.grad is not None])
    optim_mb = mb([t for st in opt.state.values() for t in st.values() if torch.is_tensor(t)])
    return params_mb, grads_mb, optim_mb


def param_init_fn(module):
    # FSDP calls this per module while sharding, to bring META tensors to life:
    # allocate real (uninitialized) tensors on THIS process's GPU...
    module.to_empty(device=torch.cuda.current_device(), recurse=False)
    # ...then set proper initial values (meta tensors carried none)
    with torch.no_grad():
        if isinstance(module, RMSNorm):
            module.g.fill_(1.0)                 # RMSNorm gain starts at 1
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()           # Linear/Embedding default init


def main():
    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    print(f"rank {rank}/{world_size} on cuda:{local_rank}", flush=True)

    # ---- data: disjoint blocks per rank (shared seed) ----
    # CPU: Config builds tokenizer; the stream reads shards into numpy (numpy = CPU RAM)
    cfg = Config()
    micro_B = 32
    # CPU: sets up the numpy batch iterator (no GPU touched here)
    train_stream, _ = get_streams(cfg, shard_dir=f"{DATA}/shards",
                                  micro_B=micro_B, seed=1234,
                                  rank=rank, world_size=world_size)

    STEPS, WARMUP, base_lr, LOG_EVERY = 1500, 100, 3e-4, 50
    torch.manual_seed(0)
    # META init: build the model with SHAPES ONLY -> ZERO bytes allocated, so the full
    # model never lands on any single GPU (this is what avoids OOM for huge models).
    with torch.device("meta"):
        model = GPT(cfg)
    print0(rank, model_info(model, cfg))

    # ---- ZeRO-3: shard params across ranks; FSDP materializes only each rank's 1/N ----
    # GPU: FULL_SHARD = ZeRO-3 (params + grads + optimizer sharded). param_init_fn builds
    # + inits each shard on the GPU, so the whole model is never assembled in one place.
    # (SHARD_GRAD_OP would be ZeRO-2; NO_SHARD would be plain DDP.)
    model = FSDP(
        model,
        # each Block is gathered/freed on its own -> peak memory = ~one Block
        auto_wrap_policy=ModuleWrapPolicy({Block}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        param_init_fn=param_init_fn,
    )
    # CPU setup over GPU-resident sharded params: MUST be built AFTER FSDP so it sees
    # the sharded (flat) params; its m/v state is allocated on GPU at first step, 1/N size
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
    print0(rank, model)

    if rank == 0:
        logf = open("/kaggle/working/metrics_zero3.csv", "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["step", "loss", "tok_per_s", "lr", "mem_mb"])

    tok_per_step = cfg.T * micro_B * world_size
    for s in range(1, STEPS + 1):
        t0 = time.time()
        # CPU: schedule math (plain floats), then push the LR onto the optimizer
        cur_lr = base_lr * lr_at(s, WARMUP, STEPS)
        for g in opt.param_groups:
            g['lr'] = cur_lr
        # CPU: pull one numpy micro-batch off the stream
        x, d = next(train_stream)
        # CPU->GPU: numpy -> GPU tensors (from_numpy is CPU, .to(device) copies over)
        tokens = torch.from_numpy(x).to(device)
        doc_id = torch.from_numpy(d).to(device)
        # GPU: build the attention/loss masks on the data's device
        attn, loss_mask = make_masks(tokens, doc_id)
        # GPU (+ GPU<->GPU comms): FSDP forward all_gathers each Block's full params
        # just-in-time, runs it, then FREES them -> peak memory ~ one Block
        loss, n = loss_fn(model(tokens, attn), tokens, loss_mask)
        # GPU: drop stale grads
        opt.zero_grad(set_to_none=True)
        # GPU (+ comms): backward gathers params again, computes grads, then
        # reduce_scatters them -> each rank KEEPS ONLY ITS SHARD of the gradients
        loss.backward()
        # GPU: each rank updates ONLY its param shard (no manual all_reduce/broadcast)
        opt.step()
        # CPU: wall-clock for this step (GPU work already synced by the collectives)
        dt = time.time() - t0

        if s % LOG_EVERY == 0:
            # GPU<->GPU: sum (loss*tokens) and tokens across ranks -> true global mean
            stats = torch.tensor([loss.item() * n, float(n)], device=device)
            dist.all_reduce(stats)
            # GPU->CPU: .item() pulls the scalar back to host for printing/logging
            avg_loss = (stats[0] / stats[1]).item()
            if rank == 0:
                # GPU->CPU: read peak memory counter
                mem = torch.cuda.max_memory_allocated() / 1e6
                tps = tok_per_step / max(dt, 1e-9)
                # CPU: write the row to disk
                writer.writerow([s, f"{avg_loss:.4f}", f"{tps:.0f}",
                                 f"{cur_lr:.2e}", f"{mem:.0f}"])
                logf.flush()
                print(f"step {s:4d} | loss {avg_loss:.4f} | {tps:.0f} tok/s | "
                      f"mem {mem:.0f}MB", flush=True)

    if rank == 0:
        logf.close()

    # per-rank holdings: params + grads + optim should ALL be ~1/world_size now
    params_mb, grads_mb, optim_mb = measure_holdings(model, opt)
    print(f"rank {rank} holdings: params {params_mb:.0f}MB grads {grads_mb:.0f}MB "
          f"optim {optim_mb:.0f}MB total {params_mb+grads_mb+optim_mb:.0f}MB", flush=True)
    if rank == 0:
        with open("/kaggle/working/holdings_zero3.csv", "w", newline="") as hf:
            w = csv.writer(hf)
            w.writerow(["component", "mb"])
            w.writerow(["params", f"{params_mb:.2f}"])
            w.writerow(["grads", f"{grads_mb:.2f}"])
            w.writerow(["optimizer", f"{optim_mb:.2f}"])

    dist.barrier()
    print0(rank, "done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# ---- after training, plot / compare (normal notebook cell) ----
#   from plot import plot_holdings, compare
#   plot_holdings("/kaggle/working/holdings_zero3.csv", "/kaggle/working/plots")
#   compare(["/kaggle/working/metrics.csv",
#            "/kaggle/working/metrics_zero1.csv",
#            "/kaggle/working/metrics_zero3.csv"], ["DDP", "ZeRO-1", "ZeRO-3"])
