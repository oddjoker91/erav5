
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

# dataset root holds the loose .py files; add it to the import path
SRC = "/kaggle/input/datasets/gona26/tinystories-slm"
# shards + bpe live under this subfolder
DATA = f"{SRC}/tinystories_data"
sys.path.insert(0, SRC)
# model/vocab sizes + the data stream factory
from data_pipeline import Config, get_streams
# the model + mask/loss helpers
from learn_pipeline import GPT, make_masks, loss_fn, lr_at


def setup():
    # GPU index for THIS process (str -> int)
    local_rank = int(os.environ["LOCAL_RANK"])
    # pin this process to that one GPU (must happen before init)
    torch.cuda.set_device(local_rank)
    # join the group; device_id tells NCCL our GPU
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return dist.get_rank(), dist.get_world_size(), local_rank


def print0(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def model_info(model, cfg):
    P = sum(p.numel() for p in model.parameters())
    emb = 2 * cfg.V * cfg.D
    return (f"model: {P/1e6:.2f}M params | L={cfg.L} D={cfg.D} H={cfg.H} "
            f"T={cfg.T} V={cfg.V} | embed+head ~{emb/1e6:.1f}M ({100*emb/P:.0f}%)")


def measure_holdings(model, opt):
    # actual bytes from live tensors (element_size() -> dtype-aware)
    mb = lambda ts: sum(t.numel() * t.element_size() for t in ts) / 1e6
    params_mb = mb(list(model.parameters()))
    grads_mb = mb([p.grad for p in model.parameters() if p.grad is not None])
    # optimizer state now covers only THIS rank's shard -> smaller
    optim_mb = mb([t for st in opt.state.values() for t in st.values() if torch.is_tensor(t)])
    return params_mb, grads_mb, optim_mb


def main():
    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    print(f"rank {rank}/{world_size} on cuda:{local_rank}", flush=True)

    # ---- data: disjoint blocks per rank (shared seed) ----
    # CPU: Config + the stream read shards into numpy (numpy is always CPU RAM)
    cfg = Config()
    micro_B = 32
    train_stream, _ = get_streams(cfg, shard_dir=f"{DATA}/shards",
                                  micro_B=micro_B, seed=1234,
                                  rank=rank, world_size=world_size)

    STEPS, WARMUP, base_lr, LOG_EVERY = 1500, 100, 3e-4, 50
    torch.manual_seed(0)
    # GPT(cfg) builds + inits all weights on CPU; .to(device) copies them to THIS GPU
    model = GPT(cfg).to(device)
    print0(rank, model_info(model, cfg))
    # GPU->GPU: copy rank 0's weights to every rank so all replicas start identical
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # ---- ZeRO-1: partition params across ranks; optimizer holds only MY shard ----
    # CPU: a Python list of references (the weight tensors themselves stay on GPU)
    params = list(model.parameters())
    # CPU: plain Python ints, one owner rank per param (no tensors involved)
    owner = [i % world_size for i in range(len(params))]
    # CPU: filter to my tensors (still just references; data stays on GPU)
    my_params = [p for i, p in enumerate(params) if owner[i] == rank]
    # CPU setup: builds the optimizer object; its m/v state is allocated on GPU
    # lazily at the FIRST opt.step(), sized to my_params only (the memory saving)
    opt = torch.optim.AdamW(my_params, lr=base_lr)

    if rank == 0:
        logf = open("/kaggle/working/metrics_zero1.csv", "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["step", "loss", "tok_per_s", "lr", "mem_mb"])

    tok_per_step = cfg.T * micro_B * world_size
    for s in range(1, STEPS + 1):
        t0 = time.time()
        cur_lr = base_lr * lr_at(s, WARMUP, STEPS)
        for g in opt.param_groups:
            g['lr'] = cur_lr
        x, d = next(train_stream)
        # CPU->GPU: batch is numpy (CPU); move it onto this rank's GPU
        tokens = torch.from_numpy(x).to(device)
        doc_id = torch.from_numpy(d).to(device)
        attn, loss_mask = make_masks(tokens, doc_id)
        loss, n = loss_fn(model(tokens, attn), tokens, loss_mask)
        # zero ALL grads (full model still does a full backward)
        for p in params:
            p.grad = None
        loss.backward()
        # gradients are still FULLY averaged across ranks (same as DDP)
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world_size
        # each rank updates ONLY its owned params (its optimizer shard)
        opt.step()
        # re-sync: each param's OWNER broadcasts its freshly-updated weights to all
        # (init broadcast used src=0; here src=owner[i] -> the rank that just stepped it)
        for i, p in enumerate(params):
            dist.broadcast(p.data, src=owner[i])
        dt = time.time() - t0

        if s % LOG_EVERY == 0:
            # true global loss across ranks (sum loss*tokens / sum tokens)
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

    # per-rank holdings: optimizer should be ~1/world_size of the DDP value
    params_mb, grads_mb, optim_mb = measure_holdings(model, opt)
    print(f"rank {rank} holdings: params {params_mb:.0f}MB grads {grads_mb:.0f}MB "
          f"optim {optim_mb:.0f}MB total {params_mb+grads_mb+optim_mb:.0f}MB", flush=True)
    if rank == 0:
        with open("/kaggle/working/holdings_zero1.csv", "w", newline="") as hf:
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
#   from plot import plot_metrics, plot_holdings, compare
#   plot_holdings("/kaggle/working/holdings_zero1.csv", "/kaggle/working/plots")
#   compare(["/kaggle/working/metrics.csv", "/kaggle/working/metrics_zero1.csv"],
#           ["DDP", "ZeRO-1"])
