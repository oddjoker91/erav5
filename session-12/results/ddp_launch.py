
# read env vars torchrun injects
import os
# csv metrics logging + step timing
import csv
import time
# add the Kaggle dataset folder to the import path
import sys
# for torch.cuda.set_device
import torch
# collectives: init/barrier/etc.
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
    # join the group; device_id tells NCCL our GPU (mutes the "guessing" warning)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    # (global id, total procs, gpu index)
    return dist.get_rank(), dist.get_world_size(), local_rank


def print0(rank, *args):
    # only the first process prints
    if rank == 0:
        # forward args like the real print()
        print(*args)


def param_checksum(model):
    # crude weight fingerprint; abs-sum so opposite-sign drift can't cancel out
    return sum(p.detach().abs().sum().item() for p in model.parameters())


def measure_holdings(model, opt):
    # actual bytes from live tensors (element_size() -> dtype-aware, no assumptions)
    mb = lambda ts: sum(t.numel() * t.element_size() for t in ts) / 1e6
    params_mb = mb(list(model.parameters()))
    grads_mb = mb([p.grad for p in model.parameters() if p.grad is not None])
    # optimizer state = Adam's m, v (and step) tensors, straight from opt.state
    optim_mb = mb([t for st in opt.state.values() for t in st.values() if torch.is_tensor(t)])
    return params_mb, grads_mb, optim_mb


def main():
    # every process runs this; only the numbers differ
    rank, world_size, local_rank = setup()
    # plain print -> see BOTH ranks
    print(f"rank {rank}/{world_size} on cuda:{local_rank} \n")

    # ---- 1.2: rank-aware data ----
    cfg = Config()
    micro_B = 32                                  # real data now -> big micro-batch
    # SAME seed on every rank + rank/world -> disjoint block partition (no overlap)
    train_stream, _ = get_streams(cfg, shard_dir=f"{DATA}/shards",
                                  micro_B=micro_B, seed=1234,
                                  rank=rank, world_size=world_size)
    # pull the first batch this rank would train on
    x, d = next(train_stream)
    # hash it; the two ranks MUST print different hashes
    print(f"rank {rank} first-batch hash: {hash(x.tobytes())}")

    # ---- 2.1 + 2.3: manual DDP + correct cross-rank logging ----
    device = torch.device(f"cuda:{local_rank}")
    # step budget + schedule + how often to log/eval
    STEPS, WARMUP, base_lr, LOG_EVERY = 1500, 200, 3e-4, 50
    # SAME seed on every rank -> identical initial weights (fair start)
    torch.manual_seed(0)
    model = GPT(cfg).to(device)
    # guarantee identical start: everyone copies rank 0's weights
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)

    # rank 0 writes metrics.csv (writable working dir) for plot.py
    if rank == 0:
        logf = open("/kaggle/working/metrics.csv", "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["step", "loss", "tok_per_s", "lr", "mem_mb"])

    tok_per_step = cfg.T * micro_B * world_size   # tokens the whole job processes per step
    for s in range(1, STEPS + 1):
        t0 = time.time()
        # warmup + cosine decay schedule
        cur_lr = base_lr * lr_at(s, WARMUP, STEPS)
        for g in opt.param_groups:
            g['lr'] = cur_lr
        x, d = next(train_stream)
        tokens = torch.from_numpy(x).to(device)
        doc_id = torch.from_numpy(d).to(device)
        attn, loss_mask = make_masks(tokens, doc_id)
        loss, n = loss_fn(model(tokens, attn), tokens, loss_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # DDP sync: average each gradient across ranks BEFORE stepping
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)       # SUM grads over ranks
                p.grad /= world_size          # -> mean
        opt.step()                            # same averaged update on every rank
        dt = time.time() - t0

        if s % LOG_EVERY == 0:
            # 2.3: TRUE global loss = sum(loss*tokens)/sum(tokens) across ranks
            # (averaging each rank's average is wrong when token counts differ)
            stats = torch.tensor([loss.item() * n, float(n)], device=device)
            dist.all_reduce(stats)            # ALL ranks must call this collective
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

    # measure ACTUAL per-GPU holdings from the live objects (grads + optim now exist)
    if rank == 0:
        params_mb, grads_mb, optim_mb = measure_holdings(model, opt)
        print(f"per-GPU holdings: params {params_mb:.0f}MB grads {grads_mb:.0f}MB "
              f"optim {optim_mb:.0f}MB total {params_mb+grads_mb+optim_mb:.0f}MB", flush=True)
        with open("/kaggle/working/holdings.csv", "w", newline="") as hf:
            w = csv.writer(hf)
            w.writerow(["component", "mb"])
            w.writerow(["params", f"{params_mb:.2f}"])
            w.writerow(["grads", f"{grads_mb:.2f}"])
            w.writerow(["optimizer", f"{optim_mb:.2f}"])

    # wall: no rank passes until all reach it
    dist.barrier()
    # rank 0 only -> printed once
    print0(rank, "all ranks synced")
    # release nccl comms; pairs with init
    dist.destroy_process_group()


# entry point torchrun runs in each process
if __name__ == "__main__":
    # start the program
    main()

# ---- after training, plot the metrics (run in a NORMAL notebook cell) ----
#   import sys; sys.path.append(BASE)      # BASE holds plot.py once you upload it
#   from plot import plot_metrics
#   plot_metrics("/kaggle/working/metrics.csv", "/kaggle/working/plots")
#   from IPython.display import Image; Image("/kaggle/working/plots/training.png")
