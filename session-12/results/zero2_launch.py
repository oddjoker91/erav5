
# read env vars torchrun injects
import os
# csv metrics logging + step timing
import csv
import time
# add the Kaggle dataset folder to the import path
import sys
# for torch.cuda.set_device
import torch
# collectives: init/reduce/broadcast
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
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            device_id=torch.device(f"cuda:{local_rank}"))
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
    # ACTUAL bytes from live tensors (element_size() -> real dtype, no formula)
    mb = lambda ts: sum(t.numel() * t.element_size() for t in ts) / 1e6
    params_mb = mb(list(model.parameters()))                          # FULL (replicated)
    grads_mb = mb([p.grad for p in model.parameters() if p.grad is not None])  # only owned -> 1/N
    optim_mb = mb([t for st in opt.state.values() for t in st.values() if torch.is_tensor(t)])
    return params_mb, grads_mb, optim_mb


def main():
    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    print(f"rank {rank}/{world_size} on cuda:{local_rank}", flush=True)

    cfg = Config()
    micro_B = 32
    train_stream, _ = get_streams(cfg, shard_dir=f"{DATA}/shards",
                                  micro_B=micro_B, seed=1234,
                                  rank=rank, world_size=world_size)

    STEPS, WARMUP, base_lr, LOG_EVERY = 1500, 100, 3e-4, 50
    torch.manual_seed(0)
    model = GPT(cfg).to(device)
    print0(rank, model_info(model, cfg))
    for p in model.parameters():
        dist.broadcast(p.data, src=0)                    # identical start on every rank

    # ---- ZeRO-2: params FULL, but grads + optimizer SHARDED ----
    params = list(model.parameters())
    owner = [i % world_size for i in range(len(params))]  # which rank owns each param
    my_params = [p for i, p in enumerate(params) if owner[i] == rank]
    opt = torch.optim.AdamW(my_params, lr=base_lr)        # optimizer over my shard only

    if rank == 0:
        logf = open("/kaggle/working/metrics_zero2.csv", "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["step", "loss", "tok_per_s", "lr", "mem_mb"])
    tok_per_step = cfg.T * micro_B * world_size

    for s in range(1, STEPS + 1):
        t0 = time.time()
        cur_lr = base_lr * lr_at(s, WARMUP, STEPS)
        for g in opt.param_groups:
            g['lr'] = cur_lr
        x, d = next(train_stream)
        tokens = torch.from_numpy(x).to(device)
        doc_id = torch.from_numpy(d).to(device)
        attn, loss_mask = make_masks(tokens, doc_id)
        loss, n = loss_fn(model(tokens, attn), tokens, loss_mask)
        for p in params:
            p.grad = None
        loss.backward()
        # ZeRO-2 grad sharding: REDUCE each grad to its owner, then non-owners FREE it
        # (ZeRO-1 used all_reduce -> every rank kept the full grad; here grad memory shrinks)
        for i, p in enumerate(params):
            if p.grad is not None:
                dist.reduce(p.grad, dst=owner[i])         # sum grad onto the owner rank
                if owner[i] == rank:
                    p.grad /= world_size                  # owner keeps the averaged grad
                else:
                    p.grad = None                         # non-owner drops it -> sharded grads
        opt.step()                                        # each owner updates its params
        for i, p in enumerate(params):
            dist.broadcast(p.data, src=owner[i])          # re-sync full params
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
    # holdings: params FULL, grads + optim ~ 1/world_size
    params_mb, grads_mb, optim_mb = measure_holdings(model, opt)
    print(f"rank {rank} holdings: params {params_mb:.0f}MB grads {grads_mb:.0f}MB "
          f"optim {optim_mb:.0f}MB total {params_mb+grads_mb+optim_mb:.0f}MB", flush=True)
    if rank == 0:
        with open("/kaggle/working/holdings_zero2.csv", "w", newline="") as hf:
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
