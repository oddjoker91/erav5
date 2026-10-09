# Session 14 MoE run: same GPT, but every FFN is a 4-expert top-2 MoE, trained from
# scratch with ZeRO-3. Twin of zero3_launch.py -- every change is marked '# MoE'.
#   torchrun --nproc_per_node=2 moe_launch.py
#   diff zero3_launch.py moe_launch.py      # shows exactly what dense -> MoE adds


# read env vars torchrun injects
import os
# csv metrics logging + step timing
import csv
import time
import math
# add the Kaggle dataset folder to the import path
import sys
# for torch.cuda.set_device
import torch
# collectives: init/all_reduce/broadcast
import torch.distributed as dist
# ZeRO-3 engine: shards params + grads + optimizer, gathers layer-by-layer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
# gathering the sharded model back into one full state dict for the checkpoint
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# dataset root holds the loose .py files; add it to the import path
SRC = "/kaggle/input/datasets/gona26/tinystories-data-540m"
# shards + bpe live under this subfolder
DATA = f"{SRC}/"
sys.path.insert(0, SRC)
# this script's folder FIRST, so data_pipeline.py / transformer.py next to it win
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# model/vocab sizes + the data stream factory
from data_pipeline import Config, get_streams
# model + a Block (the unit we shard) + RMSNorm (needs custom init) + helpers
from transformer import RMSNorm, make_masks, loss_fn, lr_at
# MoE: the model + its block (the unit we shard) + routing stats
from moe_pipeline import MoEGPT, MoEBlock, aux_loss, expert_load, param_counts

# MoE: knobs
E, TOP_K = 4, 2           # 4 experts per block, each token uses 2 of them
AUX_COEF = 0.01           # weight of the load-balancing loss


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


@torch.no_grad()
def evaluate(model, val_stream, device, max_batches):
    """Token-weighted mean loss over up to max_batches of the held-out val set,
    summed across ranks -> a true GLOBAL validation loss (perplexity = e^loss)."""
    model.eval()
    tot = torch.zeros(2, device=device)              # [sum loss*tok, sum tok]
    for i, (x, d) in enumerate(val_stream()):
        if i >= max_batches:
            break
        tokens = torch.from_numpy(x).to(device)
        doc_id = torch.from_numpy(d).to(device)
        attn, loss_mask = make_masks(tokens, doc_id)
        loss, n = loss_fn(model(tokens, attn), tokens, loss_mask)
        tot[0] += loss.item() * n
        tot[1] += n
    dist.all_reduce(tot)                             # GPU<->GPU: global sums
    model.train()
    return (tot[0] / tot[1].clamp(min=1)).item()


def main():
    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    print(f"rank {rank}/{world_size} on cuda:{local_rank}  \n", flush=True)

    # ---- data: disjoint blocks per rank (shared seed) ----
    # CPU: Config builds tokenizer; the stream reads shards into numpy (numpy = CPU RAM)
    cfg = Config()
    # guarantee the ~19.6M ("20M") config no matter which data_pipeline is uploaded
    cfg.D, cfg.H, cfg.L = 384, 8, 7
    cfg.T = 256                                    # longer context (fits a whole story)
    cfg.HD, cfg.DFF = cfg.D // cfg.H, cfg.D * 4
    # 64 in BOTH dense and MoE runs: the MoE (top-2 -> ~2x FFN activations) OOMs at 120
    micro_B = 64
    # ---- token budget: the ONE knob; everything below derives from it ----
    TRAIN_TOKENS = 55_000_000                     # unique train tokens loaded into RAM
    # CPU: sets up the numpy batch iterators (no GPU touched here); keep val too
    train_stream, val_stream = get_streams(cfg, shard_dir=f"{DATA}/shards",
                                       micro_B=micro_B, seed=1234,
                                       train_tokens=TRAIN_TOKENS,
                                       val_tokens=None,
                                       rank=rank, world_size=world_size)

    # STEPS = one pass over the budget: TRAIN_TOKENS / (T * micro_B * world_size)
    # (floor -> never spills into a 2nd epoch; adapts to any world_size)
    STEPS = TRAIN_TOKENS // (cfg.T * micro_B * world_size)
    WARMUP, base_lr, LOG_EVERY = 300, 3e-4, 50
    print0(rank, f"budget {TRAIN_TOKENS/1e6:.0f}M tok | {STEPS} steps x "
                 f"{cfg.T}T x {micro_B}B x {world_size} ranks")
    # val ppl on every log line; 20 batches x B x T x ranks (~164k tok on 2 GPUs)
    # keeps eval overhead ~10-15% of training time (50 batches would be ~30%)
    EVAL_EVERY, EVAL_BATCHES = LOG_EVERY, 20
    torch.manual_seed(0)
    # META init: build the model with SHAPES ONLY -> ZERO bytes allocated, so the full
    # model never lands on any single GPU (this is what avoids OOM for huge models).
    with torch.device("meta"):
        model = MoEGPT(cfg, E, TOP_K)                # MoE: was GPT(cfg)
    print0(rank, model_info(model, cfg))
    P_total, P_active = param_counts(model, E, TOP_K)   # MoE: only top-k experts run per token
    print0(rank, f"MoE: {E} experts, top-{TOP_K} | {P_total/1e6:.2f}M total, "
                 f"{P_active/1e6:.2f}M active per token")

    # ---- ZeRO-3: shard params across ranks; FSDP materializes only each rank's 1/N ----
    # GPU: FULL_SHARD = ZeRO-3 (params + grads + optimizer sharded). param_init_fn builds
    # + inits each shard on the GPU, so the whole model is never assembled in one place.
    # (SHARD_GRAD_OP would be ZeRO-2; NO_SHARD would be plain DDP.)
    model = FSDP(
        model,
        # each Block is gathered/freed on its own -> peak memory = ~one Block
        auto_wrap_policy=ModuleWrapPolicy({MoEBlock}),   # MoE: was Block
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        param_init_fn=param_init_fn,
    )
    # CPU setup over GPU-resident sharded params: MUST be built AFTER FSDP so it sees
    # the sharded (flat) params; its m/v state is allocated on GPU at first step, 1/N size
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
    print0(rank, model)

    # one timestamp per run -> every run writes NEW files, never overwrites old ones
    # (only rank 0 writes files, so only rank 0's clock matters)
    RUN_ID = time.strftime("%Y%m%d_%H%M%S")
    OUT_DIR = "/kaggle/working"
    metrics_path = f"{OUT_DIR}/metrics_moe_{RUN_ID}.csv"
    holdings_path = f"{OUT_DIR}/holdings_moe_{RUN_ID}.csv"

    if rank == 0:
        print(f"run {RUN_ID} -> {metrics_path}", flush=True)
        logf = open(metrics_path, "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["step", "tokens", "loss_tokens", "loss", "ppl", "val_loss",
                         "val_ppl", "aux_loss", "load_min", "load_max",   # MoE
                         "grad_norm", "tok_per_s", "lr",
                         "alloc_mb", "peak_mb", "reserved_mb", "gpu_used_mb", "gpu_total_mb"])

    # token counters, built from the ACTUAL batch shapes (not assumed T*B)
    local_seen = 0          # this rank: positions fed to the model (B*T per batch)
    local_trained = 0.0     # this rank: positions that counted in the loss (n)
    win_tok, win_t0 = 0, time.time()   # window since last log -> steady tok/s
    for s in range(1, STEPS + 1):
        # CPU: schedule math (plain floats), then push the LR onto the optimizer
        cur_lr = base_lr * lr_at(s, WARMUP, STEPS)
        for g in opt.param_groups:
            g['lr'] = cur_lr
        # CPU: pull one numpy micro-batch off the stream
        x, d = next(train_stream)
        # CPU: verify every sequence is exactly T long, then count what we really got
        assert x.shape == (micro_B, cfg.T) and d.shape == x.shape, \
            f"step {s}: bad batch shape tok={x.shape} doc={d.shape}"
        local_seen += x.shape[0] * x.shape[1]      # rows x sequence length
        win_tok += x.shape[0] * x.shape[1]
        # CPU->GPU: numpy -> GPU tensors (from_numpy is CPU, .to(device) copies over)
        tokens = torch.from_numpy(x).to(device)
        doc_id = torch.from_numpy(d).to(device)
        # GPU: build the attention/loss masks on the data's device
        attn, loss_mask = make_masks(tokens, doc_id)
        # GPU (+ GPU<->GPU comms): FSDP forward all_gathers each Block's full params
        # just-in-time, runs it, then FREES them -> peak memory ~ one Block
        loss, n = loss_fn(model(tokens, attn), tokens, loss_mask)
        # GPU: running count of positions that actually got a loss (stays on GPU,
        # no .item() here -> no per-step CPU/GPU sync)
        local_trained += n
        # GPU: drop stale grads
        opt.zero_grad(set_to_none=True)
        # GPU (+ comms): backward gathers params again, computes grads, then
        # reduce_scatters them -> each rank KEEPS ONLY ITS SHARD of the gradients
        # MoE: + load-balancing loss so the router doesn't send every token to 1 expert
        aux = aux_loss(model)
        (loss + AUX_COEF * aux).backward()
        # GPU (+ comms): FSDP-aware global grad-norm clip; returns the total norm
        gnorm = model.clip_grad_norm_(1.0)
        # GPU: each rank updates ONLY its param shard (no manual all_reduce/broadcast)
        opt.step()

        if s % LOG_EVERY == 0 or s == STEPS:          # always log the final step too
            # GPU<->GPU: one all_reduce for everything we sum across ranks:
            # [loss*n this step, n this step, total seen, total trained, window tokens]
            stats = torch.tensor([loss.item() * float(n), float(n), float(local_seen),
                                  float(local_trained), float(win_tok),
                                  aux.item()],                       # MoE
                                 device=device, dtype=torch.float64)
            dist.all_reduce(stats)
            # GPU->CPU: .item() pulls the scalars back to host for printing/logging
            avg_loss = (stats[0] / stats[1]).item()
            ppl = math.exp(min(avg_loss, 20))            # train perplexity = e^loss
            seen = int(stats[2].item())                  # global positions fed (all ranks)
            trained = int(stats[3].item())               # global positions scored by loss
            # CPU: steady throughput over the whole window, not one noisy step
            win_dt = time.time() - win_t0
            tps = stats[4].item() / max(win_dt, 1e-9)
            eta_min = (STEPS - s) * (win_dt / LOG_EVERY) / 60
            # MoE: routing health. Share of slots per expert as a multiple of uniform
            # (1/E): 1.00 = perfectly balanced, min near 0 = a dead expert
            aux_avg = stats[5].item() / world_size
            load = expert_load(model).to(device)
            dist.all_reduce(load)
            load = load / world_size * E                      # (L,E)
            load_min, load_max = load.min().item(), load.max().item()

            # GPU memory, 4 levels (each contains the one before), read BEFORE eval so
            # the numbers describe training only:
            #   alloc  = tensors alive right now
            #   peak   = highest tensor usage in this window (reset after each log)
            #   resv   = what PyTorch has claimed from the GPU (incl. its free cache)
            #   used   = GPU's own view (like nvidia-smi): + CUDA context + NCCL buffers
            free_b, total_b = torch.cuda.mem_get_info()
            mem = torch.tensor([torch.cuda.memory_allocated(),
                                torch.cuda.max_memory_allocated(),
                                torch.cuda.memory_reserved(),
                                total_b - free_b, total_b],
                               device=device, dtype=torch.float64) / 1e6
            # GPU<->GPU: MAX across ranks -> report the fullest GPU, not just rank 0
            dist.all_reduce(mem, op=dist.ReduceOp.MAX)
            alloc_mb, peak_mb, resv_mb, used_mb, total_mb = mem.tolist()

            # held-out validation perplexity (the real-world quality signal)
            val_loss = val_ppl = ""
            if s % EVAL_EVERY == 0 or s == STEPS:
                vl = evaluate(model, val_stream, device, EVAL_BATCHES)
                val_loss, val_ppl = f"{vl:.4f}", f"{math.exp(min(vl, 20)):.2f}"

            if rank == 0:
                # CPU: write the row to disk
                writer.writerow([s, seen, trained, f"{avg_loss:.4f}", f"{ppl:.2f}",
                                 val_loss, val_ppl,
                                 f"{aux_avg:.4f}", f"{load_min:.2f}", f"{load_max:.2f}",
                                 f"{float(gnorm):.2f}",
                                 f"{tps:.0f}", f"{cur_lr:.2e}",
                                 f"{alloc_mb:.0f}", f"{peak_mb:.0f}", f"{resv_mb:.0f}",
                                 f"{used_mb:.0f}", f"{total_mb:.0f}"])
                logf.flush()
                msg = (f"step {s:5d}/{STEPS} | {seen/1e6:6.1f}M tok "
                       f"({trained/1e6:.1f}M in loss) | "
                       f"loss {avg_loss:.4f} ppl {ppl:.1f}")
                if val_ppl:
                    msg += f" | val_ppl {val_ppl}"
                msg += f" | aux {aux_avg:.3f} load {load_min:.2f}-{load_max:.2f}"   # MoE
                msg += (f" | gnorm {float(gnorm):.2f} | {tps:.0f} tok/s | "
                        f"mem peak {peak_mb:.0f} / resv {resv_mb:.0f} / "
                        f"gpu {used_mb:.0f} of {total_mb:.0f}MB | eta {eta_min:.0f}m")
                print(msg, flush=True)

            # reset the throughput window AND the peak counter AFTER eval/print, so the
            # next window's tok/s and peak memory describe training steps only
            win_tok, win_t0 = 0, time.time()
            torch.cuda.reset_peak_memory_stats()

    # final token audit: counted tokens must equal the budget math
    total = torch.tensor([float(local_seen)], device=device, dtype=torch.float64)
    dist.all_reduce(total)
    expected = STEPS * cfg.T * micro_B * world_size
    print0(rank, f"token audit: counted {int(total.item()):,} | expected {expected:,} | "
                 f"budget {TRAIN_TOKENS:,}")

    if rank == 0:
        logf.close()

    # ---- save a checkpoint for inference ----
    # Under ZeRO-3 no rank holds the whole model, so FSDP must GATHER it first.
    # FULL_STATE_DICT = reassemble every shard into normal (un-sharded) weights with
    # the original parameter names; rank0_only + offload_to_cpu = only rank 0 keeps
    # the result, in CPU RAM (so no GPU ever holds a second full copy).
    # EVERY rank must call state_dict() -- it's a collective (all ranks send shards).
    save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
        full_state = model.state_dict()
    if rank == 0:
        ckpt_path = f"{OUT_DIR}/ckpt_moe_{RUN_ID}.pt"
        torch.save({
            "model": full_state,
            # the exact shape knobs used here (they were overridden above), so
            # inference rebuilds the SAME architecture without guessing
            "cfg": {"V": cfg.V, "T": cfg.T, "D": cfg.D, "H": cfg.H,
                    "HD": cfg.HD, "DFF": cfg.DFF, "L": cfg.L},
            "moe": {"E": E, "k": TOP_K},                   # MoE
            "run_id": RUN_ID, "steps": STEPS, "train_tokens": TRAIN_TOKENS,
        }, ckpt_path)
        print(f"saved checkpoint -> {ckpt_path}", flush=True)

    # per-rank holdings: params + grads + optim should ALL be ~1/world_size now
    params_mb, grads_mb, optim_mb = measure_holdings(model, opt)
    print(f"rank {rank} holdings: params {params_mb:.0f}MB grads {grads_mb:.0f}MB "
          f"optim {optim_mb:.0f}MB total {params_mb+grads_mb+optim_mb:.0f}MB", flush=True)
    if rank == 0:
        with open(holdings_path, "w", newline="") as hf:
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

# ---- after BOTH runs, compare (normal notebook cell) ----
#   from plot_moe import compare
#   compare("/kaggle/working/metrics_zero3_<RUN_ID>.csv",
#           "/kaggle/working/metrics_moe_<RUN_ID>.csv", "/kaggle/working/plots")
