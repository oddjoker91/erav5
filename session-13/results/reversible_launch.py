
"""
reversible_launch.py -- trainer for the 20M model (single-GPU or multi-GPU DDP).

Runs the SAME ~20M-param model three ways so you can fill in the assignment table:

  1) baseline (store activations)     -- fix a batch you can run
  2) reversible, SAME batch           -- isolate the reversibility effect
  3) reversible, MAX batch            -- spend the freed memory on throughput

Single GPU:
    python reversible_launch.py --micro-B 16 --tag baseline

Multi-GPU (data-parallel across all GPUs) -- launch with torchrun:
    torchrun --nproc_per_node=2 reversible_launch.py --reversible --variant euler \
             --micro-B 16 --tag rev_same

NOTE on ZeRO-3: FSDP is NOT used here. It shards params/optimizer (useless for a
tiny 20M model) and its just-in-time param hooks are bypassed by the reversible
custom autograd.Function. We use plain data parallelism instead: each GPU holds a
full replica over a disjoint data slice, and grads are averaged by an explicit
all_reduce (the DDP wrapper's bucketed reduction can miss the reversible grads,
which are produced inside a nested backward).

Each run writes metrics_<tag>.csv and prints a final summary line
(final loss | tok/s | peak memory) that drops straight into README.md.

Needs token shards first:  python data_pipeline.py
(default --shard-dir is the Kaggle dataset shards, so the flag is optional there)
"""

import argparse
import csv
import math
import os
import time

import torch
import torch.distributed as dist

from data_pipeline import Config, get_streams
from learn_pipeline import GPT, make_masks, loss_fn, lr_at
from reversible_pipeline import ReversibleGPT

# Kaggle dataset root holding the token shards (matches zero3_launch.py); the
# default --shard-dir points here so you can omit the flag on Kaggle.
SRC = "/kaggle/input/datasets/gona26/tinystories-data-540m"
DEFAULT_SHARDS = f"{SRC}/shards"


def ensure_bpe(src=f"{SRC}/bpe", dst="data/bpe"):
    """Tokenizer/Config hard-code data/bpe; symlink it to the dataset so the run
    is self-contained (no separate setup cell needed). Idempotent + rank-safe."""
    if os.path.exists(dst) or not os.path.isdir(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.symlink(src, dst)
    except FileExistsError:                      # another rank won the race
        pass


def setup_dist():
    """Join the process group if launched by torchrun; else run single-process.

    Returns (rank, world_size, local_rank, distributed?).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl",
                                device_id=torch.device(f"cuda:{local_rank}"))
        return dist.get_rank(), dist.get_world_size(), local_rank, True
    return 0, 1, 0, False


def build_config():
    # pin the ~19.6M ("20M") config regardless of data_pipeline defaults
    cfg = Config()
    cfg.D, cfg.H, cfg.L = 384, 8, 7
    cfg.T = 256
    cfg.HD, cfg.DFF = cfg.D // cfg.H, cfg.D * 4
    return cfg


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def model_info(model, cfg):
    P = n_params(model)
    emb = 2 * cfg.V * cfg.D                     # tok_emb + head dominate a tiny model
    return (f"model: {P/1e6:.2f}M params | L={cfg.L} D={cfg.D} H={cfg.H} "
            f"T={cfg.T} V={cfg.V} | embed+head ~{emb/1e6:.1f}M ({100*emb/P:.0f}%)")


def measure_holdings(model, opt):
    # live bytes held per GPU: full replica here (NOT sharded, unlike ZeRO-3)
    mb = lambda ts: sum(t.numel() * t.element_size() for t in ts) / 1e6
    params_mb = mb(list(model.parameters()))
    grads_mb = mb([p.grad for p in model.parameters() if p.grad is not None])
    optim_mb = mb([t for st in opt.state.values() for t in st.values() if torch.is_tensor(t)])
    return params_mb, grads_mb, optim_mb


@torch.no_grad()
def recon_error(model, tokens, mask):
    """End-to-end reversibility drift: run the block stack forward, invert it, and
    measure how far reconstruction misses the input (exact rev -> ~1e-5 in fp32)."""
    pos = torch.arange(tokens.shape[1], device=tokens.device)
    x = model.tok_emb(tokens) + model.pos_emb(pos)
    x1 = x2 = x
    for blk in model.blocks:
        x1, x2 = blk.forward(x1, x2, mask)
    for blk in reversed(model.blocks):
        x1, x2 = blk.inverse(x1, x2, mask)
    return (x1 - x).abs().max().item() + (x2 - x).abs().max().item()


@torch.no_grad()
def evaluate(model, val_stream, device, max_batches, distributed=False):
    """Token-weighted mean val loss over up to max_batches (perplexity = e^loss).

    Under DDP, sums (loss*tok, tok) across ranks -> a true GLOBAL val loss.
    """
    model.eval()
    tot, ntok = 0.0, 0
    for i, (x, d) in enumerate(val_stream()):
        if i >= max_batches:
            break
        tokens = torch.from_numpy(x).to(device)
        doc_id = torch.from_numpy(d).to(device)
        attn, loss_mask = make_masks(tokens, doc_id)
        loss, n = loss_fn(model(tokens, attn), tokens, loss_mask)
        tot += loss.item() * n
        ntok += n
    model.train()
    if distributed:
        t = torch.tensor([tot, float(ntok)], device=device)
        dist.all_reduce(t)
        return (t[0] / t[1].clamp(min=1)).item()
    return tot / max(1, ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reversible", action="store_true",
                    help="use the reversible stack (else the baseline GPT)")
    ap.add_argument("--variant", choices=["euler", "midpoint"], default="euler")
    ap.add_argument("--micro-B", type=int, default=16, help="batch (sequences) per step")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--train-tokens", type=int, default=50_000_000)
    ap.add_argument("--shard-dir", default=DEFAULT_SHARDS,
                    help="token-shard folder (defaults to the Kaggle dataset shards)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    rank, world_size, local_rank, distributed = setup_dist()
    is0 = rank == 0
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)      # SAME on every rank -> identical model replicas

    ensure_bpe()                      # link data/bpe -> dataset before Tokenizer loads
    cfg = build_config()
    # each rank gets a DISJOINT slice of blocks (shared seed + rank stride)
    train_stream, val_stream = get_streams(
        cfg, shard_dir=args.shard_dir, micro_B=args.micro_B, seed=1234,
        train_tokens=args.train_tokens, val_tokens=None,
        rank=rank, world_size=world_size)

    # GLOBAL tokens per optimizer step (all ranks) -> steps to cover the budget
    tok_per_step = cfg.T * args.micro_B * args.accum * world_size
    STEPS = math.ceil(args.train_tokens / tok_per_step)

    if args.reversible:
        model = ReversibleGPT(cfg, variant=args.variant).to(device)
        kind = f"reversible/{args.variant}"
        fn_evals = 2 if args.variant == "euler" else 3   # F,G vs G,F,G per block
    else:
        model = GPT(cfg).to(device)
        kind = "baseline"
        fn_evals = 2
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if is0:
        print(model_info(model, cfg), flush=True)
        print(f"{kind} | world_size={world_size} micro_B={args.micro_B} "
              f"accum={args.accum} -> global batch "
              f"{args.micro_B * args.accum * world_size} | {fn_evals} fn-evals/block "
              f"| STEPS={STEPS} | budget {args.train_tokens/1e6:.0f}M tok | {device}",
              flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if is0:
        logf = open(f"metrics_{args.tag}.csv", "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["step", "tokens", "loss_tokens", "loss", "ppl", "val_loss",
                         "val_ppl", "grad_norm", "tok_per_s", "lr", "alloc_mb",
                         "peak_mb", "reserved_mb", "gpu_used_mb", "gpu_total_mb",
                         "recon_err"])

    # token counters from ACTUAL batch shapes; window counters -> steady tok/s
    local_seen = 0
    local_trained = 0.0
    win_tok, win_t0 = 0, time.time()
    tps_hist = []
    final_loss = float("nan")
    last_recon = ""
    t_train = time.time()
    for s in range(1, STEPS + 1):
        cur_lr = args.lr * lr_at(s, args.warmup, STEPS)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        run_loss, step_tok = 0.0, 0
        for _ in range(args.accum):
            x, d = next(train_stream)
            local_seen += x.shape[0] * x.shape[1]
            win_tok += x.shape[0] * x.shape[1]
            tokens = torch.from_numpy(x).to(device)
            doc_id = torch.from_numpy(d).to(device)
            attn, loss_mask = make_masks(tokens, doc_id)
            loss, n = loss_fn(model(tokens, attn), tokens, loss_mask)
            (loss / args.accum).backward()
            run_loss += loss.item() * n
            step_tok += n
            local_trained += n

        if distributed:
            # manual DDP: average grads across ranks (the reversible custom
            # autograd.Function bypasses DDP's bucketed hooks, so reduce by hand)
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world_size
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % args.log_every == 0 or s == STEPS:
            # one reduce for: sum loss*tok, sum tok, global seen, global trained, window tok
            red = torch.tensor([run_loss, float(step_tok), float(local_seen),
                                float(local_trained), float(win_tok)],
                               device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(red)
            avg_loss = (red[0] / red[1].clamp(min=1)).item()
            seen = int(red[2].item())
            trained = int(red[3].item())
            final_loss = avg_loss
            ppl = math.exp(min(avg_loss, 20))

            win_dt = time.time() - win_t0
            tps = red[4].item() / max(win_dt, 1e-9)       # steady window throughput
            tps_hist.append(tps)
            eta_min = (STEPS - s) * (win_dt / args.log_every) / 60

            # 4 memory levels (each contains the one before), + GPU total; MAX over ranks
            if device.type == "cuda":
                free_b, total_b = torch.cuda.mem_get_info()
                mem = torch.tensor([torch.cuda.memory_allocated(),
                                    torch.cuda.max_memory_allocated(),
                                    torch.cuda.memory_reserved(),
                                    total_b - free_b, total_b],
                                   device=device, dtype=torch.float64) / 1e6
                if distributed:
                    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
                alloc_mb, peak_mb, resv_mb, used_mb, total_mb = mem.tolist()
            else:
                alloc_mb = peak_mb = resv_mb = used_mb = total_mb = 0.0

            val_loss = val_ppl = ""
            if s % args.eval_every == 0 or s == STEPS:
                vl = evaluate(model, val_stream, device, args.eval_batches, distributed)
                val_loss, val_ppl = f"{vl:.4f}", f"{math.exp(min(vl, 20)):.2f}"

            # reversibility-specific: confirm inverse() still reconstructs the input
            recon = ""
            if args.reversible and (s % args.eval_every == 0 or s == STEPS):
                recon = f"{recon_error(model, tokens, attn):.2e}"
                last_recon = recon

            if is0:
                writer.writerow([s, seen, trained, f"{avg_loss:.4f}", f"{ppl:.2f}",
                                 val_loss, val_ppl, f"{float(gnorm):.2f}",
                                 f"{tps:.0f}", f"{cur_lr:.2e}",
                                 f"{alloc_mb:.0f}", f"{peak_mb:.0f}", f"{resv_mb:.0f}",
                                 f"{used_mb:.0f}", f"{total_mb:.0f}", recon])
                logf.flush()
                msg = (f"step {s:5d}/{STEPS} | {seen/1e6:5.1f}M tok "
                       f"({trained/1e6:.1f}M in loss) | loss {avg_loss:.4f} ppl {ppl:.1f}")
                if val_ppl:
                    msg += f" | val_ppl {val_ppl}"
                msg += (f" | gnorm {float(gnorm):.2f} | {tps:.0f} tok/s | "
                        f"mem peak {peak_mb:.0f} / resv {resv_mb:.0f} / "
                        f"gpu {used_mb:.0f} of {total_mb:.0f}MB | eta {eta_min:.0f}m")
                if recon:
                    msg += f" | recon {recon}"
                print(msg, flush=True)

            win_tok, win_t0 = 0, time.time()     # reset window after logging

    # final token audit: counted positions vs the budget math
    total_seen = torch.tensor([float(local_seen)], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(total_seen)
    expected = STEPS * cfg.T * args.micro_B * args.accum * world_size

    total_s = time.time() - t_train
    avg_tps = (sum(tps_hist) / len(tps_hist)) if tps_hist else 0.0
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6
               if device.type == "cuda" else 0.0)
    if distributed:
        mp = torch.tensor([peak_mb], device=device)
        dist.all_reduce(mp, op=dist.ReduceOp.MAX)
        peak_mb = mp.item()

    # per-GPU holdings (full replica here -- contrast with ZeRO-3's 1/N shards)
    params_mb, grads_mb, optim_mb = measure_holdings(model, opt)

    if is0:
        logf.close()
        with open(f"holdings_{args.tag}.csv", "w", newline="") as hf:
            w = csv.writer(hf)
            w.writerow(["component", "mb"])
            w.writerow(["params", f"{params_mb:.2f}"])
            w.writerow(["grads", f"{grads_mb:.2f}"])
            w.writerow(["optimizer", f"{optim_mb:.2f}"])
        print(f"\ntoken audit: counted {int(total_seen.item()):,} | "
              f"expected {expected:,} | budget {args.train_tokens:,}", flush=True)
        print("==== SUMMARY (" + kind + ") ====", flush=True)
        print(f"tag           : {args.tag}", flush=True)
        print(f"final loss    : {final_loss:.4f}  (ppl {math.exp(min(final_loss,20)):.2f})",
              flush=True)
        print(f"speed         : {avg_tps:.0f} tok/s (avg, global)", flush=True)
        print(f"peak memory   : {peak_mb:.0f} MB (per GPU)", flush=True)
        print(f"holdings      : params {params_mb:.0f} + grads {grads_mb:.0f} + "
              f"optim {optim_mb:.0f} = {params_mb+grads_mb+optim_mb:.0f} MB "
              f"(per GPU, full replica)", flush=True)
        if args.reversible:
            print(f"recon error   : {last_recon or 'n/a'}  (exact-reversibility drift)",
                  flush=True)
            print(f"fn-evals/blk  : {fn_evals}  (variant {args.variant})", flush=True)
        print(f"wall time     : {total_s/60:.1f} min  ({STEPS} steps)", flush=True)
        print(f"batch         : micro_B={args.micro_B} accum={args.accum} "
              f"world_size={world_size} global={args.micro_B*args.accum*world_size}",
              flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
