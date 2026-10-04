# Session 13 — Reversible Transformers

Train a ~20M LLM on ~55M tokens, then again with reversible layers at a large batch. Compare loss, speed and memory.

**Hardware:** Kaggle, 2× Tesla T4 (15.6 GB each), launched with `torchrun --nproc_per_node=2`. **Context:** 256 tokens.

## Dataset

| | |
|---|---|
| Corpus | **TinyStories**: short, simple English stories, one story per `.txt` file |
| Source | Kaggle dataset `gona26/tinystories-data-540m` (raw text, BPE files, pre-built shards) |
| Tokenizer | Byte-level BPE trained on the corpus, 4,000 learned tokens + `<pad>` + `<eos>` → **V = 4,002** |
| Pre-tokenized | Tokenized once into `uint16` token shards (1M tokens each) with matching `uint32` doc-id shards |
| Train | First **55 shards = 55,000,000 tokens** → 214,843 blocks of 256 |
| Val | 6 shards = **5,616,737 tokens** → 21,940 blocks (held-out stories) |

**How it's fed:**
- Each story ends with `<eos>`, and stories are packed back to back into 256-token blocks (a TinyStory is ~200 tokens).
- Doc-id shards build a **same-document causal mask**, so tokens never attend across story boundaries, and the loss skips cross-document targets.
- Blocks don't overlap and are shuffled with one shared seed. Rank *r* takes `blocks[r::2]`, so the two GPUs see disjoint data.
- Every run covers the same ~55M-token budget in **one pass (no repeated data)**, so only batch size and the number of steps change between runs.

## Results

| | Baseline | Baseline | Reversible |
|---|---|---|---|
| Batch (sequences per GPU) | 64 | 120 | 512 |
| Global batch (× 2 GPUs) | 128 | 240 | 1,024 |
| Tokens / step (global batch × 256) | 32,768 | 61,440 | 262,144 |
| Steps | 1,678 | 895 | 210 |
| Tokens seen | 54.98M | 54.99M | 55.05M |
| **Final train loss** | 2.184 | 2.567 | 3.901 |
| **Final val loss** | **2.172** | **2.553** | **3.895** |
| Final val perplexity | 8.77 | 12.85 | 49.16 |
| Final grad norm | 0.46 | 0.35 | 0.11 |
| **Speed (tokens/s)** | **39.3k** | **39.6k** | **24.7k** |
| Wall time (approx.) | 23 min | 23 min | 37 min |
| **Peak memory** | **7.5 GB** | **14.0 GB** | **11.2 GB** |
| Reserved memory | 7.9 GB | 14.6 GB | 14.3 GB |
| GPU used (of 15.6 GB) | 8.1 GB | 14.9 GB | 14.6 GB |
| Activation memory per sequence | 115 MB | 116 MB | **21 MB** |
| Rebuild error | — | — | 2.1e-5 |

*Every sequence is 256 tokens. Memory figures are per GPU. Activation memory per sequence = (peak − allocated weights/optimizer) ÷ per-GPU batch.*

## Memory held by the model (per GPU)

| Component | Baseline B=64 | Baseline B=120 | Reversible B=512 |
|---|---|---|---|
| Params | 39.4 MB | 39.4 MB | 78.8 MB |
| Grads | 39.4 MB | 39.4 MB | 78.8 MB |
| Adam (m + v) | 78.8 MB | 78.8 MB | 157.5 MB |
| **Model states total** | **158 MB** | **158 MB** | **315 MB** |
| Allocated at rest | 180 MB | 186 MB | 370 MB |
| **Peak** | **7,544 MB** | **14,049 MB** | **11,195 MB** |
| Activations (peak − rest) | 7,364 MB | 13,863 MB | 10,825 MB |
| Model states as % of peak | 2.1% | 1.1% | 2.8% |

Reversible run: 78.8 MB ÷ 4 bytes ≈ 19.7M params, so 4 + 4 + 8 = **16 bytes per param** (fp32). The baseline runs were logged as `zero3` and hold exactly half, a sharded copy. Either way, model states are tiny; **activations are ~97–99% of peak**.

## Finding the max batch

| Model | Batch | Result | Peak memory (of 15.6 GB) |
|---|---|---|---|
| Baseline | 64 | ✅ trained | 7.5 GB |
| Baseline | 128 | ❌ out of memory (CUDA OOM, no step completed) | — |
| Baseline | **120** | ✅ trained, **max for baseline** | 14.0 GB |
| Reversible | **512** | ✅ trained, **max tested for reversible** | 11.2 GB |
| Reversible | 768 | ❌ out of memory (CUDA OOM, no step completed) | — |

*Batch = per-GPU micro-batch (2 GPUs, so the global batch is 2×).*

Both failures are a `torch.OutOfMemoryError` on both ranks before the first step finished:

- **Baseline 128:** tried to allocate 500 MiB with 457 MiB free (13.39 GiB already allocated). 128 × 115 MB ≈ 14.7 GB of activations + model states + allocator overhead doesn't fit.
- **Reversible 768:** tried to allocate 1.12 GiB with 1.09 GiB free (11.92 GiB already allocated). Reversibility removes the stored per-layer activations, but the working memory for one layer plus the output logits still grows with batch size.

So the reversible max lies between 512 and 768. 21 MB per sequence suggests ~700 might fit, but that wasn't tested.

## Train loss vs tokens

| Tokens | Baseline B=64 | Baseline B=120 | Reversible B=512 |
|---|---|---|---|
| ~13M | 3.37 | 4.15 | 4.99 |
| ~26M | 2.54 | 3.26 | 4.19 |
| ~39M | 2.28 | 2.70 | 3.98 |
| ~52M | 2.21 | 2.52 | 3.91 |

## Variant used

**Symplectic Euler** (RevNet coupling):

```
forward:  q = q + Attn(p)      backward:  p = p - MLP(q)
          p = p + MLP(q)                  q = q - Attn(p)
```

Each layer's input is rebuilt during backward instead of stored. A plain residual `x = x + f(x)` can't do this, because `x` appears on both sides when you try to undo it.

## Findings

1. **Memory:** reversible needs **5.5× less** activation memory per sequence (21 vs 115 MB).
2. **Max batch:** baseline 128 ran out of memory and 120 worked. Reversible 768 also ran out of memory and 512 worked, so reversible fits **4.3× the baseline's max batch**.
3. **Speed:** reversible is **37% slower**, because every layer is recomputed in backward.
4. **A bigger batch ≠ faster:** batch 64 → 120 gave the same speed. The T4 is already full at 64.
5. **Loss follows steps:** with a fixed token budget, a bigger batch means fewer updates (1,678 → 895 → 210 steps gives val loss 2.17 → 2.55 → 3.90).
6. **Warmup** was 300 steps for the baseline (18% of the B=64 run, 34% of the B=120 run) and 20 steps (10%) for reversible B=512.

**Takeaway:** reversibility saves memory as promised. For a 20M model on T4s, though, the GPU is already busy, so it costs speed and the larger batch costs loss.

## Summary

| Metric | Baseline B=64 | Baseline B=120 (max) | Reversible B=512 |
|---|---|---|---|
| Val loss | **2.17** | 2.55 | 3.90 |
| Speed | 39.3k tok/s | 39.6k tok/s | 24.7k tok/s |
| Peak memory | 7.5 GB | 14.0 GB | 11.2 GB |
| Memory / sequence | 115 MB | 116 MB | **21 MB** |
| Model states | 158 MB | 158 MB | 315 MB |
| Steps | 1,678 | 895 | 210 |
| Max batch | — | **120** (128 OOM) | **512** (768 OOM) |

**In one line:** reversible = **5.5× less memory per sequence, 37% slower**, and at a fixed 55M tokens a bigger batch means fewer steps and higher loss.

## My notes

<!-- 3–5 lines in your own words -->

## Setup

| | |
|---|---|
| Model | 19.69M params · L=7 · D=384 · H=8 · T=256 · V=4002 · RMSNorm + SwiGLU |
| Data | TinyStories, 55M train / 5.6M val tokens (see [Dataset](#dataset)) |
| Baseline | FSDP (ZeRO-3) across 2 GPUs (`zero3_launch.py`) |
| Reversible | Plain data parallel, explicit grad all-reduce (`reversible_launch.py`) |
| Optimizer | AdamW, peak LR 3e-4, cosine to 3e-5 · warmup 300 steps (baseline), 20 steps (reversible B=512) |

## References

- Gal et al., *Reversing Large Language Models for Efficient Training and Fine-Tuning*, arXiv:2512.02056
- Gomez et al., *The Reversible Residual Network*, NeurIPS 2017
