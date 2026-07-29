# India-First 120B LLM — Data Mixture & Training Plan
### ERA V5 · Assignment 5

---

## Objective

Train a 120B-parameter LLM optimized for coding, agentic tasks, reasoning, and Indic languages — in three progressive stages (3B → 20B → 120B), where each stage's trained weights initialize the next larger model, and each stage is benchmark-gated before moving on.

**In one glance:**
- 3 stages, 3 model sizes: 3B → 20B → 120B, one model growing through all three
- Tokens per stage at 40x model size: 120B → 800B → 4.8T (~5.72T total)
- Indic fixed at 12% of every stage
- Benchmark gate at the end of every stage
- Mixture validated by a cheap proxy experiment at 3B before scaling

---

## Data Sources

| Category | Sources | From |
|---|---|---|
| General web | FineWeb-Edu, DCLM, Nemotron-CC, RedPajama-v2 | HuggingFace |
| Code | The Stack v2, permissive GitHub | HuggingFace, GitHub |
| Math / reasoning | OpenWebMath, MATH, Nemotron-CC-Math | HuggingFace |
| Agentic | nebius SWE-agent-trajectories, NVIDIA Open-SWE-Traces, SWE-smith, ToolBench | HuggingFace |
| Long-context | Books, long arXiv papers, legal corpora | HuggingFace, arXiv |
| Multilingual (non-Indic) | mC4, CC-100 | HuggingFace |
| Reference | Wikipedia, arXiv abstracts | Wikimedia, arXiv |
| Indic — real | Sangraha, IndicCorp v2 | AI4Bharat (HuggingFace) |
| Indic — translated | IndicTrans2 pipeline over English sources | AI4Bharat |
| Indic — synthetic | Topic-seeded generation, rewriting | Self-generated |
| Instruction/dialogue | UltraChat-200k, Tulu 3 SFT mix, SmolTalk | HuggingFace, Ai2 |

## Benchmarks

| Benchmark | Measures | Used at | From |
|---|---|---|---|
| Perplexity + fertility | Tokenizer correctness | Stage 1 | Internal |
| MMLU | General knowledge | Stage 1 (subset) | HuggingFace |
| HumanEval | Code generation | All stages | OpenAI (HuggingFace) |
| MILU | Indic understanding | All stages | AI4Bharat |
| SWE-bench | Agentic coding | Stages 2–3 | Princeton (HuggingFace) |
| GSM8K | Multi-step math | Stages 2–3 | OpenAI (HuggingFace) |
| MATH | Long-chain reasoning | Stages 2–3 | HuggingFace |
| IndicGenBench | Indic generation | Stages 2–3 | Google (HuggingFace) |
| Long-context needle (RULER) | Long-input retrieval | Stages 2–3 | NVIDIA (open tooling) |
| IndicXTREME + Indic-translated reasoning sets | India-first differentiator | Stage 3 final | AI4Bharat, HuggingFace |


## Capability Lanes — Data That Fills Them, Benchmarks They Must Win

| Lane | Filled by (from inventory) | Benchmark it is meant to win |
|---|---|---|
| Code | The Stack v2, permissive GitHub, PR diffs | HumanEval |
| Agentic | nebius SWE-agent-trajectories, NVIDIA Open-SWE-Traces, SWE-smith, ToolBench + self-play rollouts | SWE-bench |
| Reasoning/math | OpenWebMath, MATH, Nemotron-CC-Math, rejection-sampled CoT | GSM8K, MATH |
| Long-context | Books, long arXiv, legal corpora | RULER needle test |
| Indic | Sangraha, IndicCorp v2, IndicTrans2 translations, generated synthetic | MILU, IndicGenBench, IndicXTREME |
| General web | FineWeb-Edu, DCLM, Nemotron-CC | MMLU |
| Instruction/dialogue | UltraChat-200k, Tulu 3, SmolTalk | (supports all; sets up post-training) |

---

## Inspiration

The plan follows the ancient Gurukul system: students learned in progressive life-stages — foundation first, then core deep study, then specialization — with mastery tested before advancing. Each training stage maps to one life-stage, and the model grows at each boundary the way a student's capacity grows with age.

---

## Training Pipeline

**Stage 1** — 3B, train
↓ benchmark: perplexity, fertility, MMLU (subset), HumanEval (subset), MILU (subset)

**Stage 2** — 3B → 20B, grow then train
↓ benchmark: HumanEval, SWE-bench, GSM8K, MATH, MILU, IndicGenBench, long-context needle

**Stage 3** — 20B → 120B, grow then train
↓ final benchmark (run once): full suite + IndicXTREME + Indic-translated reasoning sets

---

## STAGE 1 — Foundation
**3B · 120B tokens**

Build the substrate: a correct shared tokenizer across English, code, and Indic scripts, then broad exposure so every capability shows up before anything is pushed deep.

**Training mix:**

| Slot | Share |
|---|---|
| General web | 40% |
| Code | 18% |
| **Indic** | **12%** — Unverified 6 · Verified 3 · Translated 2 · Synthetic 1 |
| Instruction/dialogue | 10% |
| Multilingual (non-Indic) | 8% |
| Math (light) | 6% |
| Reference | 6% |

**Order of data shown to the model** (what it sees early in the stage → what it sees late):
1. First: simple documents in one language at a time
2. Then: documents mixing scripts and languages (code-switched text)
3. Last: increasingly complex structure — single facts → multi-paragraph explanations → basic multi-step Q&A

**Benchmark gate** (all must pass before growing):

| Benchmark | Pass criteria |
|---|---|
| Perplexity / fertility | No Indic language an outlier vs. English |
| MMLU subset | Above random, no domain absent |
| HumanEval subset | Non-zero pass@1 |
| MILU subset | Tracks Indic token share |

*(Gurukul mapping: the Preparatory and Vedanga years, ages 7–12 — language, memory, and logic mastered before heavy texts were allowed.)*

---

## STAGE 2 — Core Capability
**20B · 800B tokens**

The heaviest learning stage: code mastery, agentic behavior, dense reasoning, long-context.

**Training mix:**

| Slot | Share |
|---|---|
| Code (advanced) | 18% |
| Reasoning/math (dense) | 15% |
| General web (top-quality) | 15% |
| Instruction/dialogue | 15% |
| Agentic | 12% |
| **Indic** | **12%** — Verified 5.5 · Unverified 2.5 · Translated 2 · Synthetic 2 |
| Long-context | 8% |
| Multilingual (non-Indic) | 5% |

**Order of data shown to the model** — reasoning difficulty rises through the stage:
1. First: short-chain problems (1–2 steps, e.g. "what is 15% of 200?")
2. Then: multi-step problems (3–5 chained steps, e.g. GSM8K-style word problems)
3. Then: long-chain multi-hop (many steps with backtracking, e.g. proofs, multi-file debugging)
4. Last: agentic multi-turn (reasoning + acting on tools with feedback, e.g. solve a GitHub issue end-to-end)

**Benchmark gate** — the primary "starved slot" detector; any capability lagging its token share redirects cleaning effort before the 120B growth:

| Benchmark | Pass criteria |
|---|---|
| HumanEval (full) | Competitive with similar-size open models |
| SWE-bench | Non-trivial resolve rate |
| GSM8K / MATH | Clear separation between the two |
| MILU / IndicGenBench | Tracks verified-tier share |
| Long-context needle | Near-ceiling recall at target length |

*(Gurukul mapping: the core academic years, ages 13–20 — the Vedas and 18 Vidyas, intensive study once the mind was primed.)*

---

## STAGE 3 — Specialization & Anneal
**120B · 4.8T tokens**

The final model. Bulk training at full scale, narrowing toward benchmark targets, ending with a held-back anneal reserve of the very best data.

**Training mix:**

| Slot | Share |
|---|---|
| Best-of code | 20% |
| Best-of reasoning | 20% |
| Best-of agentic | 15% |
| Instruction/dialogue | 15% |
| **Indic** | **12%** — Verified 6 · Synthetic 3 · Translated 2 · Unverified 1 |
| General (top-decile) | 10% |
| Long-context (refined) | 8% |

**Order of data shown to the model:**
1. First: broad reinforcement across all slots
2. Then: progressively narrows toward the exact skills the final benchmarks test
3. Last: the **anneal reserve** — the top 3–5% of Stage 3's own inventory (verified-correct code, answer-checked CoT, success-only trajectories, spot-checked Indic), carved out before the stage starts, held back untouched, and spent only here — because data seen at the very end (as the learning rate decays) sticks in the weights the most

**Final benchmark (run once, submitted result):** HumanEval · SWE-bench · GSM8K · MATH · MILU · IndicGenBench · long-context suite (RULER) · IndicXTREME + Indic-translated reasoning sets.

*(Gurukul mapping: the specialization years, ages 16+ — Upavedas and 64 Kalas, the final polish before entering the world.)*

---

## Data Mixing

Every batch is drawn by sampling documents according to the mixture weights — e.g., in Stage 2 each batch is ~18% code, ~12% agentic, ~12% Indic in expectation. The mixture is enforced per batch, never as sequential blocks (sequential blocks cause catastrophic forgetting). Curriculum order is implemented by shifting sampling weights across the stage — short-chain weighted up early, long-chain and agentic late — so difficulty ramps smoothly while every batch still sees every slot.

## Protected Floor & Anneal Reserve

| Mechanism | Value | Rule |
|---|---|---|
| Protected floor | Indic 12% (all stages) + Code 15% minimum | Selector can never dip below these |
| Anneal reserve | Final 3–5% of Stage 3 | Held back untouched until cooldown |

## Proxy Experiment (1B and 3B)

Two cheap runs before any number is trusted at scale:

**1B sweep (cheapest, widest):** train 3–4 candidate mixtures at 1B (~40B tokens each). Purpose: rank mixtures against each other — rankings from small runs transfer to larger scale even when absolute scores don't. Kill the losers here.

**3B confirmation (Stage 1 itself):** the surviving mixture *is* Stage 1's 3B run — the pipeline's first stage doubles as the second proxy. Compare it against one flat-mixture control at equal tokens.

- **Hypothesis**: the staged curriculum beats a flat, unordered mixture at equal tokens — especially in Indic-synthetic, where real supply is thinnest
- **Metric**: MILU + IndicGenBench (staged vs. flat), plus per-lane perplexity at the 1B sweep
- **Decision**: 1B ranking picks the mixture → 3B confirms it → grow to 20B. If staged doesn't beat flat at 3B, simplify the curriculum before growing. Re-check the ranking once more at the Stage 2 gate before the 120B growth.

---

## Synthetic & Custom Data

Three slots can't be filled from real supply alone.

**Indic synthetic — need ~160B tokens total** (1.2B + 16B + 144B across stages; ~6x Cosmopedia, the largest public synthetic corpus)
- How: topic-seeded generation (Cosmopedia method) · rewrite noisy real text (Nemotron-CC method) · translate-and-verify via IndicTrans2
- If we can't generate: shift share to the translated tier (cheap, near-unlimited) · repeat real verified data · partner with Bhashini / AI4Bharat / Karya for real commissioned data

**Reasoning/CoT — need 120B (Stage 2) + 960B (Stage 3)**
- How: rejection sampling (DeepSeekMath/Minerva method) · self-consistency filtering
- If we can't generate: use open CoT corpora directly (OpenThoughts, OpenR1-Math, Nemotron post-training sets) · reformat existing solution-annotated data into CoT

**Agentic — need 96B (Stage 2) + 720B (Stage 3) · hardest slot**
- Open supply today is only a few billion tokens (nebius 80k + NVIDIA 200k+ trajectories + SWE-smith) — far below need; runs repetition-heavy no matter what
- How: self-play rollouts against sandboxed tools · AgentInstruct-style synthetic pipelines · bootstrap from open sets first, extend with self-play
- If we can't generate: backfill with advanced code data (PR diffs, issue→fix pairs) · repeat open sets · defer agentic depth to post-training — what most labs do today

---

## Honest Supply Note (Indic)

| Stage | Stage tokens | Indic (12%) |
|---|---|---|
| 1 | 120B | 14.4B |
| 2 | 800B | 96B |
| 3 | 4.8T | 576B |

Stage 3's 576B exceeds real verified supply (Sangraha + IndicCorp v2 combined). Hitting the shares there requires multiple epochs over real corpora or heavier synthetic reliance — stated openly rather than left implicit.

---

## Quick Reference

| Stage | Model | Tokens | Focus |
|---|---|---|---|
| 1. Foundation | 3B | 120B | Tokenizer, broad coverage |
| 2. Core Capability | 20B | 800B | Code, agentic, reasoning, long-context |
| 3. Specialization & Anneal | 120B | 4.8T | Best-of data, benchmark alignment, anneal |

Total: ~5.72T tokens across the pipeline.
