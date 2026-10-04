
"""
data_pipeline.py -- PIPELINE A: raw documents -> token shards (run ONCE).

Responsibilities:
  SETUP    Tokenizer, Config (shared vocab/model sizes)
  STAGE 0  RawCorpus   -- load raw text documents from a folder
  STAGE 1  ShardWriter -- tokenize docs -> token shards saved to disk
  STAGE 2  ShardStream -- load a token budget of shards into RAM -> batches

Run once (offline):
    python data_pipeline.py

Tokens are pre-tokenized and SAVED here, never tokenized on the fly.
learn_pipeline.py starts from the shards this produces.
"""

import os
import glob

import numpy as np
from tokenizers import ByteLevelBPETokenizer


# where the trained BPE lives, what it trains on, and the target vocab size
BPE_DIR = "data/bpe"
RAW_DIR = "data/raw"
VOCAB_SIZE = 4000        # target; actual may be smaller on a tiny corpus


# ---- SETUP -----------------------------------------------------------------
class Tokenizer:
    """Byte-level BPE trained on YOUR corpus (small vocab -> small embeddings).

    First construction trains the BPE from RAW_DIR and saves it to BPE_DIR;
    later constructions just load it. PAD/EOS are reserved ids just past the
    learned vocab (so the model sizes its embedding to V = vocab + 2).
    """

    def __init__(self, bpe_dir=BPE_DIR, raw_dir=RAW_DIR, vocab_size=VOCAB_SIZE):
        vocab_json = os.path.join(bpe_dir, "vocab.json")
        merges_txt = os.path.join(bpe_dir, "merges.txt")
        if not (os.path.exists(vocab_json) and os.path.exists(merges_txt)):
            self._train(bpe_dir, raw_dir, vocab_size)
        self.bpe = ByteLevelBPETokenizer(vocab_json, merges_txt)
        n = self.bpe.get_vocab_size()
        self.PAD = n          # reserved ids just past the learned vocab
        self.EOS = n + 1
        self.V = n + 2        # total vocab the model must size its embedding to

    def _train(self, bpe_dir, raw_dir, vocab_size):
        files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
        if not files:
            raise SystemExit(f"no .txt in {raw_dir}/ to train the BPE on")
        total_mb = sum(os.path.getsize(f) for f in files) / 1e6
        print(f"[BPE] training on {len(files)} files ({total_mb:.1f} MB), "
              f"target vocab {vocab_size} ...", flush=True)
        os.makedirs(bpe_dir, exist_ok=True)
        bpe = ByteLevelBPETokenizer()
        # min_frequency=1 lets it keep merging until it HITS vocab_size exactly
        bpe.train(files=files, vocab_size=vocab_size, min_frequency=1)
        bpe.save_model(bpe_dir)
        got = bpe.get_vocab_size()
        print(f"[BPE] done: {got} tokens (target {vocab_size}) -> {bpe_dir}/", flush=True)
        if got < vocab_size:
            print(f"[BPE] WARNING: only {got} < {vocab_size} -- corpus too small", flush=True)

    def encode(self, text):
        return self.bpe.encode(text).ids

    def decode(self, ids):
        ids = [i for i in ids if i not in (self.PAD, self.EOS)]
        return self.bpe.decode(ids)

    def piece(self, i):
        if i == self.PAD: return "<pad>"
        if i == self.EOS: return "<eos>"
        return self.bpe.decode([i])


class Config:
    def __init__(self):
        self.TOK = Tokenizer()
        self.V = self.TOK.V  # vocabulary size (from the BPE tokenizer)
        self.T = 256     # context window (fits a whole TinyStory ~200 tok)
        self.D = 256     # model dimension
        self.H = 8      # number of attention heads
        self.HD = self.D // self.H  # head dimension
        self.DFF = self.D*4  # feed-forward dimension
        self.L = 6      # number of layers


# ---- STAGE 0: load raw documents ------------------------------------------
class RawCorpus:
    """A folder of raw text files; each file is one document.

    Drop your own .txt files into `folder` (one document per file) and the
    rest of the pipeline picks them up. docs() streams files lazily so the
    whole corpus never needs to sit in RAM.
    """

    def __init__(self, folder, pattern="*.txt"):
        self.folder = folder
        self.pattern = pattern

    def paths(self):
        return sorted(glob.glob(os.path.join(self.folder, self.pattern)))

    def docs(self):
        for p in self.paths():
            with open(p, "r", encoding="utf-8") as f:
                yield f.read()

    def __len__(self):
        return len(self.paths())


# ---- STAGE 1: tokenize -> token shards on disk (run once) -----------------
class ShardWriter:
    """Tokenize a corpus -> fixed-size train/val shards.

    Each shard i -> two parallel files under out_dir/<split>/:
        shard_{i}.tok.npy  token ids (uint16)
        shard_{i}.doc.npy  doc ids   (uint32)  -> lets the trainer rebuild the
                                                   cross-document attention mask.
    Real pipelines use ~100M tokens/shard; we keep it tiny to see the mechanics.
    """

    TOKEN_DTYPE = np.uint16
    DOCID_DTYPE = np.uint32

    def __init__(self, tokenizer, out_dir="data/shards", shard_size=1000,
                 val_frac=0.1):
        self.tok = tokenizer
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.val_frac = val_frac

    def _write_split(self, docs, split):
        split_dir = os.path.join(self.out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        tok_buf, doc_buf, idx = [], [], 0

        def flush(tok_buf, doc_buf, idx):
            base = os.path.join(split_dir, f"shard_{idx:05d}")
            np.save(base + ".tok.npy", np.array(tok_buf, dtype=self.TOKEN_DTYPE))
            np.save(base + ".doc.npy", np.array(doc_buf, dtype=self.DOCID_DTYPE))
            print(f"  {split}/shard_{idx:05d}  {len(tok_buf):>6} tokens")

        for d, text in enumerate(docs):
            ids = self.tok.encode(text) + [self.tok.EOS]   # EOS separates docs
            tok_buf.extend(ids)
            doc_buf.extend([d] * len(ids))
            while len(tok_buf) >= self.shard_size:         # emit full shards
                flush(tok_buf[:self.shard_size], doc_buf[:self.shard_size], idx)
                tok_buf = tok_buf[self.shard_size:]
                doc_buf = doc_buf[self.shard_size:]
                idx += 1
        if tok_buf:                                        # final partial shard
            flush(tok_buf, doc_buf, idx)
        return sorted(glob.glob(os.path.join(split_dir, "shard_*.tok.npy")))

    def build(self, corpus):
        docs = list(corpus.docs())
        n_val = max(1, int(len(docs) * self.val_frac))
        val_docs, train_docs = docs[:n_val], docs[n_val:]
        print(f"docs: {len(train_docs)} train / {len(val_docs)} val")
        print("writing train shards:")
        train_paths = self._write_split(train_docs, "train")
        print("writing val shards:")
        val_paths = self._write_split(val_docs, "val")
        print(f"done: {len(train_paths)} train + {len(val_paths)} val shards")
        return train_paths, val_paths


# ---- STAGE 2: load a token budget of shards into RAM -> batches -----------
class ShardStream:
    """Stream (tokens, doc_id) (B, T) batches from a split's shards.

    - Loads only as many shards as needed to reach max_tokens (None = all),
      straight into RAM ONCE. 50M tokens = ~100 MB tokens + ~200 MB doc ids.
      After that, every batch is pure in-memory indexing: no file opens.
    - Non-overlapping blocks: the loaded tokens are cut into back-to-back
      T-length blocks so no token window is ever reused.
    - Rank partition: all ranks share one seed -> identical global block shuffle,
      then rank r takes blocks[r::world_size] -> disjoint data, no overlap.
    - Defaults rank=0/world_size=1 -> single process gets ALL blocks.
    - Yields numpy int64 arrays; the trainer wraps them in torch tensors.
    """

    def __init__(self, split_dir, B, T, seed=0, max_tokens=None, max_shards=None,
                 rank=0, world_size=1):
        paths = sorted(glob.glob(os.path.join(split_dir, "shard_*.tok.npy")))
        if max_shards is not None:                 # None = all shards; int = first N
            paths = paths[:max_shards]
        if not paths:
            raise SystemExit(f"no shards in {split_dir}/")

        toks, docs, have = [], [], 0
        for p in paths:                            # stop once the budget is met
            if max_tokens is not None and have >= max_tokens:
                break
            toks.append(np.load(p))
            docs.append(np.load(p.replace(".tok.npy", ".doc.npy")))
            have += len(toks[-1])
        self.toks = np.concatenate(toks)
        self.docs = np.concatenate(docs)
        if max_tokens is not None:                 # trim to the exact budget
            self.toks = self.toks[:max_tokens]
            self.docs = self.docs[:max_tokens]

        self.B, self.T = B, T
        self.rank, self.world_size = rank, world_size
        self.rng = np.random.default_rng(seed)     # SHARED across ranks (same seed)
        self.n_blocks = len(self.toks) // T        # global block id runs 0..n_blocks-1
        self.offsets = np.arange(T)
        print(f"{split_dir}: {len(toks)} shards, {len(self.toks):,} tokens, "
              f"{self.n_blocks:,} blocks", flush=True)

    def _batch(self, block_ids):
        idx = block_ids[:, None] * self.T + self.offsets   # (B, T) positions
        return self.toks[idx].astype(np.int64), self.docs[idx].astype(np.int64)

    def epoch(self):
        order = self.rng.permutation(self.n_blocks)      # same on all ranks (shared seed)
        mine = order[self.rank::self.world_size]          # disjoint stride partition
        for i in range(0, len(mine) - self.B + 1, self.B):
            yield self._batch(mine[i:i + self.B])         # one vectorized gather

    def iter_batches(self):
        """Infinite stream: loop epochs forever (for a fixed step budget)."""
        while True:
            yield from self.epoch()


def preprocess_to_shards(cfg, raw_dir="data/raw", shard_dir="data/shards",
                         shard_size=1000, val_frac=0.1, overwrite=False):
    """One-time preprocess: read the raw folder and write token shards.

    Skips if shards already exist (pass overwrite=True to force a rebuild).
    """
    have = glob.glob(os.path.join(shard_dir, "train", "shard_*.tok.npy"))
    if have and not overwrite:
        print(f"shards already in {shard_dir}/ -- skip (overwrite=True to rebuild)")
        return
    corpus = RawCorpus(raw_dir)
    if len(corpus) == 0:
        raise SystemExit(f"no .txt files in {raw_dir}/ -- drop documents there first")
    print(f"loaded {len(corpus)} docs from {raw_dir}/")
    ShardWriter(cfg.TOK, out_dir=shard_dir, shard_size=shard_size,
                val_frac=val_frac).build(corpus)


def preprocess_split(cfg, raw_dir, split, shard_dir="data/shards", shard_size=1_000_000):
    """Tokenize ALL docs in raw_dir into shards under shard_dir/<split>/.

    Unlike preprocess_to_shards (which splits ONE folder by val_frac), this writes
    a given folder to a named split -- use it to build train and val from the two
    separate TinyStories files (no leakage, real held-out val).
    """
    corpus = RawCorpus(raw_dir)
    if len(corpus) == 0:
        raise SystemExit(f"no .txt files in {raw_dir}/")
    writer = ShardWriter(cfg.TOK, out_dir=shard_dir, shard_size=shard_size)
    paths = writer._write_split(list(corpus.docs()), split)
    print(f"{split}: {len(paths)} shards from {raw_dir}/")
    return paths


def get_streams(cfg, shard_dir="data/shards", micro_B=4, seed=0,
                train_tokens=50_000_000, val_tokens=None, max_shards=None,
                rank=0, world_size=1):
    """Load token budgets into RAM; return (train_stream, val_stream).

    train_tokens: tokens to load for training (default 50M). None = all shards.
    val_tokens:   tokens to load for validation. None = all val shards.
    max_shards:   optional extra cap on shard count (e.g. 1 for quick debugging).
    rank/world_size: pass the SAME seed on every rank; each rank gets a disjoint
                     slice of blocks. Defaults (0, 1) = single process, all data.
    train_stream: infinite micro-batch iterator for the training loop.
    val_stream:   a callable returning one finite pass over val.
    """
    train = ShardStream(os.path.join(shard_dir, "train"), B=micro_B, T=cfg.T,
                        seed=seed, max_tokens=train_tokens, max_shards=max_shards,
                        rank=rank, world_size=world_size)
    val = ShardStream(os.path.join(shard_dir, "val"), B=micro_B, T=cfg.T,
                      seed=seed + 123, max_tokens=val_tokens, max_shards=max_shards,
                      rank=rank, world_size=world_size)
    return train.iter_batches(), val.epoch


if __name__ == "__main__":
    # Run once (offline): train BPE on data/raw, then raw docs -> token shards.
    # overwrite=True so old (gpt2-tokenized) shards are rebuilt with the new vocab.
    cfg = Config()
    print(f"vocab V = {cfg.V}  (BPE {cfg.V - 2} + PAD + EOS)")
    preprocess_to_shards(cfg, overwrite=True)
