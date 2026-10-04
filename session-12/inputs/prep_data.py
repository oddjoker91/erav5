%%writefile prep_data.py

"""
prep_data.py -- one-time data prep from the TWO TinyStories files (already on
Kaggle, no download):
  train file -> data/raw       (BPE trains here) -> shards/train
  valid file -> data/raw_val                     -> shards/val

    python prep_data.py

Cap the train side with MAX_TRAIN_STORIES so the ~2GB file doesn't take forever.
"""

import os
import glob

import numpy as np
from tokenizers import ByteLevelBPETokenizer


# ---- inlined from data_pipeline.py so this file is fully self-contained ----
class Tokenizer:
    """Byte-level BPE trained on data/raw (small vocab -> small embeddings)."""

    def __init__(self, bpe_dir="data/bpe", raw_dir="data/raw", vocab_size=4000):
        vocab_json = os.path.join(bpe_dir, "vocab.json")
        merges_txt = os.path.join(bpe_dir, "merges.txt")
        if not (os.path.exists(vocab_json) and os.path.exists(merges_txt)):
            files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
            if not files:
                raise SystemExit(f"no .txt in {raw_dir}/ to train the BPE on")
            total_mb = sum(os.path.getsize(f) for f in files) / 1e6
            print(f"[BPE] training on {len(files)} files ({total_mb:.1f} MB), "
                  f"target vocab {vocab_size} ...", flush=True)
            os.makedirs(bpe_dir, exist_ok=True)
            bpe = ByteLevelBPETokenizer()
            # min_frequency=1 lets it keep merging until it HITS vocab_size exactly
            # (higher min_frequency can stop early -> fewer tokens than the target)
            bpe.train(files=files, vocab_size=vocab_size, min_frequency=1)
            bpe.save_model(bpe_dir)
            got = bpe.get_vocab_size()
            print(f"[BPE] done: {got} tokens (target {vocab_size}) -> {bpe_dir}/", flush=True)
            if got < vocab_size:
                print(f"[BPE] WARNING: only {got} < {vocab_size} -- corpus too small; "
                      f"add more data to reach the target", flush=True)
        self.bpe = ByteLevelBPETokenizer(vocab_json, merges_txt)
        n = self.bpe.get_vocab_size()
        self.PAD, self.EOS, self.V = n, n + 1, n + 2
        print(f"[BPE] loaded vocab {n}  ->  V = {self.V} (with PAD+EOS)", flush=True)

    def encode(self, text):
        return self.bpe.encode(text).ids


class Config:
    def __init__(self):
        self.TOK = Tokenizer()
        self.V = self.TOK.V
        self.T = 128
        self.D = 256
        self.H = 8
        self.HD = self.D // self.H
        self.DFF = self.D * 4
        self.L = 6


class RawCorpus:
    def __init__(self, folder, pattern="*.txt"):
        self.folder, self.pattern = folder, pattern

    def paths(self):
        return sorted(glob.glob(os.path.join(self.folder, self.pattern)))

    def docs(self):
        for p in self.paths():
            with open(p, "r", encoding="utf-8") as f:
                yield f.read()

    def __len__(self):
        return len(self.paths())


class ShardWriter:
    TOKEN_DTYPE = np.uint16
    DOCID_DTYPE = np.uint32

    def __init__(self, tokenizer, out_dir="data/shards", shard_size=1_000_000):
        self.tok, self.out_dir, self.shard_size = tokenizer, out_dir, shard_size

    def write_split(self, docs, split):
        split_dir = os.path.join(self.out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        tok_buf, doc_buf, idx = [], [], 0

        def flush(tb, db, idx):
            base = os.path.join(split_dir, f"shard_{idx:05d}")
            np.save(base + ".tok.npy", np.array(tb, dtype=self.TOKEN_DTYPE))
            np.save(base + ".doc.npy", np.array(db, dtype=self.DOCID_DTYPE))
            print(f"  {split}/shard_{idx:05d}  {len(tb):>8} tokens")

        for d, text in enumerate(docs):
            ids = self.tok.encode(text) + [self.tok.EOS]   # EOS separates docs
            tok_buf.extend(ids)
            doc_buf.extend([d] * len(ids))
            while len(tok_buf) >= self.shard_size:
                flush(tok_buf[:self.shard_size], doc_buf[:self.shard_size], idx)
                tok_buf, doc_buf, idx = tok_buf[self.shard_size:], doc_buf[self.shard_size:], idx + 1
        if tok_buf:
            flush(tok_buf, doc_buf, idx)
        return sorted(glob.glob(os.path.join(split_dir, "shard_*.tok.npy")))


def preprocess_split(cfg, raw_dir, split, shard_dir="data/shards", shard_size=1_000_000):
    corpus = RawCorpus(raw_dir)
    if len(corpus) == 0:
        raise SystemExit(f"no .txt files in {raw_dir}/")
    paths = ShardWriter(cfg.TOK, out_dir=shard_dir, shard_size=shard_size) \
        .write_split(list(corpus.docs()), split)
    print(f"{split}: {len(paths)} shards from {raw_dir}/")
    return paths


# ---- TinyStories prep ------------------------------------------------------
DATA = "/kaggle/input/notebooks/gona26/tinystoriesv2-cpy/data"
TRAIN_SRC = f"{DATA}/TinyStoriesV2-GPT4-train.txt"
VAL_SRC = f"{DATA}/TinyStoriesV2-GPT4-valid.txt"

SEP = "<|endoftext|>"           # TinyStories separates stories with this token
STORIES_PER_FILE = 500          # group stories -> a manageable number of raw files
RAW_TRAIN = "data/raw"          # BPE trains on this folder (train stories)
RAW_VAL = "data/raw_val"
MAX_TRAIN_STORIES = 200_000     # cap the 2GB train file; None = all ~2.7M
MAX_VAL_STORIES = None          # valid is small; use all


def write_docs(src, out_dir, max_stories):
    os.makedirs(out_dir, exist_ok=True)
    stories = [s.strip() for s in open(src, encoding="utf-8").read().split(SEP) if s.strip()]
    if max_stories:
        stories = stories[:max_stories]
    for fi in range(0, len(stories), STORIES_PER_FILE):
        chunk = stories[fi:fi + STORIES_PER_FILE]
        with open(os.path.join(out_dir, f"doc_{fi // STORIES_PER_FILE:05d}.txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n\n".join(chunk))
    n = len(glob.glob(os.path.join(out_dir, "*.txt")))
    print(f"{out_dir}: {len(stories)} stories -> {n} files")


if __name__ == "__main__":
    write_docs(TRAIN_SRC, RAW_TRAIN, MAX_TRAIN_STORIES)   # -> data/raw
    write_docs(VAL_SRC, RAW_VAL, MAX_VAL_STORIES)         # -> data/raw_val
    cfg = Config()                                        # trains BPE on data/raw (train)
    print(f"vocab V = {cfg.V}  (BPE {cfg.V - 2} + PAD + EOS)")
    preprocess_split(cfg, RAW_TRAIN, "train", shard_size=1_000_000)
    preprocess_split(cfg, RAW_VAL, "val", shard_size=1_000_000)
