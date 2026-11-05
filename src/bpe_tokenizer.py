"""
bpe_tokenizer.py — Pure-Python Byte-Pair Encoding tokenizer
============================================================
Zero external dependencies. Trains on raw text, saves/loads as JSON.

Usage:
    from bpe_tokenizer import BPETokenizer

    tok = BPETokenizer(vocab_size=4000)
    tok.train(list_of_strings)
    tok.save("models/bpe_tokenizer.json")

    tok = BPETokenizer.load("models/bpe_tokenizer.json")
    ids  = tok.encode("hello world")
    text = tok.decode(ids)
    frag = tok.fragmentation("hello world")  # avg subwords per word
"""

import json
import re
from collections import defaultdict


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer.

    Special tokens (fixed IDs):
        0  <pad>
        1  <unk>
        2  <sos>
        3  <eos>

    Word boundary is marked with the ▁ suffix on the final subword of each
    word (stored as '</w>' internally for readability in the JSON).
    """

    SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]
    WORD_END = "</w>"

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, vocab_size: int = 4000):
        self.vocab_size   = vocab_size
        self.merges: list = []          # ordered list of (a, b) merge rules
        self.vocab: dict  = {}          # token → id
        self.id2tok: dict = {}          # id   → token
        self._merge_rank: dict = {}     # (a,b) → rank (for fast encode)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, texts: list):
        """
        Train BPE on a list of raw strings.
        Mutates self in place; call save() afterwards.
        """
        # Build initial character-level word-frequency table
        # Each word is stored as space-separated chars + WORD_END
        word_freq: dict = defaultdict(int)
        for text in texts:
            for word in self._pre_tokenize(text):
                key = " ".join(list(word) + [self.WORD_END])
                word_freq[key] += 1

        vocab = dict(word_freq)

        # Collect all base characters
        base_chars: set = set()
        for word in vocab:
            for sym in word.split():
                base_chars.add(sym)

        # How many merge operations to learn
        n_special = len(self.SPECIAL_TOKENS)
        n_base    = len(base_chars)
        n_merges  = max(0, self.vocab_size - n_special - n_base)

        print(f"[BPE] base chars={n_base}  merges to learn={n_merges}  "
              f"target vocab={self.vocab_size}")

        self.merges = []
        for step in range(n_merges):
            pairs = self._count_pairs(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._apply_merge(best, vocab)
            self.merges.append(best)
            if (step + 1) % 500 == 0:
                print(f"[BPE]   merge {step+1}/{n_merges}  best='{best[0]}'+'{best[1]}'  "
                      f"freq={pairs[best]}")

        # Build final token set from merged vocab
        all_tokens: set = set()
        for word in vocab:
            for tok in word.split():
                all_tokens.add(tok)

        # Assign IDs: specials first (fixed), then sorted regular tokens
        self.vocab   = {}
        self.id2tok  = {}
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            self.vocab[tok]  = i
            self.id2tok[i]   = tok

        offset = len(self.SPECIAL_TOKENS)
        for i, tok in enumerate(sorted(all_tokens - set(self.SPECIAL_TOKENS))):
            self.vocab[tok]    = i + offset
            self.id2tok[i + offset] = tok

        self._merge_rank = {m: i for i, m in enumerate(self.merges)}

        print(f"[BPE] Training complete. Actual vocab size: {len(self.vocab)}")

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, text: str, add_special: bool = False) -> list:
        """
        Encode text to a list of integer IDs.

        add_special=True wraps with <sos> … <eos> (use for target sequences
        during training).
        """
        unk_id = self.vocab.get("<unk>", 1)
        ids = []
        if add_special:
            ids.append(self.vocab.get("<sos>", 2))

        for word in self._pre_tokenize(text):
            for subword in self._tokenize_word(word):
                ids.append(self.vocab.get(subword, unk_id))

        if add_special:
            ids.append(self.vocab.get("<eos>", 3))
        else:
            ids.append(self.vocab.get("<eos>", 3))

        return ids

    def decode(self, ids: list, skip_special: bool = True) -> str:
        """Decode a list of IDs back to a human-readable string."""
        special = set(self.SPECIAL_TOKENS)
        tokens  = []
        for i in ids:
            tok = self.id2tok.get(i, "<unk>")
            if skip_special and tok in special:
                continue
            tokens.append(tok)

        # Reconstruct words from subwords:
        # Subwords ending with WORD_END mark end-of-word → add space after.
        # All other subwords are concatenated directly.
        text  = ""
        for tok in tokens:
            if tok.endswith(self.WORD_END):
                text += tok[: -len(self.WORD_END)] + " "
            else:
                text += tok
        return text.strip()

    def fragmentation(self, text: str) -> float:
        """
        Return the average number of BPE subword tokens per word.
        A value close to 1.0 means all words are in-vocab (familiar text).
        A high value (> 2.0) means many unseen / OOV words → trigger web search.
        """
        words = self._pre_tokenize(text)
        if not words:
            return 1.0
        total_subwords = sum(len(self._tokenize_word(w)) for w in words)
        return total_subwords / len(words)

    def vocab_size_actual(self) -> int:
        return len(self.vocab)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "merges":     self.merges,
            "vocab":      self.vocab,
            "id2tok":     {str(k): v for k, v in self.id2tok.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[BPE] Saved to {path}  "
              f"({len(self.merges)} merges, {len(self.vocab)} tokens)")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok           = cls(vocab_size=data["vocab_size"])
        tok.merges    = [tuple(m) for m in data["merges"]]
        tok.vocab     = data["vocab"]
        tok.id2tok    = {int(k): v for k, v in data["id2tok"].items()}
        tok._merge_rank = {m: i for i, m in enumerate(tok.merges)}
        print(f"[BPE] Loaded from {path}  "
              f"({len(tok.merges)} merges, {len(tok.vocab)} tokens)")
        return tok

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _pre_tokenize(text: str) -> list:
        """Lowercase + split on whitespace (keeps punctuation attached)."""
        return text.lower().split()

    def _tokenize_word(self, word: str) -> list:
        """
        Apply learned BPE merges to a single word and return its subword list.
        The last subword of every word gets WORD_END appended so the
        tokenizer can reconstruct word boundaries during decoding.
        """
        # Start with character-level segmentation
        chars = list(word)
        chars[-1] = chars[-1] + self.WORD_END   # mark word boundary

        # Apply merges greedily in training order
        while len(chars) > 1:
            # Find all adjacent pairs and their merge rank
            best_rank  = len(self.merges)   # sentinel: infinity
            best_pos   = -1
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                rank = self._merge_rank.get(pair, len(self.merges))
                if rank < best_rank:
                    best_rank = rank
                    best_pos  = i

            if best_pos == -1:
                break   # no applicable merge found

            # Apply the best merge at best_pos
            merged = chars[best_pos] + chars[best_pos + 1]
            chars  = chars[:best_pos] + [merged] + chars[best_pos + 2:]

        return chars

    @staticmethod
    def _count_pairs(vocab: dict) -> dict:
        """Count frequency of every adjacent symbol pair across the vocab."""
        pairs: dict = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _apply_merge(pair: tuple, vocab: dict) -> dict:
        """Replace every occurrence of `pair` with the merged symbol in vocab."""
        a, b        = pair
        bigram      = a + " " + b
        replacement = a + b
        out = {}
        for word, freq in vocab.items():
            out[word.replace(bigram, replacement)] = freq
        return out


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = [
        "what is a noun", "a noun is a naming word",
        "what is a verb", "a verb is an action word",
        "explain photosynthesis", "photosynthesis converts sunlight to energy",
        "what is python", "python is a programming language",
        "hello how are you", "i am doing well thank you",
    ] * 20   # repeat so BPE has enough frequency data

    tok = BPETokenizer(vocab_size=200)
    tok.train(sample)

    test = "what is photosynthesis"
    ids  = tok.encode(test)
    dec  = tok.decode(ids)
    frag = tok.fragmentation(test)
    print(f"\nTest encode: '{test}'")
    print(f"  IDs  : {ids}")
    print(f"  Decoded: '{dec}'")
    print(f"  Fragmentation: {frag:.2f}")

    unknown = "supercalifragilistic"
    frag_unk = tok.fragmentation(unknown)
    print(f"\nUnknown word fragmentation ('{unknown}'): {frag_unk:.2f}")
    print("  (should be >> 1.0 — confirms OOV detection works)")