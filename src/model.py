"""
Hema AI — ChatbotModel v4  (GPT Decoder-Only, fixed)
=====================================================
Fixes over v3 based on architecture review:

  ISSUE 1 — embed_size=128 too small for 177K vocab
    → embed_size=256  (doubles representational capacity per token)
    → Each of 177K words now gets 256 numbers instead of 128

  ISSUE 2 — hidden_size=256 (FFN only 2× d_model)
    → hidden_size=1024  (proper 4× ratio, standard since GPT-2)
    → FFN is where factual associations are stored — halving it cuts capacity

  ISSUE 3 — weight tying at 177K vocab hurts both matrices
    → Untied fc_out  (embedding and output projection have different jobs at scale)
    → Weight tying is beneficial at 10K–50K vocab; at 177K they diverge too much

  ISSUE 4 — generate() is O(T²): recomputes full context each step
    → Context truncated to MAX_SEQ_LEN tokens at each step

  ISSUE 5 — beam_search has same O(T²) × beam_size problem
    → Same context truncation fix applied

  IMPROVEMENT — embedding init
    → nn.init.normal_(0, 0.02) instead of Xavier for embedding layers
    → Matches GPT-2 paper and empirically trains better

  UNCHANGED
  ✓ TransformerEncoderLayer + causal mask (correct GPT architecture)
  ✓ Pre-LayerNorm (norm_first=True)
  ✓ PositionalEncoding
  ✓ return_hidden flag for contrastive loss
  ✓ Top-K + Top-P + repetition penalty in generate()
  ✓ Beam search with length penalty

  VRAM at new config (batch=16, T=512, T4):
    ~8.0 GB estimated — 6.5 GB headroom on T4
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice to current sequence length
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            x = x[:, :self.pe.size(1)]
            seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class ChatbotModel(nn.Module):
    """
    GPT-style decoder-only transformer.
    TransformerEncoderLayer stack + causal mask.
    No cross-attention, no dummy memory.
    """
    def __init__(
        self,
        vocab_size:  int,
        pad_idx:     int,
        embed_size:  int   = 256,    # FIX 1: was 128 — too small for 177K vocab
        num_heads:   int   = 4,      # 256/4 = 64 per head — efficient
        hidden_size: int   = 1024,   # FIX 2: was 256 — now proper 4× FFN ratio
        num_layers:  int   = 4,      # was 2 — more depth, better capacity
        dropout:     float = 0.35,
        max_seq_len: int   = 512,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.pad_idx     = pad_idx
        self.embed_size  = embed_size
        self._max_len    = max_seq_len   # used for context truncation in generate()

        self.embedding   = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_size, dropout, max_seq_len)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads,
            dim_feedforward=hidden_size, dropout=dropout,
            activation="gelu", batch_first=True,
            norm_first=True,          # Pre-LayerNorm — stable gradients
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(embed_size),
            enable_nested_tensor=False
        )

        # FIX 3: Untied output projection
        # At 177K vocab, embedding and fc_out have fundamentally different jobs.
        # Tying them forces a single 177K×256 matrix to handle both input
        # representation AND output prediction — capacity conflict at this scale.
        self.fc_out = nn.Linear(embed_size, vocab_size, bias=False)
        # NOTE: NOT tying weights — fc_out.weight != self.embedding.weight

        self._emb_scale = math.sqrt(embed_size)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "embedding" in name and "weight" in name:
                # FIX 5: GPT-2 style init for embeddings
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _make_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((size, size), device=device, dtype=torch.bool), diagonal=1
        )


    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """
        x: (B, T) unified token sequence
        return_hidden=True: returns (logits, hidden) for contrastive loss
        """
        causal_mask = self._make_causal_mask(x.size(1), x.device)
        pad_mask    = (x == self.pad_idx)
        emb         = self.pos_encoder(self.embedding(x) * self._emb_scale)
        hidden      = self.transformer(emb, mask=causal_mask,
                                       src_key_padding_mask=pad_mask)
        logits      = self.fc_out(hidden)
        if return_hidden:
            return logits, hidden
        return logits

    def generate(
        self,
        prompt:             torch.Tensor = None,   # (1, T_prompt)
        eos_idx:            int = None,
        max_new_tokens:     int   = 80,
        temperature:        float = 0.75,
        top_k:              int   = 40,
        top_p:              float = 0.92,
        repetition_penalty: float = 1.3,
        idx:                torch.Tensor = None,
        eos_id:             int = None,
        return_full_sequence: bool = False,
    ):
        """
        Continue generation from a prompt tensor.
        Returns new token ids list by default (or full tensor if idx or return_full_sequence is used).

        FIX 4: Context truncated to _max_len each step → O(T) instead of O(T²)
        """
        p = prompt if prompt is not None else idx
        if p is None:
            raise ValueError("Prompt tensor required for generate().")
        eos = eos_idx if eos_idx is not None else (eos_id if eos_id is not None else 3)

        self.eval()
        device     = p.device
        generated  = p[0].tolist()
        prompt_len = len(generated)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # FIX 4: Truncate to last _max_len tokens — avoids O(T²) growth
                ctx    = generated[-self._max_len:]
                x      = torch.tensor([ctx], dtype=torch.long, device=device)
                logits = self.forward(x)
                lg     = logits[0, -1, :] / max(temperature, 1e-8)

                # Repetition penalty on all generated tokens
                for tok in set(generated):
                    lg[tok] /= repetition_penalty

                # Top-K filter
                if top_k > 0:
                    kth = torch.topk(lg, min(top_k, lg.size(-1))).values[-1]
                    lg  = lg.masked_fill(lg < kth, float("-inf"))

                # Top-P nucleus filter
                probs = torch.softmax(lg, dim=-1)
                sp, si = torch.sort(probs, descending=True)
                sp[torch.cumsum(sp, dim=-1) - sp > top_p] = 0.0
                sp /= sp.sum().clamp(min=1e-9)
                next_tok = si[torch.multinomial(sp, 1)].item()

                if next_tok == eos:
                    break
                generated.append(next_tok)

        if return_full_sequence or idx is not None:
            return torch.tensor([generated], dtype=torch.long, device=device)
        return generated[prompt_len:]

    def beam_search(
        self,
        prompt:         torch.Tensor,   # (1, T_prompt)
        eos_idx:        int,
        beam_size:      int   = 4,
        max_new_tokens: int   = 80,
        length_penalty: float = 0.7,
    ) -> list:
        """
        Beam search from a prompt tensor.
        Returns new token ids only (answer tokens, eos excluded).

        FIX 4 (applied here too): Context truncated to _max_len each step.
        """
        self.eval()
        device     = prompt.device
        prompt_ids = prompt[0].tolist()
        prompt_len = len(prompt_ids)

        with torch.no_grad():
            beams: list     = [(0.0, list(prompt_ids))]
            completed: list = []

            for _ in range(max_new_tokens):
                if not beams:
                    break
                candidates: list = []
                for lp, seq in beams:
                    if len(seq) > prompt_len and seq[-1] == eos_idx:
                        completed.append((lp, seq))
                        continue
                    # FIX 4: Truncate context
                    ctx = seq[-self._max_len:]
                    x   = torch.tensor([ctx], dtype=torch.long, device=device)
                    lp_all = torch.log_softmax(self.forward(x)[0, -1, :], dim=-1)
                    top_lp, top_idx = torch.topk(lp_all, beam_size)
                    for delta, idx in zip(top_lp.tolist(), top_idx.tolist()):
                        candidates.append((lp + delta, seq + [idx]))

                if not candidates:
                    break
                ans_len = lambda c: max(1, len(c[1]) - prompt_len)
                candidates.sort(
                    key=lambda c: c[0] / ans_len(c) ** length_penalty,
                    reverse=True
                )
                beams = candidates[:beam_size]
                if all(s[-1] == eos_idx for _, s in beams if len(s) > prompt_len):
                    completed.extend(beams)
                    beams = []

            all_c = completed + beams
            if not all_c:
                return []
            best = max(
                all_c,
                key=lambda c: c[0] / max(1, len(c[1]) - prompt_len) ** length_penalty
            )[1]

        return [t for t in best[prompt_len:] if t != eos_idx]
 
 
class HemaModel(ChatbotModel):
    """
    HemaModel: Decoder-only Transformer wrapper with max_len argument compatibility.
    """
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        max_len: int = 1024,
        embed_size: int = 256,
        num_heads: int = 4,
        hidden_size: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.35,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            embed_size=embed_size,
            num_heads=num_heads,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_len,
        )