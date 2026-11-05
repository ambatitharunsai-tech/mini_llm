"""
Hema AI — Stage 3: Reward Model
=================================
Trains a reward model from thumbs up/down feedback collected by app.py.
The reward model scores responses so PPO can optimise toward high-scoring outputs.

Architecture:
  Same transformer backbone as ChatbotModel, but the final layer is a
  scalar score (1 number) instead of a vocab projection.

Training data:
  memory/feedback.csv  — collected by the thumbs up/down in app.py
  Columns: question, good_response, bad_response
  (good = thumbs up, bad = thumbs down)

Usage:
  1. Collect feedback via app.py (thumbs up/down)
  2. python reward_model.py   ← trains reward model
  3. python rlhf.py           ← fine-tunes with PPO

Output: models/reward_model.pth
"""

import os, csv, math, time, random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

_on_cuda = torch.cuda.is_available()

PAD_TOKEN = "<pad>"; SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"; UNK_TOKEN = "<unk>"

# ════════════════════════════════════════════════════════════════
# REWARD MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════
class RewardModel(nn.Module):
    """
    Wraps the pretrained transformer and adds a scalar head.
    Input: token sequence (question + response)
    Output: scalar score (higher = better response)
    """
    def __init__(self, base_model):
        super().__init__()
        self.transformer  = base_model.transformer
        self.embedding    = base_model.embedding
        self.pos_encoder  = base_model.pos_encoder
        self._emb_scale   = base_model._emb_scale
        self.pad_idx      = base_model.pad_idx
        embed_size        = base_model.embed_size

        # Scalar head: mean-pool hidden states → 1 score
        self.score_head = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.GELU(),
            nn.Linear(embed_size // 2, 1)
        )

    def forward(self, x):
        """
        x: (B, T) token ids
        Returns: (B,) scalar scores
        """
        from model import ChatbotModel
        causal_mask = torch.triu(
            torch.full((x.size(1), x.size(1)), float("-inf"), device=x.device),
            diagonal=1
        )
        pad_mask = (x == self.pad_idx)
        emb      = self.pos_encoder(self.embedding(x) * self._emb_scale)
        hidden   = self.transformer(emb, mask=causal_mask,
                                    src_key_padding_mask=pad_mask)
        # Mean-pool over non-padding positions
        mask_f = (~pad_mask).float().unsqueeze(-1)          # (B, T, 1)
        pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # (B, D)
        score  = self.score_head(pooled).squeeze(-1)        # (B,)
        return score


# ════════════════════════════════════════════════════════════════
# FEEDBACK DATASET
# ════════════════════════════════════════════════════════════════
class FeedbackDataset(Dataset):
    """
    Each sample is a (good_response, bad_response) pair.
    Loss: reward(good) should be > reward(bad)   [Bradley-Terry / pairwise ranking]
    """
    def __init__(self, feedback_csv, word2idx, max_len=128):
        self.pairs = []
        unk_id = word2idx.get(UNK_TOKEN, 0)
        sos_id = word2idx.get(SOS_TOKEN, 1)
        eos_id = word2idx.get(EOS_TOKEN, 2)

        def encode(text):
            ids = [sos_id] + [word2idx.get(w, unk_id)
                              for w in text.lower().split()][:max_len - 2] + [eos_id]
            return torch.tensor(ids, dtype=torch.long)

        if not os.path.exists(feedback_csv):
            print(f"⚠️  No feedback file at {feedback_csv}")
            print("   Collect feedback via app.py thumbs up/down first.")
            return

        with open(feedback_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q    = row.get("question", "").strip()
                good = row.get("good_answer", "").strip()
                bad  = row.get("bad_answer",  "").strip()
                if q and good and bad and good != bad:
                    good_seq = encode(f"{q} {good}")
                    bad_seq  = encode(f"{q} {bad}")
                    self.pairs.append((good_seq, bad_seq))

        print(f"FeedbackDataset: {len(self.pairs):,} preference pairs")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]


def collate_pairwise(batch):
    good_list, bad_list = zip(*batch)
    return (pad_sequence(good_list, batch_first=True, padding_value=0),
            pad_sequence(bad_list,  batch_first=True, padding_value=0))


# ════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════
def train_reward_model():
    # Load base model + vocab
    ckpt_path = "models/rnn_chatbot.pth"
    if not os.path.exists(ckpt_path):
        # Fall back to pretrained if SFT not done yet
        ckpt_path = "models/pretrained.pth"
    if not os.path.exists(ckpt_path):
        print("❌ No base model found. Run pretrain.py or train.py first.")
        return

    ckpt     = torch.load(ckpt_path, map_location="cpu")
    word2idx = ckpt["word2idx"]
    idx2word = ckpt["idx2word"]
    vocab_size = len(word2idx)
    pad_idx    = word2idx.get(PAD_TOKEN, 0)

    device = torch.device("cuda" if _on_cuda else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    from model import ChatbotModel
    base = ChatbotModel(vocab_size, pad_idx=pad_idx).to(device)
    base.load_state_dict(ckpt["model"])

    reward_model = RewardModel(base).to(device)
    print(f"Reward model ready on {device}")

    # Load feedback data
    ds = FeedbackDataset("memory/feedback.csv", word2idx)
    if len(ds) == 0:
        print("\n💡 No feedback data yet.")
        print("   Run app.py, ask questions, use thumbs up/down,")
        print("   then run reward_model.py again.")
        return

    dl = DataLoader(ds, batch_size=8, shuffle=True,
                    collate_fn=collate_pairwise, drop_last=False)

    optimizer = optim.AdamW(reward_model.parameters(), lr=1e-5, weight_decay=0.01)

    print(f"\nTraining reward model on {len(ds)} preference pairs...")
    print("=" * 50)

    for epoch in range(1, 4):   # 3 epochs is usually enough
        total_loss, correct, total = 0.0, 0, 0
        reward_model.train()

        for good_ids, bad_ids in dl:
            good_ids = good_ids.to(device)
            bad_ids  = bad_ids.to(device)

            good_score = reward_model(good_ids)   # (B,)
            bad_score  = reward_model(bad_ids)    # (B,)

            # Bradley-Terry pairwise ranking loss:
            # log sigmoid(good_score - bad_score) → push good > bad
            loss = -torch.nn.functional.logsigmoid(
                good_score - bad_score
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct    += (good_score > bad_score).float().sum().item()
            total      += len(good_ids)

        acc = correct / max(1, total) * 100
        print(f"Ep {epoch} | Loss {total_loss/len(dl):.4f} | "
              f"Ranking accuracy {acc:.1f}%")

    torch.save({
        "model":     reward_model.state_dict(),
        "word2idx":  word2idx,
        "idx2word":  idx2word,
    }, "models/reward_model.pth")
    print("\n✅ Saved models/reward_model.pth")
    print("Next: python rlhf.py")


if __name__ == "__main__":
    train_reward_model()