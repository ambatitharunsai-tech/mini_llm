"""
Hema AI — Training Pipeline v4  (Super Supervised Learning)
============================================================
New techniques over v3:

  SUPERVISED LEARNING UPGRADES
  ✓ Scheduled Sampling     — teacher forcing ratio decays epoch by epoch
                             Model learns to use its OWN outputs, not just
                             gold tokens. Closes the train→inference gap.
  ✓ Contrastive Loss       — for each batch, pull correct (src,ans) pairs
                             together in embedding space, push wrong pairs apart.
                             Forces the encoder to build meaningful representations.
  ✓ Answer Quality Weighting — long, informative answers get higher loss weight.
                             Short/low-quality answers contribute less gradient.
  ✓ MixUp on Embeddings    — interpolate between two training examples in
                             embedding space. Teaches smoother decision boundaries.
  ✓ Hard Example Mining    — track per-pair loss history; over-sample pairs
                             the model still gets wrong.

  KEPT FROM v3
  ✓ Quality filter, 5× augmentation, curriculum learning
  ✓ OneCycleLR, dropout annealing, early stopping min_delta
  ✓ In-memory best weights, gradient tracking
"""

import os, csv, math, time, random, re, copy
from collections import defaultdict
os.environ["PYTORCH_CUDA_ALLOC_CONF"]        = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]    = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
# ── NCCL stability fixes for Kaggle T4×2 ────────────────────────────────────
# Default timeout is 10min — long sequences cause one GPU to lag behind.
# Raise to 60min and disable async error handling for cleaner crash messages.
os.environ["NCCL_TIMEOUT"]              = "3600"   # 60 minutes
os.environ["NCCL_BLOCKING_WAIT"]        = "1"       # block instead of async fail
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"       # cleaner error messages
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from model import ChatbotModel

# ════════════════════════════════════════════════════════════════
# DDP INIT  — works on Kaggle dual-T4 via torchrun or mp.spawn
# Launch:  torchrun --nproc_per_node=2 train.py
# ════════════════════════════════════════════════════════════════
_ddp_available = False   # DDP disabled — Kaggle NCCL too unreliable on T4×2
# Single GPU uses cuda:0 only. Second T4 stays idle.
# To re-enable DDP later: set _ddp_available = (...) and launch with torchrun.

if False:  # DDP relaunch block — disabled
    dist.init_process_group(
        backend="nccl",
        timeout=__import__("datetime").timedelta(minutes=60)
    )
    _rank       = dist.get_rank()          # 0 or 1
    _world_size = dist.get_world_size()    # 2
    _local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(_local_rank)
    _is_main    = (_rank == 0)             # only rank-0 prints / saves
else:
    _rank       = 0
    _world_size = 1
    _local_rank = 0
    _is_main    = True

def _print(*args, **kwargs):
    """Print only on rank-0 to avoid duplicate output."""
    if _is_main:
        print(*args, **kwargs)

# ── CUDA speed flags ────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    torch.backends.cudnn.deterministic    = False

# ════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════
import torch as _torch
_on_cuda = _torch.cuda.is_available()   # used throughout for device-aware settings

# ── Single GPU mode ───────────────────────────────────────────────────────────
# DDP on Kaggle T4×2 causes persistent NCCL collective timeouts (SIGABRT at
# step ~4535). The shared network fabric is too unreliable for gradient syncs
# on variable-length sequences. Single GPU is more stable and nearly as fast
# since the model is embedding-dominated, not compute-bound.
# ── 75k Parameter LLM Model Architecture ──────────────────────────────────────
# Exactly ~75,000 parameters:
#   vocab=600, embed_size=32, num_heads=2, hidden_size=128, num_layers=3
#   Embedding (600×32) + 3 Transformer Layers + Projection (32×600) = 76,576 params
MODEL_PROFILE = os.environ.get("MODEL_PROFILE", "75k")

if MODEL_PROFILE == "75k":
    EMBED_SIZE   = 32
    NUM_HEADS    = 2
    HIDDEN_SIZE  = 128
    NUM_LAYERS   = 3
    TARGET_VOCAB = 600
    _default_bs  = 32
    _default_acc = 2
else:
    EMBED_SIZE   = 256
    NUM_HEADS    = 4
    HIDDEN_SIZE  = 1024
    NUM_LAYERS   = 4
    TARGET_VOCAB = None
    _default_bs  = 32 if _on_cuda else 4
    _default_acc = 2 if _on_cuda else 8

# Effective batch = 32 × 2 = 64
BATCH_SIZE         = int(os.environ.get("BATCH_SIZE", str(_default_bs)))
ACCUMULATION_STEPS = int(os.environ.get("ACCUMULATION_STEPS", str(_default_acc)))
EPOCHS                   = int(os.environ.get("EPOCHS", "50" if os.environ.get("TRAINING_STAGE", "1") == "1" else "250"))
EARLY_STOPPING_PATIENCE  = 25
EARLY_STOPPING_MIN_DELTA = 0.001
BASE_LR                  = 5e-5
MAX_LR                   = 8e-4
WEIGHT_DECAY             = 0.01
CLIP                     = 0.5
LABEL_SMOOTHING          = 0.08
WORD_DROPOUT_RATE        = 0.15
VAL_SPLIT                = 0.10
TEST_SPLIT               = 0.10
RESUME                   = False  # scratch — no checkpoint loaded
CHECKPOINT_EVERY         = 25
NUM_WORKERS              = 4 if _on_cuda else 0
PREFETCH_FACTOR          = 2 if _on_cuda else None
PERSISTENT_WORKERS       = True if _on_cuda else False
DROPOUT_START            = 0.35
DROPOUT_END              = 0.10
CURRICULUM_EPOCHS        = 30
MAX_SEQ_LEN              = 256

# ════════════════════════════════════════════════════════════════
# TRAINING STAGE  ← 1 = pretrain | 2 = SFT | 3 = DPO
# ════════════════════════════════════════════════════════════════
TRAINING_STAGE = int(os.environ.get("TRAINING_STAGE", "1"))   # ← 1 = pretrain (user selected)

if TRAINING_STAGE == 1:
    MAX_LR          = 1e-3    # higher LR for pretraining
    EPOCHS          = int(os.environ.get("EPOCHS", "50"))      # pretraining converges fast
    LABEL_SMOOTHING = 0.0     # no smoothing during pretraining
    CURRICULUM_EPOCHS = 0     # no curriculum during pretraining
    _STAGE_CONTRASTIVE = 0.0  # no contrastive during pretraining
elif TRAINING_STAGE == 2:
    _STAGE_CONTRASTIVE = None # use CONTRASTIVE_WEIGHT default
elif TRAINING_STAGE == 3:
    MAX_LR          = 1e-4    # low LR — avoid forgetting SFT
    EPOCHS          = 30      # short DPO run
    _STAGE_CONTRASTIVE = 0.0  # no contrastive during DPO


# ── Scheduled Sampling ───────────────────────────────────────────────────────
# NOT USED in GPT decoder-only training.
# GPT trains on the full unified sequence with a causal mask — there is no
# encoder-decoder gap to close. Teacher forcing vs self-feeding is not a
# concept here; the causal mask handles left-to-right generation natively.
# SS_START_RATIO / SS_MIN_RATIO / SS_END_EPOCH removed.

# ── Contrastive Loss ─────────────────────────────────────────────────────────
CONTRASTIVE_WEIGHT  = 0.15   # how much contrastive loss adds to total loss
CONTRASTIVE_MARGIN  = 0.5    # triplet margin — push negatives at least this far
# Temperature 0.07 amplifies logits ÷0.07 = 14× → massive gradient spikes (GN>2)
# 0.20 reduces to 5× amplification — stable gradients, still effective InfoNCE
CONTRASTIVE_TEMP    = 0.20

# ── Answer Quality Weighting ─────────────────────────────────────────────────
QUALITY_WEIGHT_MIN  = 0.5    # minimum weight for short/simple answers
QUALITY_WEIGHT_MAX  = 2.0    # maximum weight for long/rich answers

# ── MixUp ────────────────────────────────────────────────────────────────────
MIXUP_ALPHA         = 0.2    # Beta(alpha, alpha) — low alpha = mostly original
MIXUP_PROB          = 0.3    # apply MixUp to 30% of batches

# ── Hard Example Mining ───────────────────────────────────────────────────────
HEM_OVERSAMPLE_WEIGHT = 3.0  # hard examples get 3× sampling weight
HEM_HARD_THRESHOLD    = 0.7  # loss percentile above which = "hard"

PAD_TOKEN = "<pad>"; SOS_TOKEN = "<sos>"; EOS_TOKEN = "<eos>"; UNK_TOKEN = "<unk>"

# ════════════════════════════════════════════════════════════════
# DATA QUALITY FILTER
# ════════════════════════════════════════════════════════════════
JUNK_SIGNALS = [
    "is a song","is an american","american film","american slasher",
    "starring ","performed by","box office","academy award",
    "directed by","written by","theme song","soundtrack",
    "television series","tv series","reality show",
    "raindrops keep fallin","happy ending","jennifer love",
    "sarah michelle","wikipedia","wikimedia","retrieved from",
    "this article","click here","read more","see also",
    "external links","references","footnotes",
]

def is_junk(inp: str, out: str) -> bool:
    out_l = out.lower()
    if any(sig in out_l for sig in JUNK_SIGNALS): return True
    # Strict filter for large datasets (>5000 pairs), relaxed for small ones
    # This prevents nearly everything being filtered on Colab test runs
    if len(inp.split()) < 1 or len(out.split()) < 1: return True   # truly empty
    if out.lower().strip() == inp.lower().strip(): return True       # exact copy
    return False

def answer_quality_weight(answer: str) -> float:
    """
    Returns a loss weight based on answer informativeness.
    Longer, more informative answers get higher weight.
    """
    words = answer.split()
    n = len(words)
    # Sigmoid-like curve: short answers low weight, long answers high
    # 5 words → ~0.5,  15 words → ~1.0,  30+ words → ~2.0
    raw = math.log1p(n) / math.log1p(20)   # normalise around 20-word answers
    return float(max(QUALITY_WEIGHT_MIN, min(QUALITY_WEIGHT_MAX, raw * 1.5)))

# ════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ════════════════════════════════════════════════════════════════
# LOAD & FILTER
# ════════════════════════════════════════════════════════════════
dataset_path = os.environ.get("DATASET_PATH", "data/chat_dataset.csv")
if dataset_path == "data/chat_dataset.csv" and os.path.exists("data/train.csv"):
    temp_check = pd.read_csv("data/chat_dataset.csv", nrows=10)
    if len(temp_check) < 10:
        dataset_path = "data/train.csv"

print(f"Loading dataset from {dataset_path}...")
data = pd.read_csv(dataset_path, dtype=str, low_memory=False).fillna("")
data = data[data["input"].str.lower().str.strip() != "input"]
data = data.drop_duplicates(subset=["input","output"])

raw_inputs  = [clean_text(r["input"])  for _, r in data.iterrows()]
raw_outputs = [clean_text(r["output"]) for _, r in data.iterrows()]

pairs = [(i, o) for i, o in zip(raw_inputs, raw_outputs) if not is_junk(i, o)]
if not pairs:
    pairs = [(i, o) for i, o in zip(raw_inputs, raw_outputs)
             if len(i.split()) >= 2 and len(o.split()) >= 3]

sample_limit = int(os.environ.get("SAMPLE_LIMIT", "0"))
if sample_limit > 0 and len(pairs) > sample_limit:
    pairs = pairs[:sample_limit]
    print(f"Limiting to first {sample_limit:,} pairs (SAMPLE_LIMIT={sample_limit})")

inputs, outputs = zip(*pairs)
quality_weights_raw = [answer_quality_weight(o) for o in outputs]
# both ranks print identical dataset stats — OK
print(f"Pairs after quality filter: {len(inputs):,}")
print(f"Answer quality weights — min:{min(quality_weights_raw):.2f}  "
      f"max:{max(quality_weights_raw):.2f}  "
      f"avg:{sum(quality_weights_raw)/len(quality_weights_raw):.2f}")

# ════════════════════════════════════════════════════════════════
# VOCABULARY
# ════════════════════════════════════════════════════════════════
words = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN}
for s in inputs + outputs:
    words.update(s.split())

if TARGET_VOCAB and len(words) > TARGET_VOCAB:
    from collections import Counter
    counts = Counter()
    for s in inputs + outputs:
        counts.update(s.split())
    specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    top_words = [w for w, _ in counts.most_common(TARGET_VOCAB - len(specials)) if w not in specials]
    vocab_words = sorted(specials + top_words)
    word2idx = {w: i for i, w in enumerate(vocab_words)}
else:
    word2idx = {w: i for i, w in enumerate(sorted(words))}

idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx)
print(f"Vocabulary size: {vocab_size:,}")

# ════════════════════════════════════════════════════════════════
# AUGMENTATION (same 5× as v3)
# ════════════════════════════════════════════════════════════════
SYNONYMS = {
    "what is":"explain","explain":"what is","define":"what is",
    "tell me about":"what is","describe":"explain",
    "how does":"explain how","what does":"define",
    "big":"large","small":"tiny","fast":"quick","smart":"intelligent",
    "make":"create","use":"apply","show":"display","find":"discover",
    "important":"significant","problem":"issue","answer":"solution",
}
Q_TEMPLATES = [
    "what is {topic}","tell me about {topic}","explain {topic}",
    "can you explain {topic}","describe {topic}","i want to know about {topic}",
]
STOP_Q = {"what","is","are","was","were","the","a","an","of","in","on","tell","me",
          "about","explain","can","you","please","give","describe","define","how",
          "why","which","do","does","did","i","let","us"}

def extract_topic(q):
    ws = [w for w in q.split() if w not in STOP_Q and len(w) > 2]
    return " ".join(ws[:4]) if ws else ""

def aug_word_dropout(s, rate=0.12):
    return " ".join(UNK_TOKEN if random.random()<rate else w for w in s.split())
def aug_word_swap(s):
    ws=s.split()
    if len(ws)<3: return s
    i,j=random.sample(range(len(ws)),2); ws[i],ws[j]=ws[j],ws[i]
    return " ".join(ws)
def aug_word_delete(s, rate=0.10):
    ws=[w for w in s.split() if random.random()>rate]
    return " ".join(ws) if ws else s
def aug_synonym(s):
    for k,v in SYNONYMS.items():
        if k in s: return s.replace(k,v,1)
    return s
def aug_paraphrase(inp):
    topic=extract_topic(inp)
    if not topic: return inp
    return random.choice(Q_TEMPLATES).format(topic=topic)

def augment_pair(inp, out):
    results = [(inp, out)]
    # In Stage 1 Pretraining, preserve authentic grammatical text sequences
    if TRAINING_STAGE == 1:
        return results
    # On CUDA keep 2x augmentation (fast epochs); on CPU/MPS use full 5x
    import torch as _t
    _fast = _t.cuda.is_available()
    if len(out.split()) > 3:
        if not _fast:
            results.append((aug_word_dropout(inp), out))
            results.append((aug_word_swap(inp),    out))
            results.append((aug_word_delete(inp),  out))
        if len(inp.split()) >= 3:
            results.append((aug_synonym(inp), out))
        if not _fast:
            t = extract_topic(inp)
            if t and t != inp:
                p = aug_paraphrase(inp)
                if p != inp: results.append((p, out))
    return results

# ════════════════════════════════════════════════════════════════
# ENCODING — GPT instruction/response format
# ════════════════════════════════════════════════════════════════
# Sequence layout:
#   <sos> ### Instruction: \n {question} \n\n ### Response: \n {answer} <eos>
#
# Labels:
#   -100 for every token up to and including "### Response:\n"
#   Token ids for answer + <eos>  ← loss only computed here
#
# This is identical to how LLaMA / Alpaca instruction tuning works.

# Instruction/response markers — must use words that survive clean_text()
# clean_text() strips # and : via re.sub(r"[^\w\s]","",text)
# So "### Instruction:" → "instruction" and "### Response:" → "response"
# Using clean words directly avoids all tokens mapping to UNK
_INSTRUCTION_PREFIX = "instruction"
_RESPONSE_PREFIX    = "response"

def _tokenize(text: str) -> list:
    """Tokenize using same cleaning as train data — ensures all tokens are in vocab."""
    unk_id = word2idx.get(UNK_TOKEN, 0)
    # Apply same cleaning as clean_text() so tokens match word2idx
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [word2idx.get(w, unk_id) for w in text.split() if w]

def encode_pair(question: str, answer: str):
    """
    Returns (input_ids, labels) based on TRAINING_STAGE.

    Stage 1 — Pretraining:
        labels = input_ids — loss on EVERY token (raw language modeling)

    Stage 2/3 — SFT / DPO:
        labels = -100 for question+headers, token ids for answer only
    """
    sos_id = word2idx[SOS_TOKEN]
    eos_id = word2idx[EOS_TOKEN]

    instr_ids    = _tokenize(_INSTRUCTION_PREFIX)
    question_ids = _tokenize(question)
    resp_ids     = _tokenize(_RESPONSE_PREFIX)
    answer_ids   = _tokenize(answer)

    prompt_ids = [sos_id] + instr_ids + question_ids + resp_ids
    input_ids  = prompt_ids + answer_ids + [eos_id]

    if TRAINING_STAGE == 1:
        # Pretraining: predict every token — no masking at all
        labels = input_ids[:]
    else:
        # SFT / DPO: only answer tokens contribute to loss
        labels = [-100] * len(prompt_ids) + answer_ids + [eos_id]

    if len(input_ids) > MAX_SEQ_LEN:
        input_ids = input_ids[:MAX_SEQ_LEN]
        labels    = labels[:MAX_SEQ_LEN]

    return input_ids, labels

# ════════════════════════════════════════════════════════════════
# DATASET — lazy GPT-instruction format
# ════════════════════════════════════════════════════════════════
class ChatDataset(Dataset):
    """
    GPTInstructionDataset — unified sequence per sample.
    Lazy encoding (not preloaded) → supports augmentation + HEM.
    """
    def __init__(self, inp_list, out_list, qw_list, is_train=False,
                 curriculum_phase=False, sample_weights=None):
        self.qweights = []
        if is_train:
            aug_pairs, aug_qw = [], []
            for i, o, qw in zip(inp_list, out_list, qw_list):
                for ai, ao in augment_pair(i, o):
                    aug_pairs.append((ai, ao))
                    aug_qw.append(qw)
            seen, unique, unique_qw = set(), [], []
            for (i, o), qw in zip(aug_pairs, aug_qw):
                if i not in seen:
                    seen.add(i); unique.append((i, o)); unique_qw.append(qw)
            self.inputs, self.outputs = [p[0] for p in unique], [p[1] for p in unique]
            self.qweights = unique_qw
        else:
            self.inputs, self.outputs = list(inp_list), list(out_list)
            self.qweights = list(qw_list)

        self.sample_weights = sample_weights or [1.0] * len(self.inputs)

        if curriculum_phase and is_train:
            order = sorted(range(len(self.inputs)), key=lambda k: len(self.inputs[k].split()))
            self.inputs   = [self.inputs[k]   for k in order]
            self.outputs  = [self.outputs[k]  for k in order]
            self.qweights = [self.qweights[k] for k in order]

    def __len__(self): return len(self.inputs)

    def __getitem__(self, idx):
        input_ids, labels = encode_pair(self.inputs[idx], self.outputs[idx])
        # Truncate to MAX_SEQ_LEN
        if len(input_ids) > MAX_SEQ_LEN:
            input_ids = input_ids[:MAX_SEQ_LEN]
            labels    = labels[:MAX_SEQ_LEN]
        qw = torch.tensor(self.qweights[idx], dtype=torch.float)
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels,    dtype=torch.long),
                qw, idx)

def collate_fn(batch):
    input_ids_b, labels_b, qw_b, idx_b = zip(*batch)
    # input_ids padded with pad_idx — model ignores via padding mask
    x_pad = pad_sequence(input_ids_b, padding_value=word2idx[PAD_TOKEN], batch_first=True)
    # labels padded with -100 — CrossEntropyLoss ignores -100 automatically
    y_pad = pad_sequence(labels_b,    padding_value=-100,                 batch_first=True)
    return x_pad, y_pad, torch.stack(qw_b), list(idx_b)

# ════════════════════════════════════════════════════════════════
# SPLIT
# ════════════════════════════════════════════════════════════════
combined = list(zip(inputs, outputs, quality_weights_raw))
random.shuffle(combined)
inputs_s, outputs_s, qw_s = zip(*combined)

n      = len(inputs_s)
n_test = max(1, int(n * TEST_SPLIT))
n_val  = max(1, int(n * VAL_SPLIT))
n_train= n - n_val - n_test

train_inp = list(inputs_s[:n_train]);             train_out = list(outputs_s[:n_train])
train_qw  = list(qw_s[:n_train])
val_inp   = list(inputs_s[n_train:n_train+n_val]); val_out  = list(outputs_s[n_train:n_train+n_val])
val_qw    = list(qw_s[n_train:n_train+n_val])
test_inp  = list(inputs_s[n_train+n_val:]);       test_out  = list(outputs_s[n_train+n_val:])
test_qw   = list(qw_s[n_train+n_val:])
print(f"Split — Train: {n_train} | Val: {n_val} | Test: {n_test}")

val_ds  = ChatDataset(val_inp,  val_out,  val_qw,  is_train=False)
test_ds = ChatDataset(test_inp, test_out, test_qw, is_train=False)

# DDP: each rank gets a non-overlapping shard of val/test via DistributedSampler
_val_sampler  = DistributedSampler(val_ds,  shuffle=False) if _ddp_available else None
_test_sampler = DistributedSampler(test_ds, shuffle=False) if _ddp_available else None

val_dl  = DataLoader(val_ds,  batch_size=BATCH_SIZE,
                     sampler=_val_sampler, shuffle=False if _ddp_available else False,
                     collate_fn=collate_fn, num_workers=NUM_WORKERS,
                     pin_memory=_on_cuda, prefetch_factor=PREFETCH_FACTOR,
                     persistent_workers=PERSISTENT_WORKERS)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE,
                     sampler=_test_sampler, shuffle=False if _ddp_available else False,
                     collate_fn=collate_fn, num_workers=NUM_WORKERS,
                     pin_memory=_on_cuda, prefetch_factor=PREFETCH_FACTOR,
                     persistent_workers=PERSISTENT_WORKERS)

# ════════════════════════════════════════════════════════════════
# DEVICE & MODEL
# ════════════════════════════════════════════════════════════════
# User instruction: Do NOT use MPS on system; use external GPU (CUDA) or CPU
if _ddp_available:
    device = torch.device(f"cuda:{_local_rank}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

_print(f"Device: {device}  |  World size: {_world_size}  |  DDP: {'✅' if _ddp_available else '❌ (single GPU)'}")
if not torch.cuda.is_available():
    _print("💡 Notice: Running on CPU (MPS disabled by user request). For full training, launch on external GPU (CUDA).")

os.makedirs("models", exist_ok=True)
model = ChatbotModel(
    vocab_size,
    pad_idx     = word2idx[PAD_TOKEN],
    embed_size  = EMBED_SIZE,
    num_heads   = NUM_HEADS,
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    dropout     = DROPOUT_START,
    max_seq_len = MAX_SEQ_LEN,
).to(device)
_print(f"Model architecture profile: {MODEL_PROFILE} ({sum(p.numel() for p in model.parameters()):,} parameters)")

# ════════════════════════════════════════════════════════════════
# RESUME  —  with vocab-expansion transplant support
# ════════════════════════════════════════════════════════════════
best_val_loss = float("inf")
best_epoch    = 0
best_weights  = None

def _load_with_vocab_transplant(ckpt_path, model, word2idx, device):
    """
    Load a checkpoint whose vocab may differ from the current model.

    Three cases:
      A) Same vocab size       → standard load_state_dict, done.
      B) Checkpoint is SMALLER → transplant: copy old embedding rows into
                                  new (larger) embedding; new words stay
                                  randomly initialised. All other weights
                                  (encoder / decoder / FFN) copy 1-for-1
                                  because they don't depend on vocab size.
      C) Checkpoint is LARGER  → shouldn't happen; fall back to fresh start.

    Returns (success: bool, info: str)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, tuple):
        return False, "legacy tuple format — skipping"

    sd_ckpt  = ckpt.get("model", ckpt)          # state dict

    # ── Normalise key prefix & strip non-persistent positional encodings ───────
    # Checkpoints saved while torch.compile was active have '_orig_mod.' prefix.
    sample_key = next(iter(sd_ckpt))
    if sample_key.startswith("_orig_mod."):
        sd_ckpt = {k.replace("_orig_mod.", "", 1): v for k, v in sd_ckpt.items()}
    sd_ckpt = {k: v for k, v in sd_ckpt.items() if not k.endswith("pos_encoder.pe")}

    sd_model = model.state_dict()

    # ── Normalise model keys too (in case model is somehow compiled) ─────────
    sample_model_key = next(iter(sd_model))
    if sample_model_key.startswith("_orig_mod."):
        sd_model = {k.replace("_orig_mod.", "", 1): v for k, v in sd_model.items()}

    ckpt_vocab  = sd_ckpt["embedding.weight"].shape[0]
    model_vocab = sd_model["embedding.weight"].shape[0]

    # Helper: re-add prefix if the live model uses it
    def _prefix_sd(sd):
        live_key = next(iter(model.state_dict()))
        if live_key.startswith("_orig_mod.") and not next(iter(sd)).startswith("_orig_mod."):
            return {"_orig_mod." + k: v for k, v in sd.items()}
        return sd

    if ckpt_vocab == model_vocab:
        # ── Case A: exact match ──────────────────────────────────────────────
        model.load_state_dict(_prefix_sd(sd_ckpt), strict=False)
        return True, f"exact vocab match ({ckpt_vocab:,}) — full weights loaded"

    if ckpt_vocab > model_vocab:
        # ── Case C: checkpoint larger than current model — shouldn't happen ──
        return False, (f"checkpoint vocab ({ckpt_vocab:,}) > model vocab "
                       f"({model_vocab:,}) — starting fresh")


    # ── Case B: vocab expansion transplant ──────────────────────────────────
    # 1. Copy all non-embedding weights that match exactly
    mismatched, copied = [], 0
    new_sd = sd_model.copy()
    for key, ckpt_tensor in sd_ckpt.items():
        if key not in new_sd:
            mismatched.append(key); continue
        if new_sd[key].shape != ckpt_tensor.shape:
            mismatched.append(key); continue
        new_sd[key] = ckpt_tensor
        copied += 1

    # 2. Transplant embedding rows: old vocab rows → new embedding
    #    New words (rows ckpt_vocab : model_vocab) stay as Xavier init
    old_emb = sd_ckpt["embedding.weight"]          # (ckpt_vocab, embed_dim)
    new_emb = new_sd["embedding.weight"].clone()   # (model_vocab, embed_dim)  random
    new_emb[:ckpt_vocab] = old_emb                 # copy old knowledge
    new_sd["embedding.weight"] = new_emb

    # 3. Transplant fc_out (output projection) the same way
    #    fc_out.weight: (vocab, embed_dim)  fc_out.bias: (vocab,)
    if "fc_out.weight" in sd_ckpt and "fc_out.weight" in new_sd:
        old_w = sd_ckpt["fc_out.weight"]           # (ckpt_vocab, embed_dim)
        new_w = new_sd["fc_out.weight"].clone()
        new_w[:ckpt_vocab] = old_w
        new_sd["fc_out.weight"] = new_w

    if "fc_out.bias" in sd_ckpt and "fc_out.bias" in new_sd:
        old_b = sd_ckpt["fc_out.bias"]             # (ckpt_vocab,)
        new_b = new_sd["fc_out.bias"].clone()
        new_b[:ckpt_vocab] = old_b
        new_sd["fc_out.bias"] = new_b

    model.load_state_dict(_prefix_sd(new_sd), strict=False)
    new_words = model_vocab - ckpt_vocab
    info = (f"vocab transplant: {ckpt_vocab:,} → {model_vocab:,} "
            f"({new_words:,} new words random-init) | "
            f"{copied} weights copied exactly")
    return True, info

if RESUME and os.path.exists("models/rnn_chatbot.pth"):
    try:
        ok, info = _load_with_vocab_transplant(
            "models/rnn_chatbot.pth", model, word2idx, device)
        if ok:
            ckpt_meta = torch.load("models/rnn_chatbot.pth", map_location="cpu")
            best_val_loss = ckpt_meta.get("best_val_loss", float("inf"))
            prev_ep       = ckpt_meta.get("epoch", 0)
            _print(f"✅ Checkpoint resumed — {info}")
            _print(f"   Previous best: epoch {prev_ep}, val_loss {best_val_loss:.4f}")
        else:
            _print(f"⚠️  Checkpoint resume skipped ({info}) — starting fresh")
    except Exception as e:
        _print(f"Checkpoint resume failed ({e})")

# ── Stage 1 → Stage 2 handoff ────────────────────────────────────────────────
# If SFT stage (Stage 2) and pretrained.pth exists, load pretrained weights.
elif TRAINING_STAGE == 2 and os.path.exists("models/pretrained.pth"):
    try:
        ok, info = _load_with_vocab_transplant(
            "models/pretrained.pth", model, word2idx, device)
        if ok:
            _print(f"✅ Pretrained weights loaded — {info}")
            _print(f"   Starting SFT from pretrained base (Stage 1 → Stage 2)")
        else:
            _print(f"⚠️  Pretrained load skipped ({info}) — starting fresh")
    except Exception as e:
        _print(f"Pretrained load failed ({e}) — starting fresh")


# ── Wrap in DDP AFTER weights are loaded ────────────────────────────────────
_raw_model = model
if _ddp_available:
    model = DDP(model, device_ids=[_local_rank], output_device=_local_rank,
                find_unused_parameters=False)
    _m = model.module   # unwrapped ref for direct attribute access (pad_idx, embedding, etc.)
    _print("DDP: enabled ✅  (2× T4)")
else:
    _m = model          # same object — no DDP wrapper
    # torch.compile disabled: costs ~2GB VRAM in static buffers on T4.
    # With 177K vocab the model is memory-bandwidth-bound, not compute-bound.
    # compile gives <5% speedup but pushes batch=16 into OOM territory.
    _print("torch.compile: disabled (saves 2GB VRAM for batch headroom)")

# ════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ════════════════════════════════════════════════════════════════
pad_idx   = word2idx[PAD_TOKEN]

def gpt_loss(logits, labels, quality_weights):
    """
    GPT causal LM loss with answer-only masking.
    logits : (B, T, vocab) — from model(x)
    labels : (B, T) — -100 for question/pad positions, token ids for answer positions
    quality_weights: (B,) per-sample loss multipliers

    Standard GPT shift: position t predicts t+1.
    shift_logits[:,:-1] predicts shift_labels[:,1:]
    CrossEntropyLoss(ignore_index=-100) skips question + pad tokens automatically.
    """
    shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, vocab)
    shift_labels = labels[:, 1:].contiguous()        # (B, T-1)

    B, T, V = shift_logits.shape

    loss_flat = nn.CrossEntropyLoss(
        ignore_index=-100, reduction="none"
    )(shift_logits.view(-1, V), shift_labels.view(-1))  # (B*T,)

    loss_2d     = loss_flat.view(B, T)
    valid       = (shift_labels != -100).float()
    per_sample  = (loss_2d * valid).sum(1) / valid.sum(1).clamp(min=1)
    weighted    = (per_sample * quality_weights.to(device)).mean()
    return weighted, per_sample.detach()


def contrastive_loss(hidden, labels, batch_size):
    """
    InfoNCE contrastive loss — now uses ANSWER token hidden states.
    For GPT: answer positions are where shifted labels (labels[:,1:]) != -100.
    hidden : (B, T, D) from model(x, return_hidden=True)
    labels : (B, T) raw labels (unshifted)
    """
    if batch_size < 2:
        return torch.tensor(0.0, device=device)

    # Shift to align with shifted loss: answer mask = labels[:,1:] != -100
    shift_labels = labels[:, 1:].contiguous()          # (B, T-1)
    shift_hidden = hidden[:, :-1, :]                   # (B, T-1, D)

    ans_mask = (shift_labels != -100).float().unsqueeze(-1)   # (B, T-1, 1)
    rep = (shift_hidden * ans_mask).sum(1) / ans_mask.sum(1).clamp(min=1)  # (B, D)
    rep = F.normalize(rep, dim=-1)

    sim    = torch.matmul(rep, rep.T) / CONTRASTIVE_TEMP
    labels_c = torch.arange(batch_size, device=device)
    return F.cross_entropy(sim, labels_c)


def dpo_loss(model, chosen_ids, rejected_ids, chosen_labels, rejected_labels,
             beta=0.1):
    """
    Direct Preference Optimization loss.
    Trains model to prefer chosen (👍) responses over rejected (👎) ones.

    chosen_ids   : (B, T) token ids for good responses
    rejected_ids : (B, T) token ids for bad responses
    chosen_labels / rejected_labels : (B, T) with -100 on prompt positions
    beta         : KL penalty coefficient (0.1 = mild, 0.5 = strong)

    Reference: Rafailov et al. 2023 "Direct Preference Optimization"
    """
    def _log_probs(ids, labs):
        logits = model(ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labs[:, 1:].contiguous()
        B, T, V = shift_logits.shape
        lp = -nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
            shift_logits.view(-1, V), shift_labels.view(-1)
        ).view(B, T)
        valid = (shift_labels != -100).float()
        return (lp * valid).sum(1) / valid.sum(1).clamp(min=1)  # (B,)

    chosen_lp   = _log_probs(chosen_ids,   chosen_labels)
    rejected_lp = _log_probs(rejected_ids, rejected_labels)
    # DPO objective: maximize margin between chosen and rejected log-probs
    loss = -F.logsigmoid(beta * (chosen_lp - rejected_lp)).mean()
    return loss


def load_dpo_pairs():
    """
    Load preference pairs from app.py feedback data.
    Reads memory/feedback.csv written by the 👍/👎 endpoint.
    Returns list of (question, good_answer, bad_answer) tuples.
    Falls back to SFT data if no feedback file exists.
    """
    feedback_path = "memory/feedback.csv"
    if not os.path.exists(feedback_path):
        print("[DPO] No feedback.csv found — using SFT pairs as chosen, augmented as rejected")
        return []
    try:
        import pandas as _pd
        fb = _pd.read_csv(feedback_path, dtype=str).fillna("")
        pairs = []
        for _, row in fb.iterrows():
            q    = str(row.get("question", "")).strip()
            good = str(row.get("good_answer", "")).strip()
            bad  = str(row.get("bad_answer",  "")).strip()
            if q and good and bad and good != bad:
                pairs.append((q, good, bad))
        print(f"[DPO] Loaded {len(pairs):,} preference pairs from {feedback_path}")
        return pairs
    except Exception as e:
        print(f"[DPO] Could not load feedback: {e}")
        return []

# ════════════════════════════════════════════════════════════════
# OPTIMISER & SCHEDULER
# ════════════════════════════════════════════════════════════════
optimizer = optim.AdamW(
    model.parameters(), lr=BASE_LR,
    weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-9
)

# ── AMP (Automatic Mixed Precision) ─────────────────────────────────────────
# Uses fp16 for most ops → ~2x memory reduction, ~1.5-2x faster on P100/T4/V100
# Quality impact: negligible — gradient scaler prevents underflow
_use_amp = _on_cuda
_scaler  = torch.cuda.amp.GradScaler(enabled=_use_amp)
_amp_ctx = torch.cuda.amp.autocast if _use_amp else (lambda: __import__('contextlib').nullcontext())
_print(f"AMP (mixed precision): {'enabled ✅' if _use_amp else 'disabled (CPU/MPS)'}")

train_ds_init = ChatDataset(train_inp, train_out, train_qw, is_train=True,
                            curriculum_phase=False)
# ── Pre-cache augmented pairs ─────────────────────────────────────────────────
_aug_inp = list(train_ds_init.inputs)
_aug_out = list(train_ds_init.outputs)
_aug_qw  = list(train_ds_init.qweights)

_train_sampler = DistributedSampler(train_ds_init, shuffle=True) if _ddp_available else None
train_dl = DataLoader(train_ds_init, batch_size=BATCH_SIZE,
                      sampler=_train_sampler,
                      shuffle=(not _ddp_available),   # DistributedSampler handles shuffle
                      collate_fn=collate_fn, drop_last=True,
                      num_workers=NUM_WORKERS, pin_memory=_on_cuda,
                      prefetch_factor=PREFETCH_FACTOR,
                      persistent_workers=PERSISTENT_WORKERS)
# DDP: each GPU sees len/world_size steps → total optimizer steps same as single GPU
steps_per_epoch = max(1, math.ceil(len(train_dl) / ACCUMULATION_STEPS))

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LR, epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch, pct_start=0.15,
    anneal_strategy="cos",
    div_factor=MAX_LR/BASE_LR, final_div_factor=100,
)

# ════════════════════════════════════════════════════════════════
# HARD EXAMPLE MINING — per-sample loss history
# ════════════════════════════════════════════════════════════════
# Maps augmented dataset index → running average loss
hem_loss_history: dict[int, float] = defaultdict(lambda: 1.0)

def compute_sample_weights(dataset, smoothing=0.9):
    """
    Returns sampling weights for WeightedRandomSampler based on per-sample loss.
    High loss examples get oversampled.
    """
    losses = [hem_loss_history[i] for i in range(len(dataset))]
    threshold = sorted(losses)[int(len(losses) * HEM_HARD_THRESHOLD)]
    weights = [HEM_OVERSAMPLE_WEIGHT if l >= threshold else 1.0 for l in losses]
    return weights

# ════════════════════════════════════════════════════════════════
# LOG
# ════════════════════════════════════════════════════════════════
log_path = "models/training_log.csv"
with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow([
        "epoch","train_loss","val_loss","train_ppl","val_ppl",
        "gap_ratio","lr","grad_norm","dropout","teacher_ratio",
        "contrastive_loss","mixup_batches"
    ])

patience_counter = 0
grad_norms = []

_STAGE_NAMES = {1: "Stage 1 — Pretraining", 2: "Stage 2 — SFT", 3: "Stage 3 — DPO"}
_print(f"\n{'='*70}")
_print(f" Hema Training v4 · {_STAGE_NAMES[TRAINING_STAGE]}  |  {_world_size}× GPU")
_print(f" {n_train} base pairs → ~{len(train_ds_init):,} augmented | {n_val} val")
if TRAINING_STAGE == 1:
    _print(f" Objective: next-token prediction on ALL tokens (no masking)")
elif TRAINING_STAGE == 2:
    _print(f" Objective: answer-only loss (question masked with -100)")
else:
    _print(f" Objective: Direct Preference Optimization (DPO)")
_print(f" Contrastive weight: {CONTRASTIVE_WEIGHT}  |  MixUp α={MIXUP_ALPHA} p={MIXUP_PROB}")
_print(f" Eff batch: {BATCH_SIZE}×{_world_size}×{ACCUMULATION_STEPS}={BATCH_SIZE*_world_size*ACCUMULATION_STEPS}  |  Target: Val PPL < 5  |  Gap < 1.5")
_print(f"{'='*70}\n")

# ════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Dropout annealing ────────────────────────────────────────────────────
    progress     = (epoch - 1) / max(1, EPOCHS - 1)
    current_drop = DROPOUT_START + (DROPOUT_END - DROPOUT_START) * progress
    for m in _raw_model.modules():
        if isinstance(m, nn.Dropout):
            m.p = current_drop

    # ── Scheduled sampling ───────────────────────────────────────────────────
    # Not applicable to GPT decoder-only — removed.
    ss_ratio = 0.0  # kept for log row compatibility only

    # ── Curriculum / Hard Example Mining DataLoader rebuild ─────────────────
    if epoch == 1 or epoch == CURRICULUM_EPOCHS + 1 or epoch % 10 == 0:
        use_curriculum = epoch <= CURRICULUM_EPOCHS
        hem_weights = compute_sample_weights(train_ds_init) if epoch > 1 else None

        if use_curriculum:
            order = sorted(range(len(_aug_inp)), key=lambda k: len(_aug_inp[k].split()))
            train_ds = ChatDataset([_aug_inp[k] for k in order],
                                   [_aug_out[k] for k in order],
                                   [_aug_qw[k]  for k in order], is_train=False)
            _samp = DistributedSampler(train_ds, shuffle=False) if _ddp_available else None
            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=_samp, shuffle=False,
                                  collate_fn=collate_fn,
                                  drop_last=(len(train_ds) > BATCH_SIZE * 2),
                                  num_workers=NUM_WORKERS, pin_memory=_on_cuda,
                                  prefetch_factor=PREFETCH_FACTOR,
                                  persistent_workers=PERSISTENT_WORKERS)
        elif hem_weights and len(_aug_inp) > BATCH_SIZE:
            train_ds = ChatDataset(_aug_inp, _aug_out, _aug_qw, is_train=False)
            if _ddp_available:
                local_n = math.ceil(len(train_ds) / _world_size)
                _samp = WeightedRandomSampler(
                    weights=hem_weights[:len(train_ds)],
                    num_samples=local_n, replacement=True)
            else:
                _samp = WeightedRandomSampler(
                    weights=hem_weights[:len(train_ds)],
                    num_samples=len(train_ds), replacement=True)
            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=_samp, collate_fn=collate_fn,
                                  drop_last=True, num_workers=NUM_WORKERS,
                                  pin_memory=_on_cuda, prefetch_factor=PREFETCH_FACTOR,
                                  persistent_workers=PERSISTENT_WORKERS)
        else:
            train_ds = ChatDataset(_aug_inp, _aug_out, _aug_qw, is_train=False)
            _samp = DistributedSampler(train_ds, shuffle=True) if _ddp_available else None
            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=_samp, shuffle=(not _ddp_available),
                                  collate_fn=collate_fn,
                                  drop_last=(len(train_ds) > BATCH_SIZE * 2),
                                  num_workers=NUM_WORKERS, pin_memory=_on_cuda,
                                  prefetch_factor=PREFETCH_FACTOR,
                                  persistent_workers=PERSISTENT_WORKERS)

        if epoch == 1:
            _print(f"Augmented training set: {len(train_ds):,} samples  |  "
                   f"Steps/epoch/GPU: {len(train_dl):,}")
        if epoch == CURRICULUM_EPOCHS + 1:
            _print(f"Epoch {epoch}: curriculum ended — HEM sampler active")

    # ── Advance DistributedSampler shuffle seed each epoch ───────────────────
    if _ddp_available and hasattr(train_dl.sampler, 'set_epoch'):
        train_dl.sampler.set_epoch(epoch)

    # ── TRAIN ────────────────────────────────────────────────────────────────
    model.train()
    epoch_loss    = 0.0
    epoch_closs   = 0.0
    epoch_gnorm   = 0.0
    mixup_count   = 0
    accum_count   = 0
    optimizer.zero_grad()

    _n_batches   = len(train_dl)
    _print_every = max(1, _n_batches // 10)

    for i, (x, labels, qw, idxs) in enumerate(train_dl):
        # x      : (B, T) unified sequence [<sos> q_words a_words <eos>]
        # labels : (B, T) -100 for question/pad, token ids for answer
        x, labels, qw = (x.to(device, non_blocking=True),
                         labels.to(device, non_blocking=True),
                         qw.to(device, non_blocking=True))
        B = x.shape[0]

        with _amp_ctx():
            # ── MixUp on unified sequence embeddings ────────────────────────────
            do_mixup = (random.random() < MIXUP_PROB) and (B > 1)

            if do_mixup:
                lam       = random.betavariate(MIXUP_ALPHA, MIXUP_ALPHA)
                perm      = torch.randperm(B, device=device)
                emb_a     = _m.embedding(x) * _m._emb_scale
                emb_b     = _m.embedding(x[perm]) * _m._emb_scale
                mixed_emb = lam * emb_a + (1 - lam) * emb_b
                causal_mask = _m._make_causal_mask(x.size(1), device)
                pad_mask    = (x == _m.pad_idx)
                hidden      = _m.transformer(_m.pos_encoder(mixed_emb),
                                             mask=causal_mask,
                                             src_key_padding_mask=pad_mask)
                logits      = _m.fc_out(hidden)
                mixup_count += 1
            else:
                # return_hidden=True: get hidden states for contrastive loss
                # — zero extra compute vs separate forward pass
                logits, hidden = model(x, return_hidden=True)

            # ── GPT causal LM loss (answer tokens only in Stage 2/3, all in Stage 1) ─
            loss, per_sample_loss = gpt_loss(logits, labels, qw)

            # ── Contrastive loss (Stage 2 only) ─────────────────────────────────
            _c_weight = (CONTRASTIVE_WEIGHT
                         if _STAGE_CONTRASTIVE is None
                         else _STAGE_CONTRASTIVE)
            if not do_mixup and _c_weight > 0:
                c_loss     = contrastive_loss(hidden, labels, B)
                total_loss = loss + _c_weight * c_loss
                epoch_closs += c_loss.item()
            else:
                total_loss = loss

        # ── Update hard example history ─────────────────────────────────────
        for k, idx in enumerate(idxs):
            old = hem_loss_history[idx]
            hem_loss_history[idx] = 0.9 * old + 0.1 * per_sample_loss[k].item()

        _scaler.scale(total_loss / ACCUMULATION_STEPS).backward()
        accum_count += 1

        if accum_count == ACCUMULATION_STEPS or (i + 1) == len(train_dl):
            _scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()
            epoch_gnorm += gnorm
            _scaler.step(optimizer)
            if getattr(scheduler, '_step_count', 0) < getattr(scheduler, 'total_steps', float('inf')):
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

        epoch_loss += loss.item()

        if (i + 1) % _print_every == 0 or i == _n_batches - 1:
            pct = (i + 1) / _n_batches * 100
            avg = epoch_loss / (i + 1)
            _print(f"  Ep {epoch:3d} | {pct:5.1f}% | step {i+1:>6}/{_n_batches} | loss {avg:.4f}")

    n_batches = max(1, len(train_dl))
    avg_train = epoch_loss / n_batches
    avg_closs = epoch_closs / n_batches
    avg_gnorm = epoch_gnorm / max(1, n_batches // ACCUMULATION_STEPS)
    grad_norms.append(avg_gnorm)

    # ── Barrier: flush all pending DDP gradient collectives before validation ─
    # Without this, rank-1 may enter its next collective (val_loss all_reduce)
    # while rank-0 still has a pending gradient broadcast → NCCL sequence mismatch
    if _ddp_available:
        dist.barrier()

    # ── VALIDATE ─────────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, labels, qw, _ in val_dl:
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with _amp_ctx():
                logits = _raw_model(x)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                val_loss += nn.CrossEntropyLoss(ignore_index=-100)(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1)
                ).item()

    # ── DDP: average val_loss across all ranks ───────────────────────────────
    if _ddp_available:
        _vl = torch.tensor(val_loss, device=device)
        dist.all_reduce(_vl, op=dist.ReduceOp.SUM)
        val_loss = _vl.item() / _world_size

    avg_val   = val_loss / max(1, len(val_dl))
    train_ppl = math.exp(min(avg_train, 50))
    val_ppl   = math.exp(min(avg_val,   50))
    gap_ratio = val_ppl / train_ppl if train_ppl > 0 else float("inf")
    current_lr = optimizer.param_groups[0]["lr"]
    elapsed    = time.time() - t0

    # ── LOG & SAVE — rank-0 only ─────────────────────────────────────────────
    if _is_main:
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{avg_train:.4f}", f"{avg_val:.4f}",
                f"{train_ppl:.2f}", f"{val_ppl:.2f}", f"{gap_ratio:.2f}",
                f"{current_lr:.2e}", f"{avg_gnorm:.3f}", f"{current_drop:.3f}",
                f"{ss_ratio:.2f}", f"{avg_closs:.4f}", mixup_count,
            ])

    # ── SAVE BEST — rank-0 only ───────────────────────────────────────────────
    saved = ""
    if _is_main:
        ckpt_dict = {
            "model": _raw_model.state_dict(), "word2idx": word2idx,
            "idx2word": idx2word, "epoch": epoch,
            "best_val_loss": best_val_loss,
            "optimizer": optimizer.state_dict(),
            "model_profile": MODEL_PROFILE,
            "embed_size": EMBED_SIZE,
            "num_heads": NUM_HEADS,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
        }
        if avg_val < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = avg_val
            best_epoch    = epoch
            patience_counter = 0
            best_weights  = copy.deepcopy(_raw_model.state_dict())
            
            save_target = "models/pretrained.pth" if TRAINING_STAGE == 1 else "models/rnn_chatbot.pth"
            torch.save(ckpt_dict, save_target)
            if TRAINING_STAGE == 1:
                torch.save(ckpt_dict, "models/rnn_chatbot.pth")

            saved = "  ⭐"
            try:
                import shutil as _sh
                if os.path.isdir("/kaggle/working"):
                    _sh.copy(save_target, f"/kaggle/working/{os.path.basename(save_target)}")
                _colab = "/content/drive/MyDrive/hema_checkpoints"
                if os.path.isdir(_colab):
                    _sh.copy(save_target, f"{_colab}/{os.path.basename(save_target)}")
            except Exception:
                pass
        else:
            patience_counter += 1

        if epoch % CHECKPOINT_EVERY == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(ckpt_dict, f"checkpoints/checkpoint_{epoch}.pth")

        if epoch % 5 == 0 or epoch == 1 or saved:
            ovf = "✓" if gap_ratio < 1.5 else ("~" if gap_ratio < 2.5 else "✗")
            gn_avg = sum(grad_norms[-10:]) / len(grad_norms[-10:])
            gh = "✓" if 0.005 < gn_avg < 3.0 else ("↓" if gn_avg <= 0.005 else "↑")
            eta = elapsed * (EPOCHS - epoch)
            eta_str = f"{int(eta//3600)}h{int((eta%3600)//60)}m"
            _print(
                f"Ep {epoch:3d}/{EPOCHS} | "
                f"TrainPPL {train_ppl:6.1f} | ValPPL {val_ppl:5.1f} | "
                f"Gap {gap_ratio:.2f}{ovf} | "
                f"SS {ss_ratio:.0%} | "
                f"CL {avg_closs:.3f} | "
                f"GN {avg_gnorm:.3f}{gh} | "
                f"LR {current_lr:.1e} | "
                f"{elapsed:.1f}s | ETA {eta_str}{saved}"
            )

    # ── Barrier: wait for rank-0 to finish saving before all ranks proceed ────
    # rank-1 would otherwise immediately call dist.broadcast (early-stop)
    # while rank-0 is still inside the if _is_main: save block → NCCL mismatch
    if _ddp_available:
        dist.barrier()

    # ── Broadcast early-stop decision from rank-0 to all ranks ──────────────
    if _ddp_available:
        _stop = torch.tensor(int(patience_counter >= EARLY_STOPPING_PATIENCE),
                             device=device)
        dist.broadcast(_stop, src=0)
        if _stop.item():
            _print(f"\n🛑 Early stopping at epoch {epoch} "
                   f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break
    else:
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            _print(f"\n🛑 Early stopping at epoch {epoch} "
                   f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break

# ════════════════════════════════════════════════════════════════
# RESTORE BEST & EVALUATE
# ════════════════════════════════════════════════════════════════
if best_weights is not None:
    _raw_model.load_state_dict(best_weights)
    _print(f"\nRestored best weights from epoch {best_epoch}")

model.eval()
test_loss = 0.0
with torch.no_grad():
    for x, labels, qw, _ in test_dl:
        x, labels = x.to(device), labels.to(device)
        logits = _raw_model(x)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        test_loss += nn.CrossEntropyLoss(ignore_index=-100)(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1)
        ).item()

if _ddp_available:
    _tl = torch.tensor(test_loss, device=device)
    dist.all_reduce(_tl, op=dist.ReduceOp.SUM)
    test_loss = _tl.item() / _world_size

test_ppl = math.exp(min(test_loss / max(1, len(test_dl)), 50))
_print(f"\n{'='*70}")
_print(f" Best model : epoch {best_epoch}  val_loss {best_val_loss:.4f}")
_print(f" Test PPL   : {test_ppl:.2f}")
_print(" ✅ Excellent" if test_ppl < 5 else (" ⚠ Acceptable" if test_ppl < 20 else " ✗ Needs more data"))
_print(f"{'='*70}")
_print("\n✅ Training complete.  Run:  python app.py")

# ── DDP cleanup ───────────────────────────────────────────────────────────────
if _ddp_available:
    dist.destroy_process_group()