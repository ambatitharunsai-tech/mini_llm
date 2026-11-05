import os
import torch
from model import ChatbotModel

# Determine checkpoint path
ckpt_path = "models/rnn_chatbot.pth"
if not os.path.exists(ckpt_path):
    ckpt_path = "models/hema_sft.pth"

if not os.path.exists(ckpt_path):
    print("❌ ERROR: No model checkpoint found at models/rnn_chatbot.pth or models/hema_sft.pth!")
    print("Please run 'python src/train.py' first to train a model.")
    exit(1)

import sys

# Device selection: use CUDA for external GPU, otherwise CPU (MPS avoided per system constraint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

if isinstance(ckpt, tuple):
    model, word2idx, idx2word = ckpt
    model = model.to(device)
else:
    word2idx = ckpt["word2idx"]
    idx2word = ckpt.get("idx2word", {v: k for k, v in word2idx.items()})
    vocab_size = len(word2idx)
    pad_idx = word2idx.get("<pad>", 0)
    embed_size = ckpt.get("embed_size", 32 if ckpt.get("model_profile") == "75k" else 256)
    num_heads = ckpt.get("num_heads", 2 if ckpt.get("model_profile") == "75k" else 4)
    hidden_size = ckpt.get("hidden_size", 128 if ckpt.get("model_profile") == "75k" else 1024)
    num_layers = ckpt.get("num_layers", 3 if ckpt.get("model_profile") == "75k" else 4)
    max_seq_len = ckpt.get("max_seq_len", 256)
    model = ChatbotModel(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        embed_size=embed_size,
        num_heads=num_heads,
        hidden_size=hidden_size,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    ).to(device)
    sd = ckpt.get("model", ckpt)
    # Strip torch.compile _orig_mod. prefix if present
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    if "pos_encoder.pe" in sd and sd["pos_encoder.pe"].shape != model.pos_encoder.pe.shape:
        del sd["pos_encoder.pe"]
    model.load_state_dict(sd, strict=False)

model.eval()

# Retrieve special tokens required by the Attention decoder
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

# Safety check
if SOS_TOKEN not in word2idx or EOS_TOKEN not in word2idx:
    print("❌ ERROR: Special tokens (<sos>, <eos>) not found in the vocabulary!")
    exit(1)


def encode(sentence):
    """Encodes user input using the trained vocabulary."""
    tokens = []
    for w in sentence.lower().split():
        if w in word2idx:
            tokens.append(word2idx[w])
        else:
            tokens.append(word2idx.get(UNK_TOKEN, 0))
            
    tokens.append(word2idx[EOS_TOKEN])
    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)


print(f"✨ Hema Attention Chatbot loaded from {ckpt_path} on {device}!")

if len(sys.argv) > 1:
    user = " ".join(sys.argv[1:]).strip()
    src = encode(user)
    tokens = model.generate(prompt=src, eos_idx=word2idx[EOS_TOKEN])
    response = " ".join([
        idx2word[t] for t in tokens 
        if idx2word.get(t, "") not in [SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN]
    ])
    print(f"Prompt: {user}")
    print(f"AI: {response}")
    sys.exit(0)

print("Type 'quit' or 'exit' to stop.")
while True:
    try:
        user = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    
    if user.lower() in ["quit", "exit"]:
        break
    if not user:
        continue

    src = encode(user)

    if src.numel() <= 1:  # Only EOS token present
        print("AI: I don't understand.")
        continue

    # Use the step-by-step autoregressive generation
    tokens = model.generate(prompt=src, eos_idx=word2idx[EOS_TOKEN])

    # Convert back to words, stripping out all special utility tokens
    response = " ".join([
        idx2word[t] for t in tokens 
        if idx2word.get(t, "") not in [SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN]
    ])

    print("AI:", response)