"""
generate_full_dataset.py — Download Alpaca + Dolly and save to chat_dataset.csv
================================================================================
Downloads:
  - Alpaca 52K  (yahma/alpaca-cleaned)
  - Dolly 15K   (databricks/databricks-dolly-15k)

Saves to:  data/chat_dataset.csv
Columns:   input, output

Run:
    pip install datasets
    python data/generate_full_dataset.py
"""

import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "data" else _HERE
OUTPUT = os.path.join(_ROOT, "data", "chat_dataset.csv")

os.makedirs(os.path.join(_ROOT, "data"), exist_ok=True)

try:
    from datasets import load_dataset
except ImportError:
    print("❌ datasets not installed. Run:  pip install datasets")
    exit(1)

all_pairs = []

# ── 1. Alpaca 52K ────────────────────────────────────────────────────────────
print("Downloading Alpaca 52K...")
try:
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    for r in alpaca:
        q = r["instruction"].strip()
        ctx = r.get("input", "").strip()
        a = r["output"].strip()
        if ctx:
            q = f"{q}\n{ctx}"
        if q and a:
            all_pairs.append({"input": q, "output": a})
    print(f"  Alpaca: {len(all_pairs):,} pairs loaded")
except Exception as e:
    print(f"  Alpaca failed: {e}")

# ── 2. Dolly 15K ─────────────────────────────────────────────────────────────
print("Downloading Dolly 15K...")
dolly_start = len(all_pairs)
try:
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    for r in dolly:
        q = r["instruction"].strip()
        ctx = r.get("context", "").strip()
        a = r["response"].strip()
        if ctx:
            q = f"{q}\n{ctx}"
        if q and a:
            all_pairs.append({"input": q, "output": a})
    print(f"  Dolly: {len(all_pairs) - dolly_start:,} pairs loaded")
except Exception as e:
    print(f"  Dolly failed: {e}")

# ── Build DataFrame ───────────────────────────────────────────────────────────
print("\nCleaning and deduplicating...")
df = pd.DataFrame(all_pairs, columns=["input", "output"])

# Drop empty rows
df = df[df["input"].str.strip().ne("") & df["output"].str.strip().ne("")]

# Deduplicate by input — keep longest answer
df["_len"] = df["output"].str.len()
df = df.sort_values("_len", ascending=False)
df = df.drop_duplicates(subset="input", keep="first")
df = df.drop(columns=["_len"]).reset_index(drop=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Save ─────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False)
print(f"\n✅ Saved {len(df):,} pairs to {OUTPUT}")
print(f"\nSample:")
for _, row in df.head(3).iterrows():
    print(f"  input:  {row['input'][:70]}...")
    print(f"  output: {row['output'][:70]}...")
    print()
print("Next step:  python src/train.py")