"""
download_pretrain_data.py — Stream from FineWeb + C4 + The Pile + Wikipedia
============================================================================
Uses HuggingFace streaming — NO full download of 800GB needed.
Samples optimally for a 23M param model (Chinchilla: ~460M tokens).

Dataset mix:
  FineWeb          150K docs  ~100M tokens   (modern clean web)
  C4               200K docs  ~150M tokens   (Google-quality web)
  The Pile         200K docs  ~150M tokens   (books+papers+web)
  Wikipedia Simple  all docs  ~120M tokens   (factual knowledge)
  TinyStories      100K docs  ~80M tokens    (grammar+flow)
  Total:           ~650K docs  ~600M tokens

Run:  pip install datasets && python data/download_pretrain_data.py
"""

import os, random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "data" else _HERE
OUT_DIR  = os.path.join(_ROOT, "data", "pretrain")
OUT_FILE = os.path.join(OUT_DIR, "pretrain_corpus.txt")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    from datasets import load_dataset
except ImportError:
    print("Run: pip install datasets"); exit(1)

all_texts = []
source_counts = {}

def stream_sample(name, hf_name, config, split, field, n, min_words=40, desc=""):
    print(f"\n{'='*55}\n  {desc}\n  Streaming {n:,} docs from {hf_name}\n{'='*55}")
    texts, skipped = [], 0
    try:
        kw = dict(split=split, streaming=True, trust_remote_code=True)
        if config: kw["name"] = config
        ds = load_dataset(hf_name, **kw)
        for row in ds:
            if len(texts) >= n: break
            text = " ".join(str(row.get(field,"")).split())
            if len(text.split()) < min_words:
                skipped += 1; continue
            texts.append(" ".join(text.split()[:300]))
            if len(texts) % 50_000 == 0:
                print(f"    {len(texts):,}/{n:,}...")
        toks = sum(len(t.split()) for t in texts)
        print(f"  OK {len(texts):,} docs  ~{toks/1e6:.0f}M tokens  (skipped {skipped:,})")
    except Exception as e:
        print(f"  FAILED: {e}")
    source_counts[name] = len(texts)
    return texts

# 1. FineWeb
all_texts += stream_sample("FineWeb", "HuggingFaceFW/fineweb",
    "sample-10BT","train","text",150_000,40,"FineWeb — modern clean web text")

# 2. C4
all_texts += stream_sample("C4", "allenai/c4",
    "en","train","text",200_000,40,"C4 — Colossal Clean Crawled Corpus")

# 3. The Pile
pile = stream_sample("The Pile", "EleutherAI/pile",
    "all","train","text",200_000,40,"The Pile — books+papers+web")
if not pile:
    pile = stream_sample("The Pile", "monology/pile-uncopyrighted",
        None,"train","text",200_000,40,"Pile-uncopyrighted (fallback)")
all_texts += pile

# 4. Wikipedia Simple
print(f"\n{'='*55}\n  Wikipedia Simple English\n{'='*55}")
try:
    wiki = load_dataset("wikimedia/wikipedia","20231101.simple",
                        split="train", trust_remote_code=True)
    wt = [" ".join(r["text"].split()[:300]) for r in wiki
          if len(r["text"].split()) >= 30]
    all_texts += wt
    source_counts["Wikipedia"] = len(wt)
    print(f"  OK {len(wt):,} articles  ~{sum(len(t.split()) for t in wt)/1e6:.0f}M tokens")
except Exception as e:
    print(f"  FAILED: {e}")

# 5. TinyStories
all_texts += stream_sample("TinyStories", "roneneldan/TinyStories",
    None,"train","text",100_000,20,"TinyStories — grammar and language flow")

# 6. Existing chat_dataset
import pandas as pd
csv_path = os.path.join(_ROOT,"data","chat_dataset.csv")
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        df = df[df["input"].str.strip().str.lower() != "input"]
        extra = [f"{r['input']} {r['output']}" for _,r in df.iterrows()
                 if len(f"{r['input']} {r['output']}".split()) >= 10]
        all_texts += extra
        source_counts["chat_dataset"] = len(extra)
        print(f"\n  + chat_dataset.csv: {len(extra):,} pairs")
    except Exception: pass

# Deduplicate + shuffle
seen, unique = set(), []
for t in all_texts:
    k = t[:80]
    if k not in seen:
        seen.add(k); unique.append(t)
random.shuffle(unique)
all_texts = unique

# Save
with open(OUT_FILE,"w",encoding="utf-8") as f:
    for t in all_texts:
        if t.strip(): f.write(t.strip()+"\n")

total_toks = sum(len(t.split()) for t in all_texts)
file_mb    = os.path.getsize(OUT_FILE)/1e6
chinchilla = 460_000_000

print(f"\n{'='*55}")
print(f" Pretrain corpus ready!")
print(f"{'='*55}")
print(f"  Documents:    {len(all_texts):,}")
print(f"  Tokens:       ~{total_toks/1e6:.0f}M  (Chinchilla optimal: 460M)")
print(f"  File size:    {file_mb:.0f} MB")
print(f"  Status:       {'ABOVE optimal' if total_toks >= chinchilla else f'{total_toks/chinchilla*100:.0f}% of optimal'}")
print(f"\n  Mix:")
for src, n in source_counts.items():
    pct = n/max(1,len(all_texts))*100
    print(f"    {src:<20} {n:>7,} docs ({pct:.0f}%)")
print(f"\n  Next: python src/pretrain.py")