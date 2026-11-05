"""
vector_db.py — Hema FAISS Vector Search  (v3)
==============================================
v3 fixes:
  - Segfault fix: caps dataset at MAX_ROWS best entries before encoding
    (261K × 384-dim float32 = ~400MB → crashes MacBook Air RAM)
  - Saves FAISS index to disk → only encodes once ever, instant on restart
  - Frees numpy embeddings array after index is built (saves ~400MB RAM)
  - IVF index for large datasets (>10K rows) → faster search, less RAM
  - Keeps all v2 fixes: dedup, dtype=str, batch encoding, cosine similarity
"""

import os
# macOS OpenMP fix — must be before faiss/torch imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import multiprocessing
import faiss
faiss.omp_set_num_threads(1)
multiprocessing.set_start_method("spawn", force=True)

# ── Config ───────────────────────────────────────────────────────────────────
MAX_ROWS       = 1500    # cap: top N rows by answer quality (prevents OOM/segfault)
BATCH_SIZE     = 64    # encode N questions at a time
CACHE_DIR      = "models"
CACHE_INDEX    = os.path.join(CACHE_DIR, "faiss_index.bin")
CACHE_ANSWERS  = os.path.join(CACHE_DIR, "vector_answers.txt")
CACHE_META     = os.path.join(CACHE_DIR, "vector_meta.txt")  # stores row count for cache validation

# ── Embedding model ──────────────────────────────────────────────────────────
print("Loading embedding model...")
embed_model = None
try:
    from sentence_transformers import SentenceTransformer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    embed_model = SentenceTransformer("all-MiniLM-L6-v2",device="cpu")
    print("Embedding model ready.")
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer: {e}")

# ── Globals ──────────────────────────────────────────────────────────────────
answers: list[str] = []
index = None

# ── Helpers ──────────────────────────────────────────────────────────────────
def _cache_valid(expected_rows: int) -> bool:
    """True if all cache files exist and match expected row count."""
    if not all(os.path.exists(f) for f in (CACHE_INDEX, CACHE_ANSWERS, CACHE_META)):
        return False
    try:
        cached_n = int(open(CACHE_META).read().strip())
        return cached_n == expected_rows
    except Exception:
        return False


def _select_best_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Pick the N most informative rows — prioritise longest answers.
    """
    df = df.copy()
    df["_score"] = df["output"].str.len()
    df = df.sort_values("_score", ascending=False).head(n)
    df = df.drop(columns=["_score"]).reset_index(drop=True)
    return df


# ── Build / load index ───────────────────────────────────────────────────────
try:
    import faiss

    if not os.path.exists("data/chat_dataset.csv"):
        print("Warning: data/chat_dataset.csv not found. Vector search disabled.")
    elif embed_model is None:
        print("Vector search disabled (no embedding model).")
    else:
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Load & deduplicate dataset
        print("Loading dataset for vector DB...")
        data = pd.read_csv("data/chat_dataset.csv", dtype=str, low_memory=False).fillna("")
        data = data[data["input"].str.lower().str.strip() != "input"]  # remove header rows

        # Dedup on input
        data["_len"] = data["output"].str.len()
        data = data.sort_values("_len", ascending=False)
        data = data.drop_duplicates(subset="input", keep="first")
        data = data.drop(columns=["_len"]).reset_index(drop=True)

        # Cap to MAX_ROWS best entries
        if len(data) > MAX_ROWS:
            print(f"Dataset has {len(data):,} rows — selecting top {MAX_ROWS:,} by quality...")
            data = _select_best_rows(data, MAX_ROWS)

        questions_use = data["input"].astype(str).tolist()
        answers_use   = data["output"].astype(str).tolist()
        n_rows = len(questions_use)
        print(f"Vector DB will index {n_rows:,} entries.")

        if _cache_valid(n_rows):
            # ── Fast path: load from disk ──────────────────────────────────
            print("Loading cached FAISS index...")
            index   = faiss.read_index(CACHE_INDEX)
            answers = open(CACHE_ANSWERS, encoding="utf-8").read().splitlines()
            print(f"FAISS index loaded from cache ({index.ntotal:,} vectors).")

        else:
            # ── Slow path: encode + build + save ──────────────────────────
            print(f"Encoding {n_rows:,} questions in batches of {BATCH_SIZE}...")
            all_emb = []
            for start in range(0, n_rows, BATCH_SIZE):
                batch = questions_use[start : start + BATCH_SIZE]
                emb   = embed_model.encode(batch, show_progress_bar=False, convert_to_numpy=True,device="cpu")
                all_emb.append(emb.astype("float32"))
                done = min(start + BATCH_SIZE, n_rows)
                print(f"  {done:,}/{n_rows:,}  ({done/n_rows*100:.0f}%)", end="\r")
            print()

            embeddings = np.vstack(all_emb).astype("float32")
            faiss.normalize_L2(embeddings)

            # Choose index type based on size
            dim = embeddings.shape[1]
            if n_rows >= 2000:
                # IVFFlat: much faster search and lower memory than Flat
                n_lists = max(16, int(n_rows ** 0.5))  # sqrt heuristic
                quantiser = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantiser, dim, n_lists,
                                            faiss.METRIC_INNER_PRODUCT)
                index.train(embeddings)
            else:
                index = faiss.IndexFlatIP(dim)  # Flat for small sets

            index.add(embeddings)

            # Free the large array immediately — saves ~200-400MB RAM
            del embeddings, all_emb

            # Save to disk
            faiss.write_index(index, CACHE_INDEX)
            with open(CACHE_ANSWERS, "w", encoding="utf-8") as f:
                f.write("\n".join(answers_use))
            with open(CACHE_META, "w") as f:
                f.write(str(n_rows))

            answers = answers_use
            print(f"FAISS index saved to {CACHE_INDEX}")
            print(f"Vector database ready! ({index.ntotal:,} vectors)")

except Exception as e:
    print(f"Error initializing vector database: {e}")
    import traceback; traceback.print_exc()


# ── Search ───────────────────────────────────────────────────────────────────
def search(query: str, top_k: int = 3, threshold: float = 0.40) -> str | None:
    """
    Semantic search over the vector index.
    Returns best matching answer string, or None if score below threshold.
    """
    if index is None or embed_model is None or not answers:
        return None
    try:
        # Set IVF probe count for better recall (if IVF index)
        if hasattr(index, "nprobe"):
            index.nprobe = min(10, index.nlist)

        query_vec = embed_model.encode([query])
        query_vec = np.array(query_vec).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = index.search(query_vec, top_k)
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and 0 <= idx < len(answers):
                return answers[idx]
    except Exception as e:
        print(f"Vector search error: {e}")
    return None


def invalidate_cache():
    """Call after updating the dataset to force re-encoding on next start."""
    import faiss as _f
    for path in (CACHE_INDEX, CACHE_ANSWERS, CACHE_META):
        if os.path.exists(path):
            os.remove(path)
            print(f"Cache removed: {path}")


# ── Test mode ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if index is not None:
        print("\nVector search ready. Type 'exit' to stop.\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            result = search(q)
            print(f"\n{'Match: ' + result if result else 'No match found.'}\n")
    else:
        print("\nVector DB inactive — check model and data.")