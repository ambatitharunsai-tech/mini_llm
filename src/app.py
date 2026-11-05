import os
# ── macOS OpenMP fix ─────────────────────────────────────────────────────────
# PyTorch and FAISS each bundle libomp.dylib — macOS aborts when both load.
# This env var must be set BEFORE torch or faiss are imported.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import difflib
import requests
import torch
import base64
import io
import random
import random

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
# File reading support
try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("pypdf not installed. PDF reading disabled. Run: pip install pypdf")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False



# Google search — install with: pip install googlesearch-python
try:
    from googlesearch import search as google_search
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Google search not installed. Run: pip install googlesearch-python")

# Playwright headless Chrome — install with:
#   pip install playwright && playwright install chromium
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("[Playwright] Not installed. Falling back to requests scraper.")
    print("  To enable: pip install playwright && playwright install chromium")

import wikipedia
from flask import Flask, request, jsonify, render_template
# ── Hema NLP Engine ──────────────────────────────────────────────────────────
try:
    from nlp_engine import HemaNLP, get_nlp
    _NLP_AVAILABLE = True
except ImportError:
    _NLP_AVAILABLE = False
    print("[WARNING] nlp_engine.py not found — falling back to legacy NLP")

from nlp_engine import ReasoningEngine

reasoner = ReasoningEngine()

from model import ChatbotModel

# ── Vector DB ────────────────────────────────────────────────────────────────
try:
    from vector_db import search as vector_search
except ImportError:
    print("Warning: vector_db.py not found.")
    def vector_search(query): return None

app = Flask(__name__)

def generate_bot_response(user_query, model, tokenizer, device):
    """
    Generates a response using the new Decoder-Only HemaModel architecture.
    """
    # 1. Format the prompt EXACTLY as it was formatted in training
    prompt = f"<sos>User: {user_query}\nBot: "
    
    # 2. Tokenize and move to device
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    # 3. Autoregressive Generation using your new model.py method
    #    This utilizes Top-K, Temperature, and Repetition Penalty for smart sampling
    try:
        eos_id = tokenizer.vocab.get("<eos>", 3)
        generated_ids = model.generate(
            idx=input_ids, 
            max_new_tokens=512,        # Allow long answers (good for code/math)
            temperature=0.7,           # 0.7 balances creativity and logic
            top_k=40, 
            eos_id=eos_id,
            repetition_penalty=1.15    # Prevents infinite while-loops in generation
        )
        
        # 4. Decode the output
        # The model returns the FULL sequence (prompt + response).
        # We decode the whole thing, then split it to get just the Bot's answer.
        full_text = tokenizer.decode(generated_ids[0].tolist())
        
        # Extract just the response part
        if "Bot: " in full_text:
            bot_response = full_text.split("Bot: ")[-1].replace("<eos>", "").strip()
        else:
            bot_response = full_text.replace("<eos>", "").strip()
            
        return bot_response

    except Exception as e:
        print(f"Inference Error: {e}")
        return "I'm sorry, my language model encountered an error while generating a response."


# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL
# ════════════════════════════════════════════════════════════════════════════
print("Loading Hema AI brain...")
model = None
tokenizer = None
word2idx, idx2word = {}, {}  # Legacy fallback empty dicts
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

try:
    from bpe_tokenizer import BPETokenizer
    from model import HemaModel
    
    tok_path = "models/bpe_tokenizer.json"
    ckpt_path = "models/hema_sft.pth" if os.path.exists("models/hema_sft.pth") else "models/rnn_chatbot.pth"
    
    if os.path.exists(tok_path) and os.path.exists(ckpt_path):
        tokenizer = BPETokenizer.load(tok_path)
        actual_vocab_size = max(tokenizer.vocab.values()) + 1
        
        model = HemaModel(vocab_size=actual_vocab_size, max_len=1024).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get("model", ckpt)
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        print(f"✅ Hema AI Model loaded from {ckpt_path}. Vocabulary: {actual_vocab_size} tokens")
    else:
        print("⚠️ Checkpoint (hema_sft.pth/rnn_chatbot.pth) or bpe_tokenizer.json not found — local AI disabled.")
except Exception as e:
    print(f"Model load error: {e}")

UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

chat_history   = []
last_exchange  = {"q": "", "a": ""}
MAX_HISTORY    = 50   # kept for chat_history compat

# ════════════════════════════════════════════════════════════════════
# PERSISTENT MEMORY — survives server restarts
# ════════════════════════════════════════════════════════════════════
_MEMORY_DIR    = "memory"
_CONV_FILE     = "memory/conversation.json"
_PROFILE_FILE  = "memory/user_profile.json"
MAX_CONV_STORE = 500    # max turns on disk
MAX_CONV_LOAD  = 100    # max turns loaded into RAM at startup

conversation = []       # [{role, text, topic, intent}]
user_profile = {
    "topics":        [],   # last 50 unique topics discussed
    "last_seen":     "",
    "session_count": 0,
    "name":          ""
}

import threading as _threading
import datetime as _datetime

def _ensure_memory_dir():
    os.makedirs(_MEMORY_DIR, exist_ok=True)

def _load_memory():
    """Load conversation history and user profile from disk at startup."""
    global conversation, user_profile
    _ensure_memory_dir()
    # Load conversation
    if os.path.exists(_CONV_FILE):
        try:
            turns = json.load(open(_CONV_FILE, encoding="utf-8"))
            conversation = turns[-MAX_CONV_LOAD:]  # only load last N turns
            print(f"[Memory] Loaded {len(conversation)} conversation turns from disk.")
        except Exception as e:
            print(f"[Memory] Could not load conversation: {e}")
    # Load user profile
    if os.path.exists(_PROFILE_FILE):
        try:
            user_profile = json.load(open(_PROFILE_FILE, encoding="utf-8"))
            user_profile["session_count"] = user_profile.get("session_count", 0) + 1
        except Exception as e:
            print(f"[Memory] Could not load profile: {e}")
    user_profile["last_seen"] = _datetime.datetime.now().isoformat(timespec="seconds")
    _save_profile_sync()

def _save_conversation_sync():
    _ensure_memory_dir()
    try:
        # Keep only last MAX_CONV_STORE turns on disk
        to_save = conversation[-MAX_CONV_STORE:]
        with open(_CONV_FILE, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False)
    except Exception as e:
        print(f"[Memory] Save conversation error: {e}")

def _save_profile_sync():
    _ensure_memory_dir()
    try:
        with open(_PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(user_profile, f, ensure_ascii=False)
    except Exception as e:
        print(f"[Memory] Save profile error: {e}")

def _save_memory_async():
    """Write conversation + profile to disk in a background thread."""
    _threading.Thread(target=_save_conversation_sync, daemon=True).start()
    _threading.Thread(target=_save_profile_sync, daemon=True).start()

def _update_profile(topic: str):
    """Track topics in user_profile (last 50 unique, most recent first)."""
    if not topic or len(topic.split()) < 2:
        return
    topics = user_profile.get("topics", [])
    topic_clean = topic.strip().lower()[:80]
    if topic_clean in topics:
        topics.remove(topic_clean)
    topics.insert(0, topic_clean)
    user_profile["topics"] = topics[:50]

def get_remembered_topics(n: int = 5) -> list:
    """Return the n most recently discussed topics."""
    return user_profile.get("topics", [])[:n]

def add_to_conversation(role, text, topic="", intent=""):
    """Append a turn, update user profile, async-save to disk."""
    turn = {
        "role":   role,
        "text":   text.strip(),
        "topic":  (topic or text).strip()[:120],
        "intent": intent,
    }
    conversation.append(turn)
    if len(conversation) > MAX_CONV_STORE:
        del conversation[:50]
    if role == "user":
        _update_profile((topic or text).strip()[:80])
    _save_memory_async()

def get_last_topic():
    """Return the most recent meaningful user topic."""
    for turn in reversed(conversation):
        if turn["role"] == "user" and len(turn["topic"].split()) >= 2:
            return turn["topic"]
    return ""

def build_context_prompt(user_text, max_turns=20):
    """
    Build a context-aware prompt by prepending recent conversation history.
    Uses ALL stored turns (up to max_turns recent ones) so Hema reads
    the full conversation every time it searches.

    Format:
        [Conversation so far]
        User: ...
        Hema: ...
        ...
        [Current question]
        User: <user_text>
    """
    if not conversation:
        return user_text

    # Take up to max_turns most recent turns (excluding current)
    recent = conversation[-(max_turns):]  if len(conversation) > 1 else []

    # Only include turns that came before this message
    # (last entry is the current user turn we just added)
    if recent and recent[-1]["role"] == "user" and recent[-1]["text"] == user_text:
        recent = recent[:-1]

    if not recent:
        return user_text

    history_lines = []
    for turn in recent:
        label = "User" if turn["role"] == "user" else "Hema"
        # Truncate very long AI replies in context to save space
        text = turn["text"]
        if turn["role"] == "ai" and len(text) > 300:
            text = text[:300] + "..."
        history_lines.append(f"{label}: {text}")

    history_str = "\n".join(history_lines)
    return (
        f"[Conversation so far]\n{history_str}\n\n"
        f"[Current question]\nUser: {user_text}"
    )



def get_topic(text):

    words = text.lower().split()

    keywords = [w for w in words if len(w) > 4]

    return " ".join(keywords[:3])

def resolve_followup(user_text):
    """
    Detect short follow-ups ("with example", "explain", "more" etc.)
    and expand them with the last topic from conversation memory.
    Returns (resolved_text, was_resolved).
    """
    FOLLOWUP_TRIGGERS = {
        "with example", "show example", "give example", "example", "examples",
        "example please", "more", "more details", "tell me more", "explain",
        "explain more", "explain it", "elaborate", "continue", "go on",
        "what else", "in detail", "detailed", "show me", "how", "why",
        "in python", "in code", "code for it", "code", "summarise", "summarize",
        "pros and cons", "advantages", "disadvantages", "compare", "vs",
        "difference", "better", "best", "worst",
    }
    stripped = user_text.strip().lower().rstrip("?!.")
    if stripped in FOLLOWUP_TRIGGERS or len(user_text.split()) <= 2:
        last = get_last_topic()
        if last:
            return f"{stripped} {last}".strip(), True
    return user_text, False

# ════════════════════════════════════════════════════════════════════════════
# SPELL CORRECTOR — fixes typos like "waht is pytohn" → "what is python"
# ════════════════════════════════════════════════════════════════════════════

COMMON_WORDS = set("""
a about above after again against age ago all also am an and any are as at
be because been before being below between both but by can cannot come could
did do does doing down during each few for from further get got had has have
having he her here hers herself him himself his how i if in into is it its
itself let me more most my myself no nor not of off on once only or other
our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves
is are was were be been being have has had do does did will would shall should
may might must can could tell give show explain describe list name define mean
tell me give me show me explain what is how does why is what are who is
chat chit talk just okay cool nice great fine sure yes no maybe alright
hello hi hey bye good morning afternoon evening night thanks thank you please
sorry love like know think feel want need try ask help information info
fun funny interesting cool awesome nice work school home life world time
people person thing things place places idea ideas question answer
""".split())

QUESTION_WORDS = {
    "waht": "what", "whta": "what", "wath": "what", "whats": "what is",
    "wha": "what", "wat": "what", "wot": "what",
    "hwo": "how", "howd": "how does", "howdo": "how do",
    "whi": "why", "whay": "why", "wgy": "why",
    "wher": "where", "wehre": "where", "whre": "where",
    "wen": "when", "wehn": "when",
    "woh": "who", "whu": "who",
    "whihc": "which", "wich": "which",
    "iz": "is", "ar": "are", "dos": "does", "deos": "does",
    "cna": "can", "cn": "can",
    "tel": "tell", "giv": "give",
    "explan": "explain", "explian": "explain", "expain": "explain",
    "defin": "define", "defne": "define",
    "desribe": "describe", "descibe": "describe",
    "pyton": "python", "pytohn": "python", "phyton": "python", "pytoon": "python",
    "javascrip": "javascript", "javscript": "javascript", "javasript": "javascript",
    "machne": "machine", "learing": "learning", "lerning": "learning",
    "artifical": "artificial", "inteligence": "intelligence", "intellgence": "intelligence",
    "algortihm": "algorithm", "algoritm": "algorithm", "algorythm": "algorithm",
    "databse": "database", "datbase": "database",
    "programing": "programming", "progrming": "programming",
    "scince": "science", "sceince": "science", "sciense": "science",
    "mathamatics": "mathematics", "mathimatics": "mathematics", "maths": "math",
    "geografy": "geography", "geograpy": "geography",
    "histroy": "history", "hisory": "history",
    "tecnology": "technology", "techology": "technology", "tecnolgy": "technology",
    "helath": "health", "healt": "health",
    "philosphy": "philosophy", "philosofy": "philosophy",
    "busines": "business", "bussiness": "business", "buisness": "business",
    "gramer": "grammar", "grammer": "grammar",
    "exmaple": "example", "exampl": "example", "exampel": "example",
    "gess": "guess", "ges": "guess",
    "yuo": "you", "yu": "you", "u": "you",
    "hav": "have", "havv": "have",
    "teh": "the", "te": "the",
    "tis": "this", "ths": "this",
    "mke": "make", "amke": "make",
    "shwo": "show", "shw": "show",
    "wriet": "write", "writ": "write",
    "recieve": "receive", "recive": "receive",
    "seperate": "separate",
    "occured": "occurred", "occurance": "occurrence",
    "definately": "definitely", "definetly": "definitely",
    "noticable": "noticeable",
    "calender": "calendar",
    "accomodate": "accommodate",
    "untill": "until",
    "begining": "beginning",
    "existance": "existence",
    "independance": "independence",
    "goverment": "government", "govenment": "government",
    "enviroment": "environment", "enviroment": "environment",
    "knowlege": "knowledge", "knwoledge": "knowledge",
    "langauge": "language", "languege": "language",
    "reserch": "research", "reaserch": "research",
    "develpment": "development", "develoment": "development",
    "diffrence": "difference", "diferent": "different",
    "intresting": "interesting", "intersting": "interesting",
    "importent": "important", "impotant": "important",
    "sucess": "success", "succees": "success",
    "becuase": "because", "becase": "because",
    "alot": "a lot", "alote": "a lot",
}

SPELL_VOCAB = set(COMMON_WORDS)

def build_spell_vocab():
    global SPELL_VOCAB
    for q, a in dataset_qa.items():
        SPELL_VOCAB.update(q.lower().split())
        SPELL_VOCAB.update(a.lower().split())
    SPELL_VOCAB = {w for w in SPELL_VOCAB if len(w) >= 2 and w.isalpha()}
    print(f"Spell checker ready: {len(SPELL_VOCAB)} words")

# Words that must NEVER be spell-corrected — common words the spell checker
# might wrongly "fix" to something unrelated (e.g. "chat" → "what")
NEVER_CORRECT = {
    # Chat / conversation
    "chat","chats","chatting","talk","talks","talking","just","fine","cool",
    "okay","ok","sure","yep","nope","yup","nah","yeah","yea","hmm","hm",
    # Common short words easily confused
    "hi","by","my","us","am","an","be","he","in","it","me","no","of","on",
    "or","so","to","up","we","as","at","do","go","if","is","he","she","they",
    # Tech/coding terms — acronyms & abbreviations
    "api","url","css","sql","git","pip","npm","js","py","ai","ml","dl",
    "app","web","dev","ops","gpu","cpu","ram","ide","cli","gui","oop",
    "nlp","cnn","rnn","gru","gan","llm","bert","gpt","vit","llm","rl",
    "tcp","udp","http","https","dns","ssh","ftp","smtp","json","xml","yaml",
    "html","csv","pdf","png","jpg","svg","md","db","os","vm","io","ui","ux",
    "amd","arm","tpu","fpga","ssd","hdd","lan","wan","vpn","cdn","aws","gcp",
    "jwt","rsa","aes","sha","md5","utf","rgb","hex","ocr","tts","stt",
    # Full tech words the spell checker might mangle
    "python","java","javascript","typescript","kotlin","swift","golang","rust",
    "django","flask","fastapi","react","angular","vue","node","express",
    "pytorch","tensorflow","keras","numpy","pandas","matplotlib","scipy",
    "sklearn","scikit","seaborn","plotly","bokeh","opencv","pillow",
    "transformer","transformers","encoder","decoder","tokenizer","tokenize",
    "embedding","embeddings","softmax","sigmoid","relu","gelu","dropout",
    "backprop","backpropagation","gradient","gradients","epoch","epochs",
    "perceptron","neuron","neurons","synapse","dendrite","axon",
    "algorithm","algorithms","recursion","recursive","iteration","iterative",
    "polymorphism","encapsulation","abstraction","inheritance","interface",
    "boolean","integer","float","string","tuple","vector","matrix","tensor",
    "async","await","callback","promise","closure","lambda","decorator",
    "compiler","interpreter","runtime","bytecode","assembly","binary",
    "kubernetes","docker","microservice","serverless","firebase","heroku",
    "blockchain","bitcoin","ethereum","cryptocurrency","cryptography",
    "regex","regexp","syntax","semantics","pragma","namespace","module",
    "linux","ubuntu","debian","fedora","macos","android","ios","windows",
    "faiss","sqlite","mongodb","postgresql","mysql","redis","elasticsearch",
    "hema","tharun","sai",
}

def correct_word(word):
    word = word.lower()
    if not word.isalpha() or len(word) < 2:
        return word
    # Never correct known-good words
    if word in NEVER_CORRECT:
        return word
    if word in SPELL_VOCAB:
        return word
    if word in QUESTION_WORDS:
        return QUESTION_WORDS[word]
    # Only correct words that are long enough — short words have too many false matches
    # e.g. "like" (4) → "life", "chat" (4) → "what" — all bad corrections
    if len(word) < 7:
        return word
    matches = difflib.get_close_matches(word, SPELL_VOCAB, n=1, cutoff=0.88)
    if matches:
        return matches[0]
    return word

def spell_correct(text):
    """
    Fix spelling in a sentence.
    Uses HemaNLP edit-distance-1 correction when available,
    falls back to legacy difflib-based correction.
    Returns (corrected_text, was_changed, original).
    """
    if _nlp is not None and _nlp._ready:
        return _nlp.spell_correct(text)
    # Legacy fallback
    words = text.lower().strip().split()
    corrected = []
    changed = False
    for word in words:
        if not word.isalpha():
            corrected.append(word)
            continue
        fixed = correct_word(word)
        corrected.append(fixed)
        if fixed != word:
            changed = True
    result = ' '.join(corrected)
    return result, changed, text



# ════════════════════════════════════════════════════════════════════════════
# 2. LOAD DATASET FOR FUZZY SEARCH + EXAMPLES
# ════════════════════════════════════════════════════════════════════════════
import pandas as pd

dataset_qa = {}   # { question: answer }
try:
    # dtype=str + low_memory=False silences DtypeWarning on mixed-type columns
    df = pd.read_csv("data/chat_dataset.csv", dtype=str, low_memory=False)
    df = df.fillna("")
    # Remove leaked header rows
    df = df[df["input"].str.lower().str.strip() != "input"]
    # Deduplicate — keep row with longest answer per question
    df["_len"] = df["output"].str.len()
    df = df.sort_values("_len", ascending=False)
    df = df.drop_duplicates(subset="input", keep="first")
    df = df.drop(columns=["_len"]).reset_index(drop=True)
    print(f"CSV rows after dedup: {len(df):,}")
    for _, row in df.iterrows():
        q = str(row["input"]).lower().strip()
        a = str(row["output"]).strip()
        if q and a:
            dataset_qa[q] = a
    print(f"Dataset loaded: {len(dataset_qa):,} unique Q&A pairs")
    build_spell_vocab()
    # Build NLP engine index
    if _NLP_AVAILABLE:
        try:
            _nlp = get_nlp()
            # fit() expects {q: (answer, example)} — wrap plain answers
            _nlp.fit({q: (a, "") for q, a in dataset_qa.items()})
        except Exception as nlp_err:
            print(f"[NLP] Index build error: {nlp_err}")
            _nlp = None
    else:
        _nlp = None
except Exception as e:
    print(f"Dataset load error: {e}")
    _nlp = None

# ════════════════════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def clean(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text)

def encode(sentence):
    """Legacy encode — kept for spell vocab building."""
    if not word2idx:
        return torch.tensor([]), 1.0
    tokens, unk_count = [], 0
    words = sentence.lower().split()
    for w in words:
        if w in word2idx:
            tokens.append(word2idx[w])
        else:
            tokens.append(word2idx.get(UNK_TOKEN, 0))
            unk_count += 1
    tokens.append(word2idx.get(EOS_TOKEN, 0))
    unk_ratio = unk_count / len(words) if words else 0
    return torch.tensor(tokens).unsqueeze(0), unk_ratio

def encode_prompt(question: str):
    """
    GPT instruction/response prompt encoding.
    Builds:  <sos> instruction {question} response
    (special chars stripped to match clean_text() vocabulary)
    Model generates answer tokens after "response" marker.
    Returns prompt tensor (1, T) and unk_ratio.
    """
    if not word2idx:
        return torch.tensor([[word2idx.get(SOS_TOKEN, 1)]]), 1.0

    unk_id  = word2idx.get(UNK_TOKEN, 0)
    sos_id  = word2idx.get(SOS_TOKEN, 1)

    def _tok(text):
        import re as _re
        text = text.lower().strip()
        text = _re.sub(r"[^\w\s]", "", text)
        text = _re.sub(r"\s+", " ", text).strip()
        return [word2idx.get(w, unk_id) for w in text.split() if w]

    instr_ids    = _tok("instruction")
    question_ids = _tok(question)
    resp_ids     = _tok("response")

    prompt_ids = [sos_id] + instr_ids + question_ids + resp_ids

    # unk_ratio: fraction of question words not in vocab
    words     = question.lower().split()
    unk_count = sum(1 for w in words if word2idx.get(w, unk_id) == unk_id)
    unk_ratio = unk_count / max(1, len(words))

    return torch.tensor([prompt_ids], dtype=torch.long), unk_ratio

def normalize(text):
    """
    Normalise query:
    - Use HemaNLP for contraction expansion + synonym normalisation
    - Fall back to regex-based legacy normaliser
    """
    if _nlp is not None:
        return _nlp.normalize(text)
    # Legacy fallback
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    replacements = {
        "who discovered": "who invented",
        "explain": "what is",
        "tell me about": "what is",
        "define": "what is",
        "meaning of": "what is",
        "give me info on": "what is",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def fuzzy_search(query, threshold=0.65):
    """
    Find best matching answer.
    1. TF-IDF semantic search (HemaNLP) — better recall across paraphrases
    2. Fall back to difflib close-match — handles exact near-duplicates
    Returns answer string or None.
    """
    # ── TF-IDF semantic search ──
    if _nlp is not None:
        result = _nlp.search(query, top_n=1, threshold=0.12)
        if result:
            answer, _ = result   # discard example from nlp engine
            return answer

    # ── Legacy difflib fallback ──
    questions = list(dataset_qa.keys())
    matches = difflib.get_close_matches(query, questions, n=1, cutoff=threshold)
    if matches:
        return dataset_qa[matches[0]]
    return None

def solve_math(text):
    """
    Safely solve math expressions — no eval() risks.
    Handles:
      - Arithmetic:       25 * 16,  sqrt(144),  2^8
      - Word operators:   25 times 16,  100 divided by 4
      - Factorials:       10 factorial,  5!,  factorial of 6
      - Number sequences: 2, 4, 8, 16, ?  (detects pattern, predicts next)
      - Equations:        2x + 5 = 15  (simple linear, single variable)
      - Combinatorics:    5 choose 3,  C(8,2),  permutations of 4 from 7
      - Powers/roots:     square root of 144,  cube root of 27,  2 to the power of 10
    Returns string result or None if not a math expression.
    """
    import ast, math as _math, operator as _op, re as _re

    _OPERATORS = {
        ast.Add:  _op.add, ast.Sub:  _op.sub,
        ast.Mult: _op.mul, ast.Div:  _op.truediv,
        ast.Pow:  _op.pow, ast.Mod:  _op.mod,
        ast.USub: _op.neg, ast.UAdd: _op.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)): return node.value
            raise ValueError
        elif isinstance(node, ast.BinOp):
            op = _OPERATORS.get(type(node.op))
            if op is None: raise ValueError
            l, r = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Div) and r == 0:
                raise ZeroDivisionError("Division by zero")
            return op(l, r)
        elif isinstance(node, ast.UnaryOp):
            op = _OPERATORS.get(type(node.op))
            if op is None: raise ValueError
            return op(_eval(node.operand))
        elif isinstance(node, ast.Call):
            fname = getattr(node.func, 'id', '')
            if fname == 'sqrt':  return _math.sqrt(_eval(node.args[0]))
            if fname == 'abs':   return abs(_eval(node.args[0]))
            if fname == 'round':
                val = _eval(node.args[0])
                dec = int(_eval(node.args[1])) if len(node.args) > 1 else 0
                return round(val, dec)
            raise ValueError
        raise ValueError

    def _fmt(n):
        """Format a number cleanly — int if whole, else 6 sig figs."""
        if isinstance(n, float) and n.is_integer(): return str(int(n))
        if isinstance(n, float): return f"{n:.6g}"
        return str(n)

    raw  = text.strip()
    expr = raw.lower()

    # ── 1. FACTORIAL ──────────────────────────────────────────────────────────
    # "10 factorial", "factorial of 10", "5!", "what is 7 factorial"
    _fact_pat = _re.search(
        r'(?:factorial\s+of\s+|factorial\s*\(?\s*)(\d+)|(\d+)\s*!\s*$|(\d+)\s+factorial',
        expr
    )
    if _fact_pat:
        n = int(next(g for g in _fact_pat.groups() if g is not None))
        if n > 20:
            return f"{n}! = {_math.factorial(n):,}  (exact)"
        return f"{n}! = {_math.factorial(n):,}"

    # ── 2. NUMBER SEQUENCE  (detects arithmetic or geometric patterns) ─────────
    # "2, 4, 8, 16, ?" or "1 3 5 7 ?"
    _seq_pat = _re.match(
        r'^[\w\s]*?([\d]+(?:[,\s]+[\d]+){2,})[,\s]*\??$', expr
    )
    if _seq_pat:
        nums_raw = _re.findall(r'\d+', _seq_pat.group(1))
        if len(nums_raw) >= 3:
            nums = [int(x) for x in nums_raw]
            # Arithmetic: constant difference
            diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
            if len(set(diffs)) == 1:
                nxt = nums[-1] + diffs[0]
                return (f"The sequence {', '.join(str(n) for n in nums)}, ... "
                        f"is arithmetic (difference {diffs[0]:+d}).\n"
                        f"**Next term: {nxt}**")
            # Geometric: constant ratio
            try:
                ratios = [nums[i+1] / nums[i] for i in range(len(nums)-1) if nums[i] != 0]
                if ratios and len(set(round(r, 6) for r in ratios)) == 1:
                    r = ratios[0]
                    nxt = nums[-1] * r
                    r_str = _fmt(r)
                    nxt_str = _fmt(nxt)
                    return (f"The sequence {', '.join(str(n) for n in nums)}, ... "
                            f"is geometric (ratio {r_str}).\n"
                            f"**Next term: {nxt_str}**")
            except ZeroDivisionError:
                pass
            # Fibonacci-like: each term = sum of two before
            fibs = all(nums[i] == nums[i-1] + nums[i-2] for i in range(2, len(nums)))
            if fibs:
                nxt = nums[-1] + nums[-2]
                return (f"The sequence {', '.join(str(n) for n in nums)}, ... "
                        f"follows a Fibonacci pattern.\n"
                        f"**Next term: {nxt}**")

    # ── 3. SIMPLE LINEAR EQUATION  e.g. "2x + 5 = 15", "3y - 9 = 0" ─────────
    _eq_pat = _re.match(
        r'^(?:solve\s+|find\s+|what\s+is\s+)?'
        r'([+-]?\s*\d*\.?\d*)\s*([a-z])\s*([+-]\s*\d+\.?\d*)\s*=\s*([+-]?\d+\.?\d*)$',
        expr.strip()
    )
    if _eq_pat:
        a_str, var, b_str, c_str = _eq_pat.groups()
        try:
            a = float(a_str.replace(' ','') or '1')
            b = float(b_str.replace(' ',''))
            c = float(c_str.replace(' ',''))
            # ax + b = c  →  x = (c - b) / a
            if a != 0:
                x = (c - b) / a
                return f"{var} = {_fmt(x)}"
        except ValueError:
            pass

    # ── 4. COMBINATORICS ─────────────────────────────────────────────────────
    # "5 choose 3", "C(8,2)", "combinations of 4 from 7"
    _comb_pat = _re.search(
        r'(\d+)\s+choose\s+(\d+)|c\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)|'
        r'combinations?\s+of\s+(\d+)\s+from\s+(\d+)',
        expr
    )
    if _comb_pat:
        g = _comb_pat.groups()
        if g[0] and g[1]:   n, k = int(g[0]), int(g[1])
        elif g[2] and g[3]: n, k = int(g[2]), int(g[3])
        else:                n, k = int(g[5]), int(g[4])
        result = _math.comb(n, k)
        return f"C({n}, {k}) = {result:,}"

    # "permutations of 3 from 5" / "P(5,3)"
    _perm_pat = _re.search(
        r'permutations?\s+of\s+(\d+)\s+from\s+(\d+)|p\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
        expr
    )
    if _perm_pat:
        g = _perm_pat.groups()
        if g[0] and g[1]: k, n = int(g[0]), int(g[1])
        else:              n, k = int(g[2]), int(g[3])
        result = _math.perm(n, k)
        return f"P({n}, {k}) = {result:,}"

    # ── 5. CUBE ROOT / NTH ROOT ──────────────────────────────────────────────
    _cube = _re.search(r'cube\s+root\s+of\s+(\d+\.?\d*)', expr)
    if _cube:
        n = float(_cube.group(1))
        r = round(n ** (1/3), 8)
        return f"Cube root of {int(n) if n == int(n) else n} = {_fmt(r)}"

    _nthroot = _re.search(r'(\d+)(?:th|st|nd|rd)?\s+root\s+of\s+(\d+\.?\d*)', expr)
    if _nthroot:
        deg, n = int(_nthroot.group(1)), float(_nthroot.group(2))
        r = round(n ** (1/deg), 8)
        return f"{deg}th root of {int(n) if n == int(n) else n} = {_fmt(r)}"

    # ── 6. STANDARD ARITHMETIC EXPRESSION ────────────────────────────────────
    # Strip question words
    expr = _re.sub(
        r'\b(what is|whats|calculate|find|evaluate|compute|solve|result of|'
        r'the answer to|the value of|what\'s)\b', '', expr
    ).strip()

    # Word → symbol
    replacements = [
        (r'\bsquare\s+root\s+of\b', 'sqrt('),
        (r'\bsquared\b', '**2'), (r'\bcubed\b', '**3'),
        (r'\bto\s+the\s+power\s+of\b', '**'),
        (r'\btimes\b', '*'), (r'\bdivided\s+by\b', '/'),
        (r'\bplus\b', '+'),   (r'\bminus\b', '-'),
        (r'\bmod(?:ulo)?\b', '%'),
        (r'\^', '**'),
        (r'×', '*'), (r'÷', '/'),
        (r'[^\d\s\+\-\*\/\(\)\.\%\*]+', ' '),
    ]
    for pat, rep in replacements:
        expr = _re.sub(pat, rep, expr)

    # Handle unclosed sqrt(
    expr = expr.strip()
    if expr.count('(') > expr.count(')'):
        expr += ')' * (expr.count('(') - expr.count(')'))

    # Must have a digit and look like math
    if not _re.search(r'\d', expr): return None
    if not _re.match(r'^[\d\s\+\-\*\/\(\)\.\%\*]+$', expr): return None

    try:
        tree   = ast.parse(expr.strip(), mode='eval')
        result = _eval(tree.body)
        return _fmt(result)
    except ZeroDivisionError:
        return "undefined (division by zero)"
    except:
        return None


# ════════════════════════════════════════════════════════════════════════════
# 4. MULTI-SOURCE SEARCH ENGINE
# ════════════════════════════════════════════════════════════════════════════

def search_wikipedia(query, sentences=3):
    """Search Wikipedia and return summary."""
    try:
        summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0], sentences=sentences)
            return summary
        except:
            return None
    except:
        return None

def search_duckduckgo(query):
    """DuckDuckGo Instant Answer API — no API key needed."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        # AbstractText is the best result
        if data.get("AbstractText"):
            return data["AbstractText"]
        # Related Topics fallback
        topics = data.get("RelatedTopics", [])
        if topics and isinstance(topics[0], dict):
            text = topics[0].get("Text", "")
            if len(text) > 30:
                return text
    except:
        pass
    return None

def _extract_text_from_html(html: str, max_sentences: int = 3, min_len: int = 40) -> str:
    """
    Strip HTML tags from a page and extract the most informative sentences.
    Removes script, style, JSON-LD, nav, header, footer noise before parsing.
    """
    import re as _re
    # Remove script/style/JSON-LD/nav blocks ENTIRELY — must come before tag strip
    # so JSON-LD {"@context":"schema.org"...} content doesn't bleed through
    html = _re.sub(r'<(script|style|noscript)[^>]*>.*?</\1>',
                   ' ', html, flags=_re.S | _re.I)
    html = _re.sub(r'<(nav|header|footer|aside|menu)[^>]*>.*?</\1>',
                   ' ', html, flags=_re.S | _re.I)
    # Strip remaining tags
    text = _re.sub(r'<[^>]+>', ' ', html)
    text = _re.sub(r'&[a-z#0-9]+;', ' ', text)   # HTML entities
    text = _re.sub(r'\s+', ' ', text).strip()
    # Split into sentences and filter meaningful ones
    sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text)
                 if len(s.strip()) >= min_len
                 and not any(noise in s.lower() for noise in
                     ['cookie', 'javascript', 'subscribe', 'sign in',
                      'privacy policy', 'terms of service', 'advertisement',
                      'click here', 'read more', 'skip to',
                      'schema.org', '@context', '@type'])
                 and not s.strip().startswith("{")
                 and not s.strip().startswith("[")]
    if not sentences:
        return ""
    snippet = " ".join(sentences[:max_sentences])
    if len(snippet) > 600:
        snippet = snippet[:600].rsplit(" ", 1)[0] + "..."
    return snippet


def _playwright_google_search(query: str, num_results: int = 3) -> str | None:
    """
    Use Playwright headless Chromium to search Google and extract rich snippets.
    Returns the best answer snippet or None if nothing useful found.

    Advantages over requests:
    - Renders JavaScript — gets AI overviews, featured snippets, answer boxes
    - Handles bot-detection redirects automatically
    - Extracts structured answer cards (definitions, facts, calculations)
    """
    if not PLAYWRIGHT_AVAILABLE:
        return None
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ]
            )
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )
            page = ctx.new_page()

            # Navigate to Google search
            search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}&hl=en&gl=us"
            page.goto(search_url, wait_until="domcontentloaded", timeout=12000)

            # ── Priority 1: Featured snippet / answer box ──────────────────
            answer = ""
            for sel in [
                "[data-attrid='wa:/description'] span",   # knowledge panel
                ".IZ6rdc",                                  # definition card
                ".hgKElc",                                  # featured snippet bold
                ".yDYNvb.lEBKkf",                          # featured snippet body
                ".LGOjhe",                                  # answer card
                ".V3FYCf",                                  # quick answer
                ".kCrYT > a > .BNeawe.s3v9rd.AP7Wnd",     # description snippet
            ]:
                try:
                    el = page.locator(sel).first
                    if el.is_visible(timeout=500):
                        t = el.inner_text().strip()
                        if len(t) > 30:
                            answer = t
                            break
                except:
                    pass

            # ── Priority 2: Organic result snippets ───────────────────────
            if not answer:
                snippets = []
                for sel in [".VwiC3b", ".s3v9rd", ".st", ".aCOpRe"]:
                    try:
                        els = page.locator(sel).all()
                        for el in els[:num_results]:
                            t = el.inner_text().strip()
                            if len(t) > 40:
                                snippets.append(t)
                    except:
                        pass
                if snippets:
                    answer = " ".join(snippets[:2])

            # ── Priority 3: Scrape first organic result URL ───────────────
            if not answer:
                first_url = None
                for sel in ["a[jsname='UWckNb']", ".yuRUbf a", "h3.LC20lb"]:
                    try:
                        href = page.locator(sel).first.get_attribute("href", timeout=500)
                        if href and href.startswith("http"):
                            first_url = href
                            break
                    except:
                        pass
                if first_url:
                    try:
                        page.goto(first_url, wait_until="domcontentloaded", timeout=8000)
                        body_html = page.content()
                        answer = _extract_text_from_html(body_html, max_sentences=3)
                    except:
                        pass

            ctx.close()
            browser.close()

            if answer and len(answer.split()) > 5:
                # Clean up extra whitespace/newlines from Playwright inner_text
                answer = re.sub(r'\s+', ' ', answer).strip()
                if len(answer) > 600:
                    answer = answer[:600].rsplit(" ", 1)[0] + "..."
                return answer

    except Exception as e:
        print(f"[Playwright] Google search error: {e}")
    return None


def _requests_google_search(query: str, num_results: int = 3) -> str | None:
    """
    Fallback: traditional requests + regex scrape when Playwright unavailable.
    """
    if not GOOGLE_AVAILABLE:
        return None
    try:
        urls = []
        for url in google_search(query, num_results=num_results, stop=num_results, pause=1.0):
            urls.append(url)
        if not urls:
            return None
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(urls[0], headers=headers, timeout=6)
        if r.status_code == 200:
            snippet = _extract_text_from_html(r.text, max_sentences=2)
            if snippet:
                return snippet
        return None
    except Exception as e:
        print(f"[Requests] Google search error: {e}")
    return None


def search_google(query, num_results=3):
    """
    Google search — tries Playwright headless Chrome first for rich results,
    falls back to requests scraper if Playwright is not installed.
    """
    # Playwright gives real rendered results including answer boxes & snippets
    result = _playwright_google_search(query, num_results)
    if result:
        return result
    # Fallback: old requests-based scraper
    return _requests_google_search(query, num_results)


def search_geeksforgeeks(query):
    """
    Scrape GeeksforGeeks for programming / CS answers.
    Strategy: Google-site-search GFG, then scrape the article intro.
    Falls back to GFG search API if direct scrape fails.
    """
    try:
        import re as _re
        # Step 1: find GFG URL via DuckDuckGo site: search
        ddg_url = "https://api.duckduckgo.com/"
        ddg_params = {
            "q":  f"site:geeksforgeeks.org {query}",
            "format": "json", "no_html": 1, "skip_disambig": 1
        }
        r = requests.get(ddg_url, params=ddg_params, timeout=5)
        data = r.json()

        gfg_url = None
        # AbstractURL is the GFG page when a direct match exists
        if data.get("AbstractURL", "").startswith("https://www.geeksforgeeks.org"):
            gfg_url = data["AbstractURL"]
        else:
            # Scan RelatedTopics for a GFG link
            for topic in data.get("RelatedTopics", []):
                if isinstance(topic, dict):
                    url = topic.get("FirstURL", "")
                    if "geeksforgeeks.org" in url:
                        gfg_url = url
                        break

        if not gfg_url:
            # Step 2: try GFG's own search endpoint
            gfg_search = f"https://www.geeksforgeeks.org/?s={requests.utils.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0"}
            sr = requests.get(gfg_search, headers=headers, timeout=5)
            # Extract first article link from search results HTML
            links = _re.findall(r'href="(https://www[.]geeksforgeeks[.]org/[^"]+/)"', sr.text)
            links = [l for l in links if not any(skip in l for skip in
                     ["/tag/", "/category/", "/?", "/author/", "/courses/"])]
            if links:
                gfg_url = links[0]

        if not gfg_url:
            return None

        # Step 3: scrape the GFG article intro
        headers = {"User-Agent": "Mozilla/5.0"}
        pg = requests.get(gfg_url, headers=headers, timeout=6)
        if pg.status_code != 200:
            return None

        html = pg.text
        # ── Strip script/style/JSON-LD FIRST — before tag removal ──────────
        # JSON-LD blocks like {"@context":"http://schema.org"...} bleed through
        # naive tag stripping because the content is raw text inside <script>.
        html = _re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=_re.S | _re.I)
        html = _re.sub(r'<style[^>]*>.*?</style>',  ' ', html, flags=_re.S | _re.I)
        html = _re.sub(r'<noscript[^>]*>.*?</noscript>', ' ', html, flags=_re.S | _re.I)

        # Strip remaining HTML tags, collapse whitespace
        raw = _re.sub(r'<[^>]+>', ' ', html)
        raw = _re.sub(r'&[a-z#0-9]+;', ' ', raw)   # HTML entities
        raw = _re.sub(r'\s+', ' ', raw).strip()

        # Find the most informative 2–3 sentence block
        sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', raw)
                     if 40 < len(s.strip()) < 600
                     and not any(noise in s.lower() for noise in
                         ["cookie", "login", "subscribe", "advertisement",
                          "javascript", "courses", "geeks for geeks",
                          "schema.org", "@context", "@type", "organization",
                          "privacy policy", "terms of use", "all rights reserved",
                          "geeksforgeeks.org"])
                     and not s.strip().startswith("{")   # reject any stray JSON
                     and not s.strip().startswith("[")]  # reject JSON arrays
        if not sentences:
            return None

        snippet = " ".join(sentences[:3])
        if len(snippet) > 500:
            snippet = snippet[:500].rsplit(" ", 1)[0] + "..."
        return f"{snippet} (GeeksforGeeks)"

    except Exception as e:
        print(f"[GFG] search error: {e}")
        return None


def search_dictionary(word):
    """Free Dictionary API for word definitions."""
    try:
        # Only use for single-word or 2-word queries that look like definitions
        clean_word = word.replace("what is ", "").replace("define ", "").strip()
        if len(clean_word.split()) > 3:
            return None
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{clean_word.replace(' ', '%20')}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            meanings = data[0].get("meanings", [])
            if meanings:
                defs = meanings[0].get("definitions", [])
                if defs:
                    definition = defs[0].get("definition", "")
                    example = defs[0].get("example", "")
                    result = f"{definition}"
                    if example:
                        result += f' — Example: "{example}"'
                    return result
    except:
        pass
    return None

# Stop-words that don't count as meaningful overlap
_STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "what","who","where","when","how","why","do","does","did",
    "can","could","would","should","will","may","might","shall",
    "i","you","he","she","it","we","they","me","him","her","us",
    "my","your","his","its","our","their","this","that","these",
    "those","and","or","but","if","in","on","at","to","for","of",
    "with","about","by","from","up","out","into","through","tell",
    "give","show","explain","describe","define","get","make","let",
    "please","just","also","more","some","any","all","no","not","yes",
    "bad","good","ok","okay","yeah","sure","hi","hello","hey"
}

def _topic_keywords(text):
    """Return meaningful keywords from text (filtered stop-words, min 4 chars)."""
    words = set(text.lower().split())
    return {w for w in words if w not in _STOP_WORDS and len(w) >= 4}

def get_examples_from_dataset(topic, n=2):
    """
    Pull related answers from dataset by keyword overlap.
    Only returns results when there are 3+ meaningful keyword overlaps.
    Returns list of (q, a).
    """
    topic_kw = _topic_keywords(topic)
    if len(topic_kw) < 1:
        return []
    results = []
    for q, a in dataset_qa.items():
        q_kw = _topic_keywords(q)
        overlap = len(topic_kw & q_kw)
        min_overlap = 3 if len(topic_kw) >= 4 else 2
        if overlap >= min_overlap and q != topic and len(a.split()) >= 5:
            results.append((q, a))
        if len(results) >= n:
            break
    return results

def extract_example_from_text(text, topic):
    """
    Try to pull out the most example-like sentence from merged text.
    Looks for sentences containing 'for example', 'such as', 'e.g.', 
    or sentences that start with the topic name.
    Falls back to the first complete sentence.
    """
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    
    # Priority: sentence that explicitly has an example keyword
    for s in sentences:
        sl = s.lower()
        if any(kw in sl for kw in ["for example", "such as", "e.g.", "like ", "instance"]):
            return s
    
    # Second: sentence that mentions the topic
    for s in sentences:
        if topic.lower() in s.lower():
            return s
    
    # Fallback: first sentence
    return sentences[0] if sentences else text[:200]


def merge_sources(results):
    """
    Merge multiple search results into one clean answer.
    No source labels shown — just the best combined text.
    Deduplicates overlapping content and returns a single paragraph.
    """
    if not results:
        return "I could not find information on that."

    # Deduplicate: remove results that are largely substrings of each other
    unique = []
    for r in results:
        r = r.strip()
        if not r:
            continue
        # Skip if 70%+ of this result already exists in a previous one
        is_duplicate = False
        for prev in unique:
            overlap = len(set(r.lower().split()) & set(prev.lower().split()))
            shorter = min(len(r.split()), len(prev.split()))
            if shorter > 0 and overlap / shorter > 0.7:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(r)

    if len(unique) == 1:
        return unique[0]

    # Merge: join with space, capping total length to ~400 words
    merged = " ".join(unique)
    words = merged.split()
    if len(words) > 400:
        merged = " ".join(words[:400]) + "..."
    return merged


# CS/programming keywords — triggers GeeksforGeeks as first source
_CS_KEYWORDS = {
    "algorithm","data structure","array","list","stack","queue","tree","graph",
    "sort","search","dynamic","recursion","python","java","c++","javascript",
    "css","html","sql","database","api","oop","class","function","loop",
    "pointer","memory","complexity","binary","linked","hash","heap","string",
    "code","program","implement","compile","runtime","syntax","variable",
    "method","object","interface","inheritance","polymorphism","exception",
    "thread","process","network","protocol","encryption","linux","git",
    "programming","language","framework","library","compiler","interpreter",
    "debugging","recursion","iteration","paradigm","object-oriented","functional",
    "typescript","kotlin","swift","rust","golang","ruby","php","scala","perl",
    "django","flask","react","angular","vue","node","express","spring","laravel",
    "machine learning","deep learning","neural","tensorflow","pytorch","numpy",
    "pandas","matplotlib","scikit","regex","exception","decorator","generator",
    "async","await","promise","callback","closure","scope","namespace","module",
    "package","import","inheritance","encapsulation","abstraction","polymorphism"
}

def _is_cs_query(query: str) -> bool:
    """Return True when the query is about programming / CS."""
    words = set(query.lower().split())
    # Direct keyword match
    if words & _CS_KEYWORDS:
        return True
    # Phrase match for multi-word CS terms
    q_lower = query.lower()
    cs_phrases = [
        "programming language","data structure","design pattern","time complexity",
        "space complexity","big o","linked list","binary tree","hash table",
        "how to code","write a program","write code","source code","open source",
        "version control","object oriented","functional programming",
    ]
    return any(p in q_lower for p in cs_phrases)


def multi_source_search(query):
    """
    Search up to 5 sources and return best combined answer.
    For CS/programming queries: GeeksforGeeks goes FIRST.
    General queries: Google → DuckDuckGo → Wikipedia → Dictionary
    """
    results = []
    is_cs = _is_cs_query(query)

    # ── CS/programming path: GFG first ───────────────────────────────────────
    if is_cs:
        # Source 1 (CS): GeeksforGeeks — most authoritative for CS topics
        gfg = search_geeksforgeeks(query)
        if gfg and len(gfg) > 40:
            results.append(gfg)

        # Source 2 (CS): Wikipedia (good for language/concept overviews)
        wiki = search_wikipedia(query, sentences=2)
        if wiki and len(wiki) > 40:
            results.append(wiki)

        # Source 3 (CS): DuckDuckGo as supplemental
        ddg = search_duckduckgo(query)
        if ddg and len(ddg) > 40:
            if not results or ddg[:60] != results[0][:60]:
                results.append(ddg)

        # If GFG returned nothing, fall back to Google
        if not results:
            google = search_google(query)
            if google and len(google) > 40:
                results.append(google)

        return results

    # ── General path: Google first ────────────────────────────────────────────
    # Source 1: Google (broadest, most relevant)
    google = search_google(query)
    if google and len(google) > 40:
        results.append(google)

    # Source 2: DuckDuckGo Instant Answers (fast structured data)
    ddg = search_duckduckgo(query)
    if ddg and len(ddg) > 40:
        if not results or ddg[:60] != results[0][:60]:
            results.append(ddg)

    # Source 3: Wikipedia (most detailed encyclopedic info)
    wiki = search_wikipedia(query, sentences=2)
    if wiki and len(wiki) > 40:
        results.append(wiki)

    # Source 5: Dictionary (for definition queries)
    is_definition = any(w in query.lower() for w in ["what is", "define", "meaning", "definition of"])
    if is_definition:
        word = re.sub(r'what is |define |meaning of |definition of ', '', query.lower()).strip()
        defi = search_dictionary(word)
        if defi:
            results.append(defi)

    return results


def build_example_block(topic):
    """Build a related answers section from dataset entries."""
    entries = get_examples_from_dataset(topic, n=2)
    if not entries:
        return ""
    block = "\n\n<b>Related:</b>"
    for q, a in entries:
        block += f"\n• <i>Q: {q.capitalize()}?</i><br>&nbsp;&nbsp;A: {a[:120]}{'...' if len(a) > 120 else ''}"
    return block


# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN RESPONSE FUNCTION
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# RESPONSE FORMATTER — all output goes through here as clean Markdown
# ════════════════════════════════════════════════════════════════════════════

def md_bold(text): return f"**{text}**"
def md_italic(text): return f"*{text}*"
def md_code(text): return f"`{text}`"
def md_rule(): return "\n---\n"

def html_to_md(text):
    """Convert legacy <b>, <br>, <i>, <small> HTML to clean Markdown."""
    import re
    text = re.sub(r"<b>(.*?)</b>", r"**\1**", text, flags=re.S)
    text = re.sub(r"<i>(.*?)</i>", r"*\1*", text, flags=re.S)
    text = re.sub(r"<small>(.*?)</small>", r"*\1*", text, flags=re.S)
    text = re.sub(r"<pre>(.*?)</pre>", r"```\n\1\n```", text, flags=re.S)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.S)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


# ── Language map for code block auto-detection ───────────────────────────────
_LANG_KEYWORDS = {
    "python":     ["def ","import ","print(","elif","lambda","__init__","self.","#"],
    "javascript": ["function ","const ","let ","var ","=>","console.log","document.","require("],
    "java":       ["public class","public static","System.out","void ","@Override","import java"],
    "c++":        ["#include","cout <<","cin >>","int main","std::","namespace"],
    "sql":        ["SELECT ","INSERT INTO","UPDATE ","DELETE FROM","CREATE TABLE","WHERE ","FROM "],
    "html":       ["<!DOCTYPE","<html","<div","<body","<head","<p>","<script"],
    "css":        ["{","margin:","padding:","color:","display:","font-size:"],
    "bash":       ["#!/","sudo ","apt-get","echo ","chmod","grep ","ls -"],
}

def _detect_language(code: str) -> str:
    code_lower = code.lower()
    scores = {}
    for lang, hints in _LANG_KEYWORDS.items():
        scores[lang] = sum(1 for h in hints if h.lower() in code_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "python"


def _detect_query_type(query: str) -> str:
    """
    Classify query into response-style bucket.
    Returns one of: definition, howto, comparison, list, code, math, casual, factual
    """
    q = query.lower().strip()
    if any(q.startswith(p) for p in ("what is ","what are ","define ","meaning of ","who is ","who was ")):
        return "definition"
    if any(q.startswith(p) for p in ("how to ","how do i ","how can i ","steps to ","how do you ","how to make ")):
        return "howto"
    if any(w in q for w in (" vs "," versus "," or ","difference between","compare ","which is better")):
        return "comparison"
    if any(q.startswith(p) for p in ("list ","give me ","name ","what are the ","types of ")):
        return "list"
    if any(w in q for w in ("code","program","write a ","implement","function","algorithm","script","syntax for")):
        return "code"
    if any(q.startswith(p) for p in ("why ","explain ","tell me about ","describe ")):
        return "factual"
    return "factual"


def _extract_code_blocks(text: str):
    """
    Extract existing ``` code blocks from text.
    Returns (text_without_code, list_of_code_strings).
    """
    import re
    blocks = re.findall(r"```[\w]*\n?(.*?)```", text, re.S)
    clean  = re.sub(r"```[\w]*\n?.*?```", "___CODE___", text, flags=re.S)
    return clean, blocks


def _auto_fenced_code(text: str) -> str:
    """
    Detect inline code snippets (4+ space indent or obvious code lines)
    and wrap in proper fenced blocks with language detection.
    """
    import re
    lines     = text.split("\n")
    result    = []
    code_buf  = []
    in_code   = False

    def flush_code():
        if not code_buf:
            return
        snippet = "\n".join(code_buf)
        lang    = _detect_language(snippet)
        result.append(f"```{lang}\n{snippet}\n```")
        code_buf.clear()

    for line in lines:
        stripped = line.lstrip()
        is_code_line = (
            line.startswith("    ") and stripped and not stripped.startswith("#")
            or stripped.startswith(">>> ")
            or re.match(r"^(def |class |import |from |if __name__|print\(|#include|public |void |SELECT |INSERT )", stripped)
        )
        if is_code_line:
            in_code = True
            code_buf.append(line.rstrip())
        else:
            if in_code:
                flush_code()
                in_code = False
            result.append(line)

    if in_code:
        flush_code()

    return "\n".join(result)


def _structure_definition(text: str, topic: str) -> str:
    """
    Format a definition response like ChatGPT:
    **Topic** is/are ... [definition sentence bold]
    Then explanation paragraphs.
    """
    import re
    clean  = html_to_md(text).strip()
    topic_cap = topic.strip().title() if topic else ""

    # Split into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if not sentences:
        return clean

    parts = []

    # First sentence → bold definition lead
    first = sentences[0]
    # If it starts with the topic, bold just the first sentence
    if topic_cap and not first.startswith("**"):
        first = f"**{first}**"
    parts.append(first)

    # Rest → group into paragraphs of ~3 sentences each
    remaining = sentences[1:]
    chunk = []
    for s in remaining:
        chunk.append(s)
        if len(chunk) == 3:
            parts.append(" ".join(chunk))
            chunk = []
    if chunk:
        parts.append(" ".join(chunk))

    return "\n\n".join(parts)


def _structure_howto(text: str, query: str) -> str:
    """
    Format a how-to response: numbered steps extracted from text.
    """
    import re
    clean     = html_to_md(text).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if len(s.strip()) > 15]

    # If already numbered, return as-is after cleanup
    if re.search(r"^\d+[\.\)]", clean, re.M):
        return clean

    # Try to detect step-like sentences (action verbs)
    _STEP_VERBS = ("open","create","run","install","add","set","write","use","call",
                   "click","go to","navigate","enter","type","configure","start",
                   "download","import","define","make","build","edit","save","push",
                   "copy","move","check","ensure","verify","select","choose")
    steps = [s for s in sentences if any(s.lower().startswith(v) for v in _STEP_VERBS)]

    if len(steps) >= 3:
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        # Add any remaining non-step sentences as a closing note
        rest = [s for s in sentences if s not in steps]
        closing = " ".join(rest[:2]) if rest else ""
        return (numbered + ("\n\n" + closing if closing else "")).strip()

    # Fallback: treat every sentence as a step
    if len(sentences) >= 2:
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences[:8]))

    return clean


def _structure_comparison(text: str, query: str) -> str:
    """Format a comparison with a simple markdown table or two-column structure."""
    import re
    clean = html_to_md(text).strip()

    # Extract two subjects from query
    m = re.search(r"(\w[\w\s]+?)\s+(?:vs|versus|or)\s+([\w\s]+)", query, re.I)
    if not m:
        return clean

    a, b = m.group(1).strip().title(), m.group(2).strip().title()

    # If text already has a table, return it
    if "|" in clean:
        return clean

    # Build a simple two-section response
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    mid = len(sentences) // 2 or 1

    a_points = " ".join(sentences[:mid])
    b_points = " ".join(sentences[mid:]) if mid < len(sentences) else ""

    result = f"**{a}**\n{a_points}"
    if b_points:
        result += f"\n\n**{b}**\n{b_points}"
    return result


def _structure_list(text: str, query: str) -> str:
    """Format list-style responses as clean bullet points."""
    import re
    clean = html_to_md(text).strip()

    # Already has bullets or numbers — clean up and return
    if re.search(r"^[\-\*\d]", clean, re.M):
        return clean

    # Split by sentence and bullet
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if len(s.strip()) > 10]
    if len(sentences) >= 3:
        return "\n".join(f"- {s}" for s in sentences[:10])

    return clean


def _structure_code_response(text: str, query: str) -> str:
    """
    Format code-related responses: detect code, wrap in fenced blocks,
    put explanation before and after.
    """
    import re
    clean = html_to_md(text).strip()

    # Already fenced
    if "```" in clean:
        return clean

    # Detect code lines vs prose
    lines = clean.split("\n")
    prose_lines, code_lines = [], []
    for line in lines:
        stripped = line.lstrip()
        is_code = (
            line.startswith("    ") and stripped
            or re.match(r"^(def |class |import |from |>>>|#include|int main|public |SELECT )", stripped)
            or stripped.startswith(">>> ")
        )
        if is_code:
            code_lines.append(line)
        else:
            prose_lines.append(line)

    prose = "\n".join(prose_lines).strip()
    code  = "\n".join(code_lines).strip()

    if code:
        lang = _detect_language(code)
        result = ""
        if prose:
            result += prose + "\n\n"
        result += f"```{lang}\n{code}\n```"
        return result.strip()

    # No indented code but query is code-related — try auto-fence
    return _auto_fenced_code(clean)


def _add_summary_line(text: str, query_type: str) -> str:
    """
    For longer factual responses, add a brief TL;DR / summary at end
    if response is > 100 words — ChatGPT often does this.
    """
    import re
    if query_type not in ("definition", "factual"):
        return text
    words = text.split()
    if len(words) < 80:
        return text

    # Extract last sentence as summary hint
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) < 4:
        return text

    # Don't double-add summary
    if "**In summary" in text or "**Key takeaway" in text:
        return text

    # First sentence is usually the definition — use it as summary
    summary = sentences[0].replace("**","").strip()
    if len(summary) > 20:
        return text + f"\n\n**In summary:** {summary}"
    return text


def structure_like_chatgpt(text: str, query: str = "", query_type: str = "") -> str:
    """
    Master ChatGPT-style formatter.
    Takes raw answer text and returns well-structured Markdown.

    Rules:
    - definition  → bold first sentence + paragraph breaks
    - howto       → numbered steps
    - comparison  → two sections or table
    - list        → bullet points
    - code        → fenced code blocks with language detection
    - factual     → paragraph breaks + summary if long
    - math        → **Answer:** prefix
    """
    import re
    if not text:
        return ""

    clean = html_to_md(text).strip()
    if not clean:
        return ""

    # Auto-detect query type if not supplied
    if not query_type and query:
        query_type = _detect_query_type(query)

    # Handle any existing code blocks first — preserve them
    stripped, code_blocks = _extract_code_blocks(clean)

    # Route to appropriate formatter
    if query_type == "definition":
        topic = re.sub(r"^(what is|what are|define|who is|who was)\s+", "", query, flags=re.I).strip()
        body = _structure_definition(stripped, topic)
    elif query_type == "howto":
        body = _structure_howto(stripped, query)
    elif query_type == "comparison":
        body = _structure_comparison(stripped, query)
    elif query_type == "list":
        body = _structure_list(stripped, query)
    elif query_type == "code":
        body = _structure_code_response(stripped, query)
    else:
        # Factual / general — clean paragraphs
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", stripped) if s.strip()]
        chunks, chunk = [], []
        for s in sentences:
            chunk.append(s)
            if len(" ".join(chunk).split()) >= 50:
                chunks.append(" ".join(chunk))
                chunk = []
        if chunk:
            chunks.append(" ".join(chunk))
        body = "\n\n".join(chunks) if chunks else stripped

    # Re-insert code blocks
    for block in code_blocks:
        lang = _detect_language(block)
        body = body.replace("___CODE___", f"```{lang}\n{block.strip()}\n```", 1)

    # Add summary for long factual answers
    body = _add_summary_line(body, query_type)

    return body.strip()


import re as _re_global

def format_response(text, response_type="answer", topic="", spell_notice="", query=""):
    """
    Master formatter — all output goes through here.
    Now routes through structure_like_chatgpt() for rich formatting.
    response_type: "answer" | "web" | "example" | "math" | "file" | "joke"
    """
    if not text:
        return ""

    prefix = f"*{html_to_md(spell_notice)}*\n\n" if spell_notice else ""
    clean  = html_to_md(text)

    if response_type == "math":
        return prefix + f"**Answer:** {clean}"

    if response_type == "joke":
        return prefix + f"😄 {clean} 😂"

    if response_type == "example":
        topic_str = topic or _re_global.sub(r"^(example of|give me example of)\s+", "", query, flags=_re_global.I).strip()
        body = structure_like_chatgpt(clean, query=f"example of {topic_str}", query_type="factual")
        return prefix + f"**Example — {topic_str.title()}:**\n\n{body}" if topic_str else prefix + body

    if response_type == "file":
        return prefix + clean   # file summaries are pre-structured

    # For both "answer" and "web" — apply full ChatGPT-style formatting
    body = structure_like_chatgpt(clean, query=query)
    return prefix + body


def format_file_summary(filename, word_count, line_count, char_count,
                         overview, key_points, intent, extra_stats=""):
    """
    Build a clean Markdown file summary.
    All scaling already computed — this just formats it.
    """
    detail   = intent.get("detail", "normal")
    n_points = len(key_points)
    ext      = get_file_ext(filename).lstrip(".").upper()

    lines = []

    # ── Header ──
    lines.append(f"## {filename}")

    if detail != "brief":
        stats = f"**{word_count:,}** words · **{line_count:,}** lines · **{char_count:,}** chars"
        if extra_stats:
            stats += f" · {extra_stats}"
        lines.append(stats)
        lines.append("")

    # ── Overview ──
    if overview and detail != "brief":
        lines.append("### Overview")
        lines.append(overview)
        lines.append("")

    # ── Key points ──
    if key_points:
        if detail == "brief":
            lines.append("**Summary:**")
        else:
            lines.append(f"### Key Points")
        for pt in key_points:
            lines.append(f"- {pt}")
        lines.append("")

    return "\n".join(lines)


def format_code_summary(filename, line_count, word_count, code_lines,
                         comment_lines, blank_lines, func_count, key_pts, intent, extra=""):
    """Clean Markdown for code file analysis."""
    detail = intent.get("detail", "normal")
    ext = get_file_ext(filename).lstrip(".").upper()

    lines = []
    lines.append(f"## 💻 Code: `{filename}`")

    if detail == "brief":
        lines.append(f"**{line_count:,}** lines · **{func_count}** functions/classes")
    else:
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total lines | {line_count:,} |")
        lines.append(f"| Code lines | {code_lines:,} |")
        lines.append(f"| Comment lines | {comment_lines:,} |")
        lines.append(f"| Blank lines | {blank_lines:,} |")
        lines.append(f"| Words | {word_count:,} |")
        lines.append(f"| Functions / Classes | {func_count} |")
        if extra:
            for part in extra.split("|"):
                part = part.strip()
                if ":" in part:
                    k,v = part.split(":",1)
                    lines.append(f"| {k.strip()} | {v.strip()} |")
        lines.append("")

    if key_pts and detail != "brief":
        lines.append("### Key Sections")
        for pt in key_pts:
            lines.append(f"- {pt}")
        lines.append("")

    return "\n".join(lines)


def format_csv_summary(filename, rows, cols, header, preview_rows, intent):
    """Clean Markdown for CSV analysis."""
    lines = []
    lines.append(f"## {filename}  `CSV`")
    lines.append(f"**{rows:,}** rows · **{cols}** columns")
    lines.append("")
    lines.append(f"**Columns:** {header}")
    lines.append("")
    if preview_rows:
        lines.append("**Preview:**")
        lines.append("```")
        for r in preview_rows:
            lines.append(r)
        lines.append("```")
    return "\n".join(lines)


def format_image_analysis(filename, img_info):
    """Clean Markdown for image analysis."""
    lines = []
    lines.append(f"## Image: {filename}")
    lines.append("")
    for line in img_info.split("\n"):
        if line.strip():
            lines.append(f"- {line.strip()}")
    return "\n".join(lines)


def is_joke_query(text):
    """Detect if user is asking for a joke."""
    joke_keywords = ["joke", "funny", "humor", "humour", "laugh", "make me laugh",
                     "tell me a joke", "something funny", "fun fact", "comic"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in joke_keywords)

def format_joke(joke_text):
    """Wrap joke with fun emojis — only place emojis are used."""
    return f"😄 {joke_text} 😂"

def get_ai_response(user_text, selected_module="hybrid"):
    global chat_history

    original_text = user_text.strip()

    # ── FOLLOW-UP RESOLUTION ──
    resolved_text, was_followup = resolve_followup(original_text)
    if was_followup:
        user_text = resolved_text
    add_to_conversation("user", original_text, topic=resolved_text if was_followup else original_text)

    # ── NLP INTENT + ENTITY DETECTION ─────────────────────────────────────
    # Classify intent and extract entities before spell correction
    # so we work on the user's original phrasing
    _intent_label, _intent_conf = ("general", 0.3)
    _entities = {}
    if _nlp is not None:
        _intent_label, _intent_conf = _nlp.classify_intent(original_text)
        _entities = _nlp.extract_entities(original_text)

    # ── PRE-SPELL SMALL TALK GUARD ─────────────────────────────────────────
    # Check on RAW original text before any correction/normalisation
    # so spell errors can't corrupt "chat" → "what", "hi" → etc.
    _raw = original_text.lower().strip()
    _raw_words = _raw.split()

    # ── EARLY MATH INTERCEPT ─────────────────────────────────────────────────
    # Run on RAW input before spell correction or normalization can mangle it.
    # Catches: "2, 4, 8, 16, ?", "10 factorial", "25×16", "2x+5=15", "5 choose 3"
    _early_math = solve_math(original_text)
    if _early_math:
        add_to_conversation("user", original_text, topic="math", intent="math")
        return format_response(_early_math, response_type="math")

    # Raw greeting check
    _raw_greets = {"hi","hello","hey","hola","sup","yo","howdy","greetings",
                   "good morning","good afternoon","good evening","good night",
                   "how are you","how r you","how are u","hows it going",
                   "how's it going","how do you do"}
    if _raw in _raw_greets:
        import random as _r0
        return format_response(_r0.choice([
            "Hey! What would you like to explore today?",
            "Hi there! Ask me anything — science, math, coding, history, or just chat.",
            "Hello! What's on your mind?",
            "Hey! What can I help you with?",
        ]))

    # "Nice to meet you" — social greeting, not an identity query
    _raw_meet = {"nice to meet you","nice to meet you hema","nice meeting you",
                 "pleased to meet you","great to meet you","good to meet you"}
    if _raw in _raw_meet or _raw.startswith("nice to meet") or _raw.startswith("pleased to meet"):
        return format_response("Nice to meet you too! What can I help you with today?")

    # Raw casual chat check — before spell correction can destroy "chat"
    _raw_casual = {
        "just chat","just talk","just talking","just chatting","chat",
        "wanna chat","want to chat","lets chat","let's chat","can we chat",
        "bored","im bored","i'm bored","talk to me","entertain me",
        "nothing","just here","just browsing","just chilling",
        "what are you doing","whats up","what's up","wassup","are you there",
        "you there","are you busy","just checking",
    }
    if _raw in _raw_casual or any(t in _raw for t in _raw_casual):
        import random as _r1
        return format_response(_r1.choice([
            "Happy to chat! What's on your mind?",
            "Sure! Ask me anything — science, coding, history, math, or just conversation.",
            "Let's go! What would you like to talk about?",
        ]))

    # Raw farewell check
    _raw_bye = {"bye","goodbye","see you","later","take care","farewell","cya","ttyl"}
    if _raw in _raw_bye:
        return format_response("Goodbye! Come back anytime.")

    # Raw thanks check
    _raw_thanks = {"thanks","thank you","ty","thx","cheers","appreciate it","thank"}
    if _raw in _raw_thanks:
        return format_response("You're welcome! Anything else I can help with?")

    # ── SAY / REPEAT COMMAND (must be before any identity/small-talk check) ──
    if _raw.startswith("say ") or _raw.startswith("repeat "):
        spoken = _raw.replace("say ", "").replace("repeat ", "").strip()
        if spoken:
            return format_response(f'You said: **{spoken}**')

    # ── SINGLE / VERY SHORT AMBIGUOUS WORD GUARD ─────────────────────────────
    # Words like "good", "start", "stop", "ok", "yes", "no", "next", "more"
    # have no meaningful answer — asking web search returns random articles.
    _AMBIGUOUS_SHORT = {
        "good","great","fine","ok","okay","yes","no","yeah","nah","nope",
        "start","stop","begin","end","go","wait","next","back","done","ready",
        "hmm","hm","oh","ah","uh","um","er","right","wrong","cool","nice",
        "wow","whoa","interesting","really","sure","maybe","perhaps","true",
        "false","correct","incorrect","exactly","indeed","well","anyway",
    }
    if _raw in _AMBIGUOUS_SHORT or (len(_raw_words) == 1 and len(_raw) <= 5):
        return format_response(
            "Could you give me a bit more context? "
            "For example: *What is machine learning?* or *Explain recursion*."
        )

    # ── SPELL CORRECTION ────────────────────────────────────────────────────
    corrected_text, was_corrected, raw_input = spell_correct(user_text)
    spell_notice = ""
    if was_corrected and corrected_text != user_text.lower().strip():
        spell_notice = f'Did you mean: "{corrected_text}"? Answering that...'
        user_text = corrected_text
    else:
        user_text = user_text

    user_text = normalize(user_text)

    # Build context-enriched prompt — used for vector/fuzzy/web search
    # so every search reads the full conversation, not just the current msg
    context_query = build_context_prompt(user_text, max_turns=20)

    # ── SMALL TALK & IDENTITY ───────────────────────────────────────────────
    # Expanded checks — catches more phrasings before any search layer runs

    _ut = user_text.strip().lower()  # shorthand for clean comparisons

    # Greetings — varied responses so it doesn't repeat the welcome screen
    import random as _rnd
    _greet_words = {"hi","hello","hey","hola","greetings","sup","yo","howdy",
                    "good morning","good afternoon","good evening","good night"}
    if _ut in _greet_words or _intent_label == "greeting":
        _greet_replies = [
            "Hey! What would you like to explore today?",
            "Hi there! Ask me anything — science, math, coding, history, or just chat.",
            "Hello! I'm ready. What's on your mind?",
            "Hey! What can I help you with?",
        ]
        return format_response(_rnd.choice(_greet_replies), spell_notice=spell_notice)

    # Farewell
    _bye_words = {"bye","goodbye","see you","later","take care","farewell","cya","ttyl","good night"}
    if _ut in _bye_words or _intent_label == "farewell":
        return format_response("Goodbye! Come back anytime — I'll be here.")

    # Thanks
    _thanks_words = {"thanks","thank you","ty","thx","appreciate it","cheers","thank"}
    if _ut in _thanks_words or _intent_label == "thanks":
        return format_response("You're welcome! Anything else I can help with?")

    # Identity — who/what is Hema
    _identity_triggers = [
        "who are you","what are you","your name","who is hema","what is hema",
        "tell me about yourself","introduce yourself","about you",
    ]
    # Only fire identity if the query is specifically ASKING about Hema's nature,
    # not just casual phrases that happen to contain "hema" as a name in greeting
    _is_identity = (
        any(t in _ut for t in _identity_triggers)
        or _intent_label == "identity"
    ) and not any(social in _ut for social in (
        "nice to meet", "pleased to meet", "good to meet", "great to meet",
        "how are you", "how r you",
    ))
    if _is_identity:
        return format_response(
            "I'm **Hema** — a hybrid AI assistant built by **Tharun Sai**! "
            "I combine a trained neural network with live web search across Google, "
            "DuckDuckGo, Wikipedia, and a Dictionary API to give you the best answers."
        )

    # Creator / origin
    _creator_triggers = [
        "who created you","who made you","who built you","who invented you",
        "who trained you","who developed you","who programmed you",
        "who is your creator","who is your developer","your creator","your developer",
        "who is tharun","tharun sai",
    ]
    if any(t in _ut for t in _creator_triggers):
        return format_response(
            "I was created by **Tharun Sai** — a developer who built me from scratch "
            "using PyTorch, Flask, FAISS, and a custom NLP engine. No external AI API involved!"
        )

    # Capability
    _cap_triggers = ["what can you do","your capabilities","what do you know",
                     "how can you help","what are your abilities","what do you do"]
    if any(t in _ut for t in _cap_triggers) or _intent_label == "capability":
        return format_response(
            "I can help you with:\n\n"
            "- **Science** — physics, biology, chemistry, astronomy\n"
            "- **Math** — arithmetic, algebra, calculus, statistics\n"
            "- **Coding** — Python, JS, algorithms, AI/ML, debugging\n"
            "- **History** — world events, people, civilisations\n"
            "- **Geography** — capitals, rivers, countries, planets\n"
            "- **Health** — diseases, nutrition, anatomy\n"
            "- **Business** — finance, startups, economics\n"
            "- **Philosophy** — ethics, logic, meaning\n"
            "- **Live Web Search** — latest info from multiple sources\n\n"
            "Switch to **Web mode** for real-time answers!"
        )

    # How are you / social status — must come BEFORE identity check
    # (classifier sometimes misfires "how are you" as identity intent)
    _how_are_you = {"how are you","how r you","how are u","hows it going",
                    "how's it going","how do you do","you good","you okay",
                    "how have you been","how r u","how ru"}
    if _ut in _how_are_you or any(t in _ut for t in ("how are you","how r you","how r u")):
        return format_response("Running great and ready to help! What would you like to know?", spell_notice=spell_notice)

    # Nice to meet you — social phrase, not an identity query
    if any(t in _ut for t in ("nice to meet","pleased to meet","great to meet","good to meet")):
        return format_response("Nice to meet you too! What can I help you with?", spell_notice=spell_notice)



    # Name introduction
    if _ut.startswith("my name is") or (_ut.startswith("i am") and len(_ut.split()) <= 4):
        name = _ut.replace("my name is","").replace("i am","").strip().title()
        if name and len(name) < 30:
            return format_response(f"Nice to meet you, **{name}**! What would you like to learn today?")

    # What are you doing / casual existence questions — no web search
    _casual_patterns = [
        "what are you doing","what r u doing","watcha doing","what you doing",
        "what are u doing","are you busy","are you there","you there",
        "what is happening","whats up","what's up","wassup",
        "just chat","just talking","just chatting","wanna chat","want to chat",
        "lets chat","let's chat","can we chat","just talk","just wanted to chat",
        "bored","i'm bored","im bored","entertain me","talk to me",
        "nothing","just browsing","just here","just chilling","just checking",
    ]
    if any(t in _ut for t in _casual_patterns) or _ut in {"just chat","just talk","chat"}:
        _casual_replies = [
            "Happy to chat! What's on your mind — a topic you're curious about, or anything else?",
            "Sure! Ask me anything — science, coding, history, math, or just conversation.",
            "Let's go! What would you like to talk about?",
        ]
        import random as _r2
        return format_response(_r2.choice(_casual_replies))

    # Guard: nonsense / incoherent queries — detect by repeated trigger words
    # e.g. "please explain can you explain what do you know about explain"
    _words = _ut.split()
    _word_freq = {}
    for w in _words:
        _word_freq[w] = _word_freq.get(w, 0) + 1
    _max_repeat = max(_word_freq.values()) if _word_freq else 0
    _trigger_words = {"explain","tell","what","please","know","can","you","do","about","how"}
    _trigger_count = sum(1 for w in _words if w in _trigger_words)
    if (len(_words) > 4 and _trigger_count >= len(_words) * 0.6) or _max_repeat >= 3:
        return format_response("It looks like your message got a bit tangled! Could you rephrase it? Try asking something like: 'What is machine learning?' or 'Explain recursion'.")


    # ── EXAMPLE REQUEST HANDLER ──────────────────────────────────────────────
    # Detect "give me an example of X" / "example of X" / "show example X"
    example_triggers = ["example of", "give me example", "show example",
                        "give example", "examples of", "example for",
                        "show me example", "give me an example"]
    is_example_request = any(t in user_text for t in example_triggers)

    if is_example_request:
        # Extract the topic they want an example of
        topic = user_text
        for t in example_triggers:
            topic = topic.replace(t, "").strip()

        # Search web for example
        if len(topic) > 2:
            sources = multi_source_search(f"example of {topic}")
            if sources:
                merged = merge_sources(sources)
                ex_text = extract_example_from_text(merged, topic)
                learn_from_search(f"example of {topic}", merged)
                return format_response(ex_text or merged, response_type="example", topic=topic, spell_notice=spell_notice)

    # ── INTENT-AWARE ROUTING ────────────────────────────────────────────────
    # If NLP says this is a greeting/farewell/thanks with high confidence,
    # we can skip the expensive search layers entirely
    if _intent_conf >= 0.85 and _intent_label in ("greeting","farewell","thanks"):
        _intent_resp = {
            "farewell": "Goodbye! Come back anytime.",
            "thanks":   "You're welcome! Anything else I can help with?",
        }.get(_intent_label)
        if _intent_resp:
            return format_response(_intent_resp, spell_notice=spell_notice)

    # ── MATH SOLVER ─────────────────────────────────────────────────────────
    # Try on original_text first (preserves "10 factorial", "2,4,8,16,?", "2x+5=15")
    # then on normalised user_text as fallback
    math_result = solve_math(original_text) or solve_math(user_text)
    if math_result:
        return format_response(math_result, response_type="math", spell_notice=spell_notice)

    # ── WEB-ONLY MODE ────────────────────────────────────────────────────────
    if selected_module == "web":
        sources = multi_source_search(original_text)
        if sources:
            merged = merge_sources(sources)
            learn_from_search(user_text, merged)
            response = merged
            response += build_example_block(user_text)
            return response
        return "Sorry, I could not find results online for that query."

    # ── HYBRID MODE: INTERNAL FIRST ─────────────────────────────────────────

    # Step 1: Vector DB
    if selected_module != "web":
        try:
            vector_answer = vector_search(context_query) or vector_search(user_text)
            if vector_answer and len(vector_answer.split()) > 3:
                examples = build_example_block(user_text)
                if is_joke_query(user_text):
                    return format_response(vector_answer, response_type="joke", spell_notice=spell_notice, query=user_text) + ("\n\n" + html_to_md(examples) if examples else "")
                return format_response(vector_answer, spell_notice=spell_notice, query=user_text) + ("\n\n" + html_to_md(examples) if examples else "")
        except Exception as e:
            print(f"Vector search error: {e}")

    # Step 2: Fuzzy match on dataset
    fuzzy_answer = fuzzy_search(context_query) or fuzzy_search(user_text)
    if fuzzy_answer and len(fuzzy_answer.split()) > 3:
        extra_md = html_to_md(build_example_block(user_text))
        if is_joke_query(user_text):
            return format_response(fuzzy_answer, response_type="joke", spell_notice=spell_notice, query=user_text) + ("\n\n" + extra_md if extra_md else "")
        return format_response(fuzzy_answer, spell_notice=spell_notice, query=user_text) + ("\n\n" + extra_md if extra_md else "")

    # Step 3: Local AI model — GPT decoder-only inference
    local_reply = ""
    unk_ratio = 0.0

    if model is not None and 'tokenizer' in globals() and tokenizer is not None:
        try:
            local_reply = generate_bot_response(user_text, model, tokenizer, device)
        except Exception as e:
            print(f"Local AI error: {e}")

    # Step 4: Multi-source web search (when model is confused)
    is_search_intent = user_text.startswith("search ")

    # ── Canned-response guard ─────────────────────────────────────────────
    # The model defaults to Hema identity sentences when it can't answer.
    # Detect and discard those replies so web search fires instead.
    _CANNED = [
        "hema ai is more than a chatbot",
        "hema ai is an intelligent assistant",
        "i combine a trained neural network",
        "combines search reasoning and machine learning",
        "intelligent assistant that combines",
        "built by tharun",
        "hybrid ai assistant",
        "i am hema",
        "i'm hema",
    ]
    _asks_about_hema = any(w in user_text.lower()
        for w in ("hema", "who are you", "what are you", "your name", "yourself",
                  "who made you", "who built you", "what can you do"))
    if local_reply and not _asks_about_hema:
        if any(p in local_reply.lower() for p in _CANNED):
            print(f"[Pipeline] Canned reply detected — discarding, will search instead")
            local_reply = ""

    # ── Topic-word unknown check ──────────────────────────────────────────
    # specifically check whether the content words (the topic) are unknown.
    _QUESTION_WORDS = {"what","who","where","when","how","why","tell","explain",
                       "define","describe","is","can","does","do","give","show",
                       "i","want","to","know","please","me","about","a","an","the"}
    _content_words = [w for w in user_text.lower().split()
                      if w not in _QUESTION_WORDS and len(w) > 2]
    _topic_unknown = False
    # (Legacy word2idx logic bypassed; BPE handles subwords natively)

    # Don't web-search purely conversational queries
    _is_conversational = (
        _intent_label in ("greeting","farewell","thanks","identity","capability","opinion")
        or len(user_text.split()) <= 2
        or (not any(c.isalpha() for c in user_text))
    )

    # ── Force-search for definition / factual queries ─────────────────────────
    # "what is X", "define X", "explain X" — if the model has no good local answer,
    # always go to web/FAISS rather than returning a generic filler.
    _FACTUAL_PREFIXES = (
        "what is","what are","who is","who are","where is","where are",
        "when did","when was","when is","why is","why does","why do",
        "how does","how do","how is","define ","explain ","describe ",
        "tell me about","what do you know about","give me info on",
    )
    _is_factual_query = any(user_text.lower().startswith(p) for p in _FACTUAL_PREFIXES)
    # A local reply is "good" only if it's a real sentence (> 4 words)
    _local_is_good = bool(local_reply) and len(local_reply.split()) > 4

    # ── Web search helper ─────────────────────────────────────────────────────
    def _fetch_web_sources(q):
        """
        Search → Process → Answer pipeline.
        Uses smart_search to score, deduplicate and structure the answer.
        """
        q = q.replace("search ", "").strip()
        if len(q) <= 3:
            return None, None
        try:
            from smart_search import smart_search as _smart
            answer = _smart(q, multi_source_search, merge_sources)
            if answer and len(answer.split()) >= 5:
                return answer, q
        except ImportError:
            pass
        # Fallback to legacy pipeline if smart_search not available
        sources = multi_source_search(q)
        if not sources:
            return None, None
        merged = merge_sources(sources)
        if not merged or len(merged.split()) < 5:
            return None, None
        return merged, q

    def _enrich_with_web(local_ans: str, web_text: str, query: str) -> str:
        """
        Merge Hema's local answer with web search text into one clean response.

        Strategy:
        1. Extract unique facts from web_text that are NOT already in local_ans
        2. Append them as additional context after Hema's answer
        3. Keep Hema's voice — web content is supporting detail, not replacement

        Returns a single formatted Markdown string.
        """
        import re as _re

        # Sentence-tokenise both
        def _sentences(text):
            return [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text)
                    if len(s.strip()) > 30]

        local_sents = _sentences(local_ans)
        web_sents   = _sentences(web_text)

        # Keep only web sentences that add NEW information
        # (not already covered by ≥60% word overlap with any local sentence)
        local_words = set(" ".join(local_sents).lower().split())
        extra = []
        for ws in web_sents:
            ws_words = set(ws.lower().split())
            overlap = len(ws_words & local_words) / max(len(ws_words), 1)
            if overlap < 0.60:
                extra.append(ws)
            if len(extra) >= 3:   # cap extra sentences
                break

        if not extra:
            # Web adds nothing new — just return local answer
            return local_ans

        # Build merged response
        body = local_ans.rstrip()
        supplement = " ".join(extra)
        # Clean trailing partial sentences
        if not supplement[-1] in ".!?":
            supplement = supplement.rsplit(" ", 1)[0] + "."

        merged = f"{body}\n\n**Additionally:** {supplement}"
        return merged

    def _do_web_search(q):
        """Fetch web, auto-learn, return formatted response or None."""
        web_text, clean_q = _fetch_web_sources(q)
        if not web_text:
            return None, None
        learn_from_search(clean_q, web_text)
        return web_text, clean_q

    # ── Step 5: Hema first — then enrich with web ────────────────────────────
    _is_joke = is_joke_query(user_text)

    if _is_joke:
        # Jokes don't need web enrichment
        return format_response(local_reply or "Why did the programmer quit? Because he didn't get arrays.",
                               response_type="joke", spell_notice=spell_notice)

    # ── CREATIVE OVERRIDE: Never web-search poems, stories, or code ──
    _CREATIVE_PREFIXES = (
        "write ", "create ", "generate ", "code ", "can you write ", 
        "tell me a story", "write a poem", "make a "
    )
    if any(user_text.lower().startswith(p) for p in _CREATIVE_PREFIXES):
        examples_md = html_to_md(build_example_block(user_text))
        return (format_response(local_reply, spell_notice=spell_notice, query=user_text)
                + (("\n\n" + examples_md) if examples_md else ""))

    should_search = (
        not _is_conversational
        and (
            is_search_intent
            or not _local_is_good
            or (unk_ratio > 0.6)
            or _topic_unknown
            or (_is_factual_query and not _local_is_good)  # "what is X" always searches
        )
    )

    if _local_is_good and not should_search:
        # Hema has a confident answer + not a search-intent query → just return it
        examples_md = html_to_md(build_example_block(user_text))
        return (format_response(local_reply, spell_notice=spell_notice, query=user_text)
                + (("\n\n" + examples_md) if examples_md else ""))

    if _local_is_good and should_search:
        # Hema has an answer but web enrichment could improve it
        web_text, clean_q = _do_web_search(user_text)
        if web_text:
            enriched = _enrich_with_web(local_reply, web_text, user_text)
            examples_md = html_to_md(build_example_block(user_text))
            return (format_response(enriched, response_type="web", spell_notice=spell_notice, query=user_text)
                    + (("\n\n" + examples_md) if examples_md else ""))
        else:
            # Web returned nothing — serve local reply as-is
            examples_md = html_to_md(build_example_block(user_text))
            return (format_response(local_reply, spell_notice=spell_notice, query=user_text)
                    + (("\n\n" + examples_md) if examples_md else ""))

    # ── Step 6: No good local reply — web only ───────────────────────────────
    if should_search:
        web_text, clean_q = _do_web_search(user_text)
        if web_text:
            examples_md = html_to_md(build_example_block(user_text))
            return (format_response(web_text, response_type="web", spell_notice=spell_notice)
                    + (("\n\n" + examples_md) if examples_md else ""))
        # Broader keyword retry
        _broad = " ".join(user_text.lower().split()[-4:])
        if len(_broad.split()) >= 2 and _broad != user_text.lower().strip():
            web_text, clean_q = _do_web_search(_broad)
            if web_text:
                return format_response(web_text, response_type="web", spell_notice=spell_notice, query=user_text)

    return format_response(
        "I searched the web but couldn't find a reliable answer for that. "
        "Try rephrasing, or switch to **Web mode** for a direct search."
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. FLASK ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data         = request.json
        user_message = data.get("message", "")
        module       = data.get("module", "hybrid").lower()
        # Optional: client can send history array to restore context after refresh
        client_history = data.get("history", [])

        if not user_message:
            return jsonify({"reply": "Message cannot be empty."})

        # Restore client-provided history into conversation if server lost it
        if client_history and len(conversation) == 0:
            for turn in client_history[-40:]:   # max 40 turns from client
                role = turn.get("role", "user")
                text = turn.get("text", "")
                if text:
                    add_to_conversation(role, text)

        ai_reply = get_ai_response(user_message, module)
        add_to_conversation("ai", ai_reply, topic=user_message)
        last_exchange["q"] = user_message
        last_exchange["a"] = ai_reply
        return jsonify({
            "reply": ai_reply,
            "context_turns": len(conversation)  # let client know how much is remembered
        })
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"reply": "Internal error. Check terminal logs."})

@app.route("/clear", methods=["POST"])
def clear():
    global chat_history
    chat_history = []
    conversation.clear()
    last_exchange["q"] = ""
    last_exchange["a"] = ""
    # Wipe disk memory so next session starts fresh
    try:
        import os as _os
        if _os.path.exists(_CONV_FILE):
            _os.remove(_CONV_FILE)
    except Exception:
        pass
    reasoner.reset_context()
    return jsonify({"status": "cleared"})

@app.route("/sources", methods=["GET"])
def sources():
    """Show which search sources are active."""
    return jsonify({
        "sources": ["Google", "DuckDuckGo", "Wikipedia", "Dictionary API", "Local Dataset (7,408 Q&A)", "Neural Network Model"],
        "mode": "hybrid"
    })

# ════════════════════════════════════════════════════════════════════════════
# AUTO-LEARNING — saves web search results back into the dataset
# ════════════════════════════════════════════════════════════════════════════

import threading
import csv

_learn_lock = threading.Lock()

def clean_for_csv(text):
    """Strip HTML tags and normalize text for clean CSV storage."""
    import re
    text = str(text).strip()
    # Reject if it looks like raw JSON (JSON-LD bleed-through)
    if text.startswith("{") or text.startswith("["):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)          # remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()      # collapse whitespace
    text = text.replace('"', "'").replace('\n', ' ')
    return text

def learn_from_search(question, answer):
    """
    Saves a new Q&A pair learned from web search into:
      1. dataset_qa dict   (instant — available to fuzzy search immediately)
      2. chat_dataset.csv  (persistent — survives restarts + included in retrain)
    Runs in a background thread so it never slows the response.
    """
    def _write():
        with _learn_lock:
            try:
                q = clean_for_csv(question.lower().strip())
                a = clean_for_csv(answer)

                # Skip if too short or already known
                if len(q.split()) < 2 or len(a.split()) < 5:
                    return

                # ── Quality filter: reject entertainment/pop-culture junk ──
                _JUNK_SIGNALS = [
                    "is a song", "is an american", "is a 19", "is a 20",
                    "american film", "american slasher", "starring ", "performed by",
                    "box office", "nominated for", "academy award",
                    "directed by", "written by", "theme song", "soundtrack",
                    "television series", "tv series", "reality show",
                    "jennifer love hewitt", "sarah michelle gellar",
                    "raindrops keep fallin", "happy ending",
                    # JSON-LD / structured data bleed-through
                    "@context", "@type", "schema.org", "geeksforgeeks.org/logo",
                    "application/ld+json", "organization", "breadcrumb",
                ]
                a_lower = a.lower()
                _is_json = a.strip().startswith("{") or a.strip().startswith("[")
                if _is_json or any(sig in a_lower for sig in _JUNK_SIGNALS):
                    print(f"[Learn] Blocked junk result for: '{q[:50]}'")
                    return

                if q in dataset_qa:
                    return   # already known, skip

                # Add to live memory immediately
                dataset_qa[q] = a
                print(f"Learned: '{q[:60]}'")

                # Persist to CSV (2-column: input, output)
                csv_path = "data/chat_dataset.csv"
                os.makedirs("data", exist_ok=True)
                file_exists = os.path.exists(csv_path)

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    if not file_exists:
                        writer.writerow(["input", "output"])
                    writer.writerow([q, a])

            except Exception as err:
                print(f"Auto-learn error: {err}")

    t = threading.Thread(target=_write, daemon=True)
    t.start()


# ════════════════════════════════════════════════════════════════════════════
# QUOTE ENGINE — searches multiple sources, merges best result
# ════════════════════════════════════════════════════════════════════════════

# Curated fallback quotes shown when web search is unavailable
FALLBACK_QUOTES = [
    "The mind is not a vessel to be filled, but a fire to be kindled.",
    "Intelligence is the ability to adapt to change.",
    "We are all just walking each other home.",
    "The function of education is to teach one to think intensively and critically.",
    "Any sufficiently advanced technology is indistinguishable from magic.",
    "The measure of intelligence is the ability to change.",
    "Not all those who wander are lost.",
    "Somewhere, something incredible is waiting to be known.",
    "The universe is under no obligation to make sense to you.",
    "We did not build minds of silicon to answer everything, but to give our wildest questions a place to echo.",
    "Education is not preparation for life; education is life itself.",
    "The greatest enemy of knowledge is not ignorance, it is the illusion of knowledge.",
    "In the middle of difficulty lies opportunity.",
    "Logic will get you from A to Z; imagination will get you everywhere.",
    "Two things are infinite: the universe and human stupidity; and I'm not sure about the universe.",
    "The important thing is not to stop questioning.",
    "An investment in knowledge pays the best interest.",
    "Tell me and I forget, teach me and I may remember, involve me and I learn.",
    "The roots of education are bitter, but the fruit is sweet.",
    "It does not matter how slowly you go as long as you do not stop.",
]

def fetch_quote_from_web():
    """
    Searches multiple sources for a quote, picks and merges the best result.
    Returns just the text — no source labels shown to user.
    """
    candidates = []

    # Source 1: Quotable API (free, no key)
    try:
        r = requests.get("https://api.quotable.io/random?tags=technology|science|education|wisdom", timeout=4)
        if r.status_code == 200:
            data = r.json()
            if data.get("content") and len(data["content"]) > 30:
                candidates.append(data["content"].strip())
    except:
        pass

    # Source 2: ZenQuotes API (free, no key)
    try:
        r = requests.get("https://zenquotes.io/api/random", timeout=4)
        if r.status_code == 200:
            data = r.json()
            if data and data[0].get("q") and len(data[0]["q"]) > 30:
                candidates.append(data[0]["q"].strip())
    except:
        pass

    # Source 3: DuckDuckGo search for "inspirational quote"
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": "inspirational quote about learning", "format": "json", "no_html": 1},
            timeout=4
        )
        data = r.json()
        abstract = data.get("AbstractText", "")
        if abstract and len(abstract) > 40:
            # Extract sentence that looks like a quote
            sentences = [s.strip() for s in abstract.split(".") if len(s.strip()) > 30]
            if sentences:
                candidates.append(sentences[0])
    except:
        pass

    if candidates:
        # Pick the most interesting one — prefer medium length (40-180 chars)
        scored = []
        for q in candidates:
            length = len(q)
            score = 0
            if 60 <= length <= 160: score += 3   # ideal length
            elif 40 <= length <= 200: score += 1  # acceptable
            if any(w in q.lower() for w in ["learn", "knowledge", "mind", "think", "science", "ai", "truth", "wisdom"]):
                score += 2
            scored.append((score, q))
        scored.sort(reverse=True)
        return scored[0][1]

    # All sources failed — use fallback
    return random.choice(FALLBACK_QUOTES)


# ════════════════════════════════════════════════════════════════════════════
# FILE TEXT EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# FILE ANALYSIS ENGINE
# ════════════════════════════════════════════════════════════════════════════

TEXT_EXTENSIONS = {".txt",".py",".js",".ts",".html",".css",".json",
                   ".csv",".md",".xml",".yaml",".yml",".java",".cpp",
                   ".c",".h",".rs",".go",".sh",".rb",".php",".swift",
                   ".kt",".r",".sql",".toml",".ini",".cfg",".log"}

IMAGE_EXTENSIONS = {".jpg",".jpeg",".png",".gif",".bmp",".webp",
                    ".svg",".ico",".tiff",".tif"}

def get_file_ext(filename):
    return ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

def extract_file_text(file_data_b64, filename, mimetype):
    """Extract raw text content from any supported file."""
    try:
        file_bytes = base64.b64decode(file_data_b64)
        ext = get_file_ext(filename)

        # ── Plain text / code files ──
        if ext in TEXT_EXTENSIONS or mimetype.startswith("text/"):
            return file_bytes.decode("utf-8", errors="replace")

        # ── PDF ──
        if ext == ".pdf" or mimetype == "application/pdf":
            if PDF_AVAILABLE:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                pages_text = []
                for i, page in enumerate(reader.pages[:10]):
                    t = page.extract_text() or ""
                    if t.strip():
                        pages_text.append(f"[Page {i+1}]\n{t}")
                full = "\n\n".join(pages_text)
                return full if full.strip() else "PDF has no extractable text."
            return "PDF support requires: pip install pypdf"

        # ── Image ──
        if ext in IMAGE_EXTENSIONS or mimetype.startswith("image/"):
            return None  # handled separately via analyze_image()

        return f"Unsupported file type: {filename}"

    except Exception as e:
        return f"Error reading file: {str(e)}"


# ════════════════════════════════════════════════════════════════════════════
# DEEP IMAGE ANALYSIS ENGINE
# ════════════════════════════════════════════════════════════════════════════

def _rgb_to_name(r, g, b):
    """Map an average RGB value to a human-readable color name."""
    colors = [
        ((255,0,0),"Red"),((200,50,50),"Dark Red"),((255,100,100),"Light Red"),
        ((255,165,0),"Orange"),((255,200,50),"Yellow-Orange"),
        ((255,255,0),"Yellow"),((200,220,50),"Yellow-Green"),
        ((0,200,0),"Green"),((0,128,0),"Dark Green"),((100,220,180),"Teal"),
        ((0,0,255),"Blue"),((70,130,180),"Steel Blue"),((100,149,237),"Cornflower Blue"),
        ((138,43,226),"Purple"),((255,20,147),"Pink"),((165,42,42),"Brown"),
        ((255,255,255),"White"),((220,220,220),"Light Grey"),
        ((128,128,128),"Grey"),((50,50,50),"Dark Grey"),((0,0,0),"Black"),
    ]
    def dist(c1,c2): return sum((a-b)**2 for a,b in zip(c1,c2))**0.5
    return min(colors, key=lambda c: dist(c[0],(r,g,b)))[1]

def _get_top_colors(pixels, n=5):
    """Extract top N dominant colors using quantization buckets."""
    from collections import Counter
    # Bucket each pixel into 32-level bins to reduce noise
    bucketed = [(r//32*32, g//32*32, b//32*32) for r,g,b in pixels]
    counts = Counter(bucketed).most_common(n*3)
    # Convert to named colors, deduplicate
    seen_names = []
    result = []
    for (r,g,b), count in counts:
        name = _rgb_to_name(r,g,b)
        if name not in seen_names:
            seen_names.append(name)
            pct = round(count / len(bucketed) * 100)
            result.append((name, pct))
        if len(result) >= n:
            break
    return result

def _classify_image_type(width, height, brightness, color_variety, edge_density, size_kb):
    """
    Heuristically classify the image type:
    screenshot, diagram, photo, chart, logo, icon, document scan
    """
    aspect = width / height if height else 1
    is_square = 0.85 <= aspect <= 1.15
    is_wide   = aspect > 1.5
    is_tall   = aspect < 0.67

    # Screenshots: usually bright, low saturation variance, wide aspect
    if brightness > 210 and color_variety < 15 and is_wide:
        return "Screenshot / UI"

    # Icons/logos: small, square, high color contrast
    if size_kb < 50 and is_square and color_variety < 10:
        return "Icon or Logo"

    # Document scans: very bright, near A4 aspect, high edge density
    if brightness > 220 and 1.2 < aspect < 1.5 and edge_density > 0.3:
        return "Document Scan"

    # Diagrams/charts: moderate brightness, low color variety, high edges
    if edge_density > 0.25 and color_variety < 20 and brightness > 150:
        return "Diagram or Chart"

    # Dark photos: low brightness, high variety
    if brightness < 100 and color_variety > 25:
        return "Dark Photography"

    # General photos: high color variety
    if color_variety > 20:
        return "Photograph"

    return "Graphic / Illustration"

def _compute_edge_density(img_gray, sample_size=100):
    """Estimate edge density by measuring pixel-to-pixel variance on a sample grid."""
    try:
        w, h = img_gray.size
        step_x = max(1, w // sample_size)
        step_y = max(1, h // sample_size)
        pixels = img_gray.load()
        diffs = []
        for y in range(0, h - step_y, step_y):
            for x in range(0, w - step_x, step_x):
                diffs.append(abs(int(pixels[x,y]) - int(pixels[x+step_x,y])))
                diffs.append(abs(int(pixels[x,y]) - int(pixels[x,y+step_y])))
        if not diffs: return 0
        avg_diff = sum(diffs) / len(diffs)
        return round(avg_diff / 255, 3)   # 0=flat, 1=max edges
    except:
        return 0

def _analyze_regions(img_rgb, grid=3):
    """
    Divide image into a grid and describe each region's dominant color.
    Returns a list of region descriptions.
    """
    try:
        w, h = img_rgb.size
        rw, rh = w // grid, h // grid
        region_names = {
            (0,0):"top-left",(1,0):"top-center",(2,0):"top-right",
            (0,1):"middle-left",(1,1):"center",(2,1):"middle-right",
            (0,2):"bottom-left",(1,2):"bottom-center",(2,2):"bottom-right"
        }
        regions = []
        for gy in range(grid):
            for gx in range(grid):
                box = (gx*rw, gy*rh, (gx+1)*rw, (gy+1)*rh)
                region = img_rgb.crop(box).resize((10,10))
                px = list(region.getdata())
                avg = tuple(sum(c[i] for c in px)//len(px) for i in range(3))
                color = _rgb_to_name(*avg)
                regions.append((region_names.get((gx,gy), f"region-{gx},{gy}"), color))
        return regions
    except:
        return []

def analyze_image(file_data_b64, filename, mimetype):
    """
    Deep image analysis using PIL:
    - Dimensions, format, file size, DPI
    - Image type classification (photo/diagram/screenshot/icon/scan)
    - Brightness, contrast, dominant colors
    - Color palette (top 5 colors with %)
    - Edge density (sharpness estimate)
    - Region map (3x3 grid color breakdown)
    - Learns description into dataset
    Returns a rich dict with all analysis results.
    """
    result = {
        "filename": filename,
        "size_kb": 0,
        "format": "",
        "mode": "",
        "width": 0, "height": 0,
        "orientation": "",
        "image_type": "Unknown",
        "brightness": 0,
        "contrast": 0,
        "dominant_colors": [],
        "edge_density": 0,
        "regions": [],
        "dpi": None,
        "has_transparency": False,
        "description": "",
    }
    try:
        file_bytes = base64.b64decode(file_data_b64)
        result["size_kb"] = len(file_bytes) // 1024

        if not PIL_AVAILABLE:
            result["description"] = f"Image file: {filename}, {result['size_kb']} KB. Install Pillow for deep analysis: pip install Pillow"
            return result

        img = Image.open(io.BytesIO(file_bytes))
        result["format"]  = (img.format or get_file_ext(filename).lstrip(".")).upper()
        result["mode"]    = img.mode
        result["width"], result["height"] = img.size
        result["has_transparency"] = img.mode in ("RGBA","LA","PA") or (
            img.mode == "P" and "transparency" in img.info
        )
        try: result["dpi"] = img.info.get("dpi")
        except: pass

        w, h = result["width"], result["height"]
        result["orientation"] = "landscape" if w > h else ("portrait" if h > w else "square")

        # ── Convert to RGB for analysis ──
        img_rgb = img.convert("RGB")
        img_gray = img.convert("L")

        # ── Sample pixels ──
        sample_img = img_rgb.resize((80, 80))
        pixels_rgb = list(sample_img.getdata())
        pixels_lum = [int(0.299*r + 0.587*g + 0.114*b) for r,g,b in pixels_rgb]

        # Brightness & contrast
        brightness = sum(pixels_lum) // len(pixels_lum)
        result["brightness"] = brightness
        mean_lum = brightness
        variance = sum((p - mean_lum)**2 for p in pixels_lum) / len(pixels_lum)
        result["contrast"] = round(variance**0.5, 1)

        # Color palette
        result["dominant_colors"] = _get_top_colors(pixels_rgb, n=5)

        # Edge density
        result["edge_density"] = _compute_edge_density(img_gray)

        # Color variety (unique hue buckets)
        hue_buckets = set((r//64, g//64, b//64) for r,g,b in pixels_rgb)
        color_variety = len(hue_buckets)

        # Image type classification
        result["image_type"] = _classify_image_type(
            w, h, brightness, color_variety,
            result["edge_density"], result["size_kb"]
        )

        # Region map
        result["regions"] = _analyze_regions(img_rgb, grid=3)

        # ── Build natural language description ──
        tone = "bright" if brightness > 200 else ("dark" if brightness < 80 else "medium-toned")
        sharpness = "sharp/detailed" if result["edge_density"] > 0.2 else ("moderate detail" if result["edge_density"] > 0.1 else "soft/low detail")
        top_color = result["dominant_colors"][0][0] if result["dominant_colors"] else "unknown"
        transparency_note = " with transparency" if result["has_transparency"] else ""
        dpi_note = f", {result['dpi'][0]} DPI" if result["dpi"] else ""

        result["description"] = (
            f"This appears to be a {result['image_type'].lower()}{transparency_note}. "
            f"It is {w}×{h} pixels ({result['orientation']}){dpi_note}, "
            f"saved as {result['format']} ({result['size_kb']} KB). "
            f"The image has a {tone} overall tone with {sharpness}. "
            f"The dominant color is {top_color}."
        )

        return result

    except Exception as e:
        result["description"] = f"Image analysis error: {str(e)}"
        return result


def format_image_report(analysis, user_msg=""):
    """
    Build a clean structured Markdown report from the image analysis dict.
    """
    a = analysis
    fname = a["filename"]
    lines = []

    # ── Header ──
    lines.append(f"## 🖼 Image Analysis: `{fname}`")
    lines.append("")

    # ── Description ──
    lines.append(f"> {a['description']}")
    lines.append("")

    # ── Stats table ──
    lines.append("### Properties")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Type | {a['image_type']} |")
    lines.append(f"| Dimensions | {a['width']} × {a['height']} px ({a['orientation']}) |")
    lines.append(f"| Format | {a['format']} · {a['mode']} mode |")
    lines.append(f"| File size | {a['size_kb']} KB |")
    if a.get("dpi"): lines.append(f"| DPI | {a['dpi'][0]} |")
    lines.append(f"| Brightness | {a['brightness']}/255 ({'bright' if a['brightness']>180 else ('dark' if a['brightness']<80 else 'medium')}) |")
    lines.append(f"| Contrast | {a['contrast']} |")
    lines.append(f"| Sharpness | {'High' if a['edge_density']>0.2 else ('Medium' if a['edge_density']>0.1 else 'Low')} (edge density: {a['edge_density']}) |")
    lines.append(f"| Transparency | {'Yes' if a['has_transparency'] else 'No'} |")
    lines.append("")

    # ── Color palette ──
    if a["dominant_colors"]:
        lines.append("### Color Palette")
        for color, pct in a["dominant_colors"]:
            bar_len = max(1, pct // 5)
            bar = "█" * bar_len
            lines.append(f"- **{color}** {bar} {pct}%")
        lines.append("")

    # ── Region map ──
    if a["regions"]:
        lines.append("### Region Breakdown")
        lines.append("How colors are distributed across the image:")
        lines.append("")
        # Build 3x3 grid as a table
        region_dict = {r: c for r, c in a["regions"]}
        lines.append("| | Left | Center | Right |")
        lines.append("|--|------|--------|-------|")
        for row_label, positions in [("Top",["top-left","top-center","top-right"]),
                                      ("Middle",["middle-left","center","middle-right"]),
                                      ("Bottom",["bottom-left","bottom-center","bottom-right"])]:
            cells = " | ".join(region_dict.get(p,"—") for p in positions)
            lines.append(f"| **{row_label}** | {cells} |")
        lines.append("")

    return "\n".join(lines)


def parse_user_intent(user_msg):
    """
    Parse user message to determine desired output depth.
    Returns dict: { points: int, detail: 'brief'|'normal'|'detailed', focus: str|None }
    """
    msg = user_msg.lower().strip() if user_msg else ""

    # Detect explicit number requests: "5 key points", "give me 3 points"
    import re
    num_match = re.search(r"(\d+)\s*(key\s*point|point|bullet|line|item|sentence)", msg)
    explicit_n = int(num_match.group(1)) if num_match else None

    # Detect brevity preference
    if any(w in msg for w in ["brief", "short", "quick", "tldr", "tl;dr", "one line", "summarise briefly"]):
        detail = "brief"
        default_n = 3
    elif any(w in msg for w in ["detail", "thorough", "full", "complete", "everything", "in depth", "comprehensive"]):
        detail = "detailed"
        default_n = 8
    else:
        detail = "normal"
        default_n = 5

    points = explicit_n if explicit_n else default_n

    # Detect focus topic: "find key points about X", "summarise the functions"
    focus = None
    focus_match = re.search(r"(about|on|for|regarding|focus on)\s+([\w\s]+?)(?:\.|$|,)", msg)
    if focus_match:
        focus = focus_match.group(2).strip()

    return {"points": min(max(points, 1), 15), "detail": detail, "focus": focus}


def detect_document_sections(text):
    """
    Detect headings and section boundaries in plain text / PDF.
    Returns list of (heading, body_text) tuples.
    """
    import re
    lines = text.split("\n")
    sections = []
    current_heading = "Introduction"
    current_body = []

    heading_pat = re.compile(
        r"^(#{1,4}\s+.+|[A-Z][A-Z\s]{4,50}[A-Z]:?$|\d+\.\d*\s+[A-Z].{3,50}$)"
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Detect heading: all caps short line, or starts with #, or numbered
        is_heading = (
            heading_pat.match(stripped) or
            (len(stripped) < 80 and stripped.isupper() and len(stripped.split()) >= 2) or
            re.match(r"^(Chapter|Section|Part|Appendix)\s+\w", stripped, re.I)
        )
        if is_heading and len(current_body) > 0:
            sections.append((current_heading, " ".join(current_body)))
            current_heading = stripped.lstrip("#").strip()
            current_body = []
        elif is_heading:
            current_heading = stripped.lstrip("#").strip()
        else:
            current_body.append(stripped)

    if current_body:
        sections.append((current_heading, " ".join(current_body)))

    return sections if sections else [("Content", text)]


def detect_tables_in_text(text):
    """Detect table-like structures (pipe-separated or tab-separated rows)."""
    import re
    tables = []
    lines = text.split("\n")
    table_lines = []
    for line in lines:
        if "|" in line and len(line.split("|")) >= 3:
            table_lines.append(line)
        else:
            if len(table_lines) >= 2:
                tables.append(table_lines[:])
            table_lines = []
    if len(table_lines) >= 2:
        tables.append(table_lines)
    return tables


def extract_key_points(text, max_points=5, min_words_per_sent=6):
    """
    TF-IDF-style key point extraction.
    Scores sentences by:
    - Term frequency of important content words
    - Position bonus (early sentences score higher)
    - Information-density bonus (longer informative sentences)
    - Deduplication (skip sentences too similar to already-chosen ones)
    """
    import re
    from collections import Counter

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) >= min_words_per_sent]

    if not sentences:
        sentences = [l.strip() for l in text.split("\n") if len(l.split()) >= 3]
    if not sentences:
        return []

    # Build word frequency table (IDF-style: rare = more important)
    all_words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    word_freq = Counter(all_words)
    total_words = len(all_words) or 1

    skip_patterns = [
        "click here", "read more", "learn more", "privacy policy",
        "terms of", "all rights reserved", "copyright", "subscribe",
        "newsletter", "cookie", "[page", "page {"
    ]

    info_words = {
        "defined","means","refers","because","therefore","however",
        "important","significant","key","main","primary","major",
        "result","output","function","method","approach","process",
        "example","instance","such","including","consists","enables",
        "provides","allows","requires","returns","class","import"
    }

    scored = []
    for i, s in enumerate(sentences):
        sl = s.lower()
        if any(p in sl for p in skip_patterns):
            continue

        # TF score: sum of rare-word weights
        words = re.findall(r"\b[a-z]{4,}\b", sl)
        tf_score = sum(1 / (word_freq.get(w, 1) ** 0.5) for w in words)

        # Info-word bonus
        info_bonus = sum(3 for w in info_words if w in sl)

        # Position bonus: first 20% of sentences
        position_bonus = 5 if i < max(1, len(sentences) // 5) else 0

        # Length bonus: 10-30 word sentences are ideal
        wlen = len(s.split())
        length_bonus = 4 if 10 <= wlen <= 30 else (2 if wlen <= 50 else 0)

        total_score = tf_score + info_bonus + position_bonus + length_bonus
        scored.append((total_score, i, s))

    scored.sort(reverse=True)

    # Select top N, deduplicated
    selected = []
    selected_words = set()
    for _, idx, s in scored:
        s_words = set(re.findall(r"\b[a-z]{4,}\b", s.lower()))
        # Skip if 60%+ overlap with already chosen
        if selected:
            max_overlap = max(
                len(s_words & set(re.findall(r"\b[a-z]{4,}\b", prev.lower()))) /
                max(len(s_words), 1)
                for prev in selected
            )
            if max_overlap > 0.6:
                continue
        selected.append(s)
        if len(selected) >= max_points:
            break

    # Re-sort by original order
    selected_set = set(selected)
    return [s for s in sentences if s in selected_set][:max_points]


def summarise_text(text, filename, intent=None):
    """
    Full document analysis pipeline:
    1. Detect sections/headings
    2. Detect tables
    3. Extract key points with TF-IDF scoring
    4. Build structured Markdown output
    5. Scale everything to file size + user intent
    """
    import re

    if intent is None:
        intent = {"points": 5, "detail": "normal", "focus": None}

    ext        = get_file_ext(filename)
    words      = text.split()
    word_count = len(words)
    char_count = len(text)
    lines_all  = text.split("\n")
    line_count = len(lines_all)
    detail     = intent.get("detail", "normal")

    # ── Scale parameters to file size + user intent ──
    desired_n = intent["points"]
    if word_count < 50:       size_cap = 2
    elif word_count < 150:    size_cap = 3
    elif word_count < 400:    size_cap = 5
    elif word_count < 1000:   size_cap = 7
    elif word_count < 3000:   size_cap = 10
    else:                     size_cap = 15

    n_points       = min(desired_n, size_cap)
    min_sent_words = 3 if word_count < 100 else (5 if word_count < 400 else 6)
    overview_n     = 1 if detail == "brief" else (4 if detail == "detailed" else 2)
    snippet_len    = 100 if detail == "brief" else (250 if detail == "detailed" else 160)

    # ════════════════════════════════════════════════
    # CSV
    # ════════════════════════════════════════════════
    if ext == ".csv":
        n_preview = 2 if detail == "brief" else (8 if detail == "detailed" else 4)
        header    = lines_all[0] if lines_all else ""
        rows      = max(0, line_count - 1)
        cols      = len(header.split(","))
        preview   = lines_all[1:n_preview+1]
        return format_csv_summary(filename, rows, cols, header, preview, intent)

    # ════════════════════════════════════════════════
    # CODE FILES
    # ════════════════════════════════════════════════
    code_exts = {".py",".js",".ts",".java",".cpp",".c",".go",".rs",
                 ".rb",".php",".sh",".swift",".kt",".sql"}
    if ext in code_exts:
        func_pat    = re.compile(r"^\s*(async def |def |function |func |public |private |protected |class |const \w+ = \(|let \w+ = \()", re.M)
        funcs       = func_pat.findall(text)
        import_pat  = re.compile(r"^\s*(import |from |require|#include|using )", re.M)
        imports     = import_pat.findall(text)
        comment_n   = sum(1 for l in lines_all if l.strip().startswith(("#","//","/*","*","--")))
        blank_n     = sum(1 for l in lines_all if not l.strip())
        code_n      = line_count - comment_n - blank_n

        key_pts = [] if detail == "brief" else extract_key_points(
            text.replace("\n", ". "), max_points=n_points, min_words_per_sent=min_sent_words
        )
        key_pts = [pt[:snippet_len] + ("..." if len(pt) > snippet_len else "") for pt in key_pts]
        return format_code_summary(filename, line_count, word_count, code_n,
                                   comment_n, blank_n, len(funcs), key_pts, intent,
                                   extra=f"Imports/includes: {len(imports)}")

    # ════════════════════════════════════════════════
    # PLAIN TEXT / PDF / MARKDOWN
    # ════════════════════════════════════════════════
    output = []
    lang_ext = get_file_ext(filename)

    # ── Header ──
    output.append(f"## 📄 Document: `{filename}`")
    if detail != "brief":
        output.append(f"**{word_count:,}** words · **{line_count:,}** lines · **{char_count:,}** characters")
    else:
        output.append(f"**{word_count:,}** words")
    output.append("")

    # ── Detect sections ──
    sections = detect_document_sections(text)
    has_sections = len(sections) > 1

    # ── Overview (first N sentences of whole doc) ──
    all_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    valid_sents   = [s for s in all_sentences if len(s.split()) > 4]
    overview      = " ".join(valid_sents[:overview_n])

    if overview and detail != "brief":
        output.append("### Overview")
        output.append(overview)
        output.append("")

    # ── Section summary ──
    if has_sections and detail != "brief":
        output.append("### Sections Detected")
        for heading, body in sections[:8]:
            word_n = len(body.split())
            heading_clean = heading[:70]
            output.append(f"- **{heading_clean}** ({word_n:,} words)")
        output.append("")

    # ── Key points (TF-IDF scored) ──
    key_pts = extract_key_points(text, max_points=n_points, min_words_per_sent=min_sent_words)
    if key_pts:
        label = "Summary" if detail == "brief" else f"Key Points ({len(key_pts)})"
        output.append(f"### {label}")
        for pt in key_pts:
            clipped = pt[:snippet_len] + ("..." if len(pt) > snippet_len else "")
            output.append(f"- {clipped}")
        output.append("")

    # ── Section deep-dive (detailed mode only) ──
    if has_sections and detail == "detailed":
        output.append("### Section Summaries")
        for heading, body in sections[:6]:
            if len(body.split()) < 10:
                continue
            body_pts = extract_key_points(body, max_points=3, min_words_per_sent=5)
            if body_pts:
                output.append(f"**{heading[:60]}**")
                for pt in body_pts:
                    output.append(f"  - {pt[:150]}")
                output.append("")

    # ── Tables found ──
    tables = detect_tables_in_text(text)
    if tables and detail != "brief":
        output.append(f"### Tables Found: {len(tables)}")
        for i, tbl in enumerate(tables[:2]):
            output.append(f"**Table {i+1}:**")
            output.append("```")
            for row in tbl[:5]:
                output.append(row)
            output.append("```")
            output.append("")

    return "\n".join(output)




@app.route("/quote", methods=["GET"])
def get_quote():
    """Returns a merged best quote from multiple web sources. No source shown."""
    try:
        quote_text = fetch_quote_from_web()
        return jsonify({"text": quote_text, "author": ""})
    except Exception as e:
        print(f"Quote fetch error: {e}")
        return jsonify({"text": random.choice(FALLBACK_QUOTES), "author": ""})


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Full file analysis:
    - Text/code/PDF → summary + key points
    - Images → dimensions, colors, tone analysis
    - User question → answered in context of file content
    """
    try:
        data     = request.json
        file_b64 = data.get("file_data", "")
        filename = data.get("filename", "file")
        mimetype = data.get("mimetype", "text/plain")
        user_msg = data.get("message", "").strip()
        module   = data.get("module", "hybrid").lower()

        if not file_b64:
            return jsonify({"reply": "No file data received."})

        ext = get_file_ext(filename)

        # ── IMAGE ────────────────────────────────────────────────────────────
        if ext in IMAGE_EXTENSIONS or mimetype.startswith("image/"):
            analysis  = analyze_image(file_b64, filename, mimetype)
            reply     = format_image_report(analysis, user_msg)

            # Answer user question in context of image properties
            if user_msg:
                ctx = analysis["description"]
                ai_ans = get_ai_response(
                    f"{user_msg}\n\n[Image context: {ctx}]", module
                )
                reply += f"\n\n---\n\n**Answer to your question:**\n\n{format_response(ai_ans)}"

            # Learn image description into dataset
            img_key   = f"what is in the image {filename.lower()}"
            img_learn = analysis["description"]
            if img_learn and "error" not in img_learn.lower():
                learn_from_search(img_key, img_learn)

            # Add to conversation context
            add_to_conversation("ai", reply, topic=f"image {filename}")

            return jsonify({
                "reply": reply,
                "filename": filename,
                "type": "image",
                "image_type": analysis.get("image_type",""),
                "width": analysis.get("width",0),
                "height": analysis.get("height",0),
            })

        # ── TEXT / CODE / PDF ────────────────────────────────────────────────
        file_text = extract_file_text(file_b64, filename, mimetype)

        if not file_text or file_text.startswith("Error") or file_text.startswith("Unsupported"):
            return jsonify({"reply": file_text or "Could not read file.", "filename": filename})

        word_count = len(file_text.split())
        intent     = parse_user_intent(user_msg)

        ctx_sizes  = {"brief": 3000, "normal": 6000, "detailed": 12000}
        MAX_CTX    = ctx_sizes[intent["detail"]]

        # Full document analysis
        structured = summarise_text(file_text, filename, intent)

        # Answer specific question in full file context
        if user_msg:
            context = file_text[:MAX_CTX]
            if len(file_text) > MAX_CTX:
                context += "\n\n[... document continues, truncated at {:,} words for context ...]".format(len(context.split()))
            ai_answer = get_ai_response(
                f"{user_msg}\n\n[Document: {filename}]\n{context}", module
            )
            reply = structured + "\n\n---\n\n### Answer\n\n" + format_response(ai_answer)
        else:
            reply = structured

        # Learn document summary into dataset
        learn_key = f"summarise document {filename.lower()}"
        # Build a short learnable summary from first key points
        kp = extract_key_points(file_text, max_points=3)
        learn_val = " ".join(kp) if kp else file_text[:300]
        learn_from_search(learn_key, learn_val)

        # Add to conversation context so follow-up questions work
        add_to_conversation("ai", reply[:400], topic=f"document {filename}")

        return jsonify({
            "reply": reply,
            "filename": filename,
            "type": "text",
            "word_count": word_count,
            "intent": intent,
            "sections": len(detect_document_sections(file_text))
        })

    except Exception as e:
        print(f"Upload error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"reply": f"Error processing file: {str(e)}"})


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Thumbs up / down on last response.
    - Thumbs up   -> reinforces the answer (saves to dataset if new)
    - Thumbs down -> auto-searches Google, returns result to frontend,
                     saves when user confirms or types manually
    """
    try:
        data       = request.json
        rating     = data.get("rating", "")
        correction = data.get("correction", "").strip()
        question   = data.get("question", last_exchange.get("q", "")).strip()
        answer     = data.get("answer",   last_exchange.get("a", "")).strip()
        confirmed  = data.get("confirmed", False)

        if not question:
            return jsonify({"status": "error", "msg": "No question to look up."})

        q_clean = clean_for_csv(question.lower())
        a_clean = clean_for_csv(answer)

        # ── Save DPO preference pair ──────────────────────────────────────────
        # Every thumbs up/down creates a (question, good, bad) triple for Stage 3
        def _save_dpo_pair(q, good_ans, bad_ans):
            """Append a preference pair to memory/feedback.csv for DPO training."""
            try:
                import csv as _csv
                os.makedirs("memory", exist_ok=True)
                path = "memory/feedback.csv"
                write_header = not os.path.exists(path)
                with open(path, "a", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=["question","good_answer","bad_answer"])
                    if write_header:
                        w.writeheader()
                    w.writerow({"question": q,
                                "good_answer": clean_for_csv(good_ans),
                                "bad_answer":  clean_for_csv(bad_ans)})
            except Exception as e:
                print(f"[DPO] Save failed: {e}")

        # Thumbs UP
        if rating == "up":
            if q_clean not in dataset_qa:
                learn_from_search(q_clean, a_clean)
            # DPO: current answer = good. Last web search result = bad (if different)
            _save_dpo_pair(q_clean, a_clean, dataset_qa.get(q_clean, a_clean))
            return jsonify({"status": "ok", "msg": "Thanks! I'll remember that."})

        # Thumbs DOWN
        elif rating == "down":

            # Manual correction typed by user -> save immediately
            if correction and not confirmed:
                c_clean = clean_for_csv(correction)
                dataset_qa[q_clean] = c_clean
                learn_from_search(q_clean, c_clean)
                # DPO: correction = good answer, original = bad answer
                _save_dpo_pair(q_clean, c_clean, a_clean)
                return jsonify({"status": "ok",
                                "msg": "Correction saved! I'll use your answer next time."})

            # User confirmed the Google-searched answer -> save it
            if confirmed and correction:
                c_clean = clean_for_csv(correction)
                dataset_qa[q_clean] = c_clean
                learn_from_search(q_clean, c_clean)
                # DPO: confirmed answer = good, original = bad
                _save_dpo_pair(q_clean, c_clean, a_clean)
                return jsonify({"status": "ok",
                                "msg": "Learned from Google! I'll use this next time."})

            # No correction yet -> search Google and return result to frontend
            print(f"[Feedback] Searching for: '{question[:60]}'")
            try:
                sources = multi_source_search(question)
                if sources:
                    merged = merge_sources(sources)
                    parts = [s.strip() for s in merged.replace("\n", " ").split(". ")
                             if len(s.strip()) > 20]
                    short = ". ".join(parts[:2]).strip()
                    if short and not short.endswith("."):
                        short += "."
                    if len(short) > 20:
                        return jsonify({"status": "found",
                                        "searched_answer": short,
                                        "msg": "Found on Google - does this look right?"})
                return jsonify({"status": "not_found",
                                "msg": "Could not find a good answer. Type it below."})
            except Exception as se:
                print(f"[Feedback] Search error: {se}")
                return jsonify({"status": "not_found",
                                "msg": "Search failed. Please type the correct answer."})

        return jsonify({"status": "error", "msg": "Invalid rating."})

    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({"status": "error", "msg": str(e)})


@app.route("/nlp_debug", methods=["GET"])
def nlp_debug():
    """Debug route — show NLP analysis of a query."""
    q = request.args.get("q", "what is machine learning")
    result = {
        "query": q,
        "nlp_available": _nlp is not None and _nlp._ready,
    }
    if _nlp and _nlp._ready:
        intent, conf  = _nlp.classify_intent(q)
        entities      = _nlp.extract_entities(q)
        corrected, changed, _ = _nlp.spell_correct(q)
        normalised    = _nlp.normalize(q)
        search_result = _nlp.search_many(q, top_n=3)
        result.update({
            "intent":       intent,
            "intent_conf":  conf,
            "entities":     entities,
            "corrected":    corrected,
            "spell_changed":changed,
            "normalised":   normalised,
            "top_matches":  [{"answer": a[:80], "score": round(s,3)} for a, e, s in search_result],
        })
    return jsonify(result)


@app.route("/context", methods=["GET"])
def show_context():
    return jsonify({
        "total_turns": len(conversation),
        "last_topic": get_last_topic(),
        "recent_10": conversation[-10:],
        "context_prompt_preview": build_context_prompt("(your next question)", max_turns=6)
    })


@app.route("/learned", methods=["GET"])
def learned():
    """Shows the most recently auto-learned Q&A pairs."""
    try:
        csv_path = "data/chat_dataset.csv"
        if not os.path.exists(csv_path):
            return jsonify({"count": 0, "recent": []})
        df = pd.read_csv(csv_path).dropna()
        recent = df.tail(20)[["input", "output"]].to_dict(orient="records")
        return jsonify({"count": len(df), "recent": recent})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    if not os.path.exists('templates'):
        os.makedirs('templates')
    _load_memory()   # restore conversation + user profile from disk
    print(f"[Memory] Session #{user_profile['session_count']} | "
          f"{len(conversation)} prior turns loaded")
    print("\nStarting Hema AI — Multi-Source Hybrid Server")
    print("Sources: Google + DuckDuckGo + Wikipedia + Dictionary + GeeksforGeeks + Local AI")
    print("Open: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=True)