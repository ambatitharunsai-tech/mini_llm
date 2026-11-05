"""
Hema NLP Engine — nlp_engine.py
=================================
Full NLP pipeline built from scratch (no NLTK/spaCy dependency):

  1. Tokenizer          — smarter than str.split(): handles contractions,
                          punctuation, numbers, camelCase splitting
  2. Stemmer            — Porter-style suffix stripping (pure Python)
  3. Stop-word filter   — domain-aware (keeps "not", "no", "never" etc.)
  4. TF-IDF vectoriser  — builds from dataset at startup, enables cosine search
  5. Intent classifier  — 18 intent classes from rule patterns + TF-IDF fallback
  6. Named Entity       — extracts PERSON, PLACE, TECH, SCIENCE, NUMBER from text
  7. Semantic search    — cosine similarity over TF-IDF vectors (replaces difflib)
  8. Query expander     — adds synonyms to boost recall

All classes are importable independently.  app.py imports the top-level
`HemaNLP` facade and replaces:
  - fuzzy_search()   →  nlp.search(query)
  - normalize()      →  nlp.normalize(query)
  - spell_correct()  →  nlp.spell_correct(text)
  - intent detection →  nlp.classify_intent(text)
"""

from __future__ import annotations
import re, math, string
from collections import Counter, defaultdict
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════
# 1. TOKENIZER
# ════════════════════════════════════════════════════════════════════════════

class Tokenizer:
    """
    Word tokenizer that handles:
    - Contractions: "don't" → ["do", "not"]
    - Possessives:  "python's" → ["python"]
    - Hyphenated:   "real-time" → ["real", "time"]
    - Numbers:      kept as-is
    - Punctuation:  stripped
    """

    CONTRACTIONS = {
        "don't":"do not","doesn't":"does not","didn't":"did not",
        "won't":"will not","wouldn't":"would not","couldn't":"could not",
        "shouldn't":"should not","isn't":"is not","aren't":"are not",
        "wasn't":"was not","weren't":"were not","haven't":"have not",
        "hasn't":"has not","hadn't":"had not","can't":"cannot",
        "it's":"it is","that's":"that is","what's":"what is",
        "there's":"there is","i'm":"i am","i've":"i have",
        "i'll":"i will","i'd":"i would","you're":"you are",
        "you've":"you have","you'll":"you will","you'd":"you would",
        "he's":"he is","she's":"she is","they're":"they are",
        "we're":"we are","let's":"let us","who's":"who is",
    }

    def tokenize(self, text: str) -> list[str]:
        text = text.lower().strip()
        # Expand contractions
        for c, exp in self.CONTRACTIONS.items():
            text = text.replace(c, exp)
        # Strip possessives
        text = re.sub(r"'s\b", "", text)
        # Replace hyphens with space
        text = text.replace("-", " ")
        # Keep alphanumeric + spaces
        text = re.sub(r"[^\w\s]", " ", text)
        # Collapse whitespace, split
        tokens = re.sub(r"\s+", " ", text).strip().split()
        return tokens

    def detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)


# ════════════════════════════════════════════════════════════════════════════
# 2. STEMMER  (Porter-style suffix stripping)
# ════════════════════════════════════════════════════════════════════════════

class Stemmer:
    """
    Lightweight suffix stemmer. Not as precise as full Porter but fast
    and sufficient for chatbot search.
    """
    SUFFIXES = [
        # longest first so we don't over-strip
        ("ational","ate"),("tional","tion"),("enci","ence"),("anci","ance"),
        ("izer","ize"),("ising","ise"),("izing","ize"),
        ("isation","isation"),("ization","ize"),
        ("ational","ate"),("alism","al"),("aliti","al"),("alist","al"),
        ("fulness","ful"),("ousness","ous"),("iveness","ive"),
        ("ication","icate"),("icate","icat"),
        ("nesses","ness"),("ments","ment"),("ment","ment"),
        ("nesses",""),("ness",""),
        ("alities","al"),("ations","ate"),("ators","ate"),
        ("iving","ive"),("izing","ize"),
        ("ings","ing"),("sings","sing"),
        ("ingly",""),("ingly",""),
        ("ically","ic"),("ical","ic"),
        ("ified","ify"),("ifies","ify"),("ifying","ify"),
        ("ational","ate"),
        ("ations","ate"),("ating","ate"),("ated","ate"),("ates","ate"),
        ("ators","ator"),("ator","ate"),
        ("alism","al"),("alisms","al"),
        ("ists","ist"),
        ("izers","ize"),("ized","ize"),("izes","ize"),
        ("ising","ise"),("ised","ise"),("ises","ise"),
        ("ings",""),("ing",""),
        ("edly","ed"),("edly",""),("ingly",""),
        ("ful",""),("ness",""),("less",""),
        ("ers","er"),("est",""),("ed",""),("er",""),
        ("ies","y"),("ied","y"),
        ("ly",""),("al",""),("ic",""),("ous",""),("ive",""),
        ("es",""),("s",""),
    ]

    _cache: dict[str, str] = {}

    def stem(self, word: str) -> str:
        if word in self._cache:
            return self._cache[word]
        result = self._stem(word)
        self._cache[word] = result
        return result

    def _stem(self, word: str) -> str:
        if len(word) <= 3:
            return word
        for suffix, replacement in self.SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                stemmed = word[:-len(suffix)] + replacement
                if len(stemmed) >= 3:
                    return stemmed
        return word


# ════════════════════════════════════════════════════════════════════════════
# 3. STOP-WORD FILTER
# ════════════════════════════════════════════════════════════════════════════

class StopWordFilter:
    """
    Domain-aware stop words. Crucially keeps negation words
    ("not", "no", "never", "without") — removing them breaks intent.
    """
    STOP_WORDS = {
        "a","an","the","is","are","was","were","be","been","being",
        "this","that","these","those","and","or","but","if","in","on",
        "at","to","for","of","with","about","by","from","up","out",
        "into","through","do","does","did","has","have","had","will",
        "would","could","should","may","might","shall","can",
        "i","you","he","she","it","we","they","me","him","her","us",
        "my","your","his","its","our","their",
        "what","who","where","when","how","which","whom",
        "am","as","at","so","than","then","there","here",
        "just","also","too","very","quite","rather","really",
        "get","got","make","let","put","take","give","go","come",
        "some","any","all","each","every","both","few","more","most",
        "other","such","same","different",
        "please","tell","show","explain","describe","define",
        "say","said","says","know","think","see","want","need",
    }
    # Never remove these — they carry sentiment/logic
    KEEP = {"not","no","never","without","neither","nor","none","nothing"}

    def filter(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t in self.KEEP or t not in self.STOP_WORDS]

    def is_stop(self, word: str) -> bool:
        return word in self.STOP_WORDS and word not in self.KEEP


# ════════════════════════════════════════════════════════════════════════════
# 4. TF-IDF VECTORISER
# ════════════════════════════════════════════════════════════════════════════

class TFIDFVectorizer:
    """
    Builds a TF-IDF index over a corpus (list of strings).
    Supports:
    - Unigrams + bigrams
    - Stemmed tokens
    - Cosine similarity search (returns top-N matches)
    """

    def __init__(self, stemmer: Optional[Stemmer] = None,
                 stop_filter: Optional[StopWordFilter] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 use_bigrams: bool = True):
        self.stemmer     = stemmer     or Stemmer()
        self.stop_filter = stop_filter or StopWordFilter()
        self.tokenizer   = tokenizer   or Tokenizer()
        self.use_bigrams = use_bigrams

        self.vocab:     dict[str, int]         = {}   # term → col idx
        self.idf:       dict[str, float]       = {}   # term → IDF weight
        self.vectors:   list[dict[str, float]] = []   # sparse TF-IDF vectors
        self.docs:      list[str]              = []   # original documents

    def _process(self, text: str) -> list[str]:
        tokens = self.tokenizer.tokenize(text)
        tokens = self.stop_filter.filter(tokens)
        stems  = [self.stemmer.stem(t) for t in tokens]
        if self.use_bigrams:
            bigrams = [f"{stems[i]}_{stems[i+1]}" for i in range(len(stems)-1)]
            return stems + bigrams
        return stems

    def fit(self, documents: list[str]) -> "TFIDFVectorizer":
        """Build IDF weights from a list of documents."""
        self.docs = documents
        N = len(documents)
        if N == 0:
            return self

        # Document frequency
        df: Counter = Counter()
        processed = []
        for doc in documents:
            terms = set(self._process(doc))
            df.update(terms)
            processed.append(list(self._process(doc)))

        # IDF = log((N+1)/(df+1)) + 1  (smoothed)
        all_terms = set(df.keys())
        self.vocab = {t: i for i, t in enumerate(sorted(all_terms))}
        self.idf   = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in all_terms}

        # TF-IDF vectors
        self.vectors = []
        for terms in processed:
            tf_counter: Counter = Counter(terms)
            total = len(terms) or 1
            vec = {}
            for term, cnt in tf_counter.items():
                tf  = cnt / total
                idf = self.idf.get(term, 1.0)
                vec[term] = tf * idf
            # L2 normalise
            norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
            self.vectors.append({t: v/norm for t, v in vec.items()})

        return self

    def _query_vec(self, text: str) -> dict[str, float]:
        """Convert a query string to a normalised TF-IDF vector."""
        terms = self._process(text)
        tf_counter: Counter = Counter(terms)
        total = len(terms) or 1
        vec = {}
        for term, cnt in tf_counter.items():
            tf  = cnt / total
            idf = self.idf.get(term, 0.5)   # 0.5 for OOV terms
            vec[term] = tf * idf
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        return {t: v/norm for t, v in vec.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        # Only iterate over the smaller set
        if len(a) > len(b):
            a, b = b, a
        return sum(a[t] * b[t] for t in a if t in b)

    def search(self, query: str, top_n: int = 3,
               threshold: float = 0.15) -> list[tuple[int, float, str]]:
        """
        Returns list of (doc_index, score, document) sorted by score desc.
        Only includes results with score >= threshold.
        """
        if not self.vectors:
            return []
        q_vec = self._query_vec(query)
        if not q_vec:
            return []
        scored = []
        for i, v in enumerate(self.vectors):
            score = self._cosine(q_vec, v)
            if score >= threshold:
                scored.append((i, score, self.docs[i]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


# ════════════════════════════════════════════════════════════════════════════
# 5. INTENT CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════

class IntentClassifier:
    """
    Intent detection with 3 improvements over the original:

    1. NEW: `followup` intent — catches bare follow-ups like "more",
       "example", "explain it" that previously fell through to "general".

    2. NEW: TF-IDF fallback — when no regex fires AND a TF-IDF index is
       fitted, classify by similarity to labelled intent exemplars.
       Resolves the documented "Falls back to TF-IDF" promise.

    3. NEW: Context-aware reclassification — if confidence is low, check
       the previous intent and promote the current one accordingly.
       (e.g. prior=definition + current=example → keep "example" with
       boosted confidence instead of falling to "general".)
    """

    INTENTS: dict[str, list[str]] = {
        "greeting":    ["hi","hello","hey","hola","greetings","good morning",
                         "good afternoon","good evening","sup","howdy","morning",
                         "evening","afternoon",
                         "how are you","how r you","how are u","hows it going",
                         "how's it going","how do you do","nice to meet you",
                         "pleased to meet you","great to meet you","good to meet you"],
        "farewell":    ["bye","goodbye","see you","later","take care","farewell",
                         "good night","cya","ttyl","night"],
        "thanks":      ["thanks","thank you","ty","thx","appreciate","cheers",
                         "thank","grateful","many thanks"],
        "identity":    ["who are you","what are you","your name","who made you",
                         "who created you","are you ai","are you a bot",
                         "are you human","introduce yourself","about yourself"],
        "capability":  ["what can you do","how can you help","what do you know",
                         "your abilities","what are you capable","help me with",
                         "can you","features","what topics"],
        "math":        ["calculate","solve","what is [0-9]",
                         r"[0-9]+\s*[+\-*/^%]\s*[0-9]",
                         "factorial","square root","sqrt","percentage",
                         "derivative","integral","divided by","times","plus","minus",
                         "equals","arithmetic","algebra","equation","formula"],
        "definition":  ["what is","define","meaning of","what does.*mean",
                         "definition of","explain what","describe what",
                         "what are","what do you mean by","clarify"],
        "explanation": ["explain","how does","how do","why does","why do",
                         "tell me about","elaborate","describe how",
                         "break it down","break down","walk me through",
                         "in simple terms","simply","simply put","basically"],
        "comparison":  ["difference between","compare","vs","versus",
                         "which is better","what is the difference",
                         "pros and cons","advantages","disadvantages",
                         "better than","worse than","similar to"],
        "example":     ["give me an example","example of","show example",
                         "for example","such as","like what","instance of",
                         "show me","demonstrate","use case","use it in"],
        "code":        ["write code","python code","javascript","function",
                         "algorithm","program","script","implement","debug",
                         "code for","snippet","syntax","in python","in java",
                         "in c++","css","html","sql","regex"],
        "factual":     ["who invented","who discovered","when was","where is",
                         "what year","which country","capital of","largest","smallest",
                         "how many","how much","how old","distance","population",
                         "who was","what happened","history of","origin of"],
        "opinion":     ["what do you think","your opinion","do you like",
                         "best way","recommend","suggest","should i",
                         "is it worth","worth it","good or bad","your thoughts",
                         "what would you"],
        "howto":       ["how to","how can i","steps to","way to","guide",
                         "tutorial","instructions","process of","procedure",
                         "method","approach","technique","strategy"],
        "summary":     ["summarise","summarize","brief","tl;dr","key points",
                         "main points","overview","recap","in short",
                         "sum up","short version","conclusion","takeaway"],
        "weather":     ["weather","temperature","forecast","rain","sunny","climate",
                         "humidity","snow","wind","storm","hot","cold","warm"],
        "joke":        ["joke","funny","laugh","humor","humour","make me laugh",
                         "tell me a joke","pun","riddle","funny story"],
        "search":      ["search for","look up","find","google","web search",
                         "latest news","current","today","recent","news about",
                         "search","browse"],
        # ── NEW: list intent ──────────────────────────────────────────────────
        "list":        ["list all","list the","give me a list","list of",
                         "name all","name the","enumerate","what are all",
                         "all types of","all kinds of","list some","list examples",
                         "give a list","types of","kinds of","varieties of"],
        # ── NEW: translate intent ─────────────────────────────────────────────
        "translate":   ["translate","translation of","how do you say",
                         "in spanish","in french","in german","in japanese",
                         "in chinese","in arabic","in hindi","in portuguese",
                         "how to say","say it in","word for"],
        # ── NEW: follow-up intent ─────────────────────────────────────────────
        # Catches bare one-or-two-word follow-ups that reference prior context
        "followup":    ["more","more details","tell me more","explain more",
                         "go on","continue","elaborate","and then","what else",
                         "keep going","next","another","again","repeat",
                         "with example","show example","give example",
                         "in detail","detailed","simpler","simplify",
                         "in code","code for it","summarise that","summarize that",
                         "difference","compare them","pros and cons",
                         "example please","one more","few more"],
    }

    # ── Intent exemplar sentences for TF-IDF fallback ────────────────────────
    # One representative sentence per intent; used when no regex fires.
    _EXEMPLARS: dict[str, str] = {
        "greeting":    "hi hello hey good morning greetings howdy how are you nice to meet you pleased to meet you",
        "farewell":    "bye goodbye see you later take care farewell",
        "thanks":      "thanks thank you appreciate that grateful",
        "identity":    "who are you what are you your name who made you",
        "capability":  "what can you do how can you help your abilities",
        "math":        "calculate solve arithmetic equation algebra formula",
        "definition":  "what is define meaning definition of what does mean",
        "explanation": "explain how does why does tell me about elaborate",
        "comparison":  "difference between compare vs versus which is better pros cons",
        "example":     "give me an example show example instance of demonstrate",
        "code":        "write code function algorithm implement script python",
        "factual":     "who invented when was where is what year history origin",
        "opinion":     "what do you think recommend suggest opinion should i",
        "howto":       "how to steps tutorial instructions guide procedure",
        "summary":     "summarise key points overview recap in short tl dr",
        "weather":     "weather temperature forecast rain sunny climate humidity",
        "joke":        "joke funny laugh humor tell me a joke pun riddle",
        "search":      "search latest news current today look up find google",
        "followup":    "more details tell me more continue elaborate go on",
        "list":        "list all name enumerate give me a list types of kinds",
        "translate":   "translate translation how do you say in spanish french",
        "general":     "question answer information topic general",
    }

    # ── Intent compatibility for context boosting ─────────────────────────────
    # If prior_intent → current matches this table, boost confidence.
    _CONTEXT_BOOST: dict[str, set] = {
        "definition":  {"example", "explanation", "howto", "comparison", "code",
                         "list", "followup"},
        "explanation": {"example", "comparison", "howto", "code", "summary",
                         "list", "followup"},
        "factual":     {"comparison", "explanation", "list", "followup"},
        "howto":       {"code", "example", "comparison", "list", "followup"},
        "code":        {"explanation", "example", "comparison", "followup"},
        "comparison":  {"example", "definition", "list", "followup"},
        "example":     {"code", "howto", "list", "followup"},
        "list":        {"example", "comparison", "followup"},
        "translate":   {"example", "followup"},
    }

    def __init__(self):
        # Pre-compile regex patterns for each intent.
        # Single word  → \b word boundary for precision
        # Multi-word   → re.escape only (\b breaks on spaces)
        # Raw regex    → compile as-is
        self._patterns: dict[str, list[re.Pattern]] = {}
        for intent, phrases in self.INTENTS.items():
            compiled = []
            for p in phrases:
                is_regex = any(c in p for c in "[]().*+?\\^${}|")
                if is_regex:
                    compiled.append(re.compile(p, re.I))
                elif " " in p:
                    compiled.append(re.compile(re.escape(p), re.I))
                else:
                    compiled.append(re.compile(r"\b" + re.escape(p) + r"\b", re.I))
            self._patterns[intent] = compiled

        # TF-IDF vectoriser fitted on exemplar sentences (fallback)
        self._tfidf_fitted    = False
        self._exemplar_vecs:  list[tuple[str, dict]] = []  # [(intent, vec)]
        self._fit_tfidf_fallback()

    def _fit_tfidf_fallback(self):
        """
        Build a tiny TF-IDF index over the intent exemplar sentences.
        Used when no regex pattern fires.
        """
        try:
            # Simple TF bag-of-words (no stemming needed for exemplars)
            from math import log
            docs  = list(self._EXEMPLARS.items())   # [(intent, text)]
            N     = len(docs)
            df:   dict[str, int]  = defaultdict(int)
            tfs:  list[dict]      = []

            for _, text in docs:
                tokens = set(text.lower().split())
                for t in tokens:
                    df[t] += 1
                tfs.append(Counter(text.lower().split()))

            idf = {t: log((N + 1) / (df[t] + 1)) + 1 for t in df}

            for (intent, _), tf in zip(docs, tfs):
                total = sum(tf.values()) or 1
                vec   = {}
                for t, cnt in tf.items():
                    vec[t] = (cnt / total) * idf.get(t, 1.0)
                norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
                self._exemplar_vecs.append((intent, {t: v/norm for t, v in vec.items()}))

            self._idf = idf
            self._tfidf_fitted = True
        except Exception:
            self._tfidf_fitted = False

    def _tfidf_classify(self, text: str) -> tuple[str, float]:
        """Classify via cosine similarity to intent exemplars."""
        if not self._tfidf_fitted:
            return "general", 0.25

        tokens = text.lower().split()
        total  = len(tokens) or 1
        tf     = Counter(tokens)
        vec    = {}
        for t, cnt in tf.items():
            vec[t] = (cnt / total) * self._idf.get(t, 0.3)
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        qvec = {t: v/norm for t, v in vec.items()}

        best_intent, best_score = "general", 0.0
        for intent, evec in self._exemplar_vecs:
            score = sum(qvec[t] * evec.get(t, 0.0) for t in qvec)
            if score > best_score:
                best_score, best_intent = score, intent

        # Only trust TF-IDF if similarity is meaningful
        if best_score < 0.08:
            return "general", 0.25
        confidence = min(0.75, best_score * 3.0 + 0.2)
        return best_intent, round(confidence, 2)

    # Bare words that are ALWAYS follow-ups regardless of regex hits
    _BARE_FOLLOWUPS: frozenset = frozenset({
        "more","explain","elaborate","continue","go on","next","again",
        "repeat","detailed","simpler","simplify","another","summarise",
        "summarize","difference","compare","example","examples","code",
        "explain it","explain more","tell me more","more details",
        "in detail","in code","in python","in java","in c++","in javascript",
        "keep going","and then","what else","one more","few more",
        "show me","show example","with example","give example",
        "summarize that","summarise that","explain that","compare them",
        "show that","prove it","give more","say more","go deeper",
        "expand on that","expand","break it down","simplify that",
        "give me steps","give me more","translate that","list them",
        "list more","list all of them","name them","enumerate them",
        "any others","what else is there","more examples","pros and cons of that",
    })

    def classify(self, text: str,
                 prior_intent: str = "") -> tuple[str, float]:
        """
        Returns (intent_label, confidence).
        confidence ∈ [0, 1].

        prior_intent: label from the previous turn (optional).
                      Used for context-aware boosting.
        """
        text_lower      = text.lower().strip()
        text_bare       = text_lower.rstrip("?!.")   # used only for followup check
        word_count      = len(text_bare.split())

        # ── Bare follow-up shortcut ───────────────────────────────────────
        # Short phrases that are unambiguous follow-ups when prior is set.
        # Check BEFORE regex so "explain it" doesn't fire "explanation".
        if (prior_intent
                and prior_intent not in ("greeting","farewell","thanks","general")
                and (text_bare in self._BARE_FOLLOWUPS
                     or (word_count <= 2 and text_bare in self._BARE_FOLLOWUPS))):
            return "followup", 0.82

        # ── Negation guard ────────────────────────────────────────────────
        # Detect negation words immediately before a trigger to avoid
        # misclassifying "I do NOT want an example" as intent=example.
        _NEGATION_WORDS = {"not","never","don't","dont","doesn't","doesnt",
                            "isn't","isnt","no","without","avoid","stop",
                            "won't","wont","can't","cant","neither","nor"}
        _neg_tokens = set(text_bare.split())
        _has_negation = bool(_neg_tokens & _NEGATION_WORDS)

        scores: dict[str, int] = defaultdict(int)
        negated: set[str]      = set()   # intents whose trigger was negated

        for intent, patterns in self._patterns.items():
            for pat in patterns:
                m = pat.search(text_lower)
                if m:
                    # Check if negation word appears within 3 tokens before match
                    match_start = m.start()
                    prefix      = text_lower[:match_start].split()[-3:]
                    if _has_negation and set(prefix) & _NEGATION_WORDS:
                        negated.add(intent)
                    else:
                        scores[intent] += 1

        # ── Regex matched at least one intent ─────────────────────────────
        if scores:
            best_score  = max(scores.values())
            _SPECIFICITY = {
                "math":10,"factual":9,"code":9,"weather":9,"joke":9,
                "translate":9,"list":8,
                "comparison":8,"example":8,"howto":8,"summary":8,
                "explanation":7,"definition":7,"followup":7,
                "opinion":6,"search":6,"capability":5,
                "identity":5,"thanks":4,"farewell":4,"greeting":4,"general":1
            }
            candidates  = [k for k, v in scores.items() if v == best_score]
            best_intent = max(candidates, key=lambda k: _SPECIFICITY.get(k, 0))
            total_hits  = sum(scores.values())

            # Multi-pattern bonus: hitting the same intent twice = high confidence
            multi_bonus = 0.1 if scores[best_intent] >= 2 else 0.0
            confidence  = min(1.0,
                scores[best_intent] / max(total_hits, 1) * 0.8 + 0.2 + multi_bonus)

            # Circular intent guard: prior == current on a content intent
            # AND the query is short (≤ 4 words) → almost certainly a follow-up
            _CONTENT_INTENTS = {
                "definition","explanation","comparison","example","howto",
                "code","factual","summary","list","translate","opinion"
            }
            if (prior_intent
                    and prior_intent == best_intent
                    and prior_intent in _CONTENT_INTENTS
                    and word_count <= 4):
                return "followup", 0.78

            # how-many / how-much / how-long etc. → factual, not howto
            _HOW_QUANTITY = re.compile(
                r"\bhow\s+(many|much|long|far|old|often|soon|tall|deep|wide|"
                r"big|small|large|heavy|fast|slow|hot|cold|high|low)\b", re.I)
            if best_intent == "howto" and _HOW_QUANTITY.search(text_lower):
                best_intent = "factual"
                confidence  = min(1.0, confidence + 0.05)

            # Context boost
            if prior_intent and confidence < 0.75:
                compatible = self._CONTEXT_BOOST.get(prior_intent, set())
                if best_intent in compatible:
                    confidence = min(1.0, confidence + 0.2)

            return best_intent, round(confidence, 2)

        # ── No regex matched (possibly negated or unknown) ─────────────────
        # If the only hits were negated, signal general intent
        if negated and not scores:
            return "general", 0.45

        # ── TF-IDF fallback ────────────────────────────────────────────────
        tfidf_intent, tfidf_conf = self._tfidf_classify(text_lower)

        # Suppress TF-IDF result when query contains negation — the model
        # can't tell "i want an example" from "i do NOT want an example"
        if _has_negation and tfidf_intent not in (
                "greeting","farewell","thanks","general","search","math"):
            return "general", 0.40

        # Context boost on TF-IDF result too
        if prior_intent and tfidf_conf < 0.6:
            compatible = self._CONTEXT_BOOST.get(prior_intent, set())
            if tfidf_intent in compatible:
                tfidf_conf = min(0.75, tfidf_conf + 0.2)

        # Short bare follow-ups (≤ 3 words, no regex) almost always mean
        # the user wants more on the previous topic
        if word_count <= 3 and prior_intent and prior_intent not in (
                "greeting", "farewell", "thanks", "general"):
            if tfidf_intent in ("general", "followup") or tfidf_conf < 0.4:
                return "followup", 0.72

        return tfidf_intent, tfidf_conf

    def is_question(self, text: str) -> bool:
        """True if the text is likely a question."""
        question_words = {"what","who","where","when","why","how","which","whose","whom"}
        first_word = text.strip().lower().split()[0] if text.strip() else ""
        return text.strip().endswith("?") or first_word in question_words


# ════════════════════════════════════════════════════════════════════════════
# 6. NAMED ENTITY RECOGNISER
# ════════════════════════════════════════════════════════════════════════════

class NamedEntityRecognizer:
    """
    Lightweight rule-based NER. Extracts:
    - PERSON   (known scientists, historical figures)
    - PLACE    (countries, capitals, planets, continents)
    - TECH     (programming languages, frameworks, tools)
    - SCIENCE  (elements, concepts, formulas)
    - NUMBER   (integers, floats, percentages)
    """

    PERSONS = {
        "einstein","newton","darwin","hawking","curie","tesla","turing",
        "shakespeare","aristotle","plato","socrates","galileo","copernicus",
        "darwin","pasteur","faraday","maxwell","bohr","heisenberg","feynman",
        "von neumann","lovelace","babbage","dijkstra","knuth","ritchie",
        "torvalds","gates","jobs","wozniak","zuckerberg","bezos","musk",
        "washington","lincoln","gandhi","churchill","napoleon","caesar",
        "cleopatra","alexander","columbus","da vinci","michelangelo",
    }

    PLACES = {
        "india","china","usa","uk","france","germany","japan","russia",
        "brazil","canada","australia","africa","europe","asia","america",
        "antarctica","arctic","amazon","nile","everest","sahara",
        "london","paris","tokyo","beijing","moscow","delhi","mumbai",
        "new york","los angeles","sydney","dubai","singapore","toronto",
        "pacific","atlantic","indian ocean","mediterranean",
        "himalayas","alps","andes","rockies",
        "mercury","venus","earth","mars","jupiter","saturn","uranus","neptune",
        "moon","sun","milky way","andromeda",
    }

    TECH = {
        "python","javascript","java","c++","c","rust","go","swift","kotlin",
        "typescript","ruby","php","scala","r","matlab","julia","haskell",
        "html","css","sql","bash","powershell","assembly",
        "pytorch","tensorflow","keras","scikit","numpy","pandas","matplotlib",
        "react","vue","angular","node","django","flask","fastapi","spring",
        "docker","kubernetes","git","linux","windows","macos","android","ios",
        "gpt","transformer","bert","lstm","cnn","rnn","gan","llm",
        "api","rest","graphql","grpc","json","xml","yaml",
        "cpu","gpu","tpu","ram","ssd","fpga","blockchain","bitcoin",
    }

    SCIENCE = {
        "hydrogen","helium","oxygen","carbon","nitrogen","iron","gold","silver",
        "uranium","plutonium","calcium","sodium","chlorine","silicon","copper",
        "dna","rna","protein","enzyme","cell","neuron","atom","electron",
        "proton","neutron","quark","photon","boson","fermion",
        "gravity","relativity","quantum","thermodynamics","entropy",
        "evolution","genetics","photosynthesis","osmosis","diffusion",
        "velocity","acceleration","momentum","energy","force","mass",
        "wavelength","frequency","amplitude","voltage","current","resistance",
        "ph","acid","base","catalyst","reaction","compound","molecule",
    }

    def extract(self, text: str) -> dict[str, list[str]]:
        """Extract named entities from text. Returns dict of entity type → list."""
        text_lower = text.lower()
        words  = set(re.findall(r"\b\w+\b", text_lower))
        bigrams = set()
        word_list = text_lower.split()
        for i in range(len(word_list) - 1):
            bigrams.add(f"{word_list[i]} {word_list[i+1]}")

        entities: dict[str, list[str]] = defaultdict(list)

        for e in self.PERSONS:
            if e in words or e in bigrams:
                entities["PERSON"].append(e)

        for e in self.PLACES:
            if e in words or e in bigrams:
                entities["PLACE"].append(e)

        for e in self.TECH:
            if e in words:
                entities["TECH"].append(e)

        for e in self.SCIENCE:
            if e in words:
                entities["SCIENCE"].append(e)

        # NUMBER: integers, floats, percentages
        numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
        if numbers:
            entities["NUMBER"] = numbers

        return dict(entities)


# ════════════════════════════════════════════════════════════════════════════
# 7. QUERY EXPANDER
# ════════════════════════════════════════════════════════════════════════════

class QueryExpander:
    """
    Adds synonyms to a query to improve recall.
    Uses a compact hand-curated synonym map.
    """

    SYNONYMS: dict[str, list[str]] = {
        # Science
        "speed":      ["velocity","rate"],
        "force":      ["power","strength","push"],
        "energy":     ["power","work","force"],
        "planet":     ["world","celestial body"],
        "star":       ["sun","stellar"],
        "atom":       ["particle","element"],
        "cell":       ["unit","organism"],
        "gene":       ["dna","chromosome"],
        "disease":    ["illness","sickness","disorder","condition"],
        "medicine":   ["drug","treatment","therapy","medication"],
        # Tech
        "program":    ["code","script","software","application"],
        "function":   ["method","procedure","subroutine","def"],
        "bug":        ["error","defect","fault","issue"],
        "database":   ["db","datastore","storage"],
        "learn":      ["train","study","teach"],
        "network":    ["internet","web","net"],
        "compute":    ["calculate","process","run"],
        # General
        "big":        ["large","huge","enormous","giant"],
        "small":      ["tiny","little","micro","mini"],
        "fast":       ["quick","rapid","swift","speedy"],
        "smart":      ["intelligent","clever","brilliant"],
        "important":  ["significant","crucial","key","major","critical"],
        "discover":   ["find","invent","create","develop"],
        "explain":    ["describe","elaborate","clarify","define"],
        "help":       ["assist","support","aid"],
        "problem":    ["issue","challenge","difficulty","question"],
        "answer":     ["solution","response","result","reply"],
        "use":        ["apply","employ","utilise","implement"],
        "make":       ["create","build","produce","generate","construct"],
        "show":       ["display","demonstrate","present","reveal"],
        "change":     ["modify","alter","update","transform","edit"],
        "remove":     ["delete","erase","drop","clear"],
        "connect":    ["link","join","associate","integrate"],
    }

    def __init__(self, stemmer: Optional[Stemmer] = None):
        self.stemmer = stemmer or Stemmer()
        # Build stemmed lookup for faster matching
        self._stemmed: dict[str, list[str]] = {
            self.stemmer.stem(k): v for k, v in self.SYNONYMS.items()
        }

    def expand(self, tokens: list[str], max_synonyms: int = 2) -> list[str]:
        """Add synonyms for content words. Returns expanded token list."""
        extra: list[str] = []
        for t in tokens:
            s = self.stemmer.stem(t)
            syns = self._stemmed.get(s, self.SYNONYMS.get(t, []))
            extra.extend(syns[:max_synonyms])
        return tokens + extra


# ════════════════════════════════════════════════════════════════════════════
# 8. HEMA NLP FACADE
# ════════════════════════════════════════════════════════════════════════════

class HemaNLP:
    """
    Top-level NLP facade used by app.py.
    Replaces: fuzzy_search, normalize, spell_correct, intent detection.

    Usage:
        nlp = HemaNLP()
        nlp.fit(dataset_qa)         # build TF-IDF index

        intent, conf = nlp.classify_intent(text)
        results = nlp.search(query, top_n=3)
        entities = nlp.extract_entities(text)
        corrected = nlp.spell_correct(text)
        normalised = nlp.normalize(text)
    """

    def __init__(self):
        self.tokenizer  = Tokenizer()
        self.stemmer    = Stemmer()
        self.stopwords  = StopWordFilter()
        self.intent     = IntentClassifier()
        self.ner        = NamedEntityRecognizer()
        self.expander   = QueryExpander(self.stemmer)
        self.tfidf      = TFIDFVectorizer(self.stemmer, self.stopwords, self.tokenizer)

        # Maps TF-IDF doc index → (answer, example)
        self._index_to_qa: dict[int, tuple[str, str]] = {}
        # Spell vocab (built from dataset)
        self._spell_vocab: set[str] = set()
        self._ready = False
        # Session intent context — auto-updated by classify_intent()
        self._prior_intent: str = ""

    # ── Build index ──────────────────────────────────────────────────────────
    def fit(self, dataset_qa: dict[str, tuple[str, str]]) -> "HemaNLP":
        """
        Build TF-IDF index from dataset_qa dict.
        dataset_qa: { question_str: (answer_str, example_str) }
        """
        questions = list(dataset_qa.keys())
        answers   = list(dataset_qa.values())

        self.tfidf.fit(questions)

        for i, (q, (a, e)) in enumerate(zip(questions, answers)):
            self._index_to_qa[i] = (a, e)
            # Build spell vocab from all text
            self._spell_vocab.update(self.tokenizer.tokenize(q))
            self._spell_vocab.update(self.tokenizer.tokenize(a))

        self._spell_vocab = {w for w in self._spell_vocab
                             if len(w) >= 2 and w.isalpha()}
        self._ready = True
        print(f"[HemaNLP] TF-IDF index: {len(questions)} docs | "
              f"vocab {len(self.tfidf.vocab)} terms | "
              f"spell vocab {len(self._spell_vocab)} words")
        return self

    # ── Semantic search ──────────────────────────────────────────────────────
    def search(self, query: str, top_n: int = 3,
               threshold: float = 0.12) -> Optional[tuple[str, str]]:
        """
        Search dataset with TF-IDF + query expansion.
        Returns (answer, example) of best match, or None.
        """
        if not self._ready:
            return None

        # Expand query with synonyms
        tokens  = self.tokenizer.tokenize(query)
        tokens  = self.expander.expand(tokens)
        q_exp   = " ".join(tokens)

        results = self.tfidf.search(q_exp, top_n=top_n, threshold=threshold)
        if not results:
            # Try without expansion (some expansions hurt precision)
            results = self.tfidf.search(query, top_n=top_n, threshold=threshold)
        if not results:
            return None

        best_idx, best_score, _ = results[0]
        return self._index_to_qa.get(best_idx)

    def search_many(self, query: str, top_n: int = 5,
                    threshold: float = 0.10) -> list[tuple[str, str, float]]:
        """Returns top N (answer, example, score) tuples."""
        if not self._ready:
            return []
        results = self.tfidf.search(query, top_n=top_n, threshold=threshold)
        out = []
        for idx, score, _ in results:
            a, e = self._index_to_qa.get(idx, ("", ""))
            out.append((a, e, score))
        return out

    # ── Intent classification ─────────────────────────────────────────────────
    def classify_intent(self, text: str,
                        prior_intent: str = "") -> tuple[str, float]:
        """
        Classify intent with automatic prior-intent tracking.
        If prior_intent is not supplied, HemaNLP uses its own session state.
        Callers that pass prior_intent explicitly override the internal state.
        """
        effective_prior = prior_intent or self._prior_intent
        label, conf     = self.intent.classify(text, prior_intent=effective_prior)
        # Update internal state (skip volatile intents so they don't pollute context)
        if label not in ("greeting", "farewell", "thanks", "general", "followup"):
            self._prior_intent = label
        return label, conf

    def reset_intent_context(self):
        """Reset session intent state (call on conversation clear)."""
        self._prior_intent = ""

    def is_question(self, text: str) -> bool:
        return self.intent.is_question(text)

    # ── Named entity extraction ───────────────────────────────────────────────
    def extract_entities(self, text: str) -> dict[str, list[str]]:
        return self.ner.extract(text)

    # ── Normalisation ─────────────────────────────────────────────────────────
    def normalize(self, text: str) -> str:
        """
        Full normalisation pipeline:
        1. Lowercase + strip
        2. Expand contractions (via tokenizer)
        3. Stem query for canonical form
        4. Apply manual replacements (legacy compat)
        """
        tokens = self.tokenizer.tokenize(text)
        # Semantic replacements (keeps compatibility with existing code)
        REPLACEMENTS = {
            "discover": "invent",
            "discovered": "invented",
            "discovered by": "invented by",
        }
        replaced = []
        for t in tokens:
            replaced.append(REPLACEMENTS.get(t, t))
        return " ".join(replaced)

    # ── Spell correction ──────────────────────────────────────────────────────
    # Words that must never be altered — acronyms, tech terms, proper names
    _SPELL_NEVER_CORRECT: set = {
        "nlp","cnn","rnn","gru","gan","llm","bert","gpt","rl","ai","ml","dl",
        "api","url","css","sql","git","pip","npm","js","py","gpu","cpu","ram",
        "ide","cli","gui","oop","tcp","udp","http","https","dns","ssh","ftp",
        "json","xml","yaml","html","csv","pdf","db","os","vm","io","ui","ux",
        "amd","arm","tpu","fpga","ssd","hdd","lan","wan","vpn","cdn","jwt",
        "rsa","aes","sha","md5","utf","rgb","hex","ocr","tts","stt",
        "python","java","javascript","typescript","kotlin","swift","golang","rust",
        "django","flask","fastapi","react","angular","vue","node","express",
        "pytorch","tensorflow","keras","numpy","pandas","matplotlib","scipy",
        "sklearn","scikit","transformer","encoder","decoder","tokenizer",
        "embedding","softmax","sigmoid","relu","gelu","dropout","backprop",
        "gradient","perceptron","neuron","algorithm","recursion","iteration",
        "polymorphism","encapsulation","abstraction","inheritance",
        "boolean","integer","tuple","vector","matrix","tensor",
        "async","await","callback","closure","lambda","decorator","compiler",
        "kubernetes","docker","blockchain","bitcoin","ethereum","cryptography",
        "regex","linux","ubuntu","macos","android","faiss","sqlite","mongodb",
        "postgresql","mysql","redis","hema","tharun","sai",
        "chat","okay","ok","sure","yeah","hmm","nah","yep","nope","yup",
    }

    def spell_correct(self, text: str) -> tuple[str, bool, str]:
        """
        Spell-correct using dataset vocab as reference.
        Returns (corrected_text, was_changed, original).
        Uses edit-distance 1 candidates — faster and more accurate than difflib.
        """
        original = text
        words    = self.tokenizer.tokenize(text)
        corrected = []
        changed   = False

        for w in words:
            if len(w) < 3 or not w.isalpha():
                corrected.append(w)
                continue
            # Never correct whitelisted terms (tech vocab, proper names, etc.)
            if w in self._SPELL_NEVER_CORRECT:
                corrected.append(w)
                continue
            if w in self._spell_vocab:
                corrected.append(w)
                continue
            # Generate edit-distance-1 candidates
            candidates = self._edits1(w)
            valid = [c for c in candidates if c in self._spell_vocab]
            if valid:
                # Pick by longest common prefix (most similar)
                best = max(valid, key=lambda c: self._lcp(c, w))
                corrected.append(best)
                if best != w:
                    changed = True
            else:
                corrected.append(w)

        return " ".join(corrected), changed, original

    @staticmethod
    def _edits1(word: str) -> set[str]:
        """All strings 1 edit from word (delete, transpose, replace, insert)."""
        letters  = "abcdefghijklmnopqrstuvwxyz"
        splits   = [(word[:i], word[i:]) for i in range(len(word)+1)]
        deletes  = [L+R[1:]        for L, R in splits if R]
        transposes=[L+R[1]+R[0]+R[2:] for L, R in splits if len(R)>1]
        replaces = [L+c+R[1:]     for L, R in splits if R for c in letters]
        inserts  = [L+c+R         for L, R in splits        for c in letters]
        return set(deletes + transposes + replaces + inserts)

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Length of longest common prefix."""
        n = 0
        for x, y in zip(a, b):
            if x == y: n += 1
            else: break
        return n

    # ── Text summary helper ───────────────────────────────────────────────────
    def keyword_score(self, text: str) -> float:
        """Score a sentence by information density (used in key-point extraction)."""
        tokens = self.tokenizer.tokenize(text)
        content = self.stopwords.filter(tokens)
        if not tokens:
            return 0.0
        unique_stems = {self.stemmer.stem(t) for t in content}
        # More unique content stems = higher score
        return len(unique_stems) / math.sqrt(len(tokens) + 1)


# ── Singleton ────────────────────────────────────────────────────────────────
_nlp_instance: Optional[HemaNLP] = None

def get_nlp() -> HemaNLP:
    global _nlp_instance
    if _nlp_instance is None:
        _nlp_instance = HemaNLP()
    return _nlp_instance


# ════════════════════════════════════════════════════════════════════════════
# 9. REASONING ENGINE
# ════════════════════════════════════════════════════════════════════════════

class ReasoningEngine:
    """
    Lightweight reasoning layer that wraps IntentClassifier with a simpler
    `.think(text) -> intent_label` interface.

    Used by app.py to pre-classify queries before the main pipeline,
    enabling fast-path returns for high-confidence intents.

    Returns one of:
        "greeting", "farewell", "thanks", "math", "joke",
        "code", "definition", "explanation", "factual",
        "comparison", "example", "howto", "summary",
        "search", "opinion", "identity", "capability",
        "list", "translate", "followup", "general"
    """

    def __init__(self):
        self._clf = IntentClassifier()
        self._threshold    = 0.6
        self._prior_intent = ""   # updated each call for context tracking

    def think(self, text: str) -> str:
        """
        Classify text into an intent label, passing prior intent as context.
        Returns 'general' when confidence is below threshold.
        """
        intent, confidence = self._clf.classify(text,
                                                 prior_intent=self._prior_intent)
        # Update prior for next call
        if intent != "general":
            self._prior_intent = intent
        if confidence >= self._threshold:
            return intent
        return "general"

    def is_conversational(self, text: str) -> bool:
        """True for greetings, farewells, thanks — no search needed."""
        return self.think(text) in ("greeting", "farewell", "thanks")

    def is_followup(self, text: str) -> bool:
        """True when the query is a short follow-up to the previous turn."""
        return self.think(text) == "followup"

    def needs_search(self, text: str) -> bool:
        """True for intents that benefit from web search."""
        return self.think(text) in (
            "factual", "search", "definition", "explanation", "factual", "howto"
        )

    def reset_context(self):
        """Call this when the conversation is cleared."""
        self._prior_intent = ""