import express, { Request, Response } from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: "25mb" }));

// Serve static assets from src/static at /static
app.use("/static", express.static(path.join(__dirname, "src", "static")));

// ============================================================================
// 1. IN-MEMORY KNOWLEDGE BASE & DATASET (Ported from Python dataset & vector_answers)
// ============================================================================

const INITIAL_QA: Record<string, string> = {
  "what is a noun": "A noun is a word that names a person, place, thing, or idea.",
  "what is a verb": "A verb is a word that expresses an action, occurrence, or state of being.",
  "what is an adjective": "An adjective is a word that describes or modifies a noun or pronoun.",
  "what is an adverb": "An adverb is a word that modifies a verb, an adjective, or another adverb, often expressing manner, place, time, or degree.",
  "what is a sentence": "A sentence is a set of words that is complete in itself, typically containing a subject and predicate, conveying a statement, question, exclamation, or command.",
  "what is grammar": "Grammar is the whole system and structure of a language or of languages in general, usually taken as consisting of syntax and morphology.",
  "what does happy mean": "Happy means feeling or showing pleasure, contentment, or joy.",
  "what does intelligent mean": "Intelligent means having or showing intelligence, especially of a high level; able to learn, understand, and think logically.",
  "what does curious mean": "Curious means eager to know or learn something.",
  "what does confident mean": "Confident means feeling or showing certainty about something or feeling sure about yourself and your abilities.",
  "what does brave mean": "Brave means ready to face and endure danger or pain; showing courage.",
  "who discovered gravity": "Sir Isaac Newton formulated the universal law of gravitation after observing an apple fall from a tree in the late 17th century.",
  "who invented the telephone": "Alexander Graham Bell was awarded the first U.S. patent for the invention of the telephone in 1876.",
  "who invented the light bulb": "Thomas Edison developed the first practical incandescent light bulb in 1879, building upon earlier work by Humphry Davy and Joseph Swan.",
  "what is the capital of france": "Paris is the capital and most populous city of France, situated on the Seine River in the north-central part of the country.",
  "what planet is known as the red planet": "Mars is known as the Red Planet because iron oxide (rust) on its surface gives it a reddish appearance.",
  "what does lol mean": "LOL stands for 'Laughing Out Loud', a common internet slang acronym used to indicate amusement.",
  "what does brb mean": "BRB stands for 'Be Right Back', used in online chat to indicate a brief absence.",
  "what does btw mean": "BTW stands for 'By The Way', used to introduce a side note or additional comment.",
  "what does idk mean": "IDK stands for 'I Don't Know'.",
  "quantum computing basics": "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics (superposition, entanglement, and quantum interference) to solve problems too complex for classical computers. While classical computers use bits (0 or 1), quantum computers use qubits, which can exist in a linear combination of both 0 and 1 simultaneously.",
  "explain general relativity": "General relativity is Albert Einstein's geometric theory of gravitation, published in 1915. It generalizes special relativity and refines Newton's law of universal gravitation, providing a unified description of gravity as a geometric property of space and time, or four-dimensional spacetime. Massive objects like stars and planets curve spacetime around them, and this curvature dictates the path that other objects and light follow.",
  "what is machine learning": "Machine learning (ML) is a branch of artificial intelligence and computer science that focuses on using data and algorithms to enable computers to learn patterns and make decisions or predictions without being explicitly programmed for every rule. Common paradigms include supervised learning, unsupervised learning, and reinforcement learning.",
  "explain recursion": "Recursion in computer science is a method of solving a computational problem where the solution depends on solutions to smaller instances of the same problem. A recursive function calls itself with modified arguments until it reaches a termination condition known as the base case.\n\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```",
  "what is python": "Python is a high-level, general-purpose, interpreted programming language created by Guido van Rossum. Its design philosophy emphasizes code readability with the use of significant indentation, and it supports multiple programming paradigms, including structured, object-oriented, and functional programming.",
  "what is javascript": "JavaScript (JS) is a lightweight, interpreted, or just-in-time compiled programming language with first-class functions. Alongside HTML and CSS, JavaScript is one of the core technologies of the World Wide Web, enabling interactive web pages and full-stack server applications via Node.js.",
  "first day of the week": "According to international standard ISO 8601, Monday is the first day of the week. It is followed by Tuesday, Wednesday, Thursday, Friday, Saturday, and Sunday.",
  "planets in the solar system": "Our solar system has eight planets in order of distance from the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. There are also five officially recognized dwarf planets including Pluto, Eris, Haumea, Makemake, and Ceres.",
  "first president of the united states": "George Washington was a Founding Father and the first president of the United States, serving from 1789 to 1797.",
  "chemical symbol for gold": "Gold is a chemical element with the chemical symbol Au (from Latin: aurum) and atomic number 79. In its pure form, it is a bright, slightly orange-yellow, dense, soft, malleable, and ductile transition metal.",
  "largest ocean on earth": "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions, covering approximately 165.25 million square kilometers (63.8 million square miles) — about 46% of Earth's water surface.",
  "who is the ceo of microsoft": "Satya Nadella is the Chairman and Chief Executive Officer (CEO) of Microsoft, having succeeded Steve Ballmer in February 2014.",
  "quantum entanglement": "Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated, interact, or share spatial proximity in a way such that the quantum state of each particle of the group cannot be described independently of the state of the others, including when the particles are separated by a large distance."
};

const datasetQa = new Map<string, string>(Object.entries(INITIAL_QA));
const learnedPairs: Array<{ input: string; output: string }> = [];

// Load vector_answers.txt if present to enrich local knowledge
try {
  const vectorPath = path.join(__dirname, "models", "vector_answers.txt");
  if (fs.existsSync(vectorPath)) {
    const lines = fs.readFileSync(vectorPath, "utf-8").split("\n").map(l => l.trim()).filter(Boolean);
    lines.forEach((line, idx) => {
      const cleaned = line.replace(/Read more/gi, "").replace(/\s+/g, " ").trim();
      if (cleaned.length > 30) {
        const key = cleaned.split(/\s+/).slice(0, 6).join(" ").toLowerCase().replace(/[^\w\s]/g, "");
        if (key && !datasetQa.has(key)) {
          datasetQa.set(key, cleaned);
        }
      }
    });
  }
} catch {
  // Ignore if file missing
}

// ============================================================================
// 2. CONVERSATION MEMORY & CONTEXT
// ============================================================================

interface Turn {
  role: string;
  text: string;
  topic: string;
  intent: string;
}

let conversation: Turn[] = [];
const lastExchange = { q: "", a: "" };
const userProfile = {
  topics: [] as string[],
  lastSeen: new Date().toISOString(),
  sessionCount: 1,
  name: ""
};

function addToConversation(role: string, text: string, topic = "", intent = "") {
  const cleanText = text.trim();
  const cleanTopic = (topic || cleanText).slice(0, 120);
  conversation.push({ role, text: cleanText, topic: cleanTopic, intent });
  if (conversation.length > 500) {
    conversation.splice(0, 50);
  }
  if (role === "user" && cleanTopic.split(/\s+/).length >= 2) {
    const tLower = cleanTopic.toLowerCase().slice(0, 80);
    userProfile.topics = [tLower, ...userProfile.topics.filter(t => t !== tLower)].slice(0, 50);
  }
}

function getLastTopic(): string {
  for (let i = conversation.length - 1; i >= 0; i--) {
    const turn = conversation[i];
    if (turn.role === "user" && turn.topic.split(/\s+/).length >= 2) {
      return turn.topic;
    }
  }
  return "";
}

function buildContextPrompt(userText: string, maxTurns = 20): string {
  if (conversation.length <= 1) return userText;
  let recent = conversation.slice(-maxTurns);
  if (recent.length > 0 && recent[recent.length - 1].role === "user" && recent[recent.length - 1].text === userText) {
    recent = recent.slice(0, -1);
  }
  if (recent.length === 0) return userText;
  const lines = recent.map(t => {
    const label = t.role === "user" ? "User" : "Hema";
    const txt = t.role === "ai" && t.text.length > 300 ? t.text.slice(0, 300) + "..." : t.text;
    return `${label}: ${txt}`;
  });
  return `[Conversation so far]\n${lines.join("\n")}\n\n[Current question]\nUser: ${userText}`;
}

const FOLLOWUP_TRIGGERS = new Set([
  "with example", "show example", "give example", "example", "examples",
  "example please", "more", "more details", "tell me more", "explain",
  "explain more", "explain it", "elaborate", "continue", "go on",
  "what else", "in detail", "detailed", "show me", "how", "why",
  "in python", "in code", "code for it", "code", "summarise", "summarize",
  "pros and cons", "advantages", "disadvantages", "compare", "vs",
  "difference", "better", "best", "worst"
]);

function resolveFollowup(userText: string): [string, boolean] {
  const stripped = userText.trim().toLowerCase().replace(/[?!.]+$/, "");
  if (FOLLOWUP_TRIGGERS.has(stripped) || userText.trim().split(/\s+/).length <= 2) {
    const last = getLastTopic();
    if (last && FOLLOWUP_TRIGGERS.has(stripped)) {
      return [`${stripped} ${last}`.trim(), true];
    }
  }
  return [userText, false];
}

// ============================================================================
// 3. HEMA NLP ENGINE (Ported from src/nlp_engine.py)
// ============================================================================

const CONTRACTIONS: Record<string, string> = {
  "don't": "do not", "doesn't": "does not", "didn't": "did not",
  "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
  "shouldn't": "should not", "isn't": "is not", "aren't": "are not",
  "wasn't": "was not", "weren't": "were not", "haven't": "have not",
  "hasn't": "has not", "hadn't": "had not", "can't": "cannot",
  "it's": "it is", "that's": "that is", "what's": "what is",
  "there's": "there is", "i'm": "i am", "i've": "i have",
  "i'll": "i will", "i'd": "i would", "you're": "you are",
  "we're": "we are", "let's": "let us", "who's": "who is"
};

const STOP_WORDS = new Set([
  "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
  "this", "that", "these", "those", "and", "or", "but", "if", "in", "on",
  "at", "to", "for", "of", "with", "about", "by", "from", "up", "out",
  "into", "through", "do", "does", "did", "has", "have", "had", "will",
  "would", "could", "should", "may", "might", "shall", "can",
  "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
  "my", "your", "his", "its", "our", "their", "what", "who", "where",
  "when", "how", "which", "whom", "am", "as", "so", "than", "then",
  "please", "tell", "show", "explain", "describe", "define"
]);

const QUESTION_TYPOS: Record<string, string> = {
  "waht": "what", "whta": "what", "wath": "what", "whats": "what is",
  "hwo": "how", "whi": "why", "wher": "where", "wehre": "where",
  "wen": "when", "woh": "who", "explan": "explain", "explian": "explain",
  "defin": "define", "pyton": "python", "pytohn": "python", "phyton": "python",
  "javascrip": "javascript", "javscript": "javascript",
  "machne": "machine", "learing": "learning", "lerning": "learning",
  "artifical": "artificial", "inteligence": "intelligence",
  "algortihm": "algorithm", "algoritm": "algorithm",
  "databse": "database", "programing": "programming",
  "scince": "science", "sceince": "science", "mathamatics": "mathematics",
  "histroy": "history", "tecnology": "technology", "exmaple": "example",
  "yuo": "you", "teh": "the", "recieve": "receive", "seperate": "separate",
  "definately": "definitely", "occured": "occurred", "goverment": "government",
  "enviroment": "environment", "knowlege": "knowledge", "langauge": "language",
  "becuase": "because"
};

function tokenize(text: string): string[] {
  let t = text.toLowerCase().trim();
  for (const [c, exp] of Object.entries(CONTRACTIONS)) {
    t = t.replaceAll(c, exp);
  }
  t = t.replace(/'s\b/g, "").replace(/-/g, " ").replace(/[^\w\s]/g, " ");
  return t.split(/\s+/).filter(Boolean);
}

function spellCorrect(text: string): [string, boolean] {
  const words = text.trim().split(/\s+/);
  let changed = false;
  const corrected = words.map(w => {
    const clean = w.toLowerCase().replace(/[^\w]/g, "");
    if (QUESTION_TYPOS[clean]) {
      changed = true;
      return QUESTION_TYPOS[clean];
    }
    return w;
  });
  return [corrected.join(" "), changed];
}

function normalizeQuery(text: string): string {
  let t = text.toLowerCase().trim();
  for (const [c, exp] of Object.entries(CONTRACTIONS)) {
    t = t.replaceAll(c, exp);
  }
  t = t.replace(/[^\w\s]/g, " ").replace(/\s+/g, " ").trim();
  return t;
}

function classifyIntent(text: string): [string, number] {
  const lower = text.toLowerCase().trim();
  if (/^(hi|hello|hey|hola|greetings|good morning|good afternoon|good evening|sup|howdy|how are you)\b/.test(lower)) {
    return ["greeting", 0.95];
  }
  if (/^(bye|goodbye|see you|later|take care|farewell|cya|ttyl)\b/.test(lower)) {
    return ["farewell", 0.95];
  }
  if (/^(thanks|thank you|ty|thx|appreciate|cheers)\b/.test(lower)) {
    return ["thanks", 0.95];
  }
  if (/(who are you|what are you|your name|who made you|who created you|introduce yourself)/.test(lower)) {
    return ["identity", 0.92];
  }
  if (/(what can you do|how can you help|your abilities|your capabilities)/.test(lower)) {
    return ["capability", 0.9];
  }
  if (/(\d+\s*[+\-*/^%]\s*\d+|factorial|square root|sqrt|calculate|solve|choose)/.test(lower)) {
    return ["math", 0.9];
  }
  if (/(difference between|compare| vs |versus|pros and cons)/.test(lower)) {
    return ["comparison", 0.85];
  }
  if (/(give me an example|example of|show example|for example)/.test(lower)) {
    return ["example", 0.85];
  }
  if (/(write code|python code|javascript|function|algorithm|implement|code for)/.test(lower)) {
    return ["code", 0.85];
  }
  if (/^(how to|how can i|steps to|guide|tutorial)\b/.test(lower)) {
    return ["howto", 0.85];
  }
  if (/^(what is|what are|define|meaning of|definition of)\b/.test(lower)) {
    return ["definition", 0.85];
  }
  if (/^(explain|how does|why does|tell me about|elaborate)\b/.test(lower)) {
    return ["explanation", 0.8];
  }
  if (/(joke|funny|make me laugh|tell me a joke|riddle)/.test(lower)) {
    return ["joke", 0.9];
  }
  return ["general", 0.5];
}

function extractEntities(text: string): Record<string, string[]> {
  const lower = text.toLowerCase();
  const entities: Record<string, string[]> = {};
  const techs = ["python", "javascript", "typescript", "java", "c++", "rust", "go", "react", "node", "flask", "pytorch", "tensorflow", "sql", "html", "css", "docker", "kubernetes", "git", "linux"];
  const sciences = ["quantum", "gravity", "relativity", "dna", "rna", "atom", "electron", "photon", "thermodynamics", "evolution", "photosynthesis", "mars", "earth", "jupiter", "solar system"];
  const foundTech = techs.filter(t => lower.includes(t));
  const foundSci = sciences.filter(s => lower.includes(s));
  const nums = text.match(/\b\d+(?:\.\d+)?%?\b/g);
  if (foundTech.length) entities.TECH = foundTech;
  if (foundSci.length) entities.SCIENCE = foundSci;
  if (nums) entities.NUMBER = nums;
  return entities;
}

function searchDataset(query: string): Array<{ answer: string; score: number }> {
  const qTokens = tokenize(query).filter(t => !STOP_WORDS.has(t));
  if (qTokens.length === 0) return [];
  const qNorm = normalizeQuery(query);

  // Direct match check
  if (datasetQa.has(qNorm)) {
    return [{ answer: datasetQa.get(qNorm)!, score: 1.0 }];
  }

  const results: Array<{ answer: string; score: number }> = [];
  for (const [k, ans] of datasetQa.entries()) {
    const kTokens = tokenize(k).filter(t => !STOP_WORDS.has(t));
    if (kTokens.length === 0) continue;
    const setK = new Set(kTokens);
    let overlap = 0;
    for (const qt of qTokens) {
      if (setK.has(qt)) overlap++;
    }
    if (overlap > 0) {
      const score = overlap / Math.sqrt(qTokens.length * kTokens.length);
      if (score >= 0.45) {
        results.push({ answer: ans, score });
      }
    }
  }
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, 3);
}

// ============================================================================
// 4. SAFE MATH SOLVER (Ported from solve_math in src/app.py)
// ============================================================================

function factorial(n: number): number {
  let res = 1;
  for (let i = 2; i <= n; i++) res *= i;
  return res;
}

function combinations(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  return Math.round(factorial(n) / (factorial(k) * factorial(n - k)));
}

function permutations(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  return Math.round(factorial(n) / factorial(n - k));
}

function formatNum(n: number): string {
  if (Number.isInteger(n)) return n.toString();
  return Number(n.toPrecision(6)).toString();
}

function solveMath(text: string): string | null {
  const raw = text.trim();
  let expr = raw.toLowerCase();

  // 1. Factorial: "10 factorial", "factorial of 6", "5!"
  const factMatch = expr.match(/(?:factorial\s+of\s+|factorial\s*\(?\s*)(\d+)|^(\d+)\s*!\s*$|^(\d+)\s+factorial$/);
  if (factMatch) {
    const nStr = factMatch[1] || factMatch[2] || factMatch[3];
    const n = parseInt(nStr, 10);
    if (n >= 0 && n <= 170) {
      return `${n}! = ${factorial(n).toLocaleString("en-US")}`;
    }
  }

  // 2. Number sequence: "2, 4, 8, 16, ?"
  const seqMatch = expr.match(/^[\w\s]*?(\d+(?:[,\s]+\d+){2,})[,\s]*\??$/);
  if (seqMatch && (expr.includes(",") || expr.endsWith("?"))) {
    const nums = (seqMatch[1].match(/\d+/g) || []).map(Number);
    if (nums.length >= 3) {
      const diffs = nums.slice(1).map((v, i) => v - nums[i]);
      if (new Set(diffs).size === 1) {
        const nxt = nums[nums.length - 1] + diffs[0];
        const sign = diffs[0] >= 0 ? `+${diffs[0]}` : `${diffs[0]}`;
        return `The sequence ${nums.join(", ")}, ... is arithmetic (difference ${sign}).\n**Next term: ${nxt}**`;
      }
      if (nums.every(n => n !== 0)) {
        const ratios = nums.slice(1).map((v, i) => Number((v / nums[i]).toFixed(6)));
        if (new Set(ratios).size === 1) {
          const r = ratios[0];
          const nxt = nums[nums.length - 1] * r;
          return `The sequence ${nums.join(", ")}, ... is geometric (ratio ${formatNum(r)}).\n**Next term: ${formatNum(nxt)}**`;
        }
      }
      const isFib = nums.slice(2).every((v, i) => v === nums[i] + nums[i + 1]);
      if (isFib) {
        const nxt = nums[nums.length - 1] + nums[nums.length - 2];
        return `The sequence ${nums.join(", ")}, ... follows a Fibonacci pattern.\n**Next term: ${nxt}**`;
      }
    }
  }

  // 3. Simple linear equation: "2x + 5 = 15"
  const eqMatch = expr.match(/^(?:solve\s+|find\s+|what\s+is\s+)?([+-]?\s*\d*\.?\d*)\s*([a-z])\s*([+-]\s*\d+\.?\d*)\s*=\s*([+-]?\d+\.?\d*)$/);
  if (eqMatch) {
    const aStr = eqMatch[1].replace(/\s+/g, "") || "1";
    const variable = eqMatch[2];
    const bStr = eqMatch[3].replace(/\s+/g, "");
    const cStr = eqMatch[4].replace(/\s+/g, "");
    const a = parseFloat(aStr === "-" ? "-1" : aStr === "+" ? "1" : aStr);
    const b = parseFloat(bStr);
    const c = parseFloat(cStr);
    if (!isNaN(a) && !isNaN(b) && !isNaN(c) && a !== 0) {
      const x = (c - b) / a;
      return `${variable} = ${formatNum(x)}`;
    }
  }

  // 4. Combinatorics: "5 choose 3", "C(8,2)"
  const combMatch = expr.match(/(\d+)\s+choose\s+(\d+)|c\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)|combinations?\s+of\s+(\d+)\s+from\s+(\d+)/);
  if (combMatch) {
    const n = parseInt(combMatch[1] || combMatch[3] || combMatch[6], 10);
    const k = parseInt(combMatch[2] || combMatch[4] || combMatch[5], 10);
    return `C(${n}, ${k}) = ${combinations(n, k).toLocaleString("en-US")}`;
  }

  const permMatch = expr.match(/permutations?\s+of\s+(\d+)\s+from\s+(\d+)|p\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)/);
  if (permMatch) {
    const n = parseInt(permMatch[2] || permMatch[3], 10);
    const k = parseInt(permMatch[1] || permMatch[4], 10);
    return `P(${n}, ${k}) = ${permutations(n, k).toLocaleString("en-US")}`;
  }

  // 5. Roots
  const sqRootMatch = expr.match(/square\s+root\s+of\s+(\d+\.?\d*)|sqrt\s*\(\s*(\d+\.?\d*)\s*\)/);
  if (sqRootMatch) {
    const n = parseFloat(sqRootMatch[1] || sqRootMatch[2]);
    return `√${n} = ${formatNum(Math.sqrt(n))}`;
  }

  const cubeMatch = expr.match(/cube\s+root\s+of\s+(\d+\.?\d*)/);
  if (cubeMatch) {
    const n = parseFloat(cubeMatch[1]);
    return `Cube root of ${n} = ${formatNum(Math.cbrt(n))}`;
  }

  // 6. Standard arithmetic expression
  let cleanExpr = expr
    .replace(/\b(what is|whats|what's|calculate|find|evaluate|compute|solve|result of|the answer to|the value of)\b/g, "")
    .replace(/\bsquared\b/g, "**2")
    .replace(/\bcubed\b/g, "**3")
    .replace(/\bto\s+the\s+power\s+of\b/g, "**")
    .replace(/\btimes\b/g, "*")
    .replace(/\bdivided\s+by\b/g, "/")
    .replace(/\bplus\b/g, "+")
    .replace(/\bminus\b/g, "-")
    .replace(/\bmod(?:ulo)?\b/g, "%")
    .replace(/\^/g, "**")
    .replace(/×/g, "*")
    .replace(/÷/g, "/")
    .trim()
    .replace(/[?=]+$/, "")
    .trim();

  if (!/\d/.test(cleanExpr)) return null;
  if (!/[+\-*/%]/.test(cleanExpr)) return null;
  if (!/^[\d\s+\-*/().%]+$/.test(cleanExpr)) return null;

  try {
    // Validated strictly to contain only digits, spaces, and arithmetic operators
    const fn = new Function(`return (${cleanExpr});`);
    const result = fn();
    if (typeof result === "number" && !Number.isNaN(result)) {
      if (!Number.isFinite(result)) return "undefined (division by zero)";
      return formatNum(result);
    }
  } catch {
    return null;
  }
  return null;
}

// ============================================================================
// 5. MULTI-SOURCE LIVE WEB SEARCH (Wikipedia + DuckDuckGo + Dictionary API)
// ============================================================================

async function searchWikipedia(query: string): Promise<string | null> {
  try {
    const cleanQ = query
      .replace(/^(what is|what are|who is|who was|where is|explain|define|tell me about|search)\s+/i, "")
      .replace(/[?!.]+$/, "")
      .trim();
    if (!cleanQ) return null;

    const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(cleanQ)}&utf8=&format=json&srlimit=1`;
    const searchRes = await fetch(searchUrl, { signal: AbortSignal.timeout(4500) });
    if (!searchRes.ok) return null;
    const searchData = (await searchRes.json()) as {
      query?: { search?: Array<{ title: string }> };
    };
    const topTitle = searchData.query?.search?.[0]?.title;
    if (!topTitle) return null;

    const summaryUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(topTitle)}`;
    const sumRes = await fetch(summaryUrl, { signal: AbortSignal.timeout(4500) });
    if (!sumRes.ok) return null;
    const sumData = (await sumRes.json()) as { extract?: string };
    if (sumData.extract && sumData.extract.length > 40) {
      return sumData.extract;
    }
  } catch {
    // Ignore timeout or network error
  }
  return null;
}

async function searchDuckDuckGo(query: string): Promise<string | null> {
  try {
    const cleanQ = query
      .replace(/^(what is|what are|who is|who was|explain|define|tell me about|search)\s+/i, "")
      .replace(/[?!.]+$/, "")
      .trim();
    const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(cleanQ)}&format=json&no_html=1&skip_disambig=1`;
    const res = await fetch(url, { signal: AbortSignal.timeout(4500) });
    if (!res.ok) return null;
    const data = (await res.json()) as {
      AbstractText?: string;
      RelatedTopics?: Array<{ Text?: string }>;
    };
    if (data.AbstractText && data.AbstractText.length > 30) {
      return data.AbstractText;
    }
    if (data.RelatedTopics && data.RelatedTopics[0]?.Text && data.RelatedTopics[0].Text.length > 30) {
      return data.RelatedTopics[0].Text;
    }
  } catch {
    // Ignore
  }
  return null;
}

async function searchDictionary(query: string): Promise<string | null> {
  try {
    const m = query.match(/^(?:define|meaning of|what does\s+(.+?)\s+mean|definition of)\s*(.*)$/i);
    const word = (m ? (m[1] || m[2]) : query).trim().split(/\s+/)[0]?.replace(/[^\w]/g, "");
    if (!word || word.length < 2) return null;

    const url = `https://api.dictionaryapi.dev/api/v2/entries/en/${encodeURIComponent(word.toLowerCase())}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(3500) });
    if (!res.ok) return null;
    const data = (await res.json()) as Array<{
      word?: string;
      meanings?: Array<{
        partOfSpeech?: string;
        definitions?: Array<{ definition?: string; example?: string }>;
      }>;
    }>;
    const entry = data?.[0];
    const meaning = entry?.meanings?.[0];
    const def = meaning?.definitions?.[0];
    if (def?.definition) {
      let out = `**${entry.word}** (*${meaning?.partOfSpeech || "word"}*): ${def.definition}`;
      if (def.example) {
        out += `\n\n*Example:* "${def.example}"`;
      }
      return out;
    }
  } catch {
    // Ignore
  }
  return null;
}

async function multiSourceSearch(query: string): Promise<string | null> {
  const [wiki, ddg, dict] = await Promise.all([
    searchWikipedia(query),
    searchDuckDuckGo(query),
    searchDictionary(query)
  ]);

  const parts: string[] = [];
  if (wiki) parts.push(wiki);
  if (ddg && (!wiki || !wiki.slice(0, 60).toLowerCase().includes(ddg.slice(0, 40).toLowerCase()))) {
    parts.push(ddg);
  }
  if (dict && parts.length === 0) {
    parts.push(dict);
  }
  if (parts.length === 0) return null;
  return parts.join("\n\n");
}

function learnFromSearch(question: string, answer: string) {
  const q = normalizeQuery(question);
  const a = answer.trim();
  if (q.split(/\s+/).length < 2 || a.split(/\s+/).length < 5) return;
  if (!datasetQa.has(q)) {
    datasetQa.set(q, a);
    learnedPairs.push({ input: q, output: a });
  }
}

// ============================================================================
// 6. MARKDOWN FORMATTER (Ported from structure_like_chatgpt & format_response)
// ============================================================================

function formatResponse(
  text: string,
  responseType: "answer" | "web" | "example" | "math" | "file" | "joke" = "answer",
  spellNotice = "",
  query = ""
): string {
  if (!text) return "";
  const prefix = spellNotice ? `*${spellNotice}*\n\n` : "";

  if (responseType === "math") {
    return `${prefix}**Answer:** ${text}`;
  }
  if (responseType === "joke") {
    return `${prefix}😄 ${text} 😂`;
  }
  if (responseType === "file") {
    return `${prefix}${text}`;
  }

  // Structure prose cleanly into paragraphs with bold lead for definitions
  if (text.includes("```") || text.includes("\n- ") || text.includes("\n1. ") || text.startsWith("##")) {
    return `${prefix}${text}`;
  }

  const sentences = text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
  if (sentences.length <= 1) {
    return `${prefix}${text}`;
  }

  const isDef = /^(what is|what are|define|who is|who was|explain)\b/i.test(query);
  const first = isDef && !sentences[0].startsWith("**") ? `**${sentences[0]}**` : sentences[0];
  const rest = sentences.slice(1);
  const paragraphs: string[] = [first];

  for (let i = 0; i < rest.length; i += 3) {
    paragraphs.push(rest.slice(i, i + 3).join(" "));
  }

  return `${prefix}${paragraphs.join("\n\n")}`;
}

// ============================================================================
// 7. MAIN AI RESPONSE PIPELINE
// ============================================================================

async function getAiResponse(userText: string, selectedModule = "hybrid"): Promise<string> {
  const originalText = userText.trim();
  if (!originalText) return "Please enter a message.";

  const [resolvedText, wasFollowup] = resolveFollowup(originalText);
  const activeText = wasFollowup ? resolvedText : originalText;
  addToConversation("user", originalText, activeText);

  const [intentLabel] = classifyIntent(originalText);
  const rawLower = originalText.toLowerCase().trim();

  // 1. Early Math Solver
  const earlyMath = solveMath(originalText);
  if (earlyMath) {
    return formatResponse(earlyMath, "math");
  }

  // 2. Greetings & Small Talk
  const rawGreets = new Set([
    "hi", "hello", "hey", "hola", "sup", "yo", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how r you", "how are u", "hows it going", "how's it going"
  ]);
  if (rawGreets.has(rawLower) || intentLabel === "greeting") {
    const replies = [
      "Hey! What would you like to explore today?",
      "Hi there! Ask me anything — science, math, coding, history, or just chat.",
      "Hello! What's on your mind?",
      "Hey! What can I help you with?"
    ];
    return replies[Math.floor(Math.random() * replies.length)];
  }

  if (rawLower.startsWith("nice to meet") || rawLower.startsWith("pleased to meet")) {
    return "Nice to meet you too! What can I help you with today?";
  }

  if (["bye", "goodbye", "see you", "later", "take care", "farewell", "cya", "ttyl"].includes(rawLower) || intentLabel === "farewell") {
    return "Goodbye! Come back anytime — I'll be here.";
  }

  if (["thanks", "thank you", "ty", "thx", "cheers", "appreciate it"].includes(rawLower) || intentLabel === "thanks") {
    return "You're welcome! Anything else I can help with?";
  }

  if (rawLower.startsWith("say ") || rawLower.startsWith("repeat ")) {
    const spoken = originalText.replace(/^(say|repeat)\s+/i, "").trim();
    if (spoken) return `You said: **${spoken}**`;
  }

  // Identity & Creator
  if (intentLabel === "identity" || /(who are you|what are you|your name|who is hema|what is hema|introduce yourself)/.test(rawLower)) {
    return (
      "I'm **Hema** — a hybrid AI assistant built by **Tharun Sai**! " +
      "I combine a local NLP & knowledge engine with live web search across Wikipedia, " +
      "DuckDuckGo, and a Dictionary API to give you structured answers."
    );
  }

  if (/(who created you|who made you|who built you|who is your creator|tharun sai)/.test(rawLower)) {
    return (
      "I was created by **Tharun Sai** — built from scratch with a custom NLP engine, " +
      "math solver, document analyzer, and multi-source hybrid search pipeline."
    );
  }

  if (intentLabel === "capability" || /(what can you do|your capabilities|how can you help)/.test(rawLower)) {
    return (
      "I can help you with:\n\n" +
      "- **Science** — physics, biology, chemistry, astronomy\n" +
      "- **Math** — arithmetic, factorials, sequences, linear equations, combinatorics\n" +
      "- **Coding** — Python, JS, algorithms, AI/ML, code file analysis\n" +
      "- **History & Geography** — world events, capitals, oceans, planets\n" +
      "- **File & Image Analysis** — attach code, CSV, Markdown, text, or image files\n" +
      "- **Live Web Search** — real-time synthesis from Wikipedia, DuckDuckGo & Dictionary API"
    );
  }

  if (intentLabel === "joke") {
    const jokes = [
      "Why do programmers prefer dark mode? Because light attracts bugs!",
      "Why did the developer go broke? Because he used up all his cache.",
      "There are 10 types of people in the world: those who understand binary, and those who don't."
    ];
    return formatResponse(jokes[Math.floor(Math.random() * jokes.length)], "joke");
  }

  // 3. Spell Correction
  const [correctedText, wasCorrected] = spellCorrect(activeText);
  const spellNotice = wasCorrected ? `Did you mean: "${correctedText}"? Answering that...` : "";
  const queryToUse = wasCorrected ? correctedText : activeText;

  // Check math again after spell correction
  const postSpellMath = solveMath(queryToUse);
  if (postSpellMath) {
    return formatResponse(postSpellMath, "math", spellNotice);
  }

  // 4. Web-Only Mode
  if (selectedModule === "web") {
    const webRes = await multiSourceSearch(queryToUse);
    if (webRes) {
      learnFromSearch(queryToUse, webRes);
      return formatResponse(webRes, "web", spellNotice, queryToUse);
    }
    return "Sorry, I could not find results online for that query.";
  }

  // 5. Local Dataset / TF-IDF Search (Hybrid or Local mode)
  const localMatches = searchDataset(queryToUse);
  if (localMatches.length > 0 && localMatches[0].score >= 0.5) {
    return formatResponse(localMatches[0].answer, "answer", spellNotice, queryToUse);
  }

  if (selectedModule === "local") {
    if (localMatches.length > 0) {
      return formatResponse(localMatches[0].answer, "answer", spellNotice, queryToUse);
    }
    return formatResponse(
      "I couldn't find an exact match in my local dataset for that query. Switch to **Hybrid** or **Web** mode to search live sources!",
      "answer",
      spellNotice
    );
  }

  // 6. Hybrid Mode Fallback to Multi-Source Live Web Search
  const webAnswer = await multiSourceSearch(queryToUse);
  if (webAnswer) {
    learnFromSearch(queryToUse, webAnswer);
    return formatResponse(webAnswer, "web", spellNotice, queryToUse);
  }

  if (localMatches.length > 0) {
    return formatResponse(localMatches[0].answer, "answer", spellNotice, queryToUse);
  }

  return formatResponse(
    "I searched my knowledge base and live web sources but couldn't find a detailed answer for that specific phrasing. Try asking about a scientific concept, historical figure, programming topic, or math problem!",
    "answer",
    spellNotice
  );
}

// ============================================================================
// 8. FILE & IMAGE ANALYSIS ENGINE (Ported from src/app.py)
// ============================================================================

function getFileExt(filename: string): string {
  const idx = filename.lastIndexOf(".");
  return idx !== -1 ? filename.slice(idx).toLowerCase() : "";
}

function extractKeyPoints(text: string, maxPoints = 5): string[] {
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(s => s.split(/\s+/).length >= 5);

  if (sentences.length === 0) {
    return text
      .split("\n")
      .map(l => l.trim())
      .filter(l => l.split(/\s+/).length >= 3)
      .slice(0, maxPoints);
  }
  return sentences.slice(0, maxPoints);
}

function summariseTextFile(text: string, filename: string): string {
  const ext = getFileExt(filename);
  const lines = text.split("\n");
  const lineCount = lines.length;
  const words = text.trim().split(/\s+/).filter(Boolean);
  const wordCount = words.length;
  const charCount = text.length;

  if (ext === ".csv") {
    const header = lines[0] || "";
    const rows = Math.max(0, lineCount - 1);
    const cols = header.split(",").length;
    const preview = lines.slice(1, 6).join("\n");
    return [
      `## ${filename} \`CSV\``,
      `**${rows.toLocaleString()}** rows · **${cols}** columns`,
      "",
      `**Columns:** ${header}`,
      "",
      "**Preview:**",
      "```",
      preview,
      "```"
    ].join("\n");
  }

  const codeExts = new Set([".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".rb", ".php", ".sh", ".sql"]);
  if (codeExts.has(ext)) {
    const funcMatches = text.match(/^\s*(async\s+def\s+|def\s+|function\s+|class\s+|const\s+\w+\s*=\s*\(|export\s+function)/gm) || [];
    const commentLines = lines.filter(l => /^\s*(#|\/\/|\/\*|\*|--)/.test(l)).length;
    const blankLines = lines.filter(l => !l.trim()).length;
    const codeLines = lineCount - commentLines - blankLines;
    return [
      `## 💻 Code: \`${filename}\``,
      "| Metric | Value |",
      "|--------|-------|",
      `| Total lines | ${lineCount.toLocaleString()} |`,
      `| Code lines | ${codeLines.toLocaleString()} |`,
      `| Comment lines | ${commentLines.toLocaleString()} |`,
      `| Blank lines | ${blankLines.toLocaleString()} |`,
      `| Words | ${wordCount.toLocaleString()} |`,
      `| Functions / Classes | ${funcMatches.length} |`
    ].join("\n");
  }

  const keyPts = extractKeyPoints(text, 5);
  const out = [
    `## 📄 Document: \`${filename}\``,
    `**${wordCount.toLocaleString()}** words · **${lineCount.toLocaleString()}** lines · **${charCount.toLocaleString()}** characters`,
    ""
  ];
  if (keyPts.length > 0) {
    out.push(`### Key Points (${keyPts.length})`);
    for (const pt of keyPts) {
      out.push(`- ${pt.slice(0, 200)}`);
    }
  }
  return out.join("\n");
}

function parseImageDimensions(buf: Buffer, ext: string): { width: number; height: number; format: string } {
  let width = 0;
  let height = 0;
  let format = ext.replace(".", "").toUpperCase() || "IMAGE";

  try {
    // PNG
    if (buf.length >= 24 && buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) {
      format = "PNG";
      width = buf.readUInt32BE(16);
      height = buf.readUInt32BE(20);
    }
    // GIF
    else if (buf.length >= 10 && buf[0] === 0x47 && buf[1] === 0x49 && buf[2] === 0x46) {
      format = "GIF";
      width = buf.readUInt16LE(6);
      height = buf.readUInt16LE(8);
    }
    // JPEG
    else if (buf.length >= 4 && buf[0] === 0xff && buf[1] === 0xd8) {
      format = "JPEG";
      let offset = 2;
      while (offset < buf.length - 8) {
        if (buf[offset] !== 0xff) break;
        const marker = buf[offset + 1];
        if (marker >= 0xc0 && marker <= 0xc3) {
          height = buf.readUInt16BE(offset + 5);
          width = buf.readUInt16BE(offset + 7);
          break;
        }
        const len = buf.readUInt16BE(offset + 2);
        offset += 2 + len;
      }
    }
  } catch {
    // Fallback if header parsing fails
  }
  return { width, height, format };
}

// ============================================================================
// 9. QUOTE ENGINE
// ============================================================================

const FALLBACK_QUOTES = [
  "The mind is not a vessel to be filled, but a fire to be kindled.",
  "Intelligence is the ability to adapt to change.",
  "Any sufficiently advanced technology is indistinguishable from magic.",
  "The measure of intelligence is the ability to change.",
  "Somewhere, something incredible is waiting to be known.",
  "We did not build minds of silicon to answer everything, but to give our wildest questions a place to echo.",
  "Logic will get you from A to Z; imagination will get you everywhere.",
  "The important thing is not to stop questioning."
];

// ============================================================================
// 10. EXPRESS ROUTES (Matching Flask routes in src/app.py)
// ============================================================================

app.get("/", (_req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, "src", "templates", "index.html"));
});

app.get("/favicon.ico", (_req: Request, res: Response) => {
  res.status(204).end();
});

app.get("/quote", (_req: Request, res: Response) => {
  const quoteText = FALLBACK_QUOTES[Math.floor(Math.random() * FALLBACK_QUOTES.length)];
  res.json({ text: quoteText, author: "" });
});

app.post("/chat", async (req: Request, res: Response) => {
  try {
    const { message = "", module = "hybrid", history = [] } = req.body || {};
    if (Array.isArray(history) && conversation.length === 0) {
      for (const turn of history.slice(-40)) {
        if (turn && typeof turn.text === "string" && turn.text.trim()) {
          addToConversation(turn.role || "user", turn.text);
        }
      }
    }

    const aiReply = await getAiResponse(String(message), String(module).toLowerCase());
    addToConversation("ai", aiReply, String(message));
    lastExchange.q = String(message);
    lastExchange.a = aiReply;

    res.json({
      reply: aiReply,
      context_turns: conversation.length
    });
  } catch (err) {
    console.error("Chat error:", err);
    res.json({ reply: "Internal error processing your message." });
  }
});

app.post("/upload", async (req: Request, res: Response) => {
  try {
    const {
      file_data = "",
      filename = "file",
      mimetype = "text/plain",
      message = "",
      module = "hybrid"
    } = req.body || {};

    if (!file_data) {
      res.json({ reply: "No file data received." });
      return;
    }

    const ext = getFileExt(filename);
    const imageExts = new Set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"]);
    const buf = Buffer.from(file_data, "base64");
    const sizeKb = Math.max(1, Math.round(buf.length / 1024));

    if (imageExts.has(ext) || String(mimetype).startsWith("image/")) {
      const { width, height, format } = parseImageDimensions(buf, ext);
      const orientation = width && height ? (width > height ? "landscape" : height > width ? "portrait" : "square") : "standard";
      const dimStr = width && height ? `${width} × ${height} px (${orientation})` : "Standard resolution";
      const desc = `This is a ${format} image (${dimStr}, ${sizeKb} KB).`;

      let reply = [
        `## 🖼 Image Analysis: \`${filename}\``,
        "",
        `> ${desc}`,
        "",
        "### Properties",
        "| Property | Value |",
        "|----------|-------|",
        `| Format | ${format} |`,
        `| Dimensions | ${dimStr} |`,
        `| File size | ${sizeKb} KB |`
      ].join("\n");

      if (message && String(message).trim()) {
        const ans = await getAiResponse(`${message}\n\n[Image context: ${desc}]`, String(module).toLowerCase());
        reply += `\n\n---\n\n**Answer to your question:**\n\n${ans}`;
      }

      addToConversation("ai", reply, `image ${filename}`);
      res.json({
        reply,
        filename,
        type: "image",
        width,
        height
      });
      return;
    }

    const fileText = buf.toString("utf-8");
    const summary = summariseTextFile(fileText, filename);
    let reply = summary;

    if (message && String(message).trim()) {
      const ans = await getAiResponse(String(message), String(module).toLowerCase());
      reply += `\n\n---\n\n### Answer\n\n${ans}`;
    }

    addToConversation("ai", reply.slice(0, 400), `document ${filename}`);
    res.json({
      reply,
      filename,
      type: "text",
      word_count: fileText.split(/\s+/).filter(Boolean).length
    });
  } catch (err) {
    res.json({ reply: `Error processing file: ${String(err)}` });
  }
});

app.post("/clear", (_req: Request, res: Response) => {
  conversation = [];
  lastExchange.q = "";
  lastExchange.a = "";
  res.json({ status: "cleared" });
});

app.post("/feedback", async (req: Request, res: Response) => {
  try {
    const {
      rating = "",
      correction = "",
      question = lastExchange.q,
      answer = lastExchange.a,
      confirmed = false
    } = req.body || {};

    if (!question) {
      res.json({ status: "error", msg: "No question to look up." });
      return;
    }

    const qClean = normalizeQuery(String(question));

    if (rating === "up") {
      learnFromSearch(qClean, String(answer));
      res.json({ status: "ok", msg: "Thanks! I'll remember that." });
      return;
    }

    if (rating === "down") {
      if (correction && String(correction).trim()) {
        datasetQa.set(qClean, String(correction).trim());
        learnFromSearch(qClean, String(correction).trim());
        res.json({
          status: "ok",
          msg: confirmed ? "Learned from web! I'll use this next time." : "Correction saved! I'll use your answer next time."
        });
        return;
      }

      const searched = await multiSourceSearch(String(question));
      if (searched) {
        const short = searched.split(/(?<=[.!?])\s+/).slice(0, 2).join(" ");
        res.json({
          status: "found",
          searched_answer: short,
          msg: "Found on the web - does this look right?"
        });
        return;
      }
      res.json({
        status: "not_found",
        msg: "Could not find a good answer. Type it below."
      });
      return;
    }

    res.json({ status: "error", msg: "Invalid rating." });
  } catch (err) {
    res.json({ status: "error", msg: String(err) });
  }
});

app.get("/sources", (_req: Request, res: Response) => {
  res.json({
    sources: ["Wikipedia", "DuckDuckGo", "Dictionary API", `Local Dataset (${datasetQa.size} Q&A)`, "Hema NLP Engine"],
    mode: "hybrid"
  });
});

app.get("/context", (_req: Request, res: Response) => {
  res.json({
    total_turns: conversation.length,
    last_topic: getLastTopic(),
    recent_10: conversation.slice(-10),
    context_prompt_preview: buildContextPrompt("(your next question)", 6)
  });
});

app.get("/learned", (_req: Request, res: Response) => {
  res.json({
    count: learnedPairs.length,
    recent: learnedPairs.slice(-20)
  });
});

app.get("/nlp_debug", (req: Request, res: Response) => {
  const q = String(req.query.q || "what is machine learning");
  const [intent, intent_conf] = classifyIntent(q);
  const entities = extractEntities(q);
  const [corrected, spell_changed] = spellCorrect(q);
  const normalised = normalizeQuery(q);
  const top_matches = searchDataset(q).map(m => ({ answer: m.answer.slice(0, 80), score: Number(m.score.toFixed(3)) }));

  res.json({
    query: q,
    nlp_available: true,
    intent,
    intent_conf,
    entities,
    corrected,
    spell_changed,
    normalised,
    top_matches
  });
});

const PORT = Number(process.env.PORT) || 3000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Hema AI server listening on http://0.0.0.0:${PORT}`);
});
