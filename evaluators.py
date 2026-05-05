"""Deterministic, non-LLM evaluators for style-aware email generation.

All metrics are computed with the standard library so they can run offline
inside `web_app.py` and the CLI evaluator. There is intentionally no call to
another language model anywhere in this module — every score is a closed-form
function of the inputs so results are reproducible and "objective" in the
sense that re-running on the same strings always returns the same numbers.

Top-level entry points:
    - score_prediction(prediction, gold, profile=None) -> dict
        Returns the full bag of metrics. Used by `web_app.handle_test`.
    - style_fingerprint(text) -> dict
        9-dim style vector for one email body.
    - profile_style_centroid(profile) -> dict
        Average style fingerprint across a user's profile emails.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b")
_CAPS_TOKEN_RE = re.compile(r"\b[A-Z]{2,}\b")
_EMDASH_RE = re.compile(r"--|—")
_CONTRACTION_RE = re.compile(r"\b\w+'\w+\b")
_PUNCT_RE = re.compile(r"[\.,;:\?!\-–—]")
_FIRST_PERSON = {"i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine",
                 "we", "we're", "we've", "we'd", "we'll", "us", "our", "ours"}

# Words that are often capitalized at sentence start but are not actually
# proper nouns. Used to denoise the entity-overlap metric.
_PROPER_NOUN_STOPWORDS = {
    "i", "the", "a", "an", "this", "that", "these", "those",
    "hi", "hello", "hey", "dear", "thanks", "thank", "best", "regards",
    "sincerely", "cheers", "yes", "no", "ok", "okay", "sure",
    "please", "as", "if", "when", "where", "why", "how", "what", "who",
    "and", "but", "or", "so", "for", "to", "from", "on", "in", "at",
    "is", "are", "was", "were", "be", "been", "being",
    "subject", "incoming", "email", "sent", "cc", "from", "to",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}


def _word_tokens(text: str) -> List[str]:
    """Lowercased word tokens — used for ROUGE / TTR / pronoun rates."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _sentences(text: str) -> List[str]:
    """Crude sentence segmentation. Good enough for stylometry."""
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Content metrics
# ---------------------------------------------------------------------------

def _ngram_f1(pred_tokens: Sequence[str], gold_tokens: Sequence[str], n: int) -> float:
    """ROUGE-N F1 implemented from scratch (no nltk / rouge_score dep)."""
    pred_grams = Counter(_ngrams(pred_tokens, n))
    gold_grams = Counter(_ngrams(gold_tokens, n))
    if not pred_grams or not gold_grams:
        return 0.0
    overlap = sum((pred_grams & gold_grams).values())
    precision = overlap / max(sum(pred_grams.values()), 1)
    recall = overlap / max(sum(gold_grams.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence between two token lists."""
    if not a or not b:
        return 0
    # Memory-efficient O(min(len(a), len(b))) DP.
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ai in a:
        curr = [0] * (len(b) + 1)
        for j, bj in enumerate(b, start=1):
            if ai == bj:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge1(prediction: str, gold: str) -> float:
    return _ngram_f1(_word_tokens(prediction), _word_tokens(gold), 1)


def rouge2(prediction: str, gold: str) -> float:
    return _ngram_f1(_word_tokens(prediction), _word_tokens(gold), 2)


def rougeL(prediction: str, gold: str) -> float:
    pred_tokens = _word_tokens(prediction)
    gold_tokens = _word_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def chrf(prediction: str, gold: str, n: int = 3) -> float:
    """Character n-gram F-score. Spaces and case are kept; punctuation included.

    chrF is the standard metric for paraphrase-aware evaluation in MT and
    handles small wording differences ("don't"/"do not", "Wendy"/"Wendy,")
    much better than word-level F1.
    """
    pred = (prediction or "").strip()
    gold = (gold or "").strip()
    if not pred or not gold:
        return 0.0
    pred_grams = Counter(pred[i:i + n] for i in range(len(pred) - n + 1))
    gold_grams = Counter(gold[i:i + n] for i in range(len(gold) - n + 1))
    if not pred_grams or not gold_grams:
        return 0.0
    overlap = sum((pred_grams & gold_grams).values())
    precision = overlap / max(sum(pred_grams.values()), 1)
    recall = overlap / max(sum(gold_grams.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _extract_entities(text: str) -> set:
    """Return the set of likely proper-noun phrases in `text`.

    Heuristic: capitalized word(s), excluding common sentence-start words and
    weekday/month names. Catches names ("Cynthia Harkness", "Wendy"),
    organizations ("EBS", "Enron"), and dates well enough for Jaccard overlap.
    """
    raw = {m.group(0).strip() for m in _PROPER_NOUN_RE.finditer(text or "")}
    cleaned = set()
    for phrase in raw:
        parts = [p for p in phrase.split() if p.lower() not in _PROPER_NOUN_STOPWORDS]
        if not parts:
            continue
        cleaned.add(" ".join(parts).lower())
    return cleaned


def entity_overlap(prediction: str, gold: str) -> float:
    """Jaccard similarity over extracted proper nouns / named entities.

    This is the metric that catches the "model recommended different people"
    failure mode that word_f1 misses.
    """
    pred = _extract_entities(prediction)
    gold = _extract_entities(gold)
    if not pred and not gold:
        return 1.0  # both empty = trivially equal
    if not pred or not gold:
        return 0.0
    return len(pred & gold) / len(pred | gold)


# ---------------------------------------------------------------------------
# Style fingerprint
# ---------------------------------------------------------------------------

# Order is fixed so vectors are comparable across calls.
STYLE_FEATURES = (
    "avg_sentence_len",   # tokens per sentence
    "avg_word_len",       # characters per token
    "type_token_ratio",   # vocabulary richness
    "punct_per_word",     # punctuation density
    "emdash_per_100w",    # em-dash / double-hyphen frequency
    "first_person_per_100w",
    "contraction_per_100w",
    "questions_per_sent",
    "caps_token_rate",    # ALL-CAPS tokens / total tokens
)


def style_fingerprint(text: str) -> Dict[str, float]:
    """Compute the 9-dim style vector for one text body.

    All features are continuous and bounded. Empty input returns zeros.
    """
    text = text or ""
    tokens = _word_tokens(text)
    sentences = _sentences(text)
    n_tokens = len(tokens)
    n_sentences = max(len(sentences), 1)

    if n_tokens == 0:
        return {name: 0.0 for name in STYLE_FEATURES}

    char_counts = sum(len(t) for t in tokens)
    types = len(set(tokens))
    punct_count = len(_PUNCT_RE.findall(text))
    emdash_count = len(_EMDASH_RE.findall(text))
    contraction_count = len(_CONTRACTION_RE.findall(text))
    first_person_count = sum(1 for t in tokens if t in _FIRST_PERSON)
    caps_token_count = len(_CAPS_TOKEN_RE.findall(text))
    question_count = sum(1 for s in sentences if s.endswith("?"))

    return {
        "avg_sentence_len": n_tokens / n_sentences,
        "avg_word_len": char_counts / n_tokens,
        "type_token_ratio": types / n_tokens,
        "punct_per_word": punct_count / n_tokens,
        "emdash_per_100w": (emdash_count * 100) / n_tokens,
        "first_person_per_100w": (first_person_count * 100) / n_tokens,
        "contraction_per_100w": (contraction_count * 100) / n_tokens,
        "questions_per_sent": question_count / n_sentences,
        "caps_token_rate": caps_token_count / n_tokens,
    }


def profile_style_centroid(profile: Sequence[dict]) -> Dict[str, float]:
    """Average style fingerprint across all profile emails (the user's "voice")."""
    bodies = [item.get("body", "") for item in (profile or []) if item.get("body")]
    if not bodies:
        return {name: 0.0 for name in STYLE_FEATURES}
    vectors = [style_fingerprint(b) for b in bodies]
    return {
        name: sum(v[name] for v in vectors) / len(vectors)
        for name in STYLE_FEATURES
    }


def style_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Per-feature normalized agreement, averaged over features.

    For each feature, similarity is `1 - |a-b| / max(|a|, |b|)`. Returns 1.0
    when vectors are identical, 0.0 when one feature is zero and the other is
    not (capped). This is more interpretable than cosine when individual
    features have different natural scales (sentence length vs. ratio).
    """
    if not a or not b:
        return 0.0
    sims = []
    for name in STYLE_FEATURES:
        av = a.get(name, 0.0)
        bv = b.get(name, 0.0)
        denom = max(abs(av), abs(bv), 1e-9)
        diff = min(abs(av - bv) / denom, 1.0)
        sims.append(1.0 - diff)
    return sum(sims) / len(sims)


# ---------------------------------------------------------------------------
# Greeting / sign-off bucket classification
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = [
    ("hi", re.compile(r"^\s*hi\b", re.IGNORECASE)),
    ("hello", re.compile(r"^\s*hello\b", re.IGNORECASE)),
    ("hey", re.compile(r"^\s*hey\b", re.IGNORECASE)),
    ("dear", re.compile(r"^\s*dear\b", re.IGNORECASE)),
    # "Marie --" / "Marie:" / "Mark," at the very start of the body.
    ("name_prefix", re.compile(r"^\s*[A-Z][a-zA-Z]+\s*[\-–—:,]")),
]


def classify_greeting(text: str) -> str:
    """Bucket the opening salutation by *type*, not just present/absent."""
    if not text:
        return "none"
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    if not first_line:
        return "none"
    for label, pattern in _GREETING_PATTERNS:
        if pattern.match(first_line):
            return label
    # Body begins with a capitalized token followed by a non-letter? Already
    # covered by name_prefix. Otherwise no greeting.
    return "none"


_SIGNOFF_KEYWORDS = {
    "thanks": ("thanks", "thank you", "thx"),
    "best_regards": ("best", "regards", "sincerely", "cheers", "warmly", "kind regards"),
}


def classify_signoff(text: str) -> str:
    """Bucket the closing line by type."""
    if not text:
        return "none"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "none"
    last = lines[-1].lower().strip(".,!- ")

    for label, keywords in _SIGNOFF_KEYWORDS.items():
        if any(kw in last for kw in keywords):
            return label

    # Single capitalized token, plausibly a first name.
    bare = lines[-1].strip(".,!- ")
    if bare and " " not in bare and bare[0:1].isupper() and bare.isalpha() and 2 <= len(bare) <= 20:
        return "firstname"

    # All-caps short token = initials.
    if bare.isupper() and bare.isalpha() and 2 <= len(bare) <= 4:
        return "initials"

    return "none"


# ---------------------------------------------------------------------------
# Top-level scoring
# ---------------------------------------------------------------------------

CONTENT_METRICS = ("rouge1", "rouge2", "rougeL", "chrf", "entity_overlap")
STYLE_METRICS = ("style_to_gold", "style_to_user", "length_ratio")
DISCRETE_METRICS = ("greeting_type_match", "signoff_type_match")
NUMERIC_METRICS = CONTENT_METRICS + STYLE_METRICS + DISCRETE_METRICS


def score_prediction(prediction: str, gold: str, profile: Optional[Sequence[dict]] = None) -> dict:
    """Compute the full evaluator bundle for one (prediction, gold) pair.

    `profile` is optional. When provided (the user's profile emails), we also
    compute `style_to_user` — how close the prediction's style is to the
    user's average voice. Without it that field is reported as null.
    """
    pred_tokens = _word_tokens(prediction)
    gold_tokens = _word_tokens(gold)

    # Content fidelity ------------------------------------------------------
    rouge1_score = _ngram_f1(pred_tokens, gold_tokens, 1)
    rouge2_score = _ngram_f1(pred_tokens, gold_tokens, 2)
    rougeL_score = rougeL(prediction, gold)
    chrf_score = chrf(prediction, gold)
    entity_score = entity_overlap(prediction, gold)

    # Length ----------------------------------------------------------------
    length_ratio = (
        min(len(pred_tokens), len(gold_tokens)) /
        max(len(pred_tokens), len(gold_tokens), 1)
    )

    # Style vectors ---------------------------------------------------------
    pred_style = style_fingerprint(prediction)
    gold_style = style_fingerprint(gold)
    style_to_gold = style_similarity(pred_style, gold_style)

    if profile:
        user_centroid = profile_style_centroid(profile)
        style_to_user = style_similarity(pred_style, user_centroid)
    else:
        user_centroid = None
        style_to_user = None

    # Discrete buckets ------------------------------------------------------
    pred_greet = classify_greeting(prediction)
    gold_greet = classify_greeting(gold)
    pred_sign = classify_signoff(prediction)
    gold_sign = classify_signoff(gold)

    # Composite -------------------------------------------------------------
    content_score = sum([rouge1_score, rougeL_score, chrf_score, entity_score]) / 4
    style_components = [style_to_gold]
    if style_to_user is not None:
        style_components.append(style_to_user)
    style_score = sum(style_components) / len(style_components)

    return {
        # Backward-compatible fields used elsewhere (CLI summary, evaluator).
        "word_f1": round(rouge1_score, 4),
        "length_ratio": round(length_ratio, 4),
        "greeting_match": float(pred_greet == gold_greet),
        "signoff_match": float(pred_sign == gold_sign),

        # New content metrics.
        "rouge1": round(rouge1_score, 4),
        "rouge2": round(rouge2_score, 4),
        "rougeL": round(rougeL_score, 4),
        "chrf": round(chrf_score, 4),
        "entity_overlap": round(entity_score, 4),

        # New style metrics.
        "style_to_gold": round(style_to_gold, 4),
        "style_to_user": None if style_to_user is None else round(style_to_user, 4),

        # Discrete style buckets.
        "greeting_type_pred": pred_greet,
        "greeting_type_gold": gold_greet,
        "greeting_type_match": float(pred_greet == gold_greet),
        "signoff_type_pred": pred_sign,
        "signoff_type_gold": gold_sign,
        "signoff_type_match": float(pred_sign == gold_sign),

        # Composite roll-ups (handy for sorting / leaderboards).
        "content_score": round(content_score, 4),
        "style_score": round(style_score, 4),

        # Raw style vectors for any UI that wants to plot them.
        "style_vectors": {
            "prediction": {k: round(v, 4) for k, v in pred_style.items()},
            "gold": {k: round(v, 4) for k, v in gold_style.items()},
            "user": (None if user_centroid is None
                     else {k: round(v, 4) for k, v in user_centroid.items()}),
        },
    }
