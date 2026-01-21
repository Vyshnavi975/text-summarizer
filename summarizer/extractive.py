"""
extractive.py
=============

A from-scratch extractive text summarizer with zero external
dependencies (standard library only). It works entirely offline, so
it is used as the "demo mode" backend whenever no LLM API key is
configured.

Two scoring strategies are implemented:

1. ``score_sentences_by_frequency`` — classic frequency-based scoring
   (Luhn-style): words are weighted by how often they appear in the
   document (after removing stopwords), and each sentence's score is
   the average weight of the words it contains. This rewards
   sentences built from the document's most "typical" vocabulary.

2. ``score_sentences_by_textrank`` — a simplified TextRank: sentences
   are nodes in a graph, edges are weighted by how similar two
   sentences are (word overlap), and importance is computed with the
   same iterative random-walk idea PageRank uses.

Both are combined (frequency and graph scores are normalized and
averaged) by default, which tends to be more robust than either
alone. The top-N highest scoring sentences are then returned in their
*original order* so the summary still reads coherently.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

# A compact, general-purpose English stopword list. Kept local (no NLTK
# dependency) so the summarizer has zero external requirements.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "also", "however", "therefore", "thus", "e.g", "i.e",
    "said", "says", "one", "two", "three", "many", "much", "get", "gets",
}

# Matches abbreviations like "Mr.", "Dr.", "U.S.", "e.g." etc. so we don't
# split sentences on their periods.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e",
    "u.s", "u.k", "st", "gen", "rep", "sen", "gov", "no", "inc", "ltd",
    "co", "corp", "vol", "approx",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(“])")


@dataclass
class ScoredSentence:
    index: int
    text: str
    score: float


def clean_text(text: str) -> str:
    """Normalize whitespace (collapse newlines/tabs/runs of spaces)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join hyphenated line breaks like "sum-\nmarize" -> "summarize"
    text = re.sub(r"-\n\s*", "", text)
    # Collapse all remaining whitespace (including newlines) to single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences_in_paragraph(paragraph: str) -> List[str]:
    """Sentence-split a single paragraph (no blank lines inside it)."""
    text = clean_text(paragraph)
    if not text:
        return []

    # Protect decimal numbers (3.14) and known abbreviations from being
    # treated as sentence boundaries by temporarily replacing their dots
    # with a placeholder token that contains no whitespace and cannot
    # otherwise appear in the text, then restoring it after splitting.
    _DOT = "\x00DOT\x00"
    protected = text

    def _protect_decimal(match: "re.Match[str]") -> str:
        return match.group(0).replace(".", _DOT)

    protected = re.sub(r"\d+\.\d+", _protect_decimal, protected)

    def _protect_abbrev(match: "re.Match[str]") -> str:
        word = match.group(1)
        if word.lower().rstrip(".") in _ABBREVIATIONS:
            return word.replace(".", _DOT) + match.group(2)
        return match.group(0)

    protected = re.sub(r"\b([A-Za-z]{1,3}\.)(\s)", _protect_abbrev, protected)

    raw_sentences = _SENTENCE_SPLIT_RE.split(protected)

    sentences = []
    for s in raw_sentences:
        s = s.replace(_DOT, ".").strip()
        if s:
            sentences.append(s)
    return sentences


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences using a lightweight regex splitter.

    This is not a full NLP sentence tokenizer, but handles common
    abbreviations, decimals, and initials well enough for real
    articles without pulling in a heavy dependency like nltk/spacy.

    Blank-line paragraph breaks (including a title/heading line
    followed by a blank line, common in plain-text articles) are
    treated as hard sentence boundaries *before* whitespace is
    collapsed, so a heading with no terminal punctuation doesn't get
    silently glued onto the sentence that follows it.
    """
    if not text or not text.strip():
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", normalized)

    sentences: List[str] = []
    for paragraph in paragraphs:
        sentences.extend(_split_sentences_in_paragraph(paragraph))
    return sentences


def tokenize_words(text: str) -> List[str]:
    """Extract lowercase word tokens (letters/apostrophes/hyphens only)."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def _word_frequencies(sentences: Sequence[str]) -> Dict[str, float]:
    """Compute normalized word frequencies across all sentences, with
    stopwords removed. Frequencies are scaled 0..1 by the max count so
    scores stay comparable regardless of document length.
    """
    counts: Counter = Counter()
    for sentence in sentences:
        for word in tokenize_words(sentence):
            if word not in STOPWORDS and len(word) > 1:
                counts[word] += 1

    if not counts:
        return {}

    max_count = max(counts.values())
    return {word: count / max_count for word, count in counts.items()}


def score_sentences_by_frequency(sentences: Sequence[str]) -> List[float]:
    """Score each sentence by the average normalized frequency of its
    (non-stopword) words. This is a from-scratch implementation of the
    classic Luhn/word-frequency extractive scoring approach.
    """
    freqs = _word_frequencies(sentences)
    scores = []
    for sentence in sentences:
        words = [w for w in tokenize_words(sentence) if w in freqs]
        if not words:
            scores.append(0.0)
            continue
        scores.append(sum(freqs[w] for w in words) / len(words))
    return scores


def _sentence_similarity(a_words: Sequence[str], b_words: Sequence[str]) -> float:
    """Cosine-like overlap similarity between two bag-of-words sentences,
    normalized by sentence length (log-scaled, as in the original
    TextRank paper) to avoid favoring very long sentences.
    """
    set_a, set_b = set(a_words), set(b_words)
    if not set_a or not set_b:
        return 0.0
    overlap = len(set_a & set_b)
    if overlap == 0:
        return 0.0
    norm = math.log(len(set_a) + 1) + math.log(len(set_b) + 1)
    if norm == 0:
        return 0.0
    return overlap / norm


def score_sentences_by_textrank(
    sentences: Sequence[str],
    damping: float = 0.85,
    max_iter: int = 50,
    tolerance: float = 1e-4,
) -> List[float]:
    """A simplified TextRank implementation: build a sentence similarity
    graph and rank sentences with an iterative random-walk (the same
    fixed-point iteration PageRank uses), all from scratch.
    """
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    word_lists = [
        [w for w in tokenize_words(s) if w not in STOPWORDS and len(w) > 1]
        for s in sentences
    ]

    # Build a weighted adjacency (similarity) matrix.
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = _sentence_similarity(word_lists[i], word_lists[j])
            sim[i][j] = s
            sim[j][i] = s

    out_weights = [sum(row) for row in sim]

    ranks = [1.0 / n] * n
    for _ in range(max_iter):
        new_ranks = []
        for i in range(n):
            incoming = 0.0
            for j in range(n):
                if i == j or out_weights[j] == 0:
                    continue
                incoming += (sim[j][i] / out_weights[j]) * ranks[j]
            new_ranks.append((1 - damping) / n + damping * incoming)

        delta = sum(abs(a - b) for a, b in zip(new_ranks, ranks))
        ranks = new_ranks
        if delta < tolerance:
            break

    max_rank = max(ranks) if ranks else 1.0
    if max_rank == 0:
        return [0.0] * n
    return [r / max_rank for r in ranks]


def _length_to_sentence_count(total_sentences: int, length: str) -> int:
    """Translate a length preset ('short'/'medium'/'long') or a bare
    sentence-count string into an actual number of sentences to keep,
    scaled to the size of the source document.
    """
    if length is None:
        length = "medium"
    length = str(length).strip().lower()

    presets = {
        "short": 0.15,
        "medium": 0.30,
        "long": 0.50,
    }

    if length in presets:
        fraction = presets[length]
        n = max(1, round(total_sentences * fraction))
    elif length.isdigit():
        n = int(length)
    else:
        # Unknown value: fall back to medium rather than erroring, so the
        # summarizer degrades gracefully.
        n = max(1, round(total_sentences * presets["medium"]))

    return min(max(n, 1), total_sentences)


def summarize_text(
    text: str,
    length: str = "medium",
    method: str = "combined",
) -> str:
    """Produce an extractive summary of ``text``.

    Args:
        text: The source document (plain text).
        length: One of ``"short"``, ``"medium"``, ``"long"``, or a
            digit string giving an explicit number of sentences.
        method: ``"frequency"``, ``"textrank"``, or ``"combined"``
            (default) — which scoring strategy to rank sentences with.

    Returns:
        A summary composed of the top-ranked sentences, re-assembled
        in their original order so the result reads coherently. Returns
        an empty string if ``text`` has no extractable sentences.
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""

    if len(sentences) <= 2:
        # Nothing meaningful to compress further.
        return " ".join(sentences)

    freq_scores = score_sentences_by_frequency(sentences)
    rank_scores = score_sentences_by_textrank(sentences)

    if method == "frequency":
        combined = freq_scores
    elif method == "textrank":
        combined = rank_scores
    else:
        max_freq = max(freq_scores) or 1.0
        norm_freq = [s / max_freq for s in freq_scores]
        combined = [(f + r) / 2 for f, r in zip(norm_freq, rank_scores)]

    # Mild bias toward earlier sentences (leads often carry the thesis in
    # news/article writing) without letting it dominate the content score.
    position_boost = [1.0 - 0.1 * (i / max(1, len(sentences) - 1)) for i in range(len(sentences))]
    combined = [c * b for c, b in zip(combined, position_boost)]

    scored = [
        ScoredSentence(index=i, text=s, score=combined[i])
        for i, s in enumerate(sentences)
    ]

    n_keep = _length_to_sentence_count(len(sentences), length)
    top = sorted(scored, key=lambda x: x.score, reverse=True)[:n_keep]
    top_in_order = sorted(top, key=lambda x: x.index)

    # A chunk with no terminal punctuation (e.g. a title/heading line
    # pulled in as its own "sentence") reads oddly when glued directly
    # onto the sentence after it, so add a separating period purely for
    # display. This only affects the joined summary string, not the
    # sentence list itself.
    pieces = []
    for s in top_in_order:
        piece = s.text
        if piece and piece[-1] not in ".!?\"'”)":
            piece += "."
        pieces.append(piece)

    return " ".join(pieces)


def summary_stats(original: str, summary: str) -> Dict[str, float]:
    """Small helper used by the CLI/web UI to report compression stats."""
    orig_words = len(tokenize_words(original))
    sum_words = len(tokenize_words(summary))
    orig_sentences = len(split_sentences(original))
    sum_sentences = len(split_sentences(summary))
    reduction = 0.0
    if orig_words:
        reduction = round(100 * (1 - sum_words / orig_words), 1)
    return {
        "original_words": orig_words,
        "summary_words": sum_words,
        "original_sentences": orig_sentences,
        "summary_sentences": sum_sentences,
        "reduction_pct": reduction,
    }
