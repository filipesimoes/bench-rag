"""Lexical metrics: ROUGE-L, Token F1, and BLEU-4."""

import math
import re
from collections import Counter

from bench_rag.core.evaluator import LexicalMetrics

_TOKEN_PATTERN = re.compile(r"\d+(?:[.,/-]\d+)*|\w+")


def tokenize_text(text: str) -> list[str]:
    """Tokenize alphanumeric words and compound numeric identifiers, normalized to lowercase."""
    return _TOKEN_PATTERN.findall(text.lower())


def _tokenize(text: str) -> list[str]:
    """Backward-compatible alias for tokenize_text."""
    return tokenize_text(text)


def calculate_token_f1(candidate: str, reference: str) -> float:
    """Compute token-level harmonic F1 score."""
    cand_tokens = tokenize_text(candidate)
    ref_tokens = tokenize_text(reference)

    if not cand_tokens or not ref_tokens:
        return 1.0 if cand_tokens == ref_tokens else 0.0

    cand_counts = Counter(cand_tokens)
    ref_counts = Counter(ref_tokens)

    overlap = sum((cand_counts & ref_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)


def calculate_rouge_l(candidate: str, reference: str) -> float:
    """Compute symmetric ROUGE-L F1 score (beta = 1.0) using Longest Common Subsequence (LCS)."""
    cand_tokens = tokenize_text(candidate)
    ref_tokens = tokenize_text(reference)

    m, n = len(cand_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 1.0 if m == n else 0.0

    # DP matrix for LCS length
    dp = [0] * (n + 1)
    for c_tok in cand_tokens:
        prev = 0
        for j, r_tok in enumerate(ref_tokens, start=1):
            temp = dp[j]
            if c_tok == r_tok:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    lcs_len = dp[n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    f_lcs = (2 * precision * recall) / (precision + recall)
    return round(f_lcs, 4)


def calculate_bleu_4(candidate: str, reference: str) -> float:
    """Compute sentence-level BLEU-4 with Chen & Cherry smoothing and short-sequence support."""
    cand_tokens = tokenize_text(candidate)
    ref_tokens = tokenize_text(reference)

    if not cand_tokens or not ref_tokens:
        return 1.0 if cand_tokens == ref_tokens else 0.0

    c_len = len(cand_tokens)
    r_len = len(ref_tokens)

    # Brevity penalty
    if c_len >= r_len:
        bp = 1.0
    else:
        bp = math.exp(1 - (r_len / c_len))

    max_n = min(4, c_len, r_len)
    precisions: list[float] = []

    for n in range(1, max_n + 1):
        cand_ngrams = [tuple(cand_tokens[i : i + n]) for i in range(c_len - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i : i + n]) for i in range(r_len - n + 1)]

        cand_counter = Counter(cand_ngrams)
        ref_counter = Counter(ref_ngrams)
        overlap = sum((cand_counter & ref_counter).values())

        if n == 1:
            if overlap == 0:
                return 0.0
            precisions.append(overlap / len(cand_ngrams))
        else:
            if overlap == 0:
                # Smoothing for zero-overlap higher-order n-grams (n > 1)
                smoothed_precision = 1.0 / (len(cand_ngrams) + 1.0)
                precisions.append(smoothed_precision)
            else:
                precisions.append(overlap / len(cand_ngrams))

    log_sum = sum(math.log(p) for p in precisions) / max_n
    bleu = bp * math.exp(log_sum)
    return round(min(1.0, bleu), 4)


def compute_lexical_metrics(candidate: str, reference: str) -> LexicalMetrics:
    """Compute comprehensive lexical metrics."""
    return LexicalMetrics(
        rouge_l=calculate_rouge_l(candidate, reference),
        token_f1=calculate_token_f1(candidate, reference),
        bleu_4=calculate_bleu_4(candidate, reference),
    )
