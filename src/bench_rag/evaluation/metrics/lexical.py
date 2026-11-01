"""Lexical metrics: ROUGE-L, Token F1, and BLEU-4."""

import math
import re
from collections import Counter

from bench_rag.core.evaluator import LexicalMetrics


def _tokenize(text: str) -> list[str]:
    """Tokenize alphanumeric words, normalized to lowercase."""
    return re.findall(r"\w+", text.lower())


def calculate_token_f1(candidate: str, reference: str) -> float:
    """Compute token-level harmonic F1 score."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

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
    """Compute ROUGE-L using Longest Common Subsequence (LCS)."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

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
    beta = 1.2
    f_lcs = ((1 + beta**2) * precision * recall) / (recall + (beta**2) * precision)
    return round(f_lcs, 4)


def calculate_bleu_4(candidate: str, reference: str) -> float:
    """Compute sentence-level BLEU-4 with add-1 smoothing."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return 1.0 if cand_tokens == ref_tokens else 0.0

    c_len = len(cand_tokens)
    r_len = len(ref_tokens)

    # Brevity penalty
    if c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1 - (r_len / c_len))

    # N-gram precisions (1-gram to 4-gram)
    precisions = []
    for n in range(1, 5):
        cand_ngrams = [tuple(cand_tokens[i : i + n]) for i in range(c_len - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i : i + n]) for i in range(r_len - n + 1)]

        if not cand_ngrams:
            precisions.append(1.0 / (c_len + 1))  # Smoothing
            continue

        cand_counter = Counter(cand_ngrams)
        ref_counter = Counter(ref_ngrams)
        overlap = sum((cand_counter & ref_counter).values())

        if n == 1 and overlap == 0:
            return 0.0

        # Add-1 smoothing
        smoothed_precision = (overlap + 1.0) / (len(cand_ngrams) + 1.0)
        precisions.append(smoothed_precision)

    log_sum = sum(math.log(p) for p in precisions) / 4.0
    bleu = bp * math.exp(log_sum)
    return round(min(1.0, bleu), 4)


def compute_lexical_metrics(candidate: str, reference: str) -> LexicalMetrics:
    """Compute comprehensive lexical metrics."""
    return LexicalMetrics(
        rouge_l=calculate_rouge_l(candidate, reference),
        token_f1=calculate_token_f1(candidate, reference),
        bleu_4=calculate_bleu_4(candidate, reference),
    )
