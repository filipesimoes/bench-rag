"""Retrieval metrics calculation against ground-truth inquiry references."""

from bench_rag.core.evaluator import RetrievalMetrics


def compute_retrieval_metrics(
    retrieved_inquiries: list[str],
    ground_truth_references: list[str],
    k: int = 5,
) -> RetrievalMetrics:
    """Compute Precision@K, Recall@K, HitRate@K, and MRR.

    Args:
        retrieved_inquiries: Ranked list of retrieved document identifiers.
        ground_truth_references: Expected ground-truth document identifiers.
        k: Cutoff rank.

    Returns:
        RetrievalMetrics dataclass.
    """
    # Normalize strings (lowercase, stripped)
    clean_gt = {ref.strip().lower() for ref in ground_truth_references if ref.strip()}
    clean_retrieved = [doc.strip().lower() for doc in retrieved_inquiries if doc.strip()]

    if not clean_gt:
        # No ground truth required
        return RetrievalMetrics(precision_at_k=1.0, recall_at_k=1.0, hit_rate=1.0, mrr=1.0)

    top_k_retrieved = clean_retrieved[:k]
    matches = [doc for doc in top_k_retrieved if doc in clean_gt]
    unique_matches = set(matches)

    # Precision@K
    effective_k = max(1, min(k, len(top_k_retrieved)) if top_k_retrieved else k)
    precision = len(unique_matches) / effective_k

    # Recall@K
    recall = len(unique_matches) / len(clean_gt)

    # Hit Rate @ K
    hit_rate = 1.0 if unique_matches else 0.0

    # Mean Reciprocal Rank (MRR)
    mrr = 0.0
    for rank, doc in enumerate(top_k_retrieved, start=1):
        if doc in clean_gt:
            mrr = 1.0 / rank
            break

    return RetrievalMetrics(
        precision_at_k=round(min(1.0, precision), 4),
        recall_at_k=round(min(1.0, recall), 4),
        hit_rate=round(hit_rate, 4),
        mrr=round(mrr, 4),
    )
