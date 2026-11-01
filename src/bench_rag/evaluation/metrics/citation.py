"""Citation attribution metrics calculation against ground-truth inquiry references."""

from bench_rag.core.evaluator import CitationMetrics


def compute_citation_metrics(
    cited_inquiries: list[str],
    ground_truth_references: list[str],
    retrieved_inquiries: list[str] | None = None,
) -> CitationMetrics:
    """Compute citation precision, recall, and context hallucination rate.

    Args:
        cited_inquiries: List of document identifiers cited by the generator.
        ground_truth_references: Expected ground-truth document identifiers.
        retrieved_inquiries: Optional list of document identifiers retrieved by the retriever.

    Returns:
        CitationMetrics dataclass.
    """
    clean_gt = {ref.strip().lower() for ref in ground_truth_references if ref.strip()}
    clean_cited = [doc.strip().lower() for doc in cited_inquiries if doc.strip()]
    unique_cited = set(clean_cited)
    clean_retrieved = (
        {doc.strip().lower() for doc in retrieved_inquiries if doc.strip()}
        if retrieved_inquiries is not None
        else None
    )

    cited_count = len(unique_cited)

    # Case 1: No ground truth references expected
    if not clean_gt:
        if not unique_cited:
            return CitationMetrics(
                precision=1.0,
                recall=1.0,
                hallucination_rate=0.0,
                cited_count=0,
            )
        hallucination_rate = (
            len(unique_cited - clean_retrieved) / cited_count
            if clean_retrieved is not None and cited_count > 0
            else 0.0
        )
        return CitationMetrics(
            precision=0.0,
            recall=1.0,
            hallucination_rate=round(hallucination_rate, 4),
            cited_count=cited_count,
        )

    # Case 2: Model cited nothing
    if not unique_cited:
        return CitationMetrics(
            precision=0.0,
            recall=0.0,
            hallucination_rate=0.0,
            cited_count=0,
        )

    # Case 3: Standard evaluation
    matches = unique_cited & clean_gt
    precision = len(matches) / cited_count
    recall = len(matches) / len(clean_gt)

    # Hallucination: cited documents that were NOT in the retrieved context
    if clean_retrieved is not None:
        hallucinated_docs = unique_cited - clean_retrieved
        hallucination_rate = len(hallucinated_docs) / cited_count
    else:
        # Fallback if retrieved context is not passed: docs not in ground truth
        hallucinated_docs = unique_cited - clean_gt
        hallucination_rate = len(hallucinated_docs) / cited_count

    return CitationMetrics(
        precision=round(min(1.0, precision), 4),
        recall=round(min(1.0, recall), 4),
        hallucination_rate=round(min(1.0, hallucination_rate), 4),
        cited_count=cited_count,
    )
