"""Document cleaners and sanitization modules."""

from typing import TYPE_CHECKING

from bench_rag.cleaners.header_footer_cleaner import HeaderFooterCleaner
from bench_rag.core.cleaner import DocumentCleaner

if TYPE_CHECKING:
    from bench_rag.config import PipelineConfig


def get_cleaner(config: "PipelineConfig") -> DocumentCleaner | None:
    """Instantiate a DocumentCleaner from a PipelineConfig specification."""
    if not getattr(config, "clean_headers_footers", False):
        return None

    return HeaderFooterCleaner(
        header_margin=getattr(config, "header_margin", 50.0),
        footer_margin=getattr(config, "footer_margin", 40.0),
        detect_repeated_headers=getattr(config, "detect_repeated_headers", True),
        repeated_header_threshold=getattr(config, "repeated_header_threshold", 0.3),
        remove_pje_metadata=getattr(config, "remove_pje_metadata", True),
        remove_page_numbers=getattr(config, "remove_page_numbers", True),
        preserve_first_header=getattr(config, "preserve_first_header", True),
        custom_header_patterns=getattr(config, "custom_header_patterns", []),
        custom_footer_patterns=getattr(config, "custom_footer_patterns", []),
    )


__all__ = ["DocumentCleaner", "HeaderFooterCleaner", "get_cleaner"]
