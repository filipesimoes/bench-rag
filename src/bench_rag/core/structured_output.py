"""Structured response models and resilient parser for RAG generators."""

import json
import re
from dataclasses import dataclass, field


@dataclass
class ParsedGeneration:
    """Normalized structured output from a generator LLM."""

    answer: str
    cited_inquiries: list[str] = field(default_factory=list)
    raw_text: str = ""


class StructuredResponseParser:
    """Resilient parser that extracts clean answer text and citation document IDs.

    Supported patterns:
    1. Strict JSON object: {"answer": "...", "citations": ["..."]}
    2. Markdown JSON code block: ```json { ... } ```
    3. Bracketed tag fallback: "Text... [Fontes: ID1, ID2]" or "[Sources: ID1, ID2]"
    4. Plain text without citations.
    """

    CITATION_REGEX = re.compile(
        r"\[(?:Fontes?|Sources?):\s*([^\]]+)\]",
        re.IGNORECASE,
    )

    @classmethod
    def parse(cls, raw_text: str) -> ParsedGeneration:
        """Parse raw generator completion into clean answer and citations."""
        if not raw_text or not raw_text.strip():
            return ParsedGeneration(answer="", cited_inquiries=[], raw_text=raw_text or "")

        text = raw_text.strip()

        # 1. Attempt JSON parsing (direct or within ```json ... ``` blocks)
        candidate_json: str | None = None
        json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_block_match:
            candidate_json = json_block_match.group(1).strip()
        else:
            # Check for outermost JSON object if curly braces are present
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                candidate_json = text[first_brace : last_brace + 1].strip()

        if candidate_json:
            try:
                data = json.loads(candidate_json)
                if isinstance(data, dict) and "answer" in data:
                    raw_ans = str(data.get("answer", "")).strip()
                    citations_field = (
                        data.get("citations")
                        or data.get("cited_documents")
                        or data.get("sources")
                        or []
                    )
                    parsed_citations: list[str] = []
                    if isinstance(citations_field, str):
                        parsed_citations = [
                            c.strip() for c in citations_field.split(",") if c.strip()
                        ]
                    elif isinstance(citations_field, list):
                        parsed_citations = [
                            str(c).strip() for c in citations_field if str(c).strip()
                        ]

                    # Strip any accidental brackets or tags inside the answer itself
                    clean_ans = cls.CITATION_REGEX.sub("", raw_ans).strip()

                    return ParsedGeneration(
                        answer=clean_ans,
                        cited_inquiries=parsed_citations,
                        raw_text=raw_text,
                    )
            except Exception:
                pass

        # 2. Fallback: Parse bracketed tags [Fontes: ...] or [Sources: ...]
        citations: list[str] = []
        matches = cls.CITATION_REGEX.findall(text)
        for match in matches:
            for item in match.split(","):
                clean_item = item.strip()
                if clean_item and clean_item not in citations:
                    citations.append(clean_item)

        clean_answer = cls.CITATION_REGEX.sub("", text).strip()
        return ParsedGeneration(
            answer=clean_answer,
            cited_inquiries=citations,
            raw_text=raw_text,
        )
