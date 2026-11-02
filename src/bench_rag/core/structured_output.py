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
        parsed_data: dict | None = None
        json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_block_match:
            try:
                obj = json.loads(json_block_match.group(1).strip())
                if isinstance(obj, dict) and "answer" in obj:
                    parsed_data = obj
            except Exception:
                pass

        if parsed_data is None:
            decoder = json.JSONDecoder()
            for match in re.finditer(r"\{", text):
                try:
                    obj, _ = decoder.raw_decode(text[match.start() :])
                    if isinstance(obj, dict) and "answer" in obj:
                        parsed_data = obj
                        break
                except json.JSONDecodeError:
                    continue

        if parsed_data is not None:
            raw_ans = str(parsed_data.get("answer", "")).strip()
            citations_field = (
                parsed_data.get("citations")
                or parsed_data.get("cited_documents")
                or parsed_data.get("sources")
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
