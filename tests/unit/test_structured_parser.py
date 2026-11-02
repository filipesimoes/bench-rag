"""Unit tests for StructuredResponseParser."""

from bench_rag.core.structured_output import StructuredResponseParser


def test_parse_strict_json():
    raw = '{"answer": "Carlos Mendes is nominee director.", "citations": ["IP_001_2023_FRAUD"]}'
    parsed = StructuredResponseParser.parse(raw)

    assert parsed.answer == "Carlos Mendes is nominee director."
    assert parsed.cited_inquiries == ["IP_001_2023_FRAUD"]
    assert parsed.raw_text == raw


def test_parse_json_markdown_block():
    raw = """Here is the result:
```json
{
  "answer": "Orion conducted cash smurfing deposits.",
  "citations": ["IP_002_2023_LAUNDERING", "IP_003_2023_CONSPIRACY"]
}
```
"""
    parsed = StructuredResponseParser.parse(raw)

    assert parsed.answer == "Orion conducted cash smurfing deposits."
    assert parsed.cited_inquiries == ["IP_002_2023_LAUNDERING", "IP_003_2023_CONSPIRACY"]


def test_parse_json_with_comma_separated_string_citations():
    raw = '{"answer": "Evidence synthesized.", "citations": "IP_001_2023_FRAUD, IP_002_2023_LAUNDERING"}'
    parsed = StructuredResponseParser.parse(raw)

    assert parsed.answer == "Evidence synthesized."
    assert parsed.cited_inquiries == ["IP_001_2023_FRAUD", "IP_002_2023_LAUNDERING"]


def test_parse_bracketed_tag_fallback_portuguese():
    raw = "Carlos Mendes é o diretor da empresa. [Fontes: IP_001_2023_FRAUD]"
    parsed = StructuredResponseParser.parse(raw)

    assert parsed.answer == "Carlos Mendes é o diretor da empresa."
    assert parsed.cited_inquiries == ["IP_001_2023_FRAUD"]


def test_parse_bracketed_tag_fallback_english():
    raw = "Wire transfers were executed. [Sources: IP_001_2023_FRAUD, IP_003_2023_CONSPIRACY]"
    parsed = StructuredResponseParser.parse(raw)

    assert parsed.answer == "Wire transfers were executed."
    assert parsed.cited_inquiries == ["IP_001_2023_FRAUD", "IP_003_2023_CONSPIRACY"]


def test_parse_plain_text_without_citations():
    raw = "Information is not present in the records."
    parsed = StructuredResponseParser.parse(raw)

    assert parsed.answer == "Information is not present in the records."
    assert parsed.cited_inquiries == []


def test_parse_empty_input():
    parsed = StructuredResponseParser.parse("")
    assert parsed.answer == ""
    assert parsed.cited_inquiries == []


def test_parse_json_with_outer_braces_in_preamble():
    raw = (
        'Note {synthetic preamble}:\n'
        '{"answer": "TEST AGENT was the designated officer.", "citations": ["IP_001_2023_FRAUD"]}\n'
        'Trailing {tag}'
    )
    parsed = StructuredResponseParser.parse(raw)
    assert parsed.answer == "TEST AGENT was the designated officer."
    assert parsed.cited_inquiries == ["IP_001_2023_FRAUD"]

