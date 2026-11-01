"""Deterministic mock generator for testing and dry-run benchmarks."""

import re

from bench_rag.core.generator import GenerationResult, GeneratorLLM


class MockGeneratorLLM(GeneratorLLM):
    """Deterministic generator that synthesizes answers from provided context."""

    def __init__(self, model_name: str = "mock-generator"):
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        # Extract context blocks formatted as [Document: <id>] ...
        doc_matches = re.findall(r"\[Document:\s*([^\]]+)\]", prompt)
        cited_docs = sorted(set(doc_matches))

        # Extract context paragraphs
        context_lines = [
            line.strip()
            for line in prompt.splitlines()
            if (
                line.strip()
                and not line.startswith("Question:")
                and not line.startswith("Context:")
                and not line.startswith("Instructions:")
                and not line.startswith("Answer:")
            )
        ]
        informative_text = (
            " ".join(context_lines[:3]) if context_lines else "Forensic evidence evaluated."
        )

        # Check language preference: Portuguese or English
        is_portuguese = (
            "fontes:" in prompt.lower()
            or "em português" in prompt.lower()
            or "pergunta" in prompt.lower()
        )

        # Check if structured JSON response is requested
        wants_json = (
            "json" in prompt.lower()
            or (system_prompt and "json" in system_prompt.lower())
            or "schema" in prompt.lower()
            or (system_prompt and "schema" in system_prompt.lower())
        )

        base_answer = (
            informative_text
            if cited_docs
            else (
                "A informação não consta nas evidências fornecidas."
                if is_portuguese
                else "The information is not present in the provided records."
            )
        )

        if wants_json:
            import json

            answer = json.dumps(
                {"answer": base_answer, "citations": cited_docs},
                ensure_ascii=False,
            )
        else:
            if cited_docs:
                citation_str = ", ".join(cited_docs)
                sources_tag = (
                    f"[Fontes: {citation_str}]" if is_portuguese else f"[Sources: {citation_str}]"
                )
                answer = f"{base_answer} {sources_tag}"
            else:
                answer = base_answer

        return GenerationResult(
            answer=answer,
            prompt=prompt,
            model_name=self.model_name,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(answer.split())},
        )
