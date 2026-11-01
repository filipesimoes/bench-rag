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

        # Synthesize a concise deterministic answer
        if cited_docs:
            citation_str = ", ".join(cited_docs)
            sources_tag = (
                f"[Fontes: {citation_str}]" if is_portuguese else f"[Sources: {citation_str}]"
            )
            answer = f"{informative_text} {sources_tag}"
        else:
            answer = (
                "A informação não consta nas evidências fornecidas."
                if is_portuguese
                else "The information is not present in the provided records."
            )

        return GenerationResult(
            answer=answer,
            prompt=prompt,
            model_name=self.model_name,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(answer.split())},
        )
