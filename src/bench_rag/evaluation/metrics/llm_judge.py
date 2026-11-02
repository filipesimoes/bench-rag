"""Mandatory LLM-as-a-Judge metric evaluation."""

import json
import logging
import re
from typing import Any

from bench_rag.config import settings
from bench_rag.core.evaluator import LLMJudgeMetrics
from bench_rag.core.generator import GeneratorLLM
from bench_rag.evaluation.metrics.lexical import calculate_token_f1
from bench_rag.generators.mock_generator import MockGeneratorLLM
from bench_rag.generators.openai_generator import OpenAIGeneratorLLM

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a rigorous, impartial forensic intelligence auditor.
Your job is to evaluate a RAG system's response to an investigative query about police inquiry reports.
Evaluate the response according to four criteria, scoring each from 0.0 to 1.0:
1. Faithfulness: Is every factual claim in the Generated Answer grounded in and substantiated by the Retrieved Context? (1.0 = fully faithful, 0.0 = severe hallucination).
2. Answer Relevance: Does the Generated Answer directly and comprehensively address the Question? (1.0 = highly relevant, 0.0 = irrelevant or evasive).
3. Context Recall: Does the Retrieved Context contain the essential investigative facts present in the Reference Answer? (1.0 = full context recall, 0.0 = retrieved context missed all key facts).
4. Answer Correctness: Does the Generated Answer capture the essential investigative facts present in the Reference Answer? (1.0 = fully accurate answer, 0.0 = missed or contradicted all key facts).

Respond ONLY with a valid JSON object matching this schema:
{
  "faithfulness": 0.85,
  "answer_relevance": 0.90,
  "context_recall": 0.80,
  "answer_correctness": 0.85,
  "critique": "Short explanation of the evaluation"
}
"""


def _clamp_score(value: Any) -> float:
    """Validate and clamp a judge metric score to [0.0, 1.0]."""
    score = float(value)
    return round(min(1.0, max(0.0, score)), 4)


class LLMJudge:
    """Evaluates RAG quality using LLM-as-a-Judge."""

    def __init__(self, generator: GeneratorLLM):
        self.generator = generator

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        reference_answer: str,
    ) -> LLMJudgeMetrics:
        """Score generated answer against question, context, and reference answer."""
        # Fast deterministic path for Mock generator
        if getattr(self.generator, "model_name", "").startswith("mock"):
            return self._mock_score(question, answer, context, reference_answer)

        prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            f"Ground Truth Reference Answer:\n{reference_answer}\n\n"
            f"Provide your JSON evaluation:"
        )

        gen_res = self.generator.generate(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
        )
        data = self._extract_json(gen_res.answer)

        required_keys = (
            "faithfulness",
            "answer_relevance",
            "context_recall",
            "answer_correctness",
        )
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise KeyError(f"LLM Judge response missing required metric keys: {missing}")

        faithfulness = _clamp_score(data["faithfulness"])
        answer_relevance = _clamp_score(data["answer_relevance"])
        context_recall = _clamp_score(data["context_recall"])
        answer_correctness = _clamp_score(data["answer_correctness"])

        return LLMJudgeMetrics(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_recall=context_recall,
            answer_correctness=answer_correctness,
            critique=str(data.get("critique", "LLM judge completed evaluation.")),
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
        )

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON object from LLM response text."""
        stripped = text.strip()
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if code_block:
            return json.loads(code_block.group(1))

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", stripped):
            try:
                obj, _ = decoder.raw_decode(stripped[match.start() :])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

        return json.loads(stripped)

    def _mock_score(
        self,
        question: str,
        answer: str,
        context: str,
        reference_answer: str,
    ) -> LLMJudgeMetrics:
        """Deterministic heuristic scoring for hermetic test execution."""
        ans_ref_f1 = calculate_token_f1(answer, reference_answer)
        ctx_ref_f1 = calculate_token_f1(context, reference_answer)
        ans_ctx_f1 = calculate_token_f1(answer, context)
        q_f1 = calculate_token_f1(answer, question)

        faithfulness = min(1.0, max(0.6, ans_ctx_f1 * 1.5 + 0.4))
        answer_relevance = min(1.0, max(0.6, q_f1 * 1.5 + 0.5))
        context_recall = min(1.0, max(0.5, ctx_ref_f1 * 1.2 + 0.3))
        answer_correctness = min(1.0, max(0.5, ans_ref_f1 * 1.2 + 0.3))

        mock_prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            f"Ground Truth Reference Answer:\n{reference_answer}\n\n"
            f"Provide your JSON evaluation:"
        )

        return LLMJudgeMetrics(
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(answer_relevance, 4),
            context_recall=round(context_recall, 4),
            answer_correctness=round(answer_correctness, 4),
            critique="Deterministic mock judge evaluation based on forensic evidence alignment.",
            prompt=mock_prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
        )


def get_judge(
    judge_type: str = "mock",
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    enable_thinking: bool | None = None,
    dialect: str | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> LLMJudge:
    """Factory function to instantiate an LLMJudge."""
    match judge_type.lower():
        case "mock":
            gen = MockGeneratorLLM(model_name=model_name or "mock-judge")
            return LLMJudge(generator=gen)
        case "openai":
            model = model_name or settings.judge_model
            resolved_url = base_url or settings.effective_judge_base_url
            resolved_key = api_key or settings.effective_judge_api_key
            gen = OpenAIGeneratorLLM(
                model=model,
                base_url=resolved_url,
                api_key=resolved_key,
                enable_thinking=(
                    enable_thinking
                    if enable_thinking is not None
                    else settings.judge_enable_thinking
                ),
                dialect=dialect or settings.judge_dialect,
                max_tokens=max_tokens or settings.judge_max_tokens,
                timeout=timeout if timeout is not None else settings.judge_timeout,
                max_retries=max_retries if max_retries is not None else settings.http_max_retries,
            )
            return LLMJudge(generator=gen)
        case _:
            raise ValueError(f"Unsupported judge type: {judge_type}. Expected 'mock' or 'openai'.")
