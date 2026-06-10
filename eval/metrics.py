"""Scoring functions for RAG evaluation."""
from __future__ import annotations
import re
from agent import AgentAnswer, Citation


def citation_accuracy(answer: AgentAnswer) -> float:
    """Fraction of claims that have a verifiable citation (citations / total sentences)."""
    sentences = re.split(r"[.!?]+", answer.answer)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[source:", s))
    return cited / len(sentences)


def citation_validity(answer: AgentAnswer, all_chunk_ids: set[str]) -> float:
    """Fraction of citations that point to real chunks (fabrication check)."""
    if not answer.citations:
        return 1.0  # no citations = no fabrications
    valid = sum(1 for c in answer.citations if c.chunk_id in all_chunk_ids)
    return valid / len(answer.citations)


def keyword_recall(answer: AgentAnswer, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer."""
    if not expected_keywords:
        return 1.0
    text = answer.answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in text)
    return found / len(expected_keywords)


def score_answer(
    answer: AgentAnswer,
    all_chunk_ids: set[str],
    expected_keywords: list[str] | None = None,
) -> dict[str, float]:
    return {
        "citation_accuracy": citation_accuracy(answer),
        "citation_validity": citation_validity(answer, all_chunk_ids),
        "keyword_recall": keyword_recall(answer, expected_keywords or []),
    }
