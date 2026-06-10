"""Evaluation harness: compare agent vs naive baseline on questions with known answers."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from agent import Agent, AgentAnswer
from chunk import Chunk
from llm import Message, get_llm
from eval.metrics import score_answer


# ── Eval dataset ────────────────────────────────────────────────────────────
# Each entry: question + keywords that a good answer should mention.
EVAL_QUESTIONS: list[dict] = [
    {
        "question": "What does this codebase do?",
        "keywords": [],  # populated dynamically per repo
    },
    {
        "question": "What are the main entry points?",
        "keywords": [],
    },
    {
        "question": "What external libraries does this project use?",
        "keywords": [],
    },
]


# ── Naive baseline ───────────────────────────────────────────────────────────
NAIVE_SYSTEM = """You are a code understanding assistant.
Answer the question based solely on the code snippets provided.
Do not cite sources."""


def naive_answer(llm, chunks: list[Chunk], question: str, k: int = 5) -> AgentAnswer:
    """Baseline: just concatenate the first k chunks and ask the LLM."""
    context = "\n\n".join(
        f"=== {c.file}:{c.start}-{c.end} ===\n{c.content}" for c in chunks[:k]
    )
    response = llm.chat(
        messages=[Message(role="user", content=f"Question: {question}\n\nCode:\n{context}")],
        system=NAIVE_SYSTEM,
        max_tokens=1024,
        thinking=False,
    )
    return AgentAnswer(
        question=question,
        answer=response.content,
        citations=[],
        selected_chunk_ids=[c.id for c in chunks[:k]],
        raw_response=response.content,
    )


# ── Runner ───────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    question: str
    agent_scores: dict[str, float]
    baseline_scores: dict[str, float]
    agent_answer: str
    baseline_answer: str


def run_eval(
    agent: Agent,
    chunks: list[Chunk],
    questions: list[dict] | None = None,
    mock: bool = False,
) -> list[EvalResult]:
    qs = questions or EVAL_QUESTIONS
    all_chunk_ids = {c.id for c in chunks}
    llm = get_llm(mock=mock)
    results: list[EvalResult] = []

    for q_entry in qs:
        question = q_entry["question"]
        keywords = q_entry.get("keywords", [])

        agent_ans = agent.ask(question)
        base_ans = naive_answer(llm, chunks, question)

        agent_scores = score_answer(agent_ans, all_chunk_ids, keywords)
        base_scores = score_answer(base_ans, all_chunk_ids, keywords)

        results.append(
            EvalResult(
                question=question,
                agent_scores=agent_scores,
                baseline_scores=base_scores,
                agent_answer=agent_ans.answer,
                baseline_answer=base_ans.answer,
            )
        )

    return results


def print_table(results: list[EvalResult]) -> None:
    metrics = ["citation_accuracy", "citation_validity", "keyword_recall"]
    header = f"{'Question':<45} {'Metric':<22} {'Agent':>8} {'Baseline':>10} {'Delta':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        q_short = r.question[:42] + "..." if len(r.question) > 42 else r.question
        for i, metric in enumerate(metrics):
            a = r.agent_scores.get(metric, 0.0)
            b = r.baseline_scores.get(metric, 0.0)
            delta = a - b
            sign = "+" if delta >= 0 else ""
            q_col = q_short if i == 0 else ""
            print(f"{q_col:<45} {metric:<22} {a:>8.2f} {b:>10.2f} {sign}{delta:>7.2f}")
        print()
