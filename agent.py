"""Core agent loop: retrieve → LLM picks chunks → answer with verified citations."""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from typing import Any

from chunk import Chunk
from llm import LLMResponse, get_llm


@dataclass
class Citation:
    chunk_id: str
    file: str
    start: int
    end: int

    def __str__(self) -> str:
        return f"{self.file}:{self.start}-{self.end}"


@dataclass
class AgentAnswer:
    question: str
    answer: str
    citations: list[Citation]
    selected_chunk_ids: list[str]
    raw_response: str


SELECTION_SYSTEM = """You are a code understanding assistant.
You will be given a list of code chunks with IDs and a user question.
Reply with a JSON array of chunk IDs (e.g. ["chunk_0","chunk_3"]) that are
most relevant to answer the question. Include at most 8 chunks. Return ONLY the JSON array."""

ANSWER_SYSTEM = """You are a code understanding assistant.
Answer the question using ONLY the provided code chunks.
Every factual claim must cite the source chunk using the format [source: chunk_ID].
Be concise and technical. If the chunks don't contain enough information, say so."""


def _format_chunks_for_selection(chunks: list[Chunk], max_preview: int = 200) -> str:
    lines = []
    for c in chunks:
        preview = c.content[:max_preview].replace("\n", " ")
        lines.append(f"[{c.id}] {c.file}:{c.start}-{c.end} ({c.chunk_type}) — {preview}")
    return "\n".join(lines)


def _format_chunks_for_answer(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"=== [{c.id}] {c.file}:{c.start}-{c.end} ===\n{c.content}"
        )
    return "\n\n".join(parts)


def _parse_selected_ids(text: str) -> list[str]:
    # Try to extract JSON array from the response
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            ids = json.loads(match.group())
            return [str(i) for i in ids if isinstance(i, str)]
        except json.JSONDecodeError:
            pass
    # Fallback: extract anything that looks like chunk_N
    return re.findall(r"chunk_\d+", text)


def _extract_citations(answer_text: str, chunk_map: dict[str, Chunk]) -> list[Citation]:
    """Extract [source: chunk_ID] markers and verify they exist in chunk_map."""
    raw_ids = re.findall(r"\[source:\s*(chunk_\d+)\]", answer_text)
    seen: set[str] = set()
    citations: list[Citation] = []
    for cid in raw_ids:
        if cid in seen:
            continue
        seen.add(cid)
        chunk = chunk_map.get(cid)
        if chunk is None:
            # fabricated — drop it
            continue
        citations.append(
            Citation(chunk_id=cid, file=chunk.file, start=chunk.start, end=chunk.end)
        )
    return citations


def _verify_citations(answer_text: str, citations: list[Citation]) -> str:
    """Remove [source: X] tags for citations that were dropped (fabricated)."""
    valid_ids = {c.chunk_id for c in citations}
    def replace(m: re.Match) -> str:
        cid = m.group(1).strip()
        return m.group(0) if cid in valid_ids else ""
    return re.sub(r"\[source:\s*(chunk_\d+)\]", replace, answer_text)


class Agent:
    def __init__(
        self,
        index,
        chunks: list[Chunk],
        llm=None,
        retrieve_k: int = 20,
        select_top: int = 8,
    ):
        self.index = index
        self.chunk_map: dict[str, Chunk] = {c.id: c for c in chunks}
        self.llm = llm or get_llm()
        self.retrieve_k = retrieve_k
        self.select_top = select_top

    def ask(self, question: str) -> AgentAnswer:
        # Step 1: retrieve candidates
        candidates = self.index.search(question, k=self.retrieve_k)
        if not candidates:
            return AgentAnswer(
                question=question,
                answer="No relevant code found for this question.",
                citations=[],
                selected_chunk_ids=[],
                raw_response="",
            )

        # Step 2: LLM picks which candidates to read
        selection_prompt = (
            f"Question: {question}\n\n"
            f"Available chunks:\n{_format_chunks_for_selection(candidates)}"
        )
        from llm import Message
        sel_response = self.llm.chat(
            messages=[Message(role="user", content=selection_prompt)],
            system=SELECTION_SYSTEM,
            max_tokens=512,
            thinking=False,
        )
        selected_ids = _parse_selected_ids(sel_response.content)
        # Clamp to available candidates
        candidate_ids = {c.id for c in candidates}
        selected_ids = [cid for cid in selected_ids if cid in candidate_ids][: self.select_top]
        if not selected_ids:
            selected_ids = [c.id for c in candidates[: self.select_top]]

        selected_chunks = [self.chunk_map[cid] for cid in selected_ids if cid in self.chunk_map]

        # Step 3: LLM answers using only selected chunks
        answer_prompt = (
            f"Question: {question}\n\n"
            f"Code chunks:\n{_format_chunks_for_answer(selected_chunks)}"
        )
        ans_response = self.llm.chat(
            messages=[Message(role="user", content=answer_prompt)],
            system=ANSWER_SYSTEM,
            max_tokens=2048,
            thinking=True,
        )

        # Step 4: verify and strip fabricated citations
        citations = _extract_citations(ans_response.content, self.chunk_map)
        clean_answer = _verify_citations(ans_response.content, citations)

        return AgentAnswer(
            question=question,
            answer=clean_answer,
            citations=citations,
            selected_chunk_ids=selected_ids,
            raw_response=ans_response.content,
        )
