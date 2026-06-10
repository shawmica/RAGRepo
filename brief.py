"""Generate an onboarding brief by running the agent over fixed questions."""
from __future__ import annotations
from dataclasses import dataclass
from agent import Agent, AgentAnswer

ONBOARDING_QUESTIONS = [
    "What does this codebase do? What is its primary purpose?",
    "What is the overall structure and architecture of this project?",
    "What are the main entry points (e.g. main(), CLI commands, API endpoints)?",
    "Where should a new developer start reading the code?",
    "What are the core data models, classes, or types used throughout the codebase?",
    "What external dependencies does this project rely on?",
    "How is the project configured and deployed?",
]


@dataclass
class Brief:
    answers: list[AgentAnswer]

    def to_markdown(self) -> str:
        sections = ["# Codebase Onboarding Brief\n"]
        for ans in self.answers:
            sections.append(f"## {ans.question}\n")
            sections.append(ans.answer)
            if ans.citations:
                refs = ", ".join(f"`{c}`" for c in {str(c) for c in ans.citations})
                sections.append(f"\n*Sources: {refs}*")
            sections.append("")
        return "\n".join(sections)

    def to_text(self) -> str:
        sections = ["CODEBASE ONBOARDING BRIEF", "=" * 40]
        for ans in self.answers:
            sections.append(f"\n{ans.question}")
            sections.append("-" * len(ans.question))
            sections.append(ans.answer)
            if ans.citations:
                refs = ", ".join(str(c) for c in ans.citations)
                sections.append(f"Sources: {refs}")
        return "\n".join(sections)


def generate_brief(agent: Agent, questions: list[str] | None = None) -> Brief:
    qs = questions or ONBOARDING_QUESTIONS
    answers: list[AgentAnswer] = []
    for q in qs:
        ans = agent.ask(q)
        answers.append(ans)
    return Brief(answers=answers)
