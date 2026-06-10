"""Gradio web demo for the RAG-over-code pipeline."""
from __future__ import annotations
import gradio as gr
from ingest import ingest
from chunk import chunk_files
from retrieve import build_index
from agent import Agent
from brief import generate_brief, ONBOARDING_QUESTIONS
from llm import get_llm

_pipeline_cache: dict[str, tuple[Agent, list]] = {}


def _get_or_build(source: str, mode: str, mock: bool) -> tuple[Agent, list]:
    key = f"{source}::{mode}::{mock}"
    if key not in _pipeline_cache:
        files = ingest(source)
        chunks = chunk_files(files)
        index = build_index(chunks, mode=mode)
        llm = get_llm(mock=mock)
        agent = Agent(index=index, chunks=chunks, llm=llm)
        _pipeline_cache[key] = (agent, chunks)
    return _pipeline_cache[key]


def load_repo(source: str, mode: str, use_mock: bool):
    if not source.strip():
        return "Please enter a GitHub URL or local path.", gr.update(interactive=False)
    try:
        _get_or_build(source.strip(), mode, use_mock)
        return f"Loaded `{source.strip()}` with {mode} index.", gr.update(interactive=True)
    except Exception as e:
        return f"Error: {e}", gr.update(interactive=False)


def ask_question(source: str, mode: str, use_mock: bool, question: str):
    if not question.strip():
        return "Please enter a question."
    try:
        agent, _ = _get_or_build(source.strip(), mode, use_mock)
        ans = agent.ask(question)
        output = ans.answer
        if ans.citations:
            output += "\n\n**Citations:**\n" + "\n".join(f"- `{c}`" for c in ans.citations)
        return output
    except Exception as e:
        return f"Error: {e}"


def generate_brief_md(source: str, mode: str, use_mock: bool):
    try:
        agent, _ = _get_or_build(source.strip(), mode, use_mock)
        b = generate_brief(agent)
        return b.to_markdown()
    except Exception as e:
        return f"Error: {e}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAG-over-Code", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# RAG-over-Code\nAsk questions about any codebase.")

        with gr.Row():
            source_box = gr.Textbox(
                label="GitHub URL or local path",
                placeholder="https://github.com/org/repo  or  /path/to/project",
                scale=4,
            )
            mode_dd = gr.Dropdown(
                choices=["bm25", "dense", "hybrid"],
                value="bm25",
                label="Retrieval mode",
                scale=1,
            )
            mock_cb = gr.Checkbox(label="Mock LLM (offline)", value=False, scale=1)

        load_btn = gr.Button("Load Repository", variant="primary")
        status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Tabs():
            with gr.Tab("Ask a Question"):
                question_box = gr.Textbox(
                    label="Question",
                    placeholder="What does this codebase do?",
                    lines=2,
                    interactive=False,
                )
                ask_btn = gr.Button("Ask", interactive=False)
                answer_box = gr.Markdown(label="Answer")

            with gr.Tab("Onboarding Brief"):
                brief_btn = gr.Button("Generate Brief", interactive=False)
                brief_box = gr.Markdown(label="Brief")

        load_btn.click(
            fn=load_repo,
            inputs=[source_box, mode_dd, mock_cb],
            outputs=[status_box, question_box],
        ).then(
            fn=lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)),
            outputs=[ask_btn, brief_btn, question_box],
        )

        ask_btn.click(
            fn=ask_question,
            inputs=[source_box, mode_dd, mock_cb, question_box],
            outputs=[answer_box],
        )

        brief_btn.click(
            fn=generate_brief_md,
            inputs=[source_box, mode_dd, mock_cb],
            outputs=[brief_box],
        )

    return demo


_demo = build_app()
app = _demo.app  # ASGI app for Vercel

if __name__ == "__main__":
    _demo.launch(share=False)
