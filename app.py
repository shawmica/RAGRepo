"""Gradio web demo for the RAG-over-code pipeline."""
from __future__ import annotations
import gradio as gr
from ingest import ingest
from chunk import chunk_files
from retrieve import build_index
from agent import Agent
from brief import generate_brief
from llm import get_llm

_pipeline_cache: dict[str, tuple[Agent, list]] = {}

CSS = """
/* ── Global ── */
body, .gradio-container {
    background: #0f1117 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Header ── */
.rag-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.rag-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6ee7b7, #3b82f6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}
.rag-header p {
    color: #8b949e;
    font-size: 1rem;
    margin: 0;
}

/* ── Cards ── */
.card {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
}

/* ── Input boxes ── */
textarea, input[type=text] {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}
textarea:focus, input[type=text]:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}

/* ── Labels ── */
label span, .label-wrap span {
    color: #8b949e !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ── Load button ── */
.load-btn button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.load-btn button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(46,160,67,0.4) !important;
}

/* ── Ask button ── */
.ask-btn button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.ask-btn button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(56,139,253,0.4) !important;
}

/* ── Brief button ── */
.brief-btn button {
    background: linear-gradient(135deg, #6e40c9, #8957e5) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.brief-btn button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(137,87,229,0.4) !important;
}

/* ── Suggestion chip buttons ── */
.suggestion-chip button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 20px !important;
    color: #8b949e !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.25rem 0.85rem !important;
    transition: all 0.15s !important;
    white-space: nowrap !important;
}
.suggestion-chip button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
    color: #58a6ff !important;
}

/* ── Status message ── */
.status-ok p  { color: #3fb950 !important; font-size: 0.9rem !important; }
.status-err p { color: #f85149 !important; font-size: 0.9rem !important; }

/* ── Answer / Brief output ── */
.answer-box, .brief-box {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 1.25rem !important;
    color: #e6edf3 !important;
    min-height: 120px;
}
.answer-box p, .brief-box p { color: #c9d1d9 !important; line-height: 1.7 !important; }
.answer-box code, .brief-box code {
    background: #21262d !important;
    color: #79c0ff !important;
    border-radius: 4px !important;
    padding: 0.1em 0.4em !important;
    font-size: 0.88em !important;
}
.answer-box h1, .answer-box h2, .answer-box h3,
.brief-box h1, .brief-box h2, .brief-box h3 {
    color: #e6edf3 !important;
    border-bottom: 1px solid #21262d !important;
    padding-bottom: 0.3rem !important;
}

/* ── Tabs ── */
.tabs > .tab-nav button {
    color: #8b949e !important;
    font-weight: 600 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.tabs > .tab-nav button.selected {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
}

/* ── Dropdown ── */
.dropdown > label > select {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

/* ── Checkbox ── */
input[type=checkbox] { accent-color: #58a6ff !important; }
"""

SUGGESTIONS = [
    "What does this codebase do?",
    "What are the main entry points?",
    "What external libraries does this use?",
    "Where should a new developer start?",
    "What are the core data models?",
]


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
        return gr.update(value="Please enter a GitHub URL or local path.", elem_classes="status-err"), gr.update(interactive=False)
    try:
        agent, chunks = _get_or_build(source.strip(), mode, use_mock)
        n_files = len(set(c.file for c in chunks))
        msg = f"Repository loaded — {n_files} files, {len(chunks)} chunks, {mode} index."
        return gr.update(value=msg, elem_classes="status-ok"), gr.update(interactive=True)
    except Exception:
        return gr.update(value="Could not load the repository. Please check the URL or path and try again.", elem_classes="status-err"), gr.update(interactive=False)


def ask_question(source: str, mode: str, use_mock: bool, question: str):
    if not question.strip():
        return "Please enter a question."
    try:
        agent, _ = _get_or_build(source.strip(), mode, use_mock)
        ans = agent.ask(question)
        output = ans.answer
        if ans.citations:
            refs = "\n".join(f"- `{c}`" for c in ans.citations)
            output += f"\n\n---\n**Citations**\n{refs}"
        return output
    except Exception:
        return "Something went wrong while processing your question. Please try again."


def generate_brief_md(source: str, mode: str, use_mock: bool):
    try:
        agent, _ = _get_or_build(source.strip(), mode, use_mock)
        b = generate_brief(agent)
        return b.to_markdown()
    except Exception:
        return "Could not generate the brief. Please make sure a repository is loaded and try again."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAGRepo") as demo:

        # ── Header ──────────────────────────────────────────────
        gr.HTML("""
        <div class="rag-header">
            <h1>RAGRepo</h1>
            <p>Ask anything about any codebase — powered by retrieval-augmented generation</p>
        </div>
        """)

        # ── Repo loader card ─────────────────────────────────────
        with gr.Group(elem_classes="card"):
            with gr.Row(equal_height=True):
                source_box = gr.Textbox(
                    label="GitHub URL or Local Path",
                    placeholder="https://github.com/org/repo   or   C:/path/to/project",
                    scale=5,
                )
                mode_dd = gr.Dropdown(
                    choices=["bm25", "dense", "hybrid"],
                    value="bm25",
                    label="Retrieval Mode",
                    scale=1,
                )
                mock_cb = gr.Checkbox(label="Mock LLM", value=False, scale=1)

            load_btn = gr.Button("Load Repository", variant="primary", elem_classes="load-btn")
            status_md = gr.Markdown(value="", elem_classes="status-ok")

        # ── Tabs ─────────────────────────────────────────────────
        with gr.Tabs():

            # ── Ask tab ──
            with gr.Tab("Ask a Question"):
                with gr.Group(elem_classes="card"):
                    question_box = gr.Textbox(
                        label="Your Question",
                        placeholder="What does this codebase do?",
                        lines=2,
                        interactive=False,
                    )
                    gr.HTML('<div style="color:#8b949e;font-size:0.78rem;font-weight:600;margin:6px 0 4px;text-transform:uppercase;letter-spacing:.05em">Quick questions</div>')
                    with gr.Row():
                        suggestion_btns = [
                            gr.Button(s, size="sm", elem_classes="suggestion-chip") for s in SUGGESTIONS
                        ]
                    ask_btn = gr.Button("Ask", variant="primary", interactive=False, elem_classes="ask-btn")

                answer_box = gr.Markdown(
                    value="Load a repository above to get started.",
                    elem_classes="answer-box",
                )

            # ── Brief tab ──
            with gr.Tab("Onboarding Brief"):
                with gr.Group(elem_classes="card"):
                    gr.Markdown("Generate a complete onboarding document covering purpose, structure, entry points, dependencies, and more.")
                    brief_btn = gr.Button("Generate Brief", variant="primary", interactive=False, elem_classes="brief-btn")

                brief_box = gr.Markdown(
                    value="Load a repository above, then click Generate Brief.",
                    elem_classes="brief-box",
                )

        # ── Events ───────────────────────────────────────────────
        load_btn.click(
            fn=load_repo,
            inputs=[source_box, mode_dd, mock_cb],
            outputs=[status_md, question_box],
        ).then(
            fn=lambda: (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            ),
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

        for btn, suggestion in zip(suggestion_btns, SUGGESTIONS):
            btn.click(fn=lambda s=suggestion: s, outputs=[question_box])

    return demo


_demo = build_app()
_demo.launch(
    prevent_thread_lock=True,
    css=CSS,
    theme=gr.themes.Base(),
)
app = _demo.app  # ASGI export for Vercel

if __name__ == "__main__":
    _demo.launch(share=False, css=CSS, theme=gr.themes.Base())
