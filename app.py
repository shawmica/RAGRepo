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
/* ── Reset & Base ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0c10 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    color: #e2e8f0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 0 16px 40px !important;
}

/* ── Header ───────────────────────────────────────────────── */
.site-header {
    padding: 48px 0 32px;
    text-align: center;
    border-bottom: 1px solid #1e2430;
    margin-bottom: 32px;
}

.site-header .wordmark {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #f8fafc;
    margin: 0 0 8px;
}

.site-header .wordmark span {
    color: #3b82f6;
}

.site-header .tagline {
    font-size: 0.925rem;
    color: #64748b;
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ── Section labels ───────────────────────────────────────── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin: 0 0 8px;
}

/* ── Panels ───────────────────────────────────────────────── */
.panel {
    background: #0f1420 !important;
    border: 1px solid #1e2430 !important;
    border-radius: 10px !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
}

/* ── Inputs ───────────────────────────────────────────────── */
textarea,
input[type="text"],
input[type="search"],
select {
    background: #070a0f !important;
    border: 1px solid #1e2430 !important;
    color: #e2e8f0 !important;
    border-radius: 7px !important;
    font-size: 0.9rem !important;
    font-family: inherit !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    padding: 10px 12px !important;
}

textarea:focus,
input[type="text"]:focus,
select:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    outline: none !important;
}

textarea::placeholder,
input[type="text"]::placeholder {
    color: #334155 !important;
}

/* ── Labels ───────────────────────────────────────────────── */
label > span,
.label-wrap > span {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #475569 !important;
}

/* ── Buttons — base ───────────────────────────────────────── */
button {
    font-family: inherit !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

/* Load */
.btn-load button {
    background: #3b82f6 !important;
    border: none !important;
    border-radius: 7px !important;
    color: #fff !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    padding: 11px 24px !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
}
.btn-load button:hover {
    background: #2563eb !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.3) !important;
    transform: translateY(-1px) !important;
}
.btn-load button:active { transform: translateY(0) !important; }

/* Ask */
.btn-ask button {
    background: #1d4ed8 !important;
    border: none !important;
    border-radius: 7px !important;
    color: #fff !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    padding: 11px 24px !important;
    width: 100% !important;
}
.btn-ask button:hover {
    background: #1e40af !important;
    box-shadow: 0 4px 16px rgba(29,78,216,0.35) !important;
    transform: translateY(-1px) !important;
}

/* Brief */
.btn-brief button {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 7px !important;
    color: #94a3b8 !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    padding: 11px 24px !important;
    width: 100% !important;
}
.btn-brief button:hover {
    background: #1e293b !important;
    border-color: #3b82f6 !important;
    color: #e2e8f0 !important;
    transform: translateY(-1px) !important;
}

/* Disabled state */
button:disabled,
button[disabled] {
    opacity: 0.38 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Quick chips ──────────────────────────────────────────── */
.chip button {
    background: transparent !important;
    border: 1px solid #1e2430 !important;
    border-radius: 100px !important;
    color: #475569 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 5px 14px !important;
    white-space: nowrap !important;
    width: auto !important;
}
.chip button:hover {
    border-color: #3b82f6 !important;
    color: #93c5fd !important;
    background: rgba(59,130,246,0.06) !important;
}

/* ── Status ───────────────────────────────────────────────── */
.status-ok  p { color: #22c55e !important; font-size: 0.82rem !important; margin: 0 !important; }
.status-err p { color: #ef4444 !important; font-size: 0.82rem !important; margin: 0 !important; }

/* ── Tabs ─────────────────────────────────────────────────── */
.tabs > .tab-nav {
    border-bottom: 1px solid #1e2430 !important;
    margin-bottom: 20px !important;
    gap: 0 !important;
}
.tabs > .tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #475569 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    margin-bottom: -1px !important;
    letter-spacing: 0.01em !important;
}
.tabs > .tab-nav button.selected {
    color: #f8fafc !important;
    border-bottom-color: #3b82f6 !important;
}
.tabs > .tab-nav button:hover:not(.selected) {
    color: #94a3b8 !important;
}

/* ── Output boxes ─────────────────────────────────────────── */
.output-box {
    background: #070a0f !important;
    border: 1px solid #1e2430 !important;
    border-radius: 8px !important;
    padding: 20px 22px !important;
    min-height: 140px !important;
    margin-top: 12px !important;
}
.output-box p {
    color: #cbd5e1 !important;
    line-height: 1.75 !important;
    font-size: 0.9rem !important;
    margin: 0 0 12px !important;
}
.output-box p:last-child { margin-bottom: 0 !important; }
.output-box code {
    background: #1e2430 !important;
    color: #7dd3fc !important;
    border-radius: 4px !important;
    padding: 0.15em 0.45em !important;
    font-size: 0.85em !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}
.output-box pre {
    background: #1e2430 !important;
    border-radius: 6px !important;
    padding: 14px !important;
    overflow-x: auto !important;
}
.output-box pre code { background: none !important; padding: 0 !important; }
.output-box h1, .output-box h2, .output-box h3 {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
    border-bottom: 1px solid #1e2430 !important;
    padding-bottom: 6px !important;
    margin: 24px 0 12px !important;
}
.output-box h1:first-child,
.output-box h2:first-child,
.output-box h3:first-child { margin-top: 0 !important; }
.output-box strong { color: #e2e8f0 !important; }
.output-box hr {
    border: none !important;
    border-top: 1px solid #1e2430 !important;
    margin: 16px 0 !important;
}

/* ── Dropdown ─────────────────────────────────────────────── */
.wrap.svelte-1p9xokt,
select {
    background: #070a0f !important;
    border: 1px solid #1e2430 !important;
    color: #e2e8f0 !important;
    border-radius: 7px !important;
}

/* ── Checkbox ─────────────────────────────────────────────── */
input[type="checkbox"] { accent-color: #3b82f6 !important; }

/* ── Divider ──────────────────────────────────────────────── */
.divider {
    border: none;
    border-top: 1px solid #1e2430;
    margin: 24px 0;
}

/* ── Mobile ───────────────────────────────────────────────── */
@media (max-width: 640px) {
    .gradio-container {
        padding: 0 12px 32px !important;
    }
    .site-header {
        padding: 32px 0 24px;
    }
    .site-header .wordmark {
        font-size: 1.4rem;
    }
    .site-header .tagline {
        font-size: 0.82rem;
    }
    .panel {
        padding: 14px !important;
    }
    .tabs > .tab-nav button {
        font-size: 0.8rem !important;
        padding: 9px 14px !important;
    }
    .chip button {
        font-size: 0.73rem !important;
        padding: 4px 10px !important;
    }
    .output-box {
        padding: 14px !important;
    }
    .output-box p {
        font-size: 0.85rem !important;
    }
    /* Stack the top row vertically on mobile */
    .top-row {
        flex-direction: column !important;
    }
}
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
        return (
            gr.update(value="Please enter a GitHub URL or local path.", elem_classes="status-err"),
            gr.update(interactive=False),
        )
    try:
        agent, chunks = _get_or_build(source.strip(), mode, use_mock)
        n_files = len(set(c.file for c in chunks))
        msg = f"Loaded successfully — {n_files} files, {len(chunks)} chunks, {mode} index."
        return gr.update(value=msg, elem_classes="status-ok"), gr.update(interactive=True)
    except Exception:
        return (
            gr.update(value="Could not load the repository. Please check the URL or path and try again.", elem_classes="status-err"),
            gr.update(interactive=False),
        )


def ask_question(source: str, mode: str, use_mock: bool, question: str):
    if not question.strip():
        return "Please enter a question."
    try:
        agent, _ = _get_or_build(source.strip(), mode, use_mock)
        ans = agent.ask(question)
        output = ans.answer
        if ans.citations:
            refs = "\n".join(f"- `{c}`" for c in ans.citations)
            output += f"\n\n---\n**Sources**\n{refs}"
        return output
    except Exception:
        return "Something went wrong. Please try again."


def generate_brief_md(source: str, mode: str, use_mock: bool):
    try:
        agent, _ = _get_or_build(source.strip(), mode, use_mock)
        b = generate_brief(agent)
        return b.to_markdown()
    except Exception:
        return "Could not generate the brief. Please make sure a repository is loaded and try again."


def build_app() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        button_primary_background_fill="#3b82f6",
        button_primary_background_fill_hover="#2563eb",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#1e293b",
        button_secondary_background_fill_hover="#334155",
        button_secondary_text_color="#94a3b8",
        button_secondary_border_color="#334155",
        body_background_fill="#0a0c10",
        body_text_color="#e2e8f0",
        background_fill_primary="#0f1420",
        background_fill_secondary="#070a0f",
        border_color_primary="#1e2430",
        border_color_accent="#3b82f6",
        input_background_fill="#070a0f",
        input_border_color="#1e2430",
        input_border_color_focus="#3b82f6",
        input_placeholder_color="#334155",
        block_background_fill="#0f1420",
        block_border_color="#1e2430",
        block_label_text_color="#475569",
        block_title_text_color="#e2e8f0",
        panel_background_fill="#0f1420",
        panel_border_color="#1e2430",
        checkbox_background_color="#070a0f",
        checkbox_border_color="#334155",
        checkbox_border_color_focus="#3b82f6",
        checkbox_background_color_selected="#3b82f6",
        color_accent_soft="#1e3a5f",
        link_text_color="#3b82f6",
        link_text_color_hover="#60a5fa",
    )
    with gr.Blocks(title="RAGRepo", css=CSS, theme=theme) as demo:

        # ── Header ──────────────────────────────────────────
        gr.HTML("""
        <div class="site-header">
            <h1 class="wordmark">RAG<span>Repo</span></h1>
            <p class="tagline">Understand any codebase instantly with retrieval-augmented generation</p>
        </div>
        """)

        # ── Repository loader ────────────────────────────────
        with gr.Group(elem_classes="panel"):
            gr.HTML('<p class="section-label">Repository</p>')
            with gr.Row(elem_classes="top-row", equal_height=True):
                source_box = gr.Textbox(
                    label="GitHub URL or Local Path",
                    placeholder="https://github.com/org/repo",
                    scale=5,
                    container=True,
                )
                mode_dd = gr.Dropdown(
                    choices=["bm25", "dense", "hybrid"],
                    value="bm25",
                    label="Retrieval Mode",
                    scale=1,
                )
                mock_cb = gr.Checkbox(
                    label="Mock LLM",
                    value=False,
                    scale=1,
                )
            load_btn = gr.Button(
                "Load Repository",
                variant="primary",
                elem_classes="btn-load",
            )
            status_md = gr.Markdown(value="", elem_classes="status-ok")

        # ── Tabs ─────────────────────────────────────────────
        with gr.Tabs():

            # Ask tab
            with gr.Tab("Ask a Question"):
                with gr.Group(elem_classes="panel"):
                    question_box = gr.Textbox(
                        label="Question",
                        placeholder="What does this codebase do?",
                        lines=2,
                        interactive=False,
                    )
                    gr.HTML('<p class="section-label" style="margin-top:12px">Quick questions</p>')
                    with gr.Row():
                        suggestion_btns = [
                            gr.Button(s, size="sm", elem_classes="chip") for s in SUGGESTIONS
                        ]
                    ask_btn = gr.Button(
                        "Ask",
                        variant="primary",
                        interactive=False,
                        elem_classes="btn-ask",
                    )

                answer_box = gr.Markdown(
                    value="Load a repository above to get started.",
                    elem_classes="output-box",
                )

            # Brief tab
            with gr.Tab("Onboarding Brief"):
                with gr.Group(elem_classes="panel"):
                    gr.HTML("""
                    <p style="color:#64748b;font-size:0.85rem;margin:0 0 14px">
                        Generates a structured document covering purpose, architecture,
                        entry points, dependencies, and where to start reading.
                    </p>
                    """)
                    brief_btn = gr.Button(
                        "Generate Brief",
                        variant="primary",
                        interactive=False,
                        elem_classes="btn-brief",
                    )

                brief_box = gr.Markdown(
                    value="Load a repository above, then click Generate Brief.",
                    elem_classes="output-box",
                )

        # ── Footer ───────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:24px 0 8px;color:#1e2430;font-size:0.75rem;
                    border-top:1px solid #1e2430;margin-top:32px;">
            RAGRepo &nbsp;·&nbsp; BM25 retrieval &nbsp;·&nbsp; Powered by Groq
        </div>
        """)

        # ── Events ───────────────────────────────────────────
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
_demo.launch(prevent_thread_lock=True)
app = _demo.app  # ASGI export for Vercel

if __name__ == "__main__":
    _demo.launch(share=False)
