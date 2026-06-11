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
    background: #eff6ff !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    color: #1e3a5f !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container {
    max-width: 920px !important;
    margin: 0 auto !important;
    padding: 0 20px 48px !important;
}

/* ── Header ───────────────────────────────────────────────── */
.site-header {
    padding: 52px 0 36px;
    text-align: center;
    margin-bottom: 28px;
}

.site-header .wordmark {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: #1e3a8a;
    margin: 0 0 10px;
}

.site-header .wordmark span {
    color: #2563eb;
}

.site-header .badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 100px;
    margin-bottom: 14px;
}

.site-header .tagline {
    font-size: 0.95rem;
    color: #3b82f6;
    margin: 0;
    font-weight: 400;
}

/* ── Section labels ───────────────────────────────────────── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #93c5fd;
    margin: 0 0 10px;
}

/* ── Cards / panels ───────────────────────────────────────── */
.panel {
    background: #ffffff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 14px !important;
    padding: 24px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 1px 4px rgba(37,99,235,0.07) !important;
}

/* ── Inputs ───────────────────────────────────────────────── */
textarea,
input[type="text"],
input[type="search"],
select {
    background: #f0f7ff !important;
    border: 1.5px solid #bfdbfe !important;
    color: #1e3a5f !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    font-family: inherit !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    padding: 10px 13px !important;
}

textarea:focus,
input[type="text"]:focus,
select:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
    outline: none !important;
    background: #fff !important;
}

textarea::placeholder,
input[type="text"]::placeholder {
    color: #93c5fd !important;
}

/* ── Labels ───────────────────────────────────────────────── */
label > span,
.label-wrap > span {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #3b82f6 !important;
}

/* ── Buttons — base ───────────────────────────────────────── */
button {
    font-family: inherit !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

/* Load */
.btn-load button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-size: 0.875rem !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
}
.btn-load button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.35) !important;
    transform: translateY(-1px) !important;
}
.btn-load button:active { transform: translateY(0) !important; }

/* Ask */
.btn-ask button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-size: 0.875rem !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
}
.btn-ask button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.35) !important;
    transform: translateY(-1px) !important;
}

/* Brief */
.btn-brief button {
    background: #eff6ff !important;
    border: 1.5px solid #93c5fd !important;
    border-radius: 8px !important;
    color: #1d4ed8 !important;
    font-size: 0.875rem !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    width: 100% !important;
}
.btn-brief button:hover {
    background: #dbeafe !important;
    border-color: #2563eb !important;
    color: #1e40af !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.15) !important;
}

/* Disabled state */
button:disabled,
button[disabled] {
    opacity: 0.4 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Quick chips ──────────────────────────────────────────── */
.chip button {
    background: #eff6ff !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 100px !important;
    color: #2563eb !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 5px 14px !important;
    white-space: nowrap !important;
    width: auto !important;
}
.chip button:hover {
    border-color: #2563eb !important;
    background: #dbeafe !important;
    color: #1d4ed8 !important;
}

/* ── Status ───────────────────────────────────────────────── */
.status-ok  p { color: #16a34a !important; font-size: 0.83rem !important; margin: 0 !important; font-weight: 500 !important; }
.status-err p { color: #dc2626 !important; font-size: 0.83rem !important; margin: 0 !important; font-weight: 500 !important; }

/* ── Tabs ─────────────────────────────────────────────────── */
.tabs > .tab-nav {
    border-bottom: 2px solid #bfdbfe !important;
    margin-bottom: 20px !important;
    gap: 0 !important;
}
.tabs > .tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #93c5fd !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    margin-bottom: -2px !important;
    letter-spacing: 0.01em !important;
}
.tabs > .tab-nav button.selected {
    color: #1d4ed8 !important;
    border-bottom-color: #2563eb !important;
}
.tabs > .tab-nav button:hover:not(.selected) {
    color: #3b82f6 !important;
}

/* ── Output boxes ─────────────────────────────────────────── */
.output-box {
    background: #f0f7ff !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 10px !important;
    padding: 22px 24px !important;
    min-height: 140px !important;
    margin-top: 14px !important;
}
.output-box p {
    color: #1e3a5f !important;
    line-height: 1.8 !important;
    font-size: 0.9rem !important;
    margin: 0 0 12px !important;
}
.output-box p:last-child { margin-bottom: 0 !important; }
.output-box code {
    background: #dbeafe !important;
    color: #1d4ed8 !important;
    border-radius: 5px !important;
    padding: 0.15em 0.45em !important;
    font-size: 0.85em !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}
.output-box pre {
    background: #1e3a8a !important;
    border-radius: 8px !important;
    padding: 16px !important;
    overflow-x: auto !important;
}
.output-box pre code {
    background: none !important;
    padding: 0 !important;
    color: #bfdbfe !important;
}
.output-box h1, .output-box h2, .output-box h3 {
    color: #1e3a8a !important;
    font-weight: 700 !important;
    border-bottom: 1.5px solid #bfdbfe !important;
    padding-bottom: 6px !important;
    margin: 24px 0 12px !important;
}
.output-box h1:first-child,
.output-box h2:first-child,
.output-box h3:first-child { margin-top: 0 !important; }
.output-box strong { color: #1e3a8a !important; }
.output-box hr {
    border: none !important;
    border-top: 1.5px solid #bfdbfe !important;
    margin: 16px 0 !important;
}

/* ── Dropdown ─────────────────────────────────────────────── */
.wrap.svelte-1p9xokt,
select {
    background: #f0f7ff !important;
    border: 1.5px solid #bfdbfe !important;
    color: #1e3a5f !important;
    border-radius: 8px !important;
}

/* ── Checkbox ─────────────────────────────────────────────── */
input[type="checkbox"] { accent-color: #2563eb !important; }

/* ── Divider ──────────────────────────────────────────────── */
.divider {
    border: none;
    border-top: 1.5px solid #bfdbfe;
    margin: 24px 0;
}

/* ── Info strip ───────────────────────────────────────────── */
.info-strip {
    display: flex;
    gap: 12px;
    margin-bottom: 28px;
    flex-wrap: wrap;
}
.info-strip .chip-stat {
    background: #dbeafe;
    color: #1d4ed8;
    border: 1px solid #93c5fd;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

/* ── Mobile ───────────────────────────────────────────────── */
@media (max-width: 640px) {
    .gradio-container {
        padding: 0 12px 32px !important;
    }
    .site-header {
        padding: 36px 0 28px;
    }
    .site-header .wordmark {
        font-size: 1.6rem;
    }
    .panel {
        padding: 16px !important;
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
        padding: 16px !important;
    }
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


def _build_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#eff6ff",
        button_secondary_background_fill_hover="#dbeafe",
        button_secondary_text_color="#1d4ed8",
        button_secondary_border_color="#93c5fd",
        body_background_fill="#eff6ff",
        body_text_color="#1e3a5f",
        background_fill_primary="#ffffff",
        background_fill_secondary="#f0f7ff",
        border_color_primary="#bfdbfe",
        border_color_accent="#2563eb",
        input_background_fill="#f0f7ff",
        input_border_color="#bfdbfe",
        input_border_color_focus="#2563eb",
        input_placeholder_color="#93c5fd",
        block_background_fill="#ffffff",
        block_border_color="#bfdbfe",
        block_label_text_color="#3b82f6",
        block_title_text_color="#1e3a5f",
        panel_background_fill="#ffffff",
        panel_border_color="#bfdbfe",
        checkbox_background_color="#f0f7ff",
        checkbox_border_color="#bfdbfe",
        checkbox_border_color_focus="#2563eb",
        checkbox_background_color_selected="#2563eb",
        color_accent_soft="#dbeafe",
        link_text_color="#2563eb",
        link_text_color_hover="#1d4ed8",
    )


def build_app() -> gr.Blocks:
    # Gradio 6: theme/css must be passed to launch(), not the Blocks constructor.
    with gr.Blocks(title="RAGRepo") as demo:

        # ── Header ──────────────────────────────────────────
        gr.HTML("""
        <div class="site-header">
            <div class="badge">RAG · Code Intelligence</div>
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
                    gr.HTML('<p class="section-label" style="margin-top:14px">Quick questions</p>')
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
                    <p style="color:#3b82f6;font-size:0.875rem;margin:0 0 16px;line-height:1.6">
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
        <div style="text-align:center;padding:24px 0 8px;color:#93c5fd;font-size:0.75rem;
                    border-top:1.5px solid #bfdbfe;margin-top:36px;">
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
_demo.launch(prevent_thread_lock=True, theme=_build_theme(), css=CSS)
app = _demo.app  # ASGI export for Vercel

if __name__ == "__main__":
    _demo.launch(share=False, theme=_build_theme(), css=CSS)
