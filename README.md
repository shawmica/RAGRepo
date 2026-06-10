# RAG-over-Code

A small RAG (Retrieval-Augmented Generation) demo for asking questions about codebases.

## Features

- Ingests a repository or local code path, chunks files, builds a retrieval index (BM25 / dense / hybrid).
- Agent selects relevant chunks and answers questions with verified citations.
- CLI and Gradio web demo included.

## Quickstart (local)

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
# On PowerShell
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2. Set your LLM environment variable (recommended):

- Anthropic (Claude):

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

- Google Gemini (Generative Language API):

```powershell
$env:GEMINI_API_KEY = "AQ..."
```

If you don't have a provider key or want to run offline, use `--mock` with the CLI to use the deterministic `MockLLM`.

3. CLI examples

```bash
# Generate a brief (mock mode)
python cli.py brief . --mock -o brief.md

# Ask a question (mock)
python cli.py ask . "What does this codebase do?" --mock
```

4. Run the Gradio demo locally

```bash
python app.py
# then open the printed URL, e.g. http://127.0.0.1:7860
```

## Environment files

- Add a `.env` file in the project root with secrets (do not commit this file):

```
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
```

The repo already includes a `.gitignore` that excludes `.env` and `eval/env`.

## Vercel deployment troubleshooting

During deployment you may see an error like:

```
Error: Found app.py but it does not export a top-level "app", "application", or "handler" variable.
```

Why this happens

- Vercel's Python serverless runtime expects a top-level ASGI/WSGI `app`/`application`/`handler` variable it can call as the entrypoint.
- `app.py` in this repository launches a local Gradio server when executed directly and does not expose a top-level ASGI app variable, so Vercel refuses to deploy it as-is.

Recommended fixes

1. Deploy elsewhere (fastest):
   - Run the Gradio demo on a VM/host or use `vercel dev` locally and keep the production deployment on a host that supports long-running processes (e.g. a VM, Heroku, Render, or a Docker container).

2. Make an ASGI entrypoint for Vercel (example using FastAPI):

- Create a file `api/index.py` (Vercel treats files under `api/` as functions) with the following pattern:

```python
from fastapi import FastAPI
import gradio as gr
from app import build_app

app = FastAPI()
# build_app returns a `gr.Blocks` object
demo = build_app()
# mount the Gradio interface at root
gr.mount_gradio_app(app, demo, path="/")

# Now Vercel will find `app` as the FastAPI ASGI application
```

Notes:
- You may need to add `fastapi` and `uvicorn` to `requirements.txt`.
- Vercel serverless functions have execution time limits — Gradio often expects a long-running process, so using a VM or a container is usually more reliable.

## CI / GitHub

I pushed this repository to `https://github.com/shawmica/RAGRepo`. If you want a GitHub Action or Vercel-specific config, I can add it.

## Next steps I can help with

- Add the `api/index.py` FastAPI wrapper and required deps and push it (I can do this for you).
- Add a short `README` badge and usage examples tailored to Vercel or Docker.
- Create a `Dockerfile` for reliable deployment.

If you'd like me to add the FastAPI wrapper and deployable files, tell me and I'll add and push them to `main`.
# RAGRepo

Lightweight RAG (retrieval-augmented generation) pipeline for exploring codebases.

## Features
- Ingest source files and split into code chunks
- Build BM25 (and optional dense/hybrid) retrieval indexes
- Agent loop: retrieve → LLM selects chunks → answer with verified citations
- Supports `MockLLM`, Anthropic, and Gemini backends

## Requirements
- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file in the repo root (this file must NOT be committed):

```
GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

2. Add `.env` to `.gitignore` (already included).

## Usage

Run the CLI from the repo root.

- Generate a brief (mock LLM):

```bash
python cli.py brief . --mock -o brief.md
```

- Ask a single question (mock LLM):

```bash
python cli.py ask . "What does this codebase do?" --mock
```

- Run evaluation harness (mock LLM):

```bash
python cli.py eval . --mock
```

To use a real model, set the appropriate env var (example for PowerShell):

```powershell
$env:GEMINI_API_KEY = "<your_gemini_key>"
python cli.py ask . "What does this codebase do?"
```

Notes:
- `ANTHROPIC_API_KEY` enables Anthropic Claude backend in `llm.py`.
- `GEMINI_API_KEY` enables Google Gemini backend in the repo's other LLM file.
- If you hit quota/billing errors, enable billing or use `--mock`.

## Contributing
- Open an issue or PR on GitHub: https://github.com/shawmica/RAGRepo

## License
MIT
