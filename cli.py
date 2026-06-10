"""Command-line interface for the RAG-over-code pipeline."""
from __future__ import annotations
import sys
import click

from ingest import ingest
from chunk import chunk_files
from retrieve import build_index
from agent import Agent
from brief import generate_brief
from llm import get_llm


def _build_pipeline(source: str, mode: str, mock: bool):
    click.echo(f"Ingesting {source} ...")
    files = ingest(source)
    click.echo(f"  {len(files)} source files found")

    click.echo("Chunking ...")
    chunks = chunk_files(files)
    click.echo(f"  {len(chunks)} chunks created")

    click.echo(f"Building {mode} index ...")
    index = build_index(chunks, mode=mode)

    llm = get_llm(mock=mock)
    agent = Agent(index=index, chunks=chunks, llm=llm)
    return agent, chunks


@click.group()
def cli():
    """RAG-over-code: ask questions about any codebase."""


@cli.command()
@click.argument("source")
@click.option("--mode", default="bm25", type=click.Choice(["bm25", "dense", "hybrid"]), show_default=True)
@click.option("--mock", is_flag=True, help="Use mock LLM (no API key needed)")
@click.option("--output", "-o", default=None, help="Write brief to a file (default: stdout)")
@click.option("--format", "fmt", default="markdown", type=click.Choice(["markdown", "text"]), show_default=True)
def brief(source: str, mode: str, mock: bool, output: str | None, fmt: str):
    """Generate an onboarding brief for a repository."""
    agent, _ = _build_pipeline(source, mode, mock)
    click.echo("Generating brief ...")
    b = generate_brief(agent)
    content = b.to_markdown() if fmt == "markdown" else b.to_text()
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"Brief written to {output}")
    else:
        click.echo(content)


@cli.command()
@click.argument("source")
@click.argument("question")
@click.option("--mode", default="bm25", type=click.Choice(["bm25", "dense", "hybrid"]), show_default=True)
@click.option("--mock", is_flag=True, help="Use mock LLM (no API key needed)")
def ask(source: str, question: str, mode: str, mock: bool):
    """Ask a single question about a repository."""
    agent, _ = _build_pipeline(source, mode, mock)
    click.echo(f"\nQ: {question}\n")
    ans = agent.ask(question)
    click.echo(ans.answer)
    if ans.citations:
        click.echo("\nCitations:")
        for c in ans.citations:
            click.echo(f"  {c}")


@cli.command()
@click.argument("source")
@click.option("--mode", default="bm25", type=click.Choice(["bm25", "dense", "hybrid"]), show_default=True)
@click.option("--mock", is_flag=True, help="Use mock LLM (no API key needed)")
def eval(source: str, mode: str, mock: bool):
    """Run the evaluation harness and print the comparison table."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from eval.run import run_eval, print_table

    agent, chunks = _build_pipeline(source, mode, mock)
    click.echo("Running evaluation ...")
    results = run_eval(agent, chunks, mock=mock)
    print_table(results)


if __name__ == "__main__":
    cli()
