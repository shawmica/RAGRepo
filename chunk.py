"""Split source files into retrievable chunks with citation metadata."""
from __future__ import annotations
import ast, re
from dataclasses import dataclass, field
from typing import Sequence

from ingest import SourceFile


@dataclass
class Chunk:
    id: str           # unique, e.g. "chunk_42"
    file: str         # relative path
    start: int        # 1-based line
    end: int          # 1-based line (inclusive)
    content: str
    chunk_type: str   # "function" | "class" | "imports" | "module" | "window"
    symbol: str = "" # function or class name if applicable

    def citation(self) -> str:
        return f"{self.file}:{self.start}-{self.end}"


# ── Python AST chunker ──────────────────────────────────────────────────────

def _line_offsets(source: str) -> list[int]:
    """Byte offsets for each line start (0-indexed)."""
    offsets = [0]
    for ch in source:
        if ch == "\n":
            offsets.append(offsets[-1] + 1)
        else:
            offsets[-1] += 1
    return offsets


def _node_lines(node: ast.AST) -> tuple[int, int]:
    return getattr(node, "lineno", 1), getattr(node, "end_lineno", 1)


def _chunk_python(file: str, content: str) -> list[Chunk]:
    lines = content.splitlines()
    total = len(lines)
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _sliding_window(file, content, "python")

    chunks: list[Chunk] = []
    covered: set[int] = set()

    # collect top-level functions and classes (and methods inside classes)
    top_nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_nodes.append(node)

    # only keep top-level and direct class members
    def _is_top_or_method(node: ast.AST) -> bool:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        if isinstance(node, ast.ClassDef):
            return True
        return False

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start, end = _node_lines(node)
        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
        sym = node.name
        snippet = "\n".join(lines[start - 1 : end])
        chunks.append(
            Chunk(
                id="",
                file=file,
                start=start,
                end=end,
                content=snippet,
                chunk_type=chunk_type,
                symbol=sym,
            )
        )
        covered.update(range(start, end + 1))

    # imports preamble — lines not yet covered at top of file
    preamble_lines = [l for i, l in enumerate(lines, 1) if i not in covered and i < (min(covered) if covered else total + 1)]
    if preamble_lines:
        first = 1
        last = len(preamble_lines)
        chunks.insert(
            0,
            Chunk(
                id="",
                file=file,
                start=first,
                end=last,
                content="\n".join(preamble_lines),
                chunk_type="imports",
            ),
        )
        covered.update(range(first, last + 1))

    # remaining lines (e.g. module-level code after functions)
    uncovered = [i for i in range(1, total + 1) if i not in covered]
    if uncovered:
        runs = _consecutive_runs(uncovered)
        for run in runs:
            snippet = "\n".join(lines[run[0] - 1 : run[-1]])
            chunks.append(
                Chunk(
                    id="",
                    file=file,
                    start=run[0],
                    end=run[-1],
                    content=snippet,
                    chunk_type="module",
                )
            )

    return chunks


def _consecutive_runs(nums: list[int]) -> list[list[int]]:
    if not nums:
        return []
    runs: list[list[int]] = [[nums[0]]]
    for n in nums[1:]:
        if n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    return runs


# ── Sliding window for non-Python ────────────────────────────────────────────

WINDOW = 60   # lines per window
OVERLAP = 15  # overlap between consecutive windows


def _sliding_window(file: str, content: str, lang: str) -> list[Chunk]:
    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return []

    chunks: list[Chunk] = []
    start = 1
    while start <= total:
        end = min(start + WINDOW - 1, total)
        snippet = "\n".join(lines[start - 1 : end])
        chunks.append(
            Chunk(
                id="",
                file=file,
                start=start,
                end=end,
                content=snippet,
                chunk_type="window",
            )
        )
        if end == total:
            break
        start = end - OVERLAP + 1

    return chunks


# ── Structure-aware sliding window (detect function-like boundaries) ─────────

_FUNC_PATTERNS = {
    "javascript": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+\w+|^\s*(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\("),
    "typescript": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+\w+|^\s*(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\("),
    "java": re.compile(r"^\s*(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+\w+\s*\("),
    "go": re.compile(r"^\s*func\s+"),
    "rust": re.compile(r"^\s*(?:pub\s+)?fn\s+"),
    "ruby": re.compile(r"^\s*def\s+"),
    "kotlin": re.compile(r"^\s*(?:fun|suspend fun)\s+"),
}


def _structure_aware_window(file: str, content: str, lang: str) -> list[Chunk]:
    pattern = _FUNC_PATTERNS.get(lang)
    if pattern is None:
        return _sliding_window(file, content, lang)

    lines = content.splitlines()
    total = len(lines)
    boundaries = [i + 1 for i, line in enumerate(lines) if pattern.match(line)]
    if not boundaries:
        return _sliding_window(file, content, lang)

    # Build segments between boundaries
    segments: list[tuple[int, int]] = []
    for idx, b in enumerate(boundaries):
        start = b
        end = boundaries[idx + 1] - 1 if idx + 1 < len(boundaries) else total
        segments.append((start, end))

    chunks: list[Chunk] = []
    # anything before first boundary
    if boundaries[0] > 1:
        pre = "\n".join(lines[: boundaries[0] - 1])
        chunks.append(Chunk(id="", file=file, start=1, end=boundaries[0] - 1, content=pre, chunk_type="window"))

    for start, end in segments:
        # if segment too large, split with overlap
        if end - start + 1 > WINDOW * 2:
            sub_chunks = _sliding_window(file, "\n".join(lines[start - 1 : end]), lang)
            for sc in sub_chunks:
                sc.start += start - 1
                sc.end += start - 1
            chunks.extend(sub_chunks)
        else:
            snippet = "\n".join(lines[start - 1 : end])
            chunks.append(Chunk(id="", file=file, start=start, end=end, content=snippet, chunk_type="window"))

    return chunks


# ── Public API ───────────────────────────────────────────────────────────────

def chunk_file(sf: SourceFile) -> list[Chunk]:
    if sf.language == "python":
        chunks = _chunk_python(sf.path, sf.content)
    elif sf.language in _FUNC_PATTERNS:
        chunks = _structure_aware_window(sf.path, sf.content, sf.language)
    else:
        chunks = _sliding_window(sf.path, sf.content, sf.language)

    # filter empty chunks
    chunks = [c for c in chunks if c.content.strip()]
    return chunks


def chunk_files(source_files: list[SourceFile]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for sf in source_files:
        all_chunks.extend(chunk_file(sf))
    # assign IDs
    for i, c in enumerate(all_chunks):
        c.id = f"chunk_{i}"
    return all_chunks
