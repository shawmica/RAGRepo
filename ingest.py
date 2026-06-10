"""Ingest a GitHub URL or local path into a list of source files."""
from __future__ import annotations
import os, shutil, subprocess, tempfile, stat
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    "node_modules", ".git", ".hg", ".svn", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".venv", "venv", "env", ".env", "dist", "build",
    ".next", ".nuxt", "coverage", ".coverage", "htmlcov", ".tox",
}

SKIP_EXTENSIONS = {
    # lock files
    ".lock", ".sum",
    # binaries / media
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".ogg", ".mov", ".avi",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".whl", ".egg",
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".pyc", ".pyo", ".class",
    ".ttf", ".woff", ".woff2", ".eot",
    # generated
    ".min.js", ".min.css", ".map",
}

SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Pipfile.lock", "poetry.lock", "Cargo.lock",
    "composer.lock", "Gemfile.lock",
    ".DS_Store", "Thumbs.db",
}

MAX_FILE_BYTES = 512 * 1024  # 512 KB


@dataclass
class SourceFile:
    path: str           # relative path from repo root
    abs_path: str       # absolute path on disk
    content: str
    line_count: int
    language: str


def _is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


def _detect_language(path: str) -> str:
    ext = Path(path).suffix.lower()
    mapping = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "javascript", ".tsx": "typescript",
        ".java": "java", ".kt": "kotlin", ".scala": "scala",
        ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
        ".h": "c", ".hpp": "cpp", ".cs": "csharp",
        ".rb": "ruby", ".php": "php", ".swift": "swift",
        ".sh": "bash", ".bash": "bash", ".zsh": "bash",
        ".html": "html", ".css": "css", ".scss": "css",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".ini": "ini", ".cfg": "ini",
        ".md": "markdown", ".rst": "rst", ".txt": "text",
        ".sql": "sql", ".graphql": "graphql",
        ".dockerfile": "dockerfile", "dockerfile": "dockerfile",
    }
    name = Path(path).name.lower()
    if name == "dockerfile":
        return "dockerfile"
    return mapping.get(ext, "text")


def _should_skip(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    # skip hidden directories and known skip dirs
    for p in parts[:-1]:
        if p in SKIP_DIRS or p.startswith("."):
            return True
    name = parts[-1]
    if name in SKIP_FILENAMES:
        return True
    # multi-part extensions like .min.js
    lower = name.lower()
    for ext in SKIP_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return False


def _read_files(root: str) -> list[SourceFile]:
    root_path = Path(root)
    files: list[SourceFile] = []
    for abs_p in root_path.rglob("*"):
        if not abs_p.is_file():
            continue
        rel = abs_p.relative_to(root_path).as_posix()
        if _should_skip(rel):
            continue
        if abs_p.stat().st_size > MAX_FILE_BYTES:
            continue
        if _is_binary(str(abs_p)):
            continue
        try:
            content = abs_p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files.append(
            SourceFile(
                path=rel,
                abs_path=str(abs_p),
                content=content,
                line_count=content.count("\n") + 1,
                language=_detect_language(rel),
            )
        )
    return sorted(files, key=lambda f: f.path)


def _fix_permissions(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _parse_github_url(url: str) -> tuple[str, str, str]:
    if url.startswith("git@github.com:"):
        url = url.replace("git@github.com:", "https://github.com/")
    if url.endswith(".git"):
        url = url[:-4]
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc not in {"github.com", "www.github.com"}:
        raise ValueError("Only GitHub URLs are supported for remote ingestion.")
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub repository URL.")
    owner, repo = parts[0], parts[1]
    branch = "main"
    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]
    return owner, repo, branch


def _download_github_archive(owner: str, repo: str, branch: str, tmp: str) -> str:
    archive_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
    archive_path = os.path.join(tmp, "repo.zip")
    try:
        urllib.request.urlretrieve(archive_url, archive_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download GitHub archive for {owner}/{repo} branch {branch}: {exc}"
        ) from exc

    extracted = os.path.join(tmp, "repo")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extracted)
    # GitHub zip archives contain a single top-level directory.
    children = [p for p in os.listdir(extracted) if os.path.isdir(os.path.join(extracted, p))]
    if len(children) == 1:
        return os.path.join(extracted, children[0])
    return extracted


def ingest_github(url: str) -> list[SourceFile]:
    tmp = tempfile.mkdtemp(prefix="rag_repo_")
    try:
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", url, tmp],
                check=True,
                capture_output=True,
                text=True,
            )
            return _read_files(tmp)
        except FileNotFoundError:
            owner, repo, branch = _parse_github_url(url)
            repo_path = _download_github_archive(owner, repo, branch, tmp)
            return _read_files(repo_path)
        except subprocess.CalledProcessError:
            owner, repo, branch = _parse_github_url(url)
            repo_path = _download_github_archive(owner, repo, branch, tmp)
            return _read_files(repo_path)
    finally:
        shutil.rmtree(tmp, onerror=_fix_permissions)


def ingest_local(path: str) -> list[SourceFile]:
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Path not found: {resolved}")
    return _read_files(resolved)


def ingest(source: str) -> list[SourceFile]:
    """Auto-detect GitHub URL vs local path."""
    if source.startswith(("http://", "https://", "git@")):
        return ingest_github(source)
    return ingest_local(source)
