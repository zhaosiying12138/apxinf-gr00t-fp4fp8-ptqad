"""Render the shared v12 reproduction commands without running any of them."""
from __future__ import annotations

import re
from pathlib import Path


PAPER = Path(__file__).resolve().parent
INCLUDES = (
    "readme_v12_supplement_commands.md",
    "readme_v12_capture_commands.md",
    "readme_v12_engine_commands.md",
)


def rebase_local_links(markdown: str, prefix: str) -> str:
    """Rebase relative Markdown destinations, leaving fenced code untouched.

    Templates use repository-root-relative links. Pass ``"../"`` when the
    rendered document lives directly inside ``docs/``. Absolute URLs and
    fragments stay unchanged; commands and their working directories do too.
    """
    if not prefix:
        return markdown
    fence: str | None = None
    lines: list[str] = []
    for line in markdown.splitlines(keepends=True):
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            lines.append(line)
            continue
        if fence is None:
            def replace(match: re.Match[str]) -> str:
                target = match.group(1)
                if target.startswith(("#", "/")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
                    return match.group(0)
                return "](" + prefix + target + ")"
            line = re.sub(r"\]\(([^\s)]+)\)", replace, line)
        lines.append(line)
    return "".join(lines)


def render_workflow(paper_dir: Path | None = None, *, link_prefix: str = "") -> str:
    """Return the complete command section, including its heading and spacing.

    This is a pure text operation. It requires neither final evidence nor any
    optional package, and is shared by the review and final README renderers.
    ``paper_dir`` is useful for callers rendering a relocated checkout.
    """
    folder = Path(paper_dir) if paper_dir is not None else PAPER
    markdown = (folder / "readme_v12_workflow.md").read_text(encoding="utf-8")
    for name in INCLUDES:
        marker = f"<!-- include: {name} -->"
        if markdown.count(marker) != 1:
            raise ValueError(f"workflow must contain exactly one {marker}")
        markdown = markdown.replace(marker, (folder / name).read_text(encoding="utf-8").strip())
    if "<!-- include:" in markdown:
        raise ValueError("workflow contains an unresolved include")
    return rebase_local_links(markdown, link_prefix)
