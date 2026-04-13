"""
CALM Markdown backend — parse headers, code blocks, TOC, links.

Pure regex parsing, no external markdown library.
"""

from __future__ import annotations

import re


def md_headers(text: str) -> list:
    """Extract all headers with their levels: [(level, text), ...]."""
    results = []
    for line in text.splitlines():
        m = re.match(r'^(#{1,6})\s+(.+?)(?:\s*#*\s*)?$', line)
        if m:
            results.append((len(m.group(1)), m.group(2).strip()))
    return results


def md_toc(text: str) -> str:
    """Generate a table of contents from headers."""
    headers = md_headers(text)
    if not headers:
        return "(no headers found)"
    lines = []
    for level, title in headers:
        indent = "  " * (level - 1)
        slug = re.sub(r'[^\w\s-]', '', title.lower()).strip().replace(' ', '-')
        lines.append(f"{indent}- [{title}](#{slug})")
    return "\n".join(lines)


def md_code_blocks(text: str) -> list:
    """Extract fenced code blocks: [(language, code), ...]."""
    blocks = []
    pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    for m in pattern.finditer(text):
        lang = m.group(1) or "plain"
        blocks.append((lang, m.group(2).rstrip("\n")))
    return blocks


def md_links(text: str) -> list:
    """Extract markdown links (not images): [(text, url), ...]."""
    pattern = re.compile(r'(?<!!)\[([^\]]*)\]\(([^)]+)\)')
    return pattern.findall(text)


def md_images(text: str) -> list:
    """Extract markdown images: [(alt, url), ...]."""
    pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    return pattern.findall(text)


def md_section_count(text: str) -> int:
    """Count the number of top-level sections (# headers)."""
    return sum(1 for level, _ in md_headers(text) if level == 1)


def md_word_count(text: str) -> int:
    """Word count of markdown content (strips markup)."""
    # Remove code blocks
    clean = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code
    clean = re.sub(r'`[^`]+`', '', clean)
    # Remove links/images markup (keep text)
    clean = re.sub(r'!?\[([^\]]*)\]\([^)]+\)', r'\1', clean)
    # Remove headers markup
    clean = re.sub(r'^#{1,6}\s+', '', clean, flags=re.MULTILINE)
    # Remove emphasis
    clean = re.sub(r'[*_]{1,3}', '', clean)
    return len(clean.split())


MARKDOWN_FUNCTIONS = {
    "md_headers": md_headers,
    "md_toc": md_toc,
    "md_code_blocks": md_code_blocks,
    "md_links": md_links,
    "md_images": md_images,
    "md_section_count": md_section_count,
    "md_word_count": md_word_count,
}
