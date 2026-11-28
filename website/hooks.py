"""Publish existing Markdown without maintaining a second copy of the docs."""
from __future__ import annotations

import html
import posixpath
import re
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/vaaleerkiin/Kyven"
MD_LINK = re.compile(r"(!?\[[^\]\n]*\]\()([^\s)]+)([^)]*\))")
HTML_LINK = re.compile(r'((?:href|src)=[\"\'])([^\"\']+)([\"\'])')


def site_target(path: str) -> str | None:
    if path == "README.md":
        return "index.md"
    if path == "docs/README.md":
        return "guide.md"
    if path.startswith("docs/"):
        return path[5:]
    if path == "hosts/nuke/icons/kyven_logo.png":
        return "assets/kyven-logo.png"
    return None


def rewrite_url(url: str, source: str, destination: str) -> str:
    parts = urlsplit(html.unescape(url))
    if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
        return url
    target = posixpath.normpath(posixpath.join(posixpath.dirname(source), parts.path))
    published = site_target(target)
    if published is not None:
        path = posixpath.relpath(published, posixpath.dirname(destination) or ".")
        return urlunsplit(("", "", path, parts.query, parts.fragment))
    # Code, installers, licenses and development sources stay on GitHub.
    kind = "tree" if (ROOT / target).is_dir() else "blob"
    base = f"{REPO}/{kind}/main/{quote(target)}"
    return urlunsplit((*urlsplit(base)[:3], parts.query, parts.fragment))


def on_files(files, config):
    files.remove(files.get_file_from_path("README.md"))
    for uri, content in (
        ("index.md", (ROOT / "README.md").read_text(encoding="utf-8")),
        ("guide.md", (ROOT / "docs/README.md").read_text(encoding="utf-8")),
        ("assets/site.css", (ROOT / "website/site.css").read_bytes()),
        ("assets/kyven-logo.png", (ROOT / "hosts/nuke/icons/kyven_logo.png").read_bytes()),
    ):
        files.append(File.generated(config, uri, content=content))
    return files


def on_page_markdown(markdown, page, config, files):
    destination = page.file.src_uri
    source = ("README.md" if destination == "index.md" else
              "docs/README.md" if destination == "guide.md" else "docs/" + destination)
    # Keep code examples literal; adapt only rendered prose links and HTML assets.
    chunks = re.split(r"(^```[^\n]*\n.*?^```[^\n]*$)", markdown, flags=re.MULTILINE | re.DOTALL)
    for index in range(0, len(chunks), 2):
        chunks[index] = re.sub(r"^   (?=\S)", "    ", chunks[index], flags=re.MULTILINE)
        text = MD_LINK.sub(lambda m: m[1] + rewrite_url(m[2], source, destination) + m[3], chunks[index])
        chunks[index] = HTML_LINK.sub(lambda m: m[1] + rewrite_url(m[2], source, destination) + m[3], text)
    markdown = "".join(chunks)
    if destination == "index.md":
        page.meta.update(template="home.html", hide=["navigation", "toc"])
        lead = re.search(r"\*\*(.*?)\*\*", markdown, flags=re.DOTALL)
        page.meta["lead"] = lead[1] if lead else config.site_description
        markdown = re.sub(r"^<h1>.*?</h1>\s*", "", markdown, count=1)
        if lead:
            markdown = markdown.replace(lead[0], "", 1)
        page.edit_url = REPO + "/edit/main/README.md"
    elif destination == "guide.md":
        page.edit_url = REPO + "/edit/main/docs/README.md"
    return markdown
