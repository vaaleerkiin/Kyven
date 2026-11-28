"""Check the built website, including project-prefix URLs, images and anchors."""
from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, path: Path):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        for name in ("href", "src"):
            if name in attributes:
                self.links.append(attributes[name])


def check_site(root: Path) -> None:
    root = root.resolve()
    pages = {path.resolve(): Page(path) for path in root.rglob("*.html")}
    assert root / "index.html" in pages, "The homepage was not generated"
    failures = []
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            relative = unquote(url.path)
            if relative.startswith("/Kyven/"):
                destination = root / relative[len("/Kyven/"):]
            elif relative.startswith("/"):
                destination = root / relative.lstrip("/")
            else:
                destination = path.parent / relative if relative else path
            if destination.is_dir():
                destination /= "index.html"
            destination = destination.resolve()
            if not destination.is_file() or not destination.is_relative_to(root):
                failures.append(f"{path.relative_to(root)}: missing {link}")
            elif url.fragment and destination in pages and unquote(url.fragment) not in pages[destination].ids:
                failures.append(f"{path.relative_to(root)}: missing anchor {link}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Checked {len(pages)} HTML pages: local links, images and anchors are valid.")


if __name__ == "__main__":
    check_site(Path(sys.argv[1]))
