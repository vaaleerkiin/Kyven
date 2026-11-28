# Documentation website

[Documentation home](README.md) · [Project overview](../README.md)

The public site is [vaaleerkiin.github.io/Kyven](https://vaaleerkiin.github.io/Kyven/).
Its text comes directly from the Markdown already in this repository:

- Root `README.md` becomes the homepage, including its real model examples.
- `docs/README.md` becomes the documentation index.
- Other `docs/*.md` and architecture decisions become individual pages.
- Images stay in `docs/images/`. Code, installers and licenses link back to GitHub.

Edit these files normally and push to `main`. The Documentation website Action
builds and publishes GitHub Pages automatically. Pull requests build and check
the site without publishing. A new guide in `docs/` is built and searchable;
add it to `nav` in `mkdocs.yml` if it should appear in the navigation.

## Preview locally

Use a separate Python environment so documentation packages do not change Kyven:

```powershell
python -m venv .runtime/docs-venv
.runtime/docs-venv/Scripts/python.exe -m pip install -r website/requirements.txt
.runtime/docs-venv/Scripts/python.exe -m mkdocs serve
```

Open the printed local URL. Markdown changes reload automatically. For a static
build, run `python -m mkdocs build --strict` in that environment; output is in
`.runtime/site/`. Run `python website/check_site.py .runtime/site` to check links.

## Website configuration

- `mkdocs.yml`: navigation, search, theme and publication URL.
- `website/hooks.py`: virtual homepage and relative-link adaptation at build time.
- `website/overrides/home.html` and `website/site.css`: presentation only.
- `.github/workflows/docs.yml`: validation and automatic deployment.

GitHub Pages uses **GitHub Actions** as its publishing source. There is no
second Markdown tree, generated HTML in Git, or model/runtime dependency in the
website build. No model weights or private work files are published.
