# PaperSignal

CLI for an LLM to **discover, download, analyze, and publish** arXiv papers as
self-contained bilingual research briefs — readable in the browser, narrated
by Edge TTS, deployable to GitHub Pages.

The Python package and CLI command are both named `paperread`; the project
itself is **PaperSignal**, and the published site lives at
[hsiaosiyuan0.github.io/PaperSignal](https://hsiaosiyuan0.github.io/PaperSignal/).

---

## What it does

Given a topic and a target organization, an LLM can:

1. **Find** recent papers by that org (`paperread discover` → OpenAlex)
2. **Download** the PDF (`paperread download` → arXiv, streaming + retries)
3. **Convert** to Markdown with LaTeX preserved (`paperread convert` → marker / markitdown)
4. **Read** the markdown, **write** an analysis (`report.en.md` + `report.zh.md`)
5. **Build** the published page (`paperread report build` → HTML + per-sentence mp3s)
6. **Refresh** the archive index (`paperread report index`)

Push to GitHub; Actions deploys to Pages.

---

## Install

```bash
git clone git@github.com:hsiaosiyuan0/PaperSignal.git
cd PaperSignal
uv sync              # installs deps + creates .venv (first run pulls torch + marker, ~5GB)
cp .env.example .env # then put your OpenAlex key in .env
```

Requires Python 3.12+, [uv](https://github.com/astral-sh/uv), and (optionally)
a free [OpenAlex API key](https://openalex.org/) for sustained `discover` usage.

---

## Quick start (the full LLM workflow)

```bash
# 1. Find recent Microsoft papers on knowledge retrieval
uv run paperread discover "knowledge retrieval" \
  --affiliation Microsoft --year 2025- --top 10

# 2. Pick one (say 2604.15597) and prep it
uv run paperread convert 2604.15597
DIR=$(uv run paperread report path 2604.15597)

# 3. Have the LLM write the analysis. It should read:
uv run paperread cache show 2604.15597 --json
# then Write to:
#   $DIR/report.en.md
#   $DIR/report.zh.md

# 4. Build (segments sentences, runs Edge TTS, renders index.html)
uv run paperread report build 2604.15597

# 5. Refresh the archive landing page + view locally
uv run paperread report index
uv run paperread report open 2604.15597
```

A sample LLM analysis ships under [`reports/1706.03762v7/`](reports/1706.03762v7/) —
"Attention Is All You Need".

---

## Commands

| Command | Purpose |
|---|---|
| `paperread discover QUERY` | Server-side affiliation + year + sort search via OpenAlex |
| `paperread search QUERY` | arXiv native search (no affiliation filter) |
| `paperread info ID [--json]` | Metadata for one paper |
| `paperread download ID` | Streaming PDF fetch, retries on truncation |
| `paperread convert ID [-b marker|markitdown]` | PDF → Markdown |
| `paperread cache list/search/show [--json]` | Query local SQLite, LLM-friendly |
| `paperread report path ID [--lang en|zh]` | Where to write the analysis MD |
| `paperread report build ID [--no-audio]` | Render HTML + generate audio |
| `paperread report open ID` | Open in browser |
| `paperread report index` | Rebuild `reports/index.html` |
| `paperread report pages-init` | Scaffold `.nojekyll` + Actions workflow |

All search / list / show / discover commands accept `--json` for machine consumption.

---

## Configuration

Set in `.env` (gitignored) or your shell:

| Variable | Purpose |
|---|---|
| `OPENALEX_API_KEY` | Required for `discover` after free trial quota |
| `PAPERREAD_REPORTS_DIR` | Root of report bundles (default `./reports`) |
| `PAPERREAD_DATA_DIR` | Where PDFs land (default `./papers`) |
| `PAPERREAD_CACHE_DIR` | SQLite cache location (default platform cache dir) |
| `PAPERREAD_TTS_VOICE` | Edge TTS voice (default `en-US-AvaMultilingualNeural`) |
| `TORCH_DEVICE` | Override marker's auto device pick (`mps` / `cuda` / `cpu`) |

---

## GitHub Pages deployment

```bash
uv run paperread report pages-init
git add -A && git commit -m "scaffold pages" && git push
```

Then in repo Settings → Pages → **Source: GitHub Actions**. The workflow
republishes whenever anything under `reports/` changes.

The published directory structure is self-contained per paper:

```
reports/
  index.html                       ← archive landing page
  .nojekyll                        ← keep "_"-prefixed paths
  <arxiv_id>/
    index.html                     ← rendered report
    report.en.md                   ← LLM source
    report.zh.md                   ← LLM source (optional)
    audio/sNNNN.mp3                ← per-sentence Edge TTS
```

---

## Architecture

```
src/paperread/
  arxiv_client.py     wraps the arxiv Python lib; custom UA to dodge rate limits;
                      streaming PDF download with retry (avoids urlretrieve truncation)
  openalex.py         /paper/search endpoint, raw_affiliation_strings filter
  cache.py            SQLite metadata cache with auto-migrated schema (v3)
  converter.py        marker / markitdown backend abstraction with lazy loading
  sentences.py        pysbd English boundary detection, filters trivial fragments
  tts.py              edge-tts async wrapper with manifest-based incremental regen
  report.py           bilingual HTML renderer; segments rendered HTML plain text
                      so sentences crossing <strong>/inline math wrap correctly
  cli.py              Typer commands + JSON contract for LLM consumption
  paths.py            platformdirs + env overrides
```

---

## Design choices worth knowing

- **Custom User-Agent on arxiv requests** — `arxiv.py/2.3.2` is aggressively
  rate-limited by arxiv.org; we override the hardcoded UA in the session.
- **OpenAlex over Semantic Scholar** — S2 requires academic/corporate email
  for an API key and its affiliation coverage is sparse (~12% in our test).
  OpenAlex indexes affiliations as a first-class filter with any email signup.
- **`/paper/search` not `/paper/search/bulk`** — bulk doesn't return
  `authors.affiliations`; the relevance endpoint does. We sort client-side.
- **marker over markitdown for default conversion** — preserves LaTeX, which
  matters for ML/CS papers. markitdown stays available as a faster fallback.
- **Sentence segmentation on rendered HTML plain text** — not the source MD.
  This way sentences crossing `<strong>` / `$math$` get multiple `<span>`
  fragments sharing `data-sentence-id`, all highlighting together on hover.
- **Edge TTS, not OpenAI / Eleven Labs** — free, no API key, decent voices.
- **Editor's Desk visual style** — Fraunces variable serif throughout,
  bone background, single oxidized vermillion accent, Roman-numeral section
  headers. Deliberately *not* the Inter + purple-gradient AI aesthetic.

---

## Tech stack credits

[arxiv.py](https://github.com/lukasschwab/arxiv.py) ·
[OpenAlex](https://openalex.org/) ·
[marker](https://github.com/datalab-to/marker) ·
[markitdown](https://github.com/microsoft/markitdown) ·
[edge-tts](https://github.com/rany2/edge-tts) ·
[pysbd](https://github.com/nipunsadvilkar/pySBD) ·
[markdown-it-py](https://github.com/executablebooks/markdown-it-py) ·
[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) ·
[Typer](https://typer.tiangolo.com/) ·
[Rich](https://github.com/Textualize/rich) ·
[Fraunces](https://fonts.google.com/specimen/Fraunces) ·
[JetBrains Mono](https://www.jetbrains.com/lp/mono/) ·
[KaTeX](https://katex.org/)

---

## License

MIT (see `LICENSE` — TBD).

> **Note on marker**: the `marker-pdf` dependency is GPL-3.0 for code and uses
> a modified Open Rail-M license for its models — free for research, personal,
> and startups under $2M revenue. Commercial self-hosting beyond that threshold
> requires a license from Datalab.
