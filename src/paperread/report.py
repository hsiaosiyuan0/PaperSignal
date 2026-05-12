"""Bilingual HTML report renderer with per-sentence audio.

Reads `report.en.md` and (optionally) `report.zh.md` from a report directory,
generates per-sentence English TTS audio via edge-tts, wraps each rendered EN
sentence in clickable span(s) tied to its mp3, and emits `index.html` with
a language toggle. The whole directory is self-contained — point GitHub Pages
at the parent and relative URLs work as-is.

The aesthetic ("The Editor's Desk") lives in this module's CSS: bone-paper
background, graphite ink, oxidized vermillion as the single accent, Fraunces
serif throughout.

Sentence wrapping strategy: we segment the *rendered HTML's plain text* (not
the source markdown) and apply each sentence's range across whatever text
nodes it touches. Sentences crossing inline tags (`<strong>`, etc.) get
multiple span fragments that share `data-sentence-id`, so JS highlights them
as a unit.
"""

from __future__ import annotations

import html as html_lib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from markdown_it import MarkdownIt

from paperread.arxiv_client import Paper
from paperread.sentences import Sentence, segment_english
from paperread.tts import AudioEntry, synthesize_sentences

_md = (
    MarkdownIt("commonmark", {"html": True, "breaks": False, "linkify": True})
    .enable("table")
    .enable("strikethrough")
)

EN_FILENAME = "report.en.md"
ZH_FILENAME = "report.zh.md"
HTML_FILENAME = "index.html"
AUDIO_DIRNAME = "audio"

# Tags whose text content participates in sentence segmentation.
_WRAPPABLE_PARENTS = {"p", "li", "blockquote", "td", "th"}
# Don't pull text from inside these — code, math, headings are non-prose.
_SKIP_ANCESTORS = {"code", "pre", "script", "style", "h1", "h2", "h3", "h4", "h5", "h6"}


@dataclass(frozen=True, slots=True)
class BuildResult:
    html_path: Path
    audio_entries: list[AudioEntry]
    has_zh: bool


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One row in the archive index page."""

    arxiv_id: str
    title: str
    authors: tuple[str, ...]
    primary_category: str
    published: str  # "YYYY-MM-DD"
    href: str  # relative URL to the per-paper index.html, e.g. "1706.03762v7/"


def render_index(entries: list[IndexEntry], output_path: Path) -> Path:
    """Write the `reports/index.html` landing page that lists all built reports."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page = _build_index_page(entries)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def build_report(
    paper: Paper,
    report_dir: Path,
    *,
    with_audio: bool = True,
    on_audio_progress: Callable[[str], None] | None = None,
) -> BuildResult:
    """Read EN (+ optional ZH) markdown from `report_dir`, render to index.html."""
    en_md = report_dir / EN_FILENAME
    zh_md = report_dir / ZH_FILENAME
    if not en_md.is_file():
        raise FileNotFoundError(en_md)

    en_text = en_md.read_text(encoding="utf-8")
    zh_text = zh_md.read_text(encoding="utf-8") if zh_md.is_file() else None

    en_html_raw = _md.render(en_text)
    zh_html_raw = _md.render(zh_text) if zh_text is not None else ""

    audio_entries: list[AudioEntry] = []
    en_html_wrapped = en_html_raw
    if with_audio:
        soup = BeautifulSoup(en_html_raw, "html.parser")
        plain, node_map = _collect_plain_text(soup)
        sentences = segment_english(plain)
        audio_dir = report_dir / AUDIO_DIRNAME
        audio_entries = synthesize_sentences(
            sentences, audio_dir, on_progress=on_audio_progress
        )
        _apply_sentence_wraps(node_map, plain, sentences, audio_entries)
        en_html_wrapped = str(soup)

    html_path = report_dir / HTML_FILENAME
    page = _build_page(
        paper=paper,
        en_body=en_html_wrapped,
        zh_body=zh_html_raw,
        has_zh=zh_text is not None,
        has_audio=bool(audio_entries),
    )
    html_path.write_text(page, encoding="utf-8")
    return BuildResult(html_path=html_path, audio_entries=audio_entries, has_zh=zh_text is not None)


# --- sentence wrapping ------------------------------------------------------


@dataclass(slots=True)
class _TextSpan:
    node: NavigableString
    start: int  # offset in the full plain text
    end: int


def _collect_plain_text(soup: BeautifulSoup) -> tuple[str, list[_TextSpan]]:
    """Walk the soup in document order, building plain text from wrappable nodes.

    `node_map` lets us map back from plain-text offsets to the originating text
    node so we can splice spans into the right place.
    """
    plain_parts: list[str] = []
    node_map: list[_TextSpan] = []
    cursor = 0
    for node in soup.descendants:
        if not isinstance(node, NavigableString):
            continue
        if _has_skipped_ancestor(node):
            continue
        if not _parent_chain_includes_wrappable(node):
            continue
        text = str(node)
        if not text:
            continue
        node_map.append(_TextSpan(node=node, start=cursor, end=cursor + len(text)))
        plain_parts.append(text)
        cursor += len(text)
    return "".join(plain_parts), node_map


def _has_skipped_ancestor(node: NavigableString) -> bool:
    cur = node.parent
    while cur is not None and getattr(cur, "name", None):
        if cur.name in _SKIP_ANCESTORS:
            return True
        cur = cur.parent
    return False


def _parent_chain_includes_wrappable(node: NavigableString) -> bool:
    cur = node.parent
    while cur is not None and getattr(cur, "name", None):
        if cur.name in _WRAPPABLE_PARENTS:
            return True
        cur = cur.parent
    return False


def _apply_sentence_wraps(
    node_map: list[_TextSpan],
    plain: str,
    sentences: list[Sentence],
    entries: list[AudioEntry],
) -> None:
    """For each sentence with a corresponding AudioEntry, wrap its plain-text
    range across whatever nodes it spans. Sentences without audio are skipped.
    """
    entries_by_idx = {e.index: e for e in entries}

    # Per text-node, collect (slice_start, slice_end, entry) so we can apply
    # everything in one pass per node (multiple sentences can touch one node).
    per_node: dict[int, list[tuple[int, int, AudioEntry]]] = {}

    cursor = 0
    for sentence in sentences:
        entry = entries_by_idx.get(sentence.index)
        if entry is None:
            continue
        target = sentence.core
        if not target:
            continue
        idx = plain.find(target, cursor)
        if idx < 0:
            continue
        end = idx + len(target)
        cursor = end
        for i, span in enumerate(node_map):
            if span.end <= idx:
                continue
            if span.start >= end:
                break
            slice_start = max(0, idx - span.start)
            slice_end = min(span.end - span.start, end - span.start)
            if slice_end <= slice_start:
                continue
            per_node.setdefault(i, []).append((slice_start, slice_end, entry))

    for i, wraps in per_node.items():
        _apply_wraps_to_node(node_map[i].node, wraps)


def _apply_wraps_to_node(
    text_node: NavigableString,
    wraps: list[tuple[int, int, AudioEntry]],
) -> None:
    """Replace one text node with [text-prefix, span, text-mid, span, text-suffix]."""
    wraps.sort(key=lambda w: w[0])
    text = str(text_node)
    soup = BeautifulSoup("", "html.parser")
    new_children: list[NavigableString | Tag] = []
    cursor = 0
    for s, e, entry in wraps:
        if s < cursor:
            # Overlap with previous sentence's slice — skip to avoid invalid HTML.
            continue
        if s > cursor:
            new_children.append(NavigableString(text[cursor:s]))
        span = soup.new_tag("span")
        span["class"] = "sentence"
        span["data-sentence-id"] = str(entry.index)
        span["data-audio"] = f"{AUDIO_DIRNAME}/{entry.filename}"
        span.string = text[s:e]
        new_children.append(span)
        cursor = e
    if cursor < len(text):
        new_children.append(NavigableString(text[cursor:]))
    if new_children:
        text_node.replace_with(*new_children)


# --- HTML composition -------------------------------------------------------


def _build_page(
    *,
    paper: Paper,
    en_body: str,
    zh_body: str,
    has_zh: bool,
    has_audio: bool,
) -> str:
    title = html_lib.escape(paper.title)
    arxiv_id = html_lib.escape(paper.arxiv_id)
    category = html_lib.escape(paper.primary_category)
    abs_url = html_lib.escape(paper.abs_url)
    published = paper.published.strftime("%Y-%m-%d")
    generated = datetime.now().astimezone().strftime("%Y-%m-%d")
    authors = " · ".join(html_lib.escape(a) for a in paper.authors)

    toggle = ""
    zh_block = ""
    if has_zh:
        toggle = (
            '<div class="lang-switch" role="tablist" aria-label="Language">'
            '<button class="active" data-lang="en" aria-selected="true">EN</button>'
            '<button data-lang="zh" aria-selected="false">中</button>'
            "</div>"
        )
        zh_block = f'<div class="body" data-lang="zh" hidden>\n{zh_body}\n</div>'

    return _TEMPLATE.format(
        title=title,
        arxiv_id=arxiv_id,
        category=category,
        abs_url=abs_url,
        published=published,
        generated=generated,
        authors=authors,
        en_body=en_body,
        zh_block=zh_block,
        toggle=toggle,
        css=_CSS,
        audio_js=_AUDIO_JS if has_audio else "",
        toggle_js=_TOGGLE_JS if has_zh else "",
    )


_CSS = """
:root {
  --bone: #F4EFE6;
  --bone-soft: #FBF7EE;
  --graphite: #1F1F1E;
  --graphite-soft: #6A6A66;
  --graphite-fade: #B7B5AE;
  --vermillion: #C9482A;
  --vermillion-soft: rgba(201, 72, 42, 0.08);
  --hairline: rgba(31, 31, 30, 0.14);
  --serif: "Fraunces", "Iowan Old Style", "Hoefler Text", Georgia, serif;
  --serif-cn: "Source Han Serif SC", "Noto Serif CJK SC", "Songti SC", serif;
  --mono: "JetBrains Mono", "SF Mono", ui-monospace, monospace;
  --reading-width: 38rem;
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }

body {
  background: var(--bone);
  color: var(--graphite);
  font-family: var(--serif);
  font-variation-settings: "opsz" 14, "SOFT" 50;
  font-size: 1.0625rem;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

article {
  max-width: var(--reading-width);
  margin: 0 auto;
  padding: 4.5rem 1.5rem 2rem;
  position: relative;
}

.lang-switch {
  position: absolute;
  top: 1.5rem;
  right: 1.5rem;
  display: flex;
  gap: 0.25rem;
  font-family: var(--mono);
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.lang-switch button {
  border: 1px solid var(--hairline);
  background: transparent;
  color: var(--graphite-soft);
  padding: 0.35rem 0.7rem;
  cursor: pointer;
  font: inherit;
  letter-spacing: inherit;
  transition: background 160ms ease, color 160ms ease, border-color 160ms ease;
}
.lang-switch button:hover { color: var(--vermillion); }
.lang-switch button.active {
  background: var(--graphite);
  color: var(--bone);
  border-color: var(--graphite);
}

.brand {
  font-family: var(--mono);
  font-size: 0.68rem;
  font-weight: 500;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--graphite-soft);
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.6rem;
}
.brand .dot { color: var(--vermillion); }
.brand-rule { border: none; border-top: 1px solid var(--hairline); margin: 0 0 3rem; }

.meta {
  display: flex;
  gap: 1.5rem;
  align-items: baseline;
  flex-wrap: wrap;
  font-family: var(--mono);
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--graphite-soft);
  margin-bottom: 2rem;
}
.meta .id {
  color: var(--vermillion);
  letter-spacing: 0.1em;
  font-weight: 600;
}

h1.title {
  font-family: var(--serif);
  font-variation-settings: "opsz" 130, "SOFT" 35;
  font-style: italic;
  font-weight: 350;
  font-size: clamp(2.1rem, 5vw, 3.2rem);
  line-height: 1.05;
  letter-spacing: -0.018em;
  margin: 0 0 1.75rem;
  color: var(--graphite);
  text-wrap: balance;
}

.authors {
  font-family: var(--serif);
  font-variation-settings: "opsz" 14, "wght" 500;
  font-size: 0.78rem;
  font-feature-settings: "smcp" on, "c2sc" on;
  letter-spacing: 0.08em;
  color: var(--graphite-soft);
  margin: 0 0 2.5rem;
  padding-bottom: 2.5rem;
  border-bottom: 1px solid var(--hairline);
  text-wrap: pretty;
}

.body { counter-reset: section; }
.body[data-lang="zh"] { font-family: var(--serif-cn); }
.body[data-lang="zh"] h2, .body[data-lang="zh"] h3 { font-family: var(--serif-cn); }

.body h1 { display: none; }

.body h2 {
  counter-increment: section;
  font-family: var(--serif);
  font-variation-settings: "opsz" 36, "wght" 500;
  font-size: 1.55rem;
  letter-spacing: -0.01em;
  line-height: 1.2;
  margin: 3.5rem 0 1.1rem;
  color: var(--graphite);
}
.body h2::before {
  content: counter(section, upper-roman);
  display: inline;
  color: var(--vermillion);
  font-family: var(--mono);
  font-size: 0.62em;
  font-weight: 500;
  letter-spacing: 0.1em;
  margin-right: 0.85em;
  vertical-align: 0.2em;
}

.body h3 {
  font-family: var(--serif);
  font-variation-settings: "opsz" 24, "wght" 500;
  font-style: italic;
  font-size: 1.18rem;
  margin: 2.5rem 0 0.9rem;
}

.body p {
  margin: 0 0 1.1em;
  font-variant-numeric: lining-nums proportional-nums;
  text-wrap: pretty;
  hanging-punctuation: first;
}

.body a {
  color: var(--vermillion);
  text-decoration: none;
  border-bottom: 1px solid rgba(201, 72, 42, 0.35);
  transition: border-color 160ms ease, background 160ms ease;
}
.body a:hover { border-bottom-color: var(--vermillion); background: var(--vermillion-soft); }

.body strong { font-weight: 600; }
.body em { font-style: italic; }

.body ul, .body ol { padding-left: 1.5rem; margin: 0 0 1.2em; }
.body li { margin-bottom: 0.35em; }
.body li::marker { color: var(--vermillion); font-family: var(--mono); }

.body code {
  font-family: var(--mono);
  font-size: 0.85em;
  padding: 0.1em 0.35em;
  background: var(--vermillion-soft);
  color: var(--graphite);
  border-radius: 2px;
}

.body pre {
  font-family: var(--mono);
  font-size: 0.82em;
  line-height: 1.6;
  background: rgba(31, 31, 30, 0.04);
  border-left: 2px solid var(--vermillion);
  padding: 1rem 1.25rem;
  overflow-x: auto;
  margin: 1.6rem 0;
}
.body pre code { background: none; padding: 0; border-radius: 0; }

.body blockquote {
  font-style: italic;
  font-variation-settings: "opsz" 18;
  margin: 1.75rem 0;
  padding: 0 1.25rem;
  border-left: 2px solid var(--vermillion);
  color: var(--graphite-soft);
}

.body table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.75rem 0;
  font-size: 0.92em;
  font-variant-numeric: tabular-nums;
}
.body th, .body td {
  text-align: left;
  padding: 0.65rem 0.75rem;
  border-bottom: 1px solid var(--hairline);
}
.body th:first-child, .body td:first-child { padding-left: 0; }
.body th:last-child, .body td:last-child { padding-right: 0; }
.body thead th {
  border-bottom: 2px solid var(--graphite);
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  font-weight: 600;
  font-family: var(--mono);
}

.body hr { border: none; border-top: 1px solid var(--hairline); margin: 2.5rem 0; }

/* KaTeX */
.body .katex-display {
  margin: 1.75em 0;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.25em 0;
}
.body .katex { font-size: 1em; color: var(--graphite); }
.body .katex-display > .katex { display: block; text-align: center; }

/* Sentence audio */
.body[data-lang="en"] .sentence {
  cursor: pointer;
  transition: background 140ms ease, color 140ms ease;
  border-radius: 1px;
  padding: 0 0.05em;
}
.body[data-lang="en"] .sentence.hover,
.body[data-lang="en"] .sentence:hover {
  background: var(--vermillion-soft);
}
.body[data-lang="en"] .sentence.playing {
  background: var(--vermillion);
  color: var(--bone);
}

footer {
  max-width: var(--reading-width);
  margin: 5rem auto 3rem;
  padding: 1.5rem;
  border-top: 1px solid var(--hairline);
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--graphite-soft);
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}
footer a { color: var(--vermillion); text-decoration: none; }

@media (max-width: 640px) {
  article { padding: 3rem 1.25rem 1.5rem; }
  .lang-switch { top: 1rem; right: 1rem; }
  .brand { font-size: 0.62rem; gap: 1rem; }
  .meta { gap: 0.9rem 1.2rem; font-size: 0.68rem; }
  h1.title { font-size: 1.95rem; }
  .authors { font-size: 0.72rem; }
}

@media (prefers-color-scheme: dark) {
  :root {
    --bone: #1B1A18;
    --bone-soft: #232220;
    --graphite: #ECE5D6;
    --graphite-soft: #948B7B;
    --graphite-fade: #524C42;
    --vermillion: #E36246;
    --vermillion-soft: rgba(227, 98, 70, 0.10);
    --hairline: rgba(236, 229, 214, 0.16);
  }
}

@media print {
  body { background: white; color: black; }
  .lang-switch { display: none; }
  article { padding: 1in 0.75in; }
  .brand, footer { color: #555; }
  .body[hidden] { display: none !important; }
}
"""


_TOGGLE_JS = """
(function() {
  const buttons = document.querySelectorAll('.lang-switch button');
  const bodies = document.querySelectorAll('.body');
  buttons.forEach(btn => btn.addEventListener('click', () => {
    const lang = btn.dataset.lang;
    buttons.forEach(b => {
      b.classList.toggle('active', b === btn);
      b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
    });
    bodies.forEach(body => {
      const matches = body.dataset.lang === lang;
      body.hidden = !matches;
    });
    if (window.location.hash !== '#' + lang) {
      history.replaceState(null, '', '#' + lang);
    }
  }));
  const initial = (window.location.hash || '#en').slice(1);
  const target = document.querySelector(`.lang-switch button[data-lang="${initial}"]`);
  if (target) target.click();
})();
"""


_AUDIO_JS = """
(function() {
  let currentId = null;
  let audio = null;

  function setPlaying(id, on) {
    const sel = `.sentence[data-sentence-id="${id}"]`;
    document.querySelectorAll(sel).forEach(el => el.classList.toggle('playing', on));
  }

  // Highlight all fragments of a sentence on hover, not just the one under cursor.
  document.addEventListener('mouseover', e => {
    const s = e.target.closest('.sentence');
    if (!s) return;
    const id = s.dataset.sentenceId;
    if (id != null) {
      document.querySelectorAll(`.sentence[data-sentence-id="${id}"]`)
        .forEach(el => el.classList.add('hover'));
    }
  });
  document.addEventListener('mouseout', e => {
    const s = e.target.closest('.sentence');
    if (!s) return;
    const id = s.dataset.sentenceId;
    if (id != null) {
      document.querySelectorAll(`.sentence[data-sentence-id="${id}"]`)
        .forEach(el => el.classList.remove('hover'));
    }
  });

  document.addEventListener('click', e => {
    const span = e.target.closest('.sentence');
    if (!span) return;
    const src = span.dataset.audio;
    const id = span.dataset.sentenceId;
    if (!src || id == null) return;
    if (currentId === id && audio && !audio.paused) {
      audio.pause();
      setPlaying(currentId, false);
      currentId = null;
      return;
    }
    if (audio) audio.pause();
    if (currentId !== null) setPlaying(currentId, false);
    audio = new Audio(src);
    audio.addEventListener('ended', () => { setPlaying(id, false); });
    audio.play().catch(err => console.warn('audio play failed', err));
    setPlaying(id, true);
    currentId = id;
  });
})();
"""


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — paperread</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT@0,9..144,300..700,30..100;1,9..144,300..700,30..100&family=JetBrains+Mono:wght@400;500;600&display=swap">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">
<style>{css}</style>
</head>
<body>
<article>
  {toggle}
  <div class="brand">
    <span>Paperread</span>
    <span class="dot">·</span>
    <span>Research Brief</span>
    <span class="dot">·</span>
    <span>{generated}</span>
  </div>
  <hr class="brand-rule">
  <div class="meta">
    <span class="id">arXiv / {arxiv_id}</span>
    <span>{category}</span>
    <span>Published {published}</span>
  </div>
  <h1 class="title">{title}</h1>
  <p class="authors">{authors}</p>
  <div class="body" data-lang="en">
{en_body}
  </div>
  {zh_block}
</article>
<footer>
  <span>Compiled from arXiv:{arxiv_id}</span>
  <a href="{abs_url}">View on arXiv →</a>
</footer>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous" onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}, {{left: '\\\\[', right: '\\\\]', display: true}}, {{left: '\\\\(', right: '\\\\)', display: false}}], throwOnError: false, ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']}})"></script>
<script>{toggle_js}</script>
<script>{audio_js}</script>
</body>
</html>
"""


# --- index page ------------------------------------------------------------


def _build_index_page(entries: list[IndexEntry]) -> str:
    from datetime import datetime as _dt

    generated = _dt.now().astimezone().strftime("%Y-%m-%d")
    if entries:
        rows = "\n".join(_index_row(e) for e in entries)
        body = f'<ol class="archive">\n{rows}\n</ol>'
    else:
        body = (
            '<p class="empty">No research briefs yet. Run '
            "<code>paperread report build &lt;id&gt;</code> to add one.</p>"
        )
    return _INDEX_TEMPLATE.format(
        body=body,
        count=len(entries),
        generated=generated,
        css=_INDEX_CSS,
    )


def _index_row(e: IndexEntry) -> str:
    arxiv_id = html_lib.escape(e.arxiv_id)
    title = html_lib.escape(e.title)
    category = html_lib.escape(e.primary_category)
    href = html_lib.escape(e.href)
    authors_short = " · ".join(html_lib.escape(a) for a in e.authors[:3])
    if len(e.authors) > 3:
        authors_short += " · et al."
    return (
        f'<li><a href="{href}">'
        f'<span class="row-id">{arxiv_id}</span>'
        f'<span class="row-title">{title}</span>'
        f'<span class="row-meta">{category} · {e.published}</span>'
        f'<span class="row-authors">{authors_short}</span>'
        f"</a></li>"
    )


_INDEX_CSS = """
:root {
  --bone: #F4EFE6;
  --bone-soft: #FBF7EE;
  --graphite: #1F1F1E;
  --graphite-soft: #6A6A66;
  --vermillion: #C9482A;
  --vermillion-soft: rgba(201, 72, 42, 0.08);
  --hairline: rgba(31, 31, 30, 0.14);
  --serif: "Fraunces", "Iowan Old Style", "Hoefler Text", Georgia, serif;
  --mono: "JetBrains Mono", "SF Mono", ui-monospace, monospace;
  --container-width: 52rem;
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }

body {
  background: var(--bone);
  color: var(--graphite);
  font-family: var(--serif);
  font-variation-settings: "opsz" 14, "SOFT" 50;
  font-size: 1.0625rem;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

main {
  max-width: var(--container-width);
  margin: 0 auto;
  padding: 5rem 1.5rem 4rem;
}

.brand {
  font-family: var(--mono);
  font-size: 0.68rem;
  font-weight: 500;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--graphite-soft);
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.6rem;
}
.brand .dot { color: var(--vermillion); }
.brand-rule { border: none; border-top: 1px solid var(--hairline); margin: 0 0 2.5rem; }

h1.archive-title {
  font-family: var(--serif);
  font-variation-settings: "opsz" 130, "SOFT" 35;
  font-style: italic;
  font-weight: 350;
  font-size: clamp(2.4rem, 6vw, 3.6rem);
  line-height: 1.05;
  letter-spacing: -0.018em;
  margin: 0 0 0.8rem;
}
.archive-sub {
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--graphite-soft);
  margin: 0 0 4rem;
}
.archive-sub .count { color: var(--vermillion); font-weight: 600; }

ol.archive {
  list-style: none;
  margin: 0;
  padding: 0;
  border-top: 1px solid var(--hairline);
}
ol.archive li { margin: 0; }
ol.archive li a {
  display: grid;
  grid-template-columns: 9rem 1fr;
  grid-template-areas:
    "id title"
    "id meta"
    "id authors";
  gap: 0.15rem 1.5rem;
  padding: 1.5rem 0.5rem;
  text-decoration: none;
  color: inherit;
  border-bottom: 1px solid var(--hairline);
  transition: background 160ms ease, padding 220ms ease;
}
ol.archive li a:hover {
  background: var(--vermillion-soft);
  padding-left: 1rem;
}
ol.archive li a:hover .row-title { color: var(--vermillion); }
.row-id {
  grid-area: id;
  font-family: var(--mono);
  font-size: 0.74rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--vermillion);
  align-self: start;
  padding-top: 0.4rem;
}
.row-title {
  grid-area: title;
  font-family: var(--serif);
  font-variation-settings: "opsz" 28, "wght" 450;
  font-style: italic;
  font-size: 1.35rem;
  line-height: 1.2;
  letter-spacing: -0.005em;
  text-wrap: balance;
  transition: color 160ms ease;
}
.row-meta {
  grid-area: meta;
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--graphite-soft);
}
.row-authors {
  grid-area: authors;
  font-family: var(--serif);
  font-feature-settings: "smcp" on, "c2sc" on;
  font-size: 0.78rem;
  letter-spacing: 0.06em;
  color: var(--graphite-soft);
  margin-top: 0.25rem;
}

.empty {
  font-style: italic;
  color: var(--graphite-soft);
  padding: 2rem 0;
}
.empty code {
  font-family: var(--mono);
  font-style: normal;
  background: var(--vermillion-soft);
  padding: 0.1em 0.4em;
  border-radius: 2px;
}

footer {
  max-width: var(--container-width);
  margin: 5rem auto 3rem;
  padding: 1.5rem;
  border-top: 1px solid var(--hairline);
  font-family: var(--mono);
  font-size: 0.66rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--graphite-soft);
  text-align: center;
}

@media (max-width: 640px) {
  main { padding: 3rem 1.25rem 2rem; }
  ol.archive li a { grid-template-columns: 1fr; grid-template-areas: "id" "title" "meta" "authors"; gap: 0.35rem; }
  .row-id { padding-top: 0; }
  .row-title { font-size: 1.2rem; }
  h1.archive-title { font-size: 2.1rem; }
}

@media (prefers-color-scheme: dark) {
  :root {
    --bone: #1B1A18;
    --bone-soft: #232220;
    --graphite: #ECE5D6;
    --graphite-soft: #948B7B;
    --vermillion: #E36246;
    --vermillion-soft: rgba(227, 98, 70, 0.10);
    --hairline: rgba(236, 229, 214, 0.16);
  }
}
"""


_INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paperread Archive</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT@0,9..144,300..700,30..100;1,9..144,300..700,30..100&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{css}</style>
</head>
<body>
<main>
  <div class="brand">
    <span>Paperread</span>
    <span class="dot">·</span>
    <span>Archive</span>
    <span class="dot">·</span>
    <span>{generated}</span>
  </div>
  <hr class="brand-rule">
  <h1 class="archive-title">Research Briefs</h1>
  <p class="archive-sub"><span class="count">{count}</span> papers · curated by LLM, narrated by Edge TTS</p>
  {body}
</main>
<footer>
  <span>Compiled with paperread</span>
</footer>
</body>
</html>
"""
