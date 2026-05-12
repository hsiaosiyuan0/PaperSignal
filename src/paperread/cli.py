from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import Annotated, Any

import arxiv as _arxiv
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from paperread.arxiv_client import ArxivClient, Paper, SortBy, SortOrder
from paperread.cache import MetadataCache
from paperread.converter import Backend, make_backend
from paperread.openalex import (
    OpenAlexClient,
    OpenAlexError,
    OpenAlexSort,
    OpenAlexWork,
    parse_year_range,
)
from paperread.paths import cache_db_path, papers_dir, reports_dir
from paperread.report import (
    EN_FILENAME,
    ZH_FILENAME,
    IndexEntry,
    build_report,
    render_index,
)

# Pick up .env (OPENALEX_API_KEY, TORCH_DEVICE, etc.) before any client reads env.
# Clients only read os.environ inside __init__, which happens when commands run.
load_dotenv()

app = typer.Typer(
    name="paperread",
    help="Search and download arXiv papers.",
    no_args_is_help=True,
    add_completion=False,
)
cache_app = typer.Typer(
    name="cache", help="Inspect or clear the metadata cache.", no_args_is_help=True
)
app.add_typer(cache_app)
report_app = typer.Typer(
    name="report",
    help="Track LLM-authored analyses of cached papers and render them as HTML.",
    no_args_is_help=True,
)
app.add_typer(report_app)

console = Console()
err_console = Console(stderr=True)


def _handle_arxiv_errors[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except _arxiv.HTTPError as exc:
            err_console.print(f"[red]arXiv HTTP {exc.status}[/red] — try again shortly.")
            raise typer.Exit(code=2) from exc
        except _arxiv.ArxivError as exc:
            err_console.print(f"[red]arXiv error:[/red] {exc}")
            raise typer.Exit(code=2) from exc

    return wrapper


def _open_cache() -> MetadataCache:
    return MetadataCache(cache_db_path())


def _fetch_paper(cache: MetadataCache, arxiv_id: str, *, refresh: bool) -> Paper:
    """Return one paper, hitting the cache first unless `refresh` is set."""
    if not refresh:
        cached = cache.get(arxiv_id)
        if cached is not None:
            return cached
    client = ArxivClient()
    results = client.get(arxiv_id)
    if not results:
        err_console.print(f"[red]No paper found for id {arxiv_id!r}.[/red]")
        raise typer.Exit(code=1)
    paper = results[0]
    cache.put(paper)
    return paper


@app.command()
@_handle_arxiv_errors
def search(
    query: Annotated[
        str,
        typer.Argument(help="Free-text query, e.g. 'ti:diffusion AND cat:cs.LG'."),
    ],
    max_results: Annotated[int, typer.Option("--max", "-n", min=1, max=2000)] = 10,
    sort_by: Annotated[SortBy, typer.Option("--sort-by")] = SortBy.RELEVANCE,
    sort_order: Annotated[SortOrder, typer.Option("--sort-order")] = SortOrder.DESCENDING,
) -> None:
    """Search arXiv and print results as a table."""
    client = ArxivClient()
    papers = list(
        client.search(
            query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
    )
    if not papers:
        err_console.print("[yellow]No results.[/yellow]")
        raise typer.Exit(code=1)
    with _open_cache() as cache:
        for p in papers:
            cache.put(p)
    _render_table(papers)


@app.command()
def discover(
    query: Annotated[str, typer.Argument(help="Topic keywords, e.g. 'knowledge retrieval'.")],
    affiliation: Annotated[
        str | None,
        typer.Option(
            "--affiliation",
            "-a",
            help="Match author affiliations server-side (e.g. 'Microsoft').",
        ),
    ] = None,
    year: Annotated[
        str | None,
        typer.Option(
            "--year",
            help="Year filter: '2026', '2024-2026', '-2024', '2024-'.",
        ),
    ] = None,
    top: Annotated[int, typer.Option("--top", "-n", min=1, max=200)] = 10,
    sort: Annotated[OpenAlexSort, typer.Option("--sort")] = OpenAlexSort.DATE_DESC,
    arxiv_only: Annotated[
        bool,
        typer.Option(
            "--arxiv-only/--include-non-arxiv",
            help="Show only papers that also have an arXiv id (so they can be downloaded).",
        ),
    ] = True,
) -> None:
    """Discover recent papers via OpenAlex.

    Affiliation filtering happens server-side — unlike arXiv, OpenAlex indexes
    parsed institution affiliations and supports them as a first-class filter.
    """
    client = OpenAlexClient()
    from_date, to_date = parse_year_range(year or "")
    try:
        works = client.search_works(
            query,
            affiliation=affiliation,
            from_date=from_date,
            to_date=to_date,
            sort=sort,
            limit=top * 3 if arxiv_only else top,
        )
    except OpenAlexError as exc:
        err_console.print(f"[red]OpenAlex error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    fetched_count = len(works)
    if arxiv_only:
        works = [w for w in works if w.arxiv_id]
    works = works[:top]

    if not works:
        err_console.print("[yellow]No matching papers.[/yellow]")
        if arxiv_only and fetched_count:
            err_console.print(
                f"[dim]Fetched {fetched_count} papers; none had a linked arXiv id. "
                f"Try --include-non-arxiv to see them anyway.[/dim]"
            )
        raise typer.Exit(code=1)
    _render_discover_table(works, affiliation=affiliation)


@app.command()
@_handle_arxiv_errors
def info(
    arxiv_id: Annotated[str, typer.Argument(help="arXiv id, e.g. '2605.05538' or '2605.05538v1'.")],
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Bypass the cache and re-fetch.")
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a single JSON object instead of a rich table."),
    ] = False,
) -> None:
    """Print metadata for a single paper."""
    with _open_cache() as cache:
        paper = _fetch_paper(cache, arxiv_id, refresh=refresh)
        pdf = cache.pdf_path(paper.arxiv_id)
        md = cache.markdown_path(paper.arxiv_id)
    if json_output:
        _emit_json(_paper_to_dict(paper, pdf, md))
    else:
        _render_detail(paper)


@app.command()
@_handle_arxiv_errors
def download(
    arxiv_id: Annotated[str, typer.Argument(help="arXiv id to download.")],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="Output directory (defaults to platform user data dir).",
        ),
    ] = None,
    filename: Annotated[str | None, typer.Option("--filename", "-f")] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help="Re-download even if cached.")] = False,
) -> None:
    """Download a paper's PDF (cached after first fetch)."""
    out_dir = output or papers_dir()
    with _open_cache() as cache:
        paper = _fetch_paper(cache, arxiv_id, refresh=refresh)
        if not refresh:
            existing = cache.pdf_path(paper.arxiv_id)
            if existing is not None and existing.is_file():
                console.print(f"[green]Already downloaded[/green] {existing}")
                return
        client = ArxivClient()
        with console.status(f"Downloading {paper.arxiv_id}…"):
            path = client.download(paper, output_dir=out_dir, filename=filename).resolve()
        cache.record_pdf(paper.arxiv_id, path)
    console.print(f"[green]Saved[/green] {path}")


@app.command()
@_handle_arxiv_errors
def convert(
    arxiv_id: Annotated[str, typer.Argument(help="arXiv id to convert to Markdown.")],
    backend: Annotated[
        Backend,
        typer.Option(
            "--backend",
            "-b",
            help="marker = high-fidelity (LaTeX equations); markitdown = fast plain text.",
        ),
    ] = Backend.MARKER,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output .md path (defaults next to the PDF)."),
    ] = None,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-convert even if a .md already exists.")
    ] = False,
) -> None:
    """Convert a paper's PDF to Markdown."""
    with _open_cache() as cache:
        paper = _fetch_paper(cache, arxiv_id, refresh=False)
        pdf_path = cache.pdf_path(paper.arxiv_id)
        if pdf_path is None or not pdf_path.is_file():
            client = ArxivClient()
            with console.status(f"Downloading {paper.arxiv_id}…"):
                pdf_path = client.download(paper, output_dir=papers_dir()).resolve()
            cache.record_pdf(paper.arxiv_id, pdf_path)

        md_path = (output or pdf_path.with_suffix(".md")).resolve()
        if not refresh and md_path.is_file():
            console.print(f"[green]Already converted[/green] {md_path}")
            return

        with console.status(f"Converting {pdf_path.name} with {backend.value}…"):
            make_backend(backend).convert(pdf_path, md_path)
        cache.record_markdown(paper.arxiv_id, md_path)
    console.print(f"[green]Markdown[/green] {md_path}")


@cache_app.command("list")
def cache_list(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a JSON array instead of a rich table."),
    ] = False,
) -> None:
    """List papers currently in the metadata cache."""
    with _open_cache() as cache:
        papers = list(cache.list_all())
        if json_output:
            payload = [
                _paper_to_dict(p, cache.pdf_path(p.arxiv_id), cache.markdown_path(p.arxiv_id))
                for p in papers
            ]
            _emit_json(payload)
            return
    if not papers:
        console.print("[dim]Cache is empty.[/dim]")
        return
    _render_table(papers)


@cache_app.command("search")
def cache_search(
    query: Annotated[str, typer.Argument(help="Substring to match against title and summary.")],
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=500)] = 25,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a JSON array instead of a rich table."),
    ] = False,
) -> None:
    """Substring-search the cached metadata (title + abstract). LLM-friendly with --json."""
    with _open_cache() as cache:
        papers = cache.search(query, limit=limit)
        if json_output:
            payload = [
                _paper_to_dict(p, cache.pdf_path(p.arxiv_id), cache.markdown_path(p.arxiv_id))
                for p in papers
            ]
            _emit_json(payload)
            return
    if not papers:
        console.print("[dim]No matches in cache.[/dim]")
        raise typer.Exit(code=1)
    _render_table(papers)


@cache_app.command("show")
def cache_show(
    arxiv_id: Annotated[str, typer.Argument(help="arXiv id (with or without version).")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a JSON object instead of a rich detail view."),
    ] = False,
) -> None:
    """Show one cached paper with all metadata and local file paths."""
    with _open_cache() as cache:
        paper = cache.get(arxiv_id)
        if paper is None:
            err_console.print(f"[red]Not in cache:[/red] {arxiv_id}")
            raise typer.Exit(code=1)
        pdf = cache.pdf_path(paper.arxiv_id)
        md = cache.markdown_path(paper.arxiv_id)
    if json_output:
        _emit_json(_paper_to_dict(paper, pdf, md))
    else:
        _render_detail(paper)
        if pdf:
            console.print(f"[dim]PDF path:[/dim] {pdf}")
        if md:
            console.print(f"[dim]Markdown path:[/dim] {md}")


@cache_app.command("clear")
def cache_clear(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Remove all cached metadata (does not delete PDF/Markdown files)."""
    if not yes:
        typer.confirm("Clear all cached metadata?", abort=True)
    with _open_cache() as cache:
        cache.clear()
    console.print("[green]Cache cleared.[/green]")


@cache_app.command("path")
def cache_path() -> None:
    """Print the SQLite cache file path."""
    console.print(str(cache_db_path()))


class _Lang(StrEnum):
    EN = "en"
    ZH = "zh"


def _report_dir_for(paper: Paper) -> Path:
    return (reports_dir() / paper.arxiv_id).resolve()


@report_app.command("path")
def report_path(
    arxiv_id: Annotated[str, typer.Argument(help="arXiv id (paper must be cached first).")],
    lang: Annotated[
        _Lang | None,
        typer.Option("--lang", help="If set, print the per-language .md path instead of the dir."),
    ] = None,
) -> None:
    """Print the directory (or per-language .md path) where the LLM should write the report.

    The LLM should create `report.en.md` and `report.zh.md` inside the directory,
    then run `paperread report build <id>` to generate audio + HTML.
    """
    with _open_cache() as cache:
        paper = cache.get(arxiv_id)
        if paper is None:
            err_console.print(f"[red]Not in cache:[/red] {arxiv_id}  (run `paperread info` first)")
            raise typer.Exit(code=1)
    dir_ = _report_dir_for(paper)
    dir_.mkdir(parents=True, exist_ok=True)
    if lang is _Lang.EN:
        console.print(str(dir_ / EN_FILENAME))
    elif lang is _Lang.ZH:
        console.print(str(dir_ / ZH_FILENAME))
    else:
        console.print(str(dir_))


@report_app.command("build")
def report_build(
    arxiv_id: Annotated[str, typer.Argument(help="arXiv id (paper must be cached first).")],
    no_audio: Annotated[
        bool,
        typer.Option("--no-audio", help="Skip TTS generation (much faster, no mp3s produced)."),
    ] = False,
) -> None:
    """Build the bilingual HTML report + per-sentence audio in the paper's report dir.

    Expects `report.en.md` (required) and optionally `report.zh.md` to exist in
    the directory printed by `paperread report path <id>`.
    """
    with _open_cache() as cache:
        paper = cache.get(arxiv_id)
        if paper is None:
            err_console.print(f"[red]Not in cache:[/red] {arxiv_id}")
            raise typer.Exit(code=1)
    report_dir_path = _report_dir_for(paper)
    en_md = report_dir_path / EN_FILENAME
    if not en_md.is_file():
        err_console.print(
            f"[red]Missing[/red] {en_md}\n"
            f"[dim]Have your LLM write {EN_FILENAME} (and optionally {ZH_FILENAME}) into "
            f"{report_dir_path} first.[/dim]"
        )
        raise typer.Exit(code=1)

    msg = f"Building {paper.arxiv_id}…"
    if no_audio:
        msg += " (no audio)"
    with console.status(msg) as status:
        progress_count = [0]

        def on_audio(filename: str) -> None:
            progress_count[0] += 1
            status.update(f"Synthesizing {filename} ({progress_count[0]} sentences done)")

        result = build_report(
            paper,
            report_dir_path,
            with_audio=not no_audio,
            on_audio_progress=on_audio if not no_audio else None,
        )

    with _open_cache() as cache:
        cache.record_report_dir(paper.arxiv_id, report_dir_path)

    audio_note = f" + {len(result.audio_entries)} audio clips" if result.audio_entries else ""
    zh_note = " + zh" if result.has_zh else ""
    console.print(f"[green]Built[/green] {result.html_path}{audio_note}{zh_note}")


@report_app.command("list")
def report_list(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit a JSON array instead of a rich table.")
    ] = False,
) -> None:
    """List all papers that have a built report directory."""
    with _open_cache() as cache:
        entries = list(cache.list_reports())
        if json_output:
            payload = [
                _paper_to_dict(p, cache.pdf_path(p.arxiv_id), cache.markdown_path(p.arxiv_id))
                | {"report_dir": str(rd)}
                for p, rd in entries
            ]
            _emit_json(payload)
            return
    if not entries:
        console.print("[dim]No reports built yet.[/dim]")
        return
    table = Table(show_lines=False, header_style="bold")
    table.add_column("arXiv ID", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Report dir", style="dim")
    for paper, rd in entries:
        table.add_row(paper.arxiv_id, paper.title, str(rd))
    console.print(table)


@report_app.command("show")
def report_show(
    arxiv_id: Annotated[str, typer.Argument(help="arXiv id (with or without version).")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit a JSON object instead of a rich detail view.")
    ] = False,
) -> None:
    """Show the built report's directory and metadata."""
    with _open_cache() as cache:
        paper = cache.get(arxiv_id)
        if paper is None:
            err_console.print(f"[red]Not in cache:[/red] {arxiv_id}")
            raise typer.Exit(code=1)
        rd = cache.report_dir(paper.arxiv_id)
        if rd is None:
            err_console.print(
                f"[red]No report built for[/red] {paper.arxiv_id}  "
                f"[dim](use `paperread report build`)[/dim]"
            )
            raise typer.Exit(code=1)
        pdf = cache.pdf_path(paper.arxiv_id)
        md = cache.markdown_path(paper.arxiv_id)
    if json_output:
        _emit_json(
            _paper_to_dict(paper, pdf, md)
            | {
                "report_dir": str(rd),
                "report_html": str(rd / "index.html"),
                "report_en": str(rd / EN_FILENAME),
                "report_zh": str(rd / ZH_FILENAME) if (rd / ZH_FILENAME).is_file() else None,
            }
        )
    else:
        _render_detail(paper)
        console.print(f"[dim]Report dir:[/dim] {rd}")
        console.print(f"[dim]HTML:[/dim] {rd / 'index.html'}")


@report_app.command("open")
def report_open(
    arxiv_id: Annotated[str, typer.Argument(help="arXiv id (with or without version).")],
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Don't open the browser; just print the path.")
    ] = False,
) -> None:
    """Open the built report's index.html in the browser (does not rebuild)."""
    with _open_cache() as cache:
        paper = cache.get(arxiv_id)
        if paper is None:
            err_console.print(f"[red]Not in cache:[/red] {arxiv_id}")
            raise typer.Exit(code=1)
        rd = cache.report_dir(paper.arxiv_id)
        if rd is None:
            err_console.print(
                f"[red]No report built for[/red] {paper.arxiv_id}  "
                f"[dim](run `paperread report build <id>`)[/dim]"
            )
            raise typer.Exit(code=1)
    html = rd / "index.html"
    if not html.is_file():
        err_console.print(f"[red]index.html missing in[/red] {rd}")
        raise typer.Exit(code=1)
    console.print(str(html))
    if not no_browser:
        import webbrowser

        webbrowser.open(html.as_uri())


@report_app.command("index")
def report_index(
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Don't open the browser; just print the path.")
    ] = False,
) -> None:
    """Rebuild reports/index.html listing every built report.

    Run this after each `report build` if you want the archive landing page
    to stay in sync. Cheap — no network, no audio.
    """
    root = reports_dir()
    root.mkdir(parents=True, exist_ok=True)
    with _open_cache() as cache:
        entries: list[IndexEntry] = []
        for paper, rd in cache.list_reports():
            try:
                rel = rd.relative_to(root.resolve())
            except ValueError:
                # Report dir isn't under the current reports root; skip from index.
                continue
            entries.append(
                IndexEntry(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=paper.authors,
                    primary_category=paper.primary_category,
                    published=paper.published.strftime("%Y-%m-%d"),
                    href=f"{rel.as_posix()}/",
                )
            )
    out = render_index(entries, root / "index.html")
    console.print(f"[green]Wrote[/green] {out}  [dim]({len(entries)} entries)[/dim]")
    if not no_browser:
        import webbrowser

        webbrowser.open(out.as_uri())


_NOJEKYLL_NOTE = "# Empty — disables Jekyll on GitHub Pages so '_'-prefixed dirs stay served.\n"
_WORKFLOW_YML = """\
name: Deploy reports to GitHub Pages

on:
  push:
    branches: [main]
    paths:
      - 'reports/**'
      - '.github/workflows/deploy-pages.yml'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
        with:
          # Auto-enable Pages on first run so we don't need a manual settings click.
          enablement: true
      - uses: actions/upload-pages-artifact@v3
        with:
          path: ./reports
      - id: deployment
        uses: actions/deploy-pages@v4
"""


@report_app.command("pages-init")
def report_pages_init(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing files."),
    ] = False,
) -> None:
    """Scaffold GitHub Pages deployment: `.nojekyll` + Actions workflow.

    Run this once at the root of the repo that holds your `reports/` directory.
    After it: enable Pages in the repo settings (Source = GitHub Actions), push
    to main, and your reports go live at https://<user>.github.io/<repo>/.
    """
    root = reports_dir()
    root.mkdir(parents=True, exist_ok=True)
    nojekyll = root / ".nojekyll"
    workflow_dir = Path(".github") / "workflows"
    workflow = workflow_dir / "deploy-pages.yml"

    files = {
        nojekyll: _NOJEKYLL_NOTE,
        workflow: _WORKFLOW_YML,
    }
    for path, body in files.items():
        if path.is_file() and not force:
            console.print(f"[dim]skip[/dim] {path}  [dim](exists; use --force)[/dim]")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        console.print(f"[green]wrote[/green] {path}")

    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Commit and push the new files.")
    console.print("  2. In GitHub: Settings → Pages → Source = GitHub Actions.")
    console.print("  3. Each push that touches reports/ will redeploy.")


def _paper_to_dict(
    paper: Paper, pdf_path: Path | None, markdown_path: Path | None
) -> dict[str, Any]:
    """Stable JSON schema for cached papers. Consumers (LLMs) depend on this."""
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": list(paper.authors),
        "summary": paper.summary,
        "published": paper.published.isoformat(),
        "updated": paper.updated.isoformat(),
        "primary_category": paper.primary_category,
        "categories": list(paper.categories),
        "pdf_url": paper.pdf_url,
        "abs_url": paper.abs_url,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "markdown_path": str(markdown_path) if markdown_path else None,
    }


def _emit_json(payload: Any) -> None:
    """Write JSON directly to stdout, bypassing rich's wrapping/coloring."""
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _render_table(papers: list[Paper]) -> None:
    table = Table(show_lines=False, header_style="bold")
    table.add_column("arXiv ID", style="cyan", no_wrap=True)
    table.add_column("Published", no_wrap=True)
    table.add_column("Title")
    table.add_column("Authors", style="dim")
    for p in papers:
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        table.add_row(p.arxiv_id, p.published.strftime("%Y-%m-%d"), p.title, authors)
    console.print(table)


def _render_discover_table(works: list[OpenAlexWork], *, affiliation: str | None) -> None:
    table = Table(show_lines=False, header_style="bold")
    table.add_column("arXiv ID", style="cyan", no_wrap=True)
    table.add_column("Date", no_wrap=True)
    table.add_column("Cites", justify="right", style="magenta")
    table.add_column("Title")
    table.add_column("Authors", style="dim")
    for w in works:
        date = w.publication_date or (str(w.year) if w.year else "—")
        authors_display = _format_discover_authors(w, affiliation)
        table.add_row(
            w.arxiv_id or "—",
            date,
            str(w.cited_by_count),
            w.title,
            authors_display,
        )
    console.print(table)


def _format_discover_authors(work: OpenAlexWork, affiliation: str | None) -> str:
    """Show top 3 authors; highlight ones whose institution matches the filter."""
    needle = affiliation.lower() if affiliation else None
    chunks: list[str] = []
    for author in work.authors[:3]:
        match = needle is not None and any(needle in inst.lower() for inst in author.institutions)
        chunks.append(f"[green]{author.name}[/green]" if match else author.name)
    suffix = " et al." if len(work.authors) > 3 else ""
    return ", ".join(chunks) + suffix


def _render_detail(paper: Paper) -> None:
    console.print(f"[bold cyan]{paper.arxiv_id}[/bold cyan]  {paper.abs_url}")
    console.print(f"[bold]{paper.title}[/bold]")
    console.print(f"[dim]Authors:[/dim] {', '.join(paper.authors)}")
    console.print(
        f"[dim]Published:[/dim] {paper.published:%Y-%m-%d}  "
        f"[dim]Updated:[/dim] {paper.updated:%Y-%m-%d}  "
        f"[dim]Category:[/dim] {paper.primary_category}"
    )
    console.print(f"[dim]PDF:[/dim] {paper.pdf_url}")
    console.print()
    console.print(paper.summary)
