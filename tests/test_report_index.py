from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from paperread.report import IndexEntry, render_index


def _entry(arxiv_id: str = "1706.03762v7", title: str = "Attention Is All You Need") -> IndexEntry:
    return IndexEntry(
        arxiv_id=arxiv_id,
        title=title,
        authors=("Vaswani", "Shazeer", "Parmar"),
        primary_category="cs.CL",
        published="2017-06-12",
        href=f"{arxiv_id}/",
    )


class TestRenderIndex:
    def test_writes_file_with_expected_structure(self, tmp_path: Path) -> None:
        out = render_index([_entry()], tmp_path / "index.html")
        assert out.is_file()
        soup = BeautifulSoup(out.read_text(encoding="utf-8"), "html.parser")
        # Title element on the page.
        assert soup.find("h1", class_="archive-title") is not None
        # Exactly one row in the archive list.
        rows = soup.select("ol.archive > li")
        assert len(rows) == 1
        link = rows[0].find("a")
        assert link is not None
        assert link["href"] == "1706.03762v7/"
        assert "Attention Is All You Need" in link.get_text()

    def test_empty_list_renders_empty_state(self, tmp_path: Path) -> None:
        out = render_index([], tmp_path / "index.html")
        soup = BeautifulSoup(out.read_text(encoding="utf-8"), "html.parser")
        assert soup.select("ol.archive li") == []
        assert soup.find("p", class_="empty") is not None

    def test_authors_truncate_with_et_al(self, tmp_path: Path) -> None:
        entry = IndexEntry(
            arxiv_id="2401.0000v1",
            title="Many Authors",
            authors=("A", "B", "C", "D", "E"),
            primary_category="cs.AI",
            published="2024-01-01",
            href="2401.0000v1/",
        )
        out = render_index([entry], tmp_path / "index.html")
        html = out.read_text(encoding="utf-8")
        assert "A · B · C · et al." in html

    def test_escapes_html_in_title(self, tmp_path: Path) -> None:
        entry = _entry(title="<script>alert('xss')</script>")
        out = render_index([entry], tmp_path / "index.html")
        html = out.read_text(encoding="utf-8")
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_count_shown_in_subhead(self, tmp_path: Path) -> None:
        entries = [_entry(f"id{i}", f"Title {i}") for i in range(3)]
        out = render_index(entries, tmp_path / "index.html")
        soup = BeautifulSoup(out.read_text(encoding="utf-8"), "html.parser")
        count = soup.find("span", class_="count")
        assert count is not None
        assert count.get_text(strip=True) == "3"


class TestPagesInitCommandShape:
    """Smoke-test the workflow-yaml constant for required sections."""

    def test_workflow_has_required_sections(self) -> None:
        from paperread.cli import _NOJEKYLL_NOTE, _WORKFLOW_YML

        assert "github-pages" in _WORKFLOW_YML
        assert "deploy-pages" in _WORKFLOW_YML
        assert "upload-pages-artifact" in _WORKFLOW_YML
        assert "./reports" in _WORKFLOW_YML
        assert _NOJEKYLL_NOTE.strip()  # not empty
