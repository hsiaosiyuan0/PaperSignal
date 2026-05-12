from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import arxiv
import pytest

from paperread.arxiv_client import ArxivClient, Paper, SortBy, SortOrder, _default_filename


def _fake_result(
    entry_id: str = "http://arxiv.org/abs/2605.05538v1",
    title: str = "  A Study of Things  ",
    authors: tuple[str, ...] = ("Ada Lovelace", "Alan Turing"),
    summary: str = "  An abstract.  ",
) -> SimpleNamespace:
    return SimpleNamespace(
        entry_id=entry_id,
        title=title,
        authors=[SimpleNamespace(name=n) for n in authors],
        summary=summary,
        published=datetime(2026, 5, 1, tzinfo=UTC),
        updated=datetime(2026, 5, 2, tzinfo=UTC),
        primary_category="cs.LG",
        categories=["cs.LG", "cs.AI"],
        pdf_url="http://arxiv.org/pdf/2605.05538v1",
    )


class TestPaperFromResult:
    def test_extracts_bare_arxiv_id(self) -> None:
        paper = Paper.from_result(_fake_result())  # type: ignore[arg-type]
        assert paper.arxiv_id == "2605.05538v1"

    def test_strips_title_and_summary(self) -> None:
        paper = Paper.from_result(_fake_result())  # type: ignore[arg-type]
        assert paper.title == "A Study of Things"
        assert paper.summary == "An abstract."

    def test_flattens_authors_and_categories(self) -> None:
        paper = Paper.from_result(_fake_result())  # type: ignore[arg-type]
        assert paper.authors == ("Ada Lovelace", "Alan Turing")
        assert paper.categories == ("cs.LG", "cs.AI")
        assert paper.primary_category == "cs.LG"


class TestArxivClientSearch:
    def test_passes_through_query_and_sort(self) -> None:
        client = ArxivClient()
        client._client = MagicMock()
        client._client.results.return_value = iter([_fake_result()])

        with patch("paperread.arxiv_client.arxiv.Search") as search_cls:
            list(
                client.search(
                    "diffusion models",
                    max_results=5,
                    sort_by=SortBy.SUBMITTED,
                    sort_order=SortOrder.ASCENDING,
                )
            )

        search_cls.assert_called_once_with(
            query="diffusion models",
            max_results=5,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Ascending,
        )

    def test_yields_paper_objects(self) -> None:
        client = ArxivClient()
        client._client = MagicMock()
        client._client.results.return_value = iter([_fake_result(), _fake_result()])

        papers = list(client.search("anything"))

        assert len(papers) == 2
        assert all(isinstance(p, Paper) for p in papers)


class TestArxivClientGet:
    def test_empty_input_returns_empty_list(self) -> None:
        client = ArxivClient()
        client._client = MagicMock()
        assert client.get([]) == []
        client._client.results.assert_not_called()

    def test_single_id_wraps_into_list(self) -> None:
        client = ArxivClient()
        client._client = MagicMock()
        client._client.results.return_value = iter([_fake_result()])

        with patch("paperread.arxiv_client.arxiv.Search") as search_cls:
            client.get("2605.05538")

        search_cls.assert_called_once_with(id_list=["2605.05538"])


class TestDefaultFilename:
    def test_sanitizes_unsafe_chars(self) -> None:
        paper = Paper.from_result(_fake_result(title="Foo/Bar: A Study?"))  # type: ignore[arg-type]
        name = _default_filename(paper)
        assert name.endswith(".pdf")
        assert "/" not in name
        assert ":" not in name
        assert "?" not in name

    def test_includes_arxiv_id(self) -> None:
        paper = Paper.from_result(_fake_result())  # type: ignore[arg-type]
        assert _default_filename(paper).startswith("2605.05538v1_")

    @pytest.mark.parametrize("length", [50, 200, 500])
    def test_truncates_long_titles(self, length: int) -> None:
        paper = Paper.from_result(_fake_result(title="x" * length))  # type: ignore[arg-type]
        # arxiv_id prefix (~14 chars) + underscore + up to 80 title chars + ".pdf"
        assert len(_default_filename(paper)) <= 14 + 1 + 80 + 4
