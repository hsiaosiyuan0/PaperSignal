from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from paperread.arxiv_client import Paper
from paperread.cache import MetadataCache, split_arxiv_id


def _make_paper(
    arxiv_id: str = "2605.05538v1",
    title: str = "A Study of Things",
) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        authors=("Ada Lovelace", "Alan Turing"),
        summary="An abstract.",
        published=datetime(2026, 5, 1, tzinfo=UTC),
        updated=datetime(2026, 5, 2, tzinfo=UTC),
        primary_category="cs.LG",
        categories=("cs.LG", "cs.AI"),
        pdf_url="http://arxiv.org/pdf/2605.05538v1",
        abs_url="http://arxiv.org/abs/2605.05538v1",
    )


@pytest.fixture
def cache(tmp_path: Path) -> MetadataCache:
    return MetadataCache(tmp_path / "cache.db")


class TestSplitArxivId:
    def test_with_version(self) -> None:
        assert split_arxiv_id("2605.05538v3") == ("2605.05538", 3)

    def test_without_version(self) -> None:
        assert split_arxiv_id("2605.05538") == ("2605.05538", None)

    def test_double_digit_version(self) -> None:
        assert split_arxiv_id("2605.05538v12") == ("2605.05538", 12)

    def test_old_style_id_with_version(self) -> None:
        assert split_arxiv_id("cs/0309136v2") == ("cs/0309136", 2)


class TestPutAndGet:
    def test_round_trip(self, cache: MetadataCache) -> None:
        paper = _make_paper()
        cache.put(paper)
        assert cache.get("2605.05538v1") == paper

    def test_get_returns_none_when_missing(self, cache: MetadataCache) -> None:
        assert cache.get("9999.99999v1") is None

    def test_put_rejects_paper_without_version(self, cache: MetadataCache) -> None:
        with pytest.raises(ValueError, match="without version"):
            cache.put(_make_paper(arxiv_id="2605.05538"))

    def test_get_by_base_id_returns_highest_version(self, cache: MetadataCache) -> None:
        cache.put(_make_paper(arxiv_id="2605.05538v1", title="v1"))
        cache.put(_make_paper(arxiv_id="2605.05538v3", title="v3"))
        cache.put(_make_paper(arxiv_id="2605.05538v2", title="v2"))

        latest = cache.get("2605.05538")
        assert latest is not None
        assert latest.arxiv_id == "2605.05538v3"
        assert latest.title == "v3"

    def test_put_is_upsert(self, cache: MetadataCache) -> None:
        cache.put(_make_paper(title="old"))
        cache.put(_make_paper(title="new"))
        result = cache.get("2605.05538v1")
        assert result is not None
        assert result.title == "new"


class TestPathTracking:
    def test_record_pdf_then_read_back(self, cache: MetadataCache, tmp_path: Path) -> None:
        cache.put(_make_paper())
        pdf = tmp_path / "paper.pdf"
        cache.record_pdf("2605.05538v1", pdf)
        assert cache.pdf_path("2605.05538v1") == pdf

    def test_record_pdf_lookup_via_base_id(self, cache: MetadataCache, tmp_path: Path) -> None:
        cache.put(_make_paper())
        cache.record_pdf("2605.05538v1", tmp_path / "paper.pdf")
        assert cache.pdf_path("2605.05538") == tmp_path / "paper.pdf"

    def test_record_markdown(self, cache: MetadataCache, tmp_path: Path) -> None:
        cache.put(_make_paper())
        md = tmp_path / "paper.md"
        cache.record_markdown("2605.05538v1", md)
        assert cache.markdown_path("2605.05538v1") == md

    def test_record_pdf_for_unknown_paper_raises(
        self, cache: MetadataCache, tmp_path: Path
    ) -> None:
        with pytest.raises(KeyError):
            cache.record_pdf("9999.99999v1", tmp_path / "x.pdf")


class TestListAndClear:
    def test_list_all(self, cache: MetadataCache) -> None:
        cache.put(_make_paper("2605.05538v1"))
        cache.put(_make_paper("2401.12345v1"))
        ids = {p.arxiv_id for p in cache.list_all()}
        assert ids == {"2605.05538v1", "2401.12345v1"}

    def test_clear_removes_all(self, cache: MetadataCache) -> None:
        cache.put(_make_paper())
        cache.clear()
        assert list(cache.list_all()) == []

    def test_delete_one(self, cache: MetadataCache) -> None:
        cache.put(_make_paper("2605.05538v1"))
        cache.put(_make_paper("2401.12345v1"))
        assert cache.delete("2605.05538v1") is True
        assert cache.get("2605.05538v1") is None
        assert cache.get("2401.12345v1") is not None

    def test_delete_missing_returns_false(self, cache: MetadataCache) -> None:
        assert cache.delete("9999.99999v1") is False


class TestSearch:
    def test_matches_title_substring_case_insensitive(self, cache: MetadataCache) -> None:
        cache.put(_make_paper("1234.00001v1", title="Attention Is All You Need"))
        cache.put(_make_paper("1234.00002v1", title="ResNet"))
        hits = cache.search("attention")
        assert [p.arxiv_id for p in hits] == ["1234.00001v1"]

    def test_matches_summary_substring(self, cache: MetadataCache) -> None:
        paper = Paper(
            arxiv_id="1234.00003v1",
            title="Some unrelated title",
            authors=("A",),
            summary="We propose a knowledge retrieval system based on graph neural networks.",
            published=datetime(2026, 1, 1, tzinfo=UTC),
            updated=datetime(2026, 1, 1, tzinfo=UTC),
            primary_category="cs.IR",
            categories=("cs.IR",),
            pdf_url=None,
            abs_url="http://arxiv.org/abs/1234.00003v1",
        )
        cache.put(paper)
        hits = cache.search("knowledge retrieval")
        assert len(hits) == 1
        assert hits[0].arxiv_id == "1234.00003v1"

    def test_empty_query_returns_empty(self, cache: MetadataCache) -> None:
        cache.put(_make_paper())
        assert cache.search("") == []
        assert cache.search("   ") == []

    def test_no_match_returns_empty(self, cache: MetadataCache) -> None:
        cache.put(_make_paper())
        assert cache.search("definitely_not_in_corpus_xyz123") == []

    def test_respects_limit(self, cache: MetadataCache) -> None:
        for i in range(5):
            cache.put(_make_paper(f"1234.0000{i}v1", title=f"shared keyword paper {i}"))
        assert len(cache.search("shared keyword", limit=2)) == 2

    def test_orders_by_cached_at_desc(self, cache: MetadataCache) -> None:
        # Two papers with the same matching title; the one inserted later should come first.
        cache.put(_make_paper("1234.00001v1", title="shared term first"))
        cache.put(_make_paper("1234.00002v1", title="shared term second"))
        hits = cache.search("shared term")
        assert hits[0].arxiv_id == "1234.00002v1"


class TestPersistenceAcrossInstances:
    def test_data_survives_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        with MetadataCache(db) as c:
            c.put(_make_paper())
        with MetadataCache(db) as c:
            assert c.get("2605.05538v1") is not None
