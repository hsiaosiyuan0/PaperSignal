"""Tests for the JSON output contract — what LLMs and machine consumers see."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from paperread.arxiv_client import Paper
from paperread.cache import MetadataCache
from paperread.cli import _paper_to_dict, app

runner = CliRunner()

EXPECTED_KEYS = {
    "arxiv_id",
    "title",
    "authors",
    "summary",
    "published",
    "updated",
    "primary_category",
    "categories",
    "pdf_url",
    "abs_url",
    "pdf_path",
    "markdown_path",
}


def _make_paper(arxiv_id: str = "1706.03762v7") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title="Attention Is All You Need",
        authors=("Vaswani", "Shazeer"),
        summary="A sequence transduction model based on attention.",
        published=datetime(2017, 6, 12, tzinfo=UTC),
        updated=datetime(2023, 8, 2, tzinfo=UTC),
        primary_category="cs.CL",
        categories=("cs.CL", "cs.LG"),
        pdf_url="http://arxiv.org/pdf/1706.03762v7",
        abs_url="http://arxiv.org/abs/1706.03762v7",
    )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point paperread at a fresh tmp cache + data dir."""
    monkeypatch.setenv("PAPERREAD_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PAPERREAD_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


class TestPaperToDictSchema:
    def test_includes_all_expected_keys(self) -> None:
        d = _paper_to_dict(_make_paper(), None, None)
        assert set(d.keys()) == EXPECTED_KEYS

    def test_serializes_to_json_without_error(self) -> None:
        d = _paper_to_dict(_make_paper(), Path("/abs/x.pdf"), Path("/abs/x.md"))
        s = json.dumps(d)
        round_tripped = json.loads(s)
        assert round_tripped["arxiv_id"] == "1706.03762v7"
        assert round_tripped["pdf_path"] == "/abs/x.pdf"
        assert round_tripped["markdown_path"] == "/abs/x.md"

    def test_null_paths_serialize_as_json_null(self) -> None:
        d = _paper_to_dict(_make_paper(), None, None)
        assert d["pdf_path"] is None
        assert d["markdown_path"] is None

    def test_datetimes_are_iso_strings(self) -> None:
        d = _paper_to_dict(_make_paper(), None, None)
        assert d["published"] == "2017-06-12T00:00:00+00:00"
        assert d["updated"] == "2023-08-02T00:00:00+00:00"

    def test_tuples_become_lists(self) -> None:
        d = _paper_to_dict(_make_paper(), None, None)
        assert isinstance(d["authors"], list)
        assert isinstance(d["categories"], list)


class TestCacheJsonCommands:
    def _seed(self, env_dir: Path) -> None:
        from paperread.paths import cache_db_path

        with MetadataCache(cache_db_path()) as c:
            c.put(_make_paper("1706.03762v7"))
            c.put(_make_paper("2401.12345v1"))

    def test_cache_list_json(self, env: Path) -> None:
        self._seed(env)
        result = runner.invoke(app, ["cache", "list", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        assert len(payload) == 2
        for item in payload:
            assert set(item.keys()) == EXPECTED_KEYS

    def test_cache_search_json_hits(self, env: Path) -> None:
        self._seed(env)
        result = runner.invoke(app, ["cache", "search", "attention", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert len(payload) == 2  # both seeded papers have "Attention" in the title

    def test_cache_search_json_no_hits(self, env: Path) -> None:
        self._seed(env)
        result = runner.invoke(app, ["cache", "search", "xyz_not_present", "--json"])
        # --json mode never prints "no matches" prose; it just emits []
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == []

    def test_cache_show_json(self, env: Path) -> None:
        self._seed(env)
        result = runner.invoke(app, ["cache", "show", "1706.03762", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["arxiv_id"] == "1706.03762v7"
        assert set(payload.keys()) == EXPECTED_KEYS

    def test_cache_show_missing_exits_nonzero(self, env: Path) -> None:
        self._seed(env)
        result = runner.invoke(app, ["cache", "show", "9999.99999", "--json"])
        assert result.exit_code == 1
