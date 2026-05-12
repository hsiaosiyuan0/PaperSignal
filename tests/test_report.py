from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from paperread.arxiv_client import Paper
from paperread.cache import MetadataCache
from paperread.report import EN_FILENAME, ZH_FILENAME, build_report


def _make_paper() -> Paper:
    return Paper(
        arxiv_id="1706.03762v7",
        title="Attention Is All You Need",
        authors=("Ashish Vaswani", "Noam Shazeer"),
        summary="abstract",
        published=datetime(2017, 6, 12, tzinfo=UTC),
        updated=datetime(2023, 8, 2, tzinfo=UTC),
        primary_category="cs.CL",
        categories=("cs.CL",),
        pdf_url="http://arxiv.org/pdf/1706.03762v7",
        abs_url="http://arxiv.org/abs/1706.03762v7",
    )


def _setup_dir(tmp_path: Path, *, with_zh: bool = False) -> Path:
    d = tmp_path / "1706.03762v7"
    d.mkdir()
    (d / EN_FILENAME).write_text(
        "## Background\n\nA test of two sentences. The second sentence follows.\n",
        encoding="utf-8",
    )
    if with_zh:
        (d / ZH_FILENAME).write_text(
            "## 背景\n\n这是中文测试段落。\n", encoding="utf-8"
        )
    return d


class TestBuildReport:
    def test_writes_index_html(self, tmp_path: Path) -> None:
        d = _setup_dir(tmp_path)
        result = build_report(_make_paper(), d, with_audio=False)
        assert result.html_path == d / "index.html"
        assert result.html_path.is_file()
        assert result.audio_entries == []
        assert result.has_zh is False

    def test_embeds_metadata(self, tmp_path: Path) -> None:
        d = _setup_dir(tmp_path)
        build_report(_make_paper(), d, with_audio=False)
        html = (d / "index.html").read_text(encoding="utf-8")
        assert "Attention Is All You Need" in html
        assert "1706.03762v7" in html
        assert "Ashish Vaswani" in html

    def test_bilingual_includes_toggle_and_both_bodies(self, tmp_path: Path) -> None:
        d = _setup_dir(tmp_path, with_zh=True)
        result = build_report(_make_paper(), d, with_audio=False)
        assert result.has_zh is True
        html = (d / "index.html").read_text(encoding="utf-8")
        assert 'data-lang="en"' in html
        assert 'data-lang="zh"' in html
        assert "lang-switch" in html
        assert "背景" in html

    def test_monolingual_no_toggle(self, tmp_path: Path) -> None:
        from bs4 import BeautifulSoup

        d = _setup_dir(tmp_path, with_zh=False)
        build_report(_make_paper(), d, with_audio=False)
        soup = BeautifulSoup((d / "index.html").read_text(encoding="utf-8"), "html.parser")
        assert soup.find("div", class_="lang-switch") is None
        assert soup.find("div", attrs={"data-lang": "zh"}) is None

    def test_with_audio_creates_mp3_and_manifest(self, tmp_path: Path) -> None:
        d = _setup_dir(tmp_path)
        with patch("paperread.tts.asyncio.run") as run_mock:
            # Pretend synthesis succeeded but create empty .mp3 files via side effect.
            def fake_run(coro):
                coro.close()
                for i in range(1, 3):  # two sentences in our fixture
                    (d / "audio" / f"s{i:04d}.mp3").write_bytes(b"\x00")
                return []  # no failures

            run_mock.side_effect = fake_run
            result = build_report(_make_paper(), d, with_audio=True)
        assert len(result.audio_entries) == 2
        assert (d / "audio" / ".manifest.json").is_file()

    def test_audio_skipped_when_with_audio_false(self, tmp_path: Path) -> None:
        d = _setup_dir(tmp_path)
        result = build_report(_make_paper(), d, with_audio=False)
        assert result.audio_entries == []
        assert not (d / "audio").exists()

    def test_missing_en_md_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with patch("paperread.tts.asyncio.run"):
            try:
                build_report(_make_paper(), d, with_audio=False)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("expected FileNotFoundError")


class TestSentenceWrapping:
    def test_audio_spans_wrap_prose_when_audio_present(self, tmp_path: Path) -> None:
        d = _setup_dir(tmp_path)
        with patch("paperread.tts.asyncio.run") as run_mock:
            def fake_run(coro):
                coro.close()
                for i in range(1, 3):
                    (d / "audio" / f"s{i:04d}.mp3").write_bytes(b"\x00")
                return []

            run_mock.side_effect = fake_run
            build_report(_make_paper(), d, with_audio=True)
        html = (d / "index.html").read_text(encoding="utf-8")
        assert 'class="sentence"' in html
        assert 'data-audio="audio/s0001.mp3"' in html

    def test_no_sentence_spans_when_audio_off(self, tmp_path: Path) -> None:
        d = _setup_dir(tmp_path)
        build_report(_make_paper(), d, with_audio=False)
        html = (d / "index.html").read_text(encoding="utf-8")
        assert 'class="sentence"' not in html


class TestCacheReportDir:
    def test_record_and_read_back(self, tmp_path: Path) -> None:
        cache = MetadataCache(tmp_path / "cache.db")
        cache.put(_make_paper())
        d = tmp_path / "report_dir"
        d.mkdir()
        cache.record_report_dir("1706.03762v7", d)
        assert cache.report_dir("1706.03762v7") == d
        cache.close()

    def test_list_reports_uses_report_dir(self, tmp_path: Path) -> None:
        cache = MetadataCache(tmp_path / "cache.db")
        cache.put(_make_paper())
        cache.put(
            Paper(
                arxiv_id="2401.12345v1",
                title="other",
                authors=("a",),
                summary="x",
                published=datetime(2024, 1, 1, tzinfo=UTC),
                updated=datetime(2024, 1, 1, tzinfo=UTC),
                primary_category="cs.AI",
                categories=("cs.AI",),
                pdf_url=None,
                abs_url="http://arxiv.org/abs/2401.12345v1",
            )
        )
        d = tmp_path / "r1"
        d.mkdir()
        cache.record_report_dir("1706.03762v7", d)
        registered = list(cache.list_reports())
        assert len(registered) == 1
        assert registered[0][0].arxiv_id == "1706.03762v7"
        assert registered[0][1] == d
        cache.close()


class TestSchemaMigration:
    def test_v2_db_upgrades_to_v3(self, tmp_path: Path) -> None:
        import sqlite3

        db = tmp_path / "v2.db"
        with sqlite3.connect(db) as conn:
            conn.executescript(
                """
                CREATE TABLE papers (
                    arxiv_id TEXT PRIMARY KEY, base_id TEXT NOT NULL, version INTEGER NOT NULL,
                    title TEXT NOT NULL, authors TEXT NOT NULL, summary TEXT NOT NULL,
                    published TEXT NOT NULL, updated TEXT NOT NULL,
                    primary_category TEXT NOT NULL, categories TEXT NOT NULL,
                    pdf_url TEXT, abs_url TEXT NOT NULL,
                    pdf_path TEXT, markdown_path TEXT, report_path TEXT,
                    cached_at TEXT NOT NULL
                );
                """
            )
            conn.execute("PRAGMA user_version = 2")
        cache = MetadataCache(db)
        cache.put(_make_paper())
        cache.record_report_dir("1706.03762v7", tmp_path)
        assert cache.report_dir("1706.03762v7") == tmp_path
        cache.close()

    def test_fresh_db_at_v3(self, tmp_path: Path) -> None:
        import sqlite3

        db = tmp_path / "fresh.db"
        MetadataCache(db).close()
        with sqlite3.connect(db) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
