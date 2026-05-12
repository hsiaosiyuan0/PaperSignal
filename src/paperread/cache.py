from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime
from pathlib import Path
from types import TracebackType

from paperread.arxiv_client import Paper

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id         TEXT PRIMARY KEY,
    base_id          TEXT NOT NULL,
    version          INTEGER NOT NULL,
    title            TEXT NOT NULL,
    authors          TEXT NOT NULL,
    summary          TEXT NOT NULL,
    published        TEXT NOT NULL,
    updated          TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    categories       TEXT NOT NULL,
    pdf_url          TEXT,
    abs_url          TEXT NOT NULL,
    pdf_path         TEXT,
    markdown_path    TEXT,
    report_path      TEXT,
    report_dir       TEXT,
    cached_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_papers_base ON papers(base_id, version DESC);
"""


_VERSION_RE = re.compile(r"^(?P<base>.+?)v(?P<version>\d+)$")


def split_arxiv_id(arxiv_id: str) -> tuple[str, int | None]:
    """Split '2605.05538v3' → ('2605.05538', 3); '2605.05538' → ('2605.05538', None)."""
    m = _VERSION_RE.match(arxiv_id)
    if m:
        return m.group("base"), int(m.group("version"))
    return arxiv_id, None


class MetadataCache:
    """SQLite-backed cache for arXiv paper metadata and local artifact paths.

    Lookups accept either a full id (`2605.05538v1`) or a base id (`2605.05538`);
    for base ids the highest-known version is returned.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def _migrate(self) -> None:
        current = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if current == SCHEMA_VERSION:
            return
        with self._conn:
            if current == 0:
                # Fresh DB — create everything at the latest schema.
                self._conn.executescript(_SCHEMA)
            else:
                # Incremental ALTERs for an existing DB.
                if current < 2:
                    # v1 → v2: introduce report_path for LLM-authored analyses.
                    self._conn.execute("ALTER TABLE papers ADD COLUMN report_path TEXT")
                if current < 3:
                    # v2 → v3: report_dir replaces report_path. Old column kept
                    # to avoid surprise data loss; new flow uses report_dir.
                    self._conn.execute("ALTER TABLE papers ADD COLUMN report_dir TEXT")
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MetadataCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- writes ----------------------------------------------------------

    def put(self, paper: Paper) -> None:
        base, version = split_arxiv_id(paper.arxiv_id)
        if version is None:
            # We only persist papers with an explicit version, since `Paper.arxiv_id`
            # always carries one when sourced from the arxiv API.
            raise ValueError(f"refusing to cache paper without version: {paper.arxiv_id!r}")
        self._conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, base_id, version,
                title, authors, summary, published, updated,
                primary_category, categories, pdf_url, abs_url, cached_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                title = excluded.title,
                authors = excluded.authors,
                summary = excluded.summary,
                published = excluded.published,
                updated = excluded.updated,
                primary_category = excluded.primary_category,
                categories = excluded.categories,
                pdf_url = excluded.pdf_url,
                abs_url = excluded.abs_url,
                cached_at = excluded.cached_at
            """,
            (
                paper.arxiv_id,
                base,
                version,
                paper.title,
                json.dumps(list(paper.authors)),
                paper.summary,
                paper.published.isoformat(),
                paper.updated.isoformat(),
                paper.primary_category,
                json.dumps(list(paper.categories)),
                paper.pdf_url,
                paper.abs_url,
                datetime.now().astimezone().isoformat(),
            ),
        )

    def record_pdf(self, arxiv_id: str, pdf_path: Path) -> None:
        self._update_path(arxiv_id, "pdf_path", pdf_path)

    def record_markdown(self, arxiv_id: str, md_path: Path) -> None:
        self._update_path(arxiv_id, "markdown_path", md_path)

    def record_report(self, arxiv_id: str, report_path: Path) -> None:
        self._update_path(arxiv_id, "report_path", report_path)

    def record_report_dir(self, arxiv_id: str, dir_path: Path) -> None:
        self._update_path(arxiv_id, "report_dir", dir_path)

    def _update_path(self, arxiv_id: str, column: str, value: Path) -> None:
        if column not in {"pdf_path", "markdown_path", "report_path", "report_dir"}:
            raise ValueError(f"unknown column {column!r}")
        # Identifier interpolation is safe here because `column` is checked above.
        cursor = self._conn.execute(
            f"UPDATE papers SET {column} = ? WHERE arxiv_id = ?",
            (str(value), arxiv_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(arxiv_id)

    def clear(self) -> None:
        self._conn.execute("DELETE FROM papers")

    def delete(self, arxiv_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM papers WHERE arxiv_id = ?", (arxiv_id,))
        return cursor.rowcount > 0

    # --- reads -----------------------------------------------------------

    def get(self, arxiv_id: str) -> Paper | None:
        row = self._row_for(arxiv_id)
        return _row_to_paper(row) if row is not None else None

    def pdf_path(self, arxiv_id: str) -> Path | None:
        row = self._row_for(arxiv_id)
        if row is None or row["pdf_path"] is None:
            return None
        return Path(row["pdf_path"])

    def markdown_path(self, arxiv_id: str) -> Path | None:
        row = self._row_for(arxiv_id)
        if row is None or row["markdown_path"] is None:
            return None
        return Path(row["markdown_path"])

    def report_path(self, arxiv_id: str) -> Path | None:
        row = self._row_for(arxiv_id)
        if row is None or row["report_path"] is None:
            return None
        return Path(row["report_path"])

    def report_dir(self, arxiv_id: str) -> Path | None:
        row = self._row_for(arxiv_id)
        if row is None or row["report_dir"] is None:
            return None
        return Path(row["report_dir"])

    def list_reports(self) -> Iterator[tuple[Paper, Path]]:
        """Yield (paper, report_dir) for every paper with a built report."""
        with closing(
            self._conn.execute(
                "SELECT * FROM papers WHERE report_dir IS NOT NULL ORDER BY cached_at DESC"
            )
        ) as cur:
            for row in cur:
                yield _row_to_paper(row), Path(row["report_dir"])

    def list_all(self) -> Iterator[Paper]:
        with closing(self._conn.execute("SELECT * FROM papers ORDER BY cached_at DESC")) as cur:
            for row in cur:
                yield _row_to_paper(row)

    def search(self, query: str, *, limit: int = 50) -> list[Paper]:
        """Case-insensitive substring search over title + summary.

        Cheap LIKE-based search; if the cache grows large enough that this is
        slow, swap to an FTS5 virtual table.
        """
        if not query.strip():
            return []
        needle = f"%{query}%"
        rows = self._conn.execute(
            """
            SELECT * FROM papers
            WHERE title LIKE ? COLLATE NOCASE
               OR summary LIKE ? COLLATE NOCASE
            ORDER BY cached_at DESC
            LIMIT ?
            """,
            (needle, needle, limit),
        ).fetchall()
        return [_row_to_paper(row) for row in rows]

    def _row_for(self, arxiv_id: str) -> sqlite3.Row | None:
        base, version = split_arxiv_id(arxiv_id)
        if version is not None:
            return self._conn.execute(
                "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
        return self._conn.execute(
            "SELECT * FROM papers WHERE base_id = ? ORDER BY version DESC LIMIT 1",
            (base,),
        ).fetchone()


def _row_to_paper(row: sqlite3.Row) -> Paper:
    return Paper(
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        authors=tuple(json.loads(row["authors"])),
        summary=row["summary"],
        published=datetime.fromisoformat(row["published"]),
        updated=datetime.fromisoformat(row["updated"]),
        primary_category=row["primary_category"],
        categories=tuple(json.loads(row["categories"])),
        pdf_url=row["pdf_url"],
        abs_url=row["abs_url"],
    )
