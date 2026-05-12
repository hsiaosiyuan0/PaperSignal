from __future__ import annotations

import os
from pathlib import Path

from platformdirs import PlatformDirs

_APP_NAME = "paperread"
_dirs = PlatformDirs(appname=_APP_NAME)


def cache_db_path() -> Path:
    """Path to the SQLite cache DB. Honors PAPERREAD_CACHE_DIR for tests."""
    base = os.environ.get("PAPERREAD_CACHE_DIR")
    root = Path(base) if base else Path(_dirs.user_cache_dir)
    return root / "cache.db"


def papers_dir() -> Path:
    """Default directory for downloaded PDFs and converted markdown.

    Defaults to `./papers` (relative to the caller's cwd) so the user-facing
    artifacts land where the user can actually find them. Override with
    PAPERREAD_DATA_DIR if you want a fixed library location.
    """
    base = os.environ.get("PAPERREAD_DATA_DIR")
    if base:
        return Path(base)
    return Path("papers")


def reports_dir() -> Path:
    """Root directory for per-paper report bundles.

    Each paper gets its own subdirectory `<root>/<arxiv_id>/` containing
    `report.en.md`, `report.zh.md`, `index.html`, and `audio/`. Defaults
    to `./reports` so the whole root can be pushed to a static host
    (GitHub Pages) as-is. Override via PAPERREAD_REPORTS_DIR.
    """
    base = os.environ.get("PAPERREAD_REPORTS_DIR")
    if base:
        return Path(base)
    return Path("reports")
