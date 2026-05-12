from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import arxiv
import requests

DEFAULT_USER_AGENT = "paperread/0.1.0 (+https://github.com/hsiaosiyuan/paperread)"


class SortBy(StrEnum):
    RELEVANCE = "relevance"
    LAST_UPDATED = "last_updated"
    SUBMITTED = "submitted"


class SortOrder(StrEnum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


_SORT_BY_MAP = {
    SortBy.RELEVANCE: arxiv.SortCriterion.Relevance,
    SortBy.LAST_UPDATED: arxiv.SortCriterion.LastUpdatedDate,
    SortBy.SUBMITTED: arxiv.SortCriterion.SubmittedDate,
}

_SORT_ORDER_MAP = {
    SortOrder.ASCENDING: arxiv.SortOrder.Ascending,
    SortOrder.DESCENDING: arxiv.SortOrder.Descending,
}


@dataclass(frozen=True, slots=True)
class Paper:
    """A normalized view of an arXiv paper that hides the upstream SDK type."""

    arxiv_id: str
    title: str
    authors: tuple[str, ...]
    summary: str
    published: datetime
    updated: datetime
    primary_category: str
    categories: tuple[str, ...]
    pdf_url: str | None
    abs_url: str

    @classmethod
    def from_result(cls, result: arxiv.Result) -> Paper:
        # entry_id looks like "http://arxiv.org/abs/2605.05538v1"; keep the bare id.
        arxiv_id = result.entry_id.rsplit("/", 1)[-1]
        return cls(
            arxiv_id=arxiv_id,
            title=result.title.strip(),
            authors=tuple(a.name for a in result.authors),
            summary=result.summary.strip(),
            published=result.published,
            updated=result.updated,
            primary_category=result.primary_category,
            categories=tuple(result.categories),
            pdf_url=result.pdf_url,
            abs_url=result.entry_id,
        )


class ArxivClient:
    """High-level client wrapping the `arxiv` library.

    The wrapper exists so callers depend on `Paper` (a stable, owned type)
    rather than on the upstream SDK's `Result` object.
    """

    def __init__(
        self,
        *,
        page_size: int = 100,
        delay_seconds: float = 3.0,
        num_retries: int = 3,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._client = arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries,
        )
        # arxiv.py hard-codes its own UA per-request, which arxiv.org throttles
        # aggressively. Swap in a session that overrides that header.
        self._client._session = _UserAgentSession(user_agent)

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        sort_by: SortBy = SortBy.RELEVANCE,
        sort_order: SortOrder = SortOrder.DESCENDING,
    ) -> Iterator[Paper]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=_SORT_BY_MAP[sort_by],
            sort_order=_SORT_ORDER_MAP[sort_order],
        )
        for result in self._client.results(search):
            yield Paper.from_result(result)

    def get(self, arxiv_ids: str | Iterable[str]) -> list[Paper]:
        ids = [arxiv_ids] if isinstance(arxiv_ids, str) else list(arxiv_ids)
        if not ids:
            return []
        search = arxiv.Search(id_list=ids)
        return [Paper.from_result(r) for r in self._client.results(search)]

    def download(
        self,
        paper: Paper,
        output_dir: Path,
        *,
        filename: str | None = None,
        chunk_size: int = 64 * 1024,
        retries: int = 3,
        timeout: float = 60.0,
    ) -> Path:
        """Stream the PDF to disk via our session.

        Bypasses arxiv.py's `download_pdf`, which uses `urllib.urlretrieve` and
        is prone to mid-stream truncation on large files.
        """
        if paper.pdf_url is None:
            raise ValueError(f"No PDF URL available for {paper.arxiv_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / (filename or _default_filename(paper))
        tmp = target.with_suffix(target.suffix + ".part")

        last_err: Exception | None = None
        for _ in range(retries):
            try:
                with self._client._session.get(paper.pdf_url, stream=True, timeout=timeout) as resp:
                    resp.raise_for_status()
                    expected = int(resp.headers.get("content-length") or 0)
                    with tmp.open("wb") as f:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            f.write(chunk)
                actual = tmp.stat().st_size
                if expected and actual < expected:
                    raise OSError(f"incomplete download: {actual}/{expected} bytes")
                tmp.replace(target)
                return target
            except (requests.RequestException, OSError) as exc:
                last_err = exc
                tmp.unlink(missing_ok=True)
        assert last_err is not None
        raise last_err


class _UserAgentSession(requests.Session):
    """Force a custom User-Agent, overriding any per-request override."""

    def __init__(self, user_agent: str) -> None:
        super().__init__()
        self._user_agent = user_agent

    def request(self, *args: Any, **kwargs: Any) -> requests.Response:
        headers = dict(kwargs.get("headers") or {})
        headers["user-agent"] = self._user_agent
        kwargs["headers"] = headers
        return super().request(*args, **kwargs)


def _default_filename(paper: Paper) -> str:
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in paper.title)
    safe_title = "_".join(safe_title.split())[:80]
    return f"{paper.arxiv_id}_{safe_title}.pdf"
