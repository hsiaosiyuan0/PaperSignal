from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import requests

API_BASE = "https://api.openalex.org"

_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)


class OpenAlexSort(StrEnum):
    DATE_DESC = "publication_date:desc"
    DATE_ASC = "publication_date:asc"
    CITATIONS_DESC = "cited_by_count:desc"
    RELEVANCE = "relevance_score:desc"


@dataclass(frozen=True, slots=True)
class OpenAlexAuthor:
    name: str
    institutions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpenAlexWork:
    work_id: str
    title: str
    publication_date: str | None
    year: int | None
    cited_by_count: int
    authors: tuple[OpenAlexAuthor, ...]
    arxiv_id: str | None
    doi: str | None
    abstract: str | None
    landing_url: str | None

    @classmethod
    def from_json(cls, item: dict[str, Any]) -> OpenAlexWork:
        ids = item.get("ids") or {}
        return cls(
            work_id=item.get("id") or "",
            title=item.get("title") or item.get("display_name") or "",
            publication_date=item.get("publication_date"),
            year=item.get("publication_year"),
            cited_by_count=int(item.get("cited_by_count") or 0),
            authors=tuple(
                OpenAlexAuthor(
                    name=(a.get("author") or {}).get("display_name") or "",
                    institutions=tuple(
                        i.get("display_name") or "" for i in a.get("institutions") or ()
                    ),
                )
                for a in item.get("authorships") or ()
            ),
            arxiv_id=_extract_arxiv_id(item),
            doi=(ids.get("doi") or "").removeprefix("https://doi.org/") or None,
            abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
            landing_url=ids.get("openalex"),
        )


class OpenAlexError(RuntimeError):
    """Raised when the OpenAlex API returns a non-recoverable error."""


class OpenAlexClient:
    """Minimal client for the OpenAlex /works endpoint.

    Affiliations are first-class: pass `affiliation="Microsoft"` and the API
    matches against author affiliation strings server-side, no client filtering
    needed. Date filtering is native too.

    OpenAlex requires an API key as of Feb 2026 for sustained use. Pass via
    `api_key=` or `OPENALEX_API_KEY` env var. Without a key you get a small
    burst of free calls, then 409 errors.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        user_agent: str = "paperread/0.1.0",
        timeout: float = 30.0,
    ) -> None:
        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent
        self._api_key = api_key or os.environ.get("OPENALEX_API_KEY")
        self._timeout = timeout

    def search_works(
        self,
        query: str,
        *,
        affiliation: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        sort: OpenAlexSort = OpenAlexSort.DATE_DESC,
        limit: int = 25,
        max_retries: int = 4,
    ) -> list[OpenAlexWork]:
        filters: list[str] = []
        if affiliation:
            filters.append(f"raw_affiliation_strings.search:{affiliation}")
        if from_date:
            filters.append(f"from_publication_date:{from_date}")
        if to_date:
            filters.append(f"to_publication_date:{to_date}")

        params: dict[str, str] = {
            "search": query,
            "sort": sort.value,
            "per-page": str(min(max(1, limit), 200)),
        }
        if filters:
            params["filter"] = ",".join(filters)
        if self._api_key:
            params["api_key"] = self._api_key

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self._session.get(f"{API_BASE}/works", params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2**attempt)
                continue

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After") or max(5, 2**attempt))
                time.sleep(retry_after)
                continue
            if resp.status_code == 409:
                # OpenAlex returns 409 when the unauthenticated quota is exhausted.
                raise OpenAlexError(
                    "OpenAlex says quota exhausted. Set OPENALEX_API_KEY "
                    "(free signup at https://openalex.org/) for sustained use."
                )
            if resp.status_code >= 500:
                time.sleep(2**attempt)
                continue
            if resp.status_code != 200:
                raise OpenAlexError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            payload = resp.json()
            return [OpenAlexWork.from_json(item) for item in (payload.get("results") or [])]

        msg = "OpenAlex request failed after retries"
        if last_exc is not None:
            raise OpenAlexError(msg) from last_exc
        raise OpenAlexError(msg)


def _extract_arxiv_id(item: dict[str, Any]) -> str | None:
    """Walk `locations` to find an arXiv landing or PDF URL and pull the id out."""
    for loc in item.get("locations") or ():
        source = (loc.get("source") or {}).get("display_name") or ""
        if "arxiv" not in source.lower():
            continue
        for url_key in ("landing_page_url", "pdf_url"):
            url = loc.get(url_key) or ""
            m = _ARXIV_URL_RE.search(url)
            if m:
                return m.group(1)
    return None


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as `{word: [positions]}` to dodge copyright concerns."""
    if not inverted:
        return None
    max_pos = -1
    for positions in inverted.values():
        for p in positions:
            if p > max_pos:
                max_pos = p
    if max_pos < 0:
        return None
    words = [""] * (max_pos + 1)
    for word, positions in inverted.items():
        for p in positions:
            if 0 <= p <= max_pos:
                words[p] = word
    return " ".join(w for w in words if w)


def parse_year_range(year: str) -> tuple[str | None, str | None]:
    """Convert a CLI-friendly year spec into (from_date, to_date) in ISO format.

    Accepts '2026', '2024-2026', '-2024', '2024-'. Returns (None, None) for empty.
    """
    if not year:
        return None, None
    if "-" in year:
        a, b = year.split("-", 1)
        from_date = f"{a}-01-01" if a else None
        to_date = f"{b}-12-31" if b else None
        return from_date, to_date
    return f"{year}-01-01", f"{year}-12-31"
