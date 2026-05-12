from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from paperread.openalex import (
    OpenAlexClient,
    OpenAlexError,
    OpenAlexSort,
    OpenAlexWork,
    parse_year_range,
)


def _work(
    title: str = "Knowledge Retrieval for the Win",
    arxiv_landing: str | None = "http://arxiv.org/abs/2501.12345",
    cites: int = 7,
    publication_date: str = "2025-09-01",
    institutions: tuple[str, ...] = ("Microsoft Research",),
    abstract_inverted: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    locations: list[dict[str, Any]] = []
    if arxiv_landing:
        locations.append(
            {
                "source": {"display_name": "arXiv (Cornell University)"},
                "landing_page_url": arxiv_landing,
                "pdf_url": "https://arxiv.org/pdf/2501.12345",
            }
        )
    return {
        "id": "https://openalex.org/W123",
        "title": title,
        "publication_date": publication_date,
        "publication_year": int(publication_date[:4]) if publication_date else None,
        "cited_by_count": cites,
        "authorships": [
            {
                "author": {"display_name": "Jane Doe"},
                "institutions": [{"display_name": i} for i in institutions],
            }
        ],
        "locations": locations,
        "ids": {
            "openalex": "https://openalex.org/W123",
            "doi": "https://doi.org/10.48550/arXiv.2501.12345",
        },
        "abstract_inverted_index": abstract_inverted,
    }


def _payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"meta": {"count": len(items)}, "results": items}


def _mock_response(status: int = 200, body: Any = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body or {}
    resp.headers = {}
    resp.text = "<error body>"
    return resp


class TestWorkFromJson:
    def test_extracts_arxiv_id_from_landing_url(self) -> None:
        w = OpenAlexWork.from_json(_work(arxiv_landing="http://arxiv.org/abs/2501.12345v2"))
        assert w.arxiv_id == "2501.12345v2"

    def test_returns_none_when_no_arxiv_location(self) -> None:
        w = OpenAlexWork.from_json(_work(arxiv_landing=None))
        assert w.arxiv_id is None

    def test_strips_doi_prefix(self) -> None:
        w = OpenAlexWork.from_json(_work())
        assert w.doi == "10.48550/arXiv.2501.12345"

    def test_flattens_authors_and_institutions(self) -> None:
        w = OpenAlexWork.from_json(_work(institutions=("Microsoft", "MIT")))
        assert w.authors[0].name == "Jane Doe"
        assert w.authors[0].institutions == ("Microsoft", "MIT")

    def test_reconstructs_abstract_from_inverted_index(self) -> None:
        inv = {"Hello": [0, 3], "world": [1], "how": [2], "are": [3], "you": [4]}
        # Position 3 has both "Hello" and "are" — last write wins in our implementation.
        w = OpenAlexWork.from_json(_work(abstract_inverted=inv))
        assert w.abstract is not None
        # Word at each position; just assert structure is sensible
        assert "Hello" in w.abstract and "world" in w.abstract

    def test_abstract_none_when_inverted_index_missing(self) -> None:
        w = OpenAlexWork.from_json(_work(abstract_inverted=None))
        assert w.abstract is None


class TestSearchWorks:
    def test_builds_filter_string_correctly(self) -> None:
        client = OpenAlexClient()
        with patch.object(
            client._session, "get", return_value=_mock_response(200, _payload([_work()]))
        ) as get:
            client.search_works(
                "knowledge retrieval",
                affiliation="Microsoft",
                from_date="2025-01-01",
                to_date="2026-12-31",
                sort=OpenAlexSort.CITATIONS_DESC,
                limit=15,
            )
        params = get.call_args.kwargs["params"]
        assert params["search"] == "knowledge retrieval"
        assert params["sort"] == "cited_by_count:desc"
        assert params["per-page"] == "15"
        # Filter parts are comma-separated; order doesn't matter, but our impl is deterministic.
        assert "raw_affiliation_strings.search:Microsoft" in params["filter"]
        assert "from_publication_date:2025-01-01" in params["filter"]
        assert "to_publication_date:2026-12-31" in params["filter"]

    def test_no_filter_param_when_no_filters_provided(self) -> None:
        client = OpenAlexClient()
        with patch.object(
            client._session, "get", return_value=_mock_response(200, _payload([]))
        ) as get:
            client.search_works("anything")
        assert "filter" not in get.call_args.kwargs["params"]

    def test_caps_per_page_at_200(self) -> None:
        client = OpenAlexClient()
        with patch.object(
            client._session, "get", return_value=_mock_response(200, _payload([]))
        ) as get:
            client.search_works("x", limit=999)
        assert get.call_args.kwargs["params"]["per-page"] == "200"

    def test_floors_per_page_at_1(self) -> None:
        client = OpenAlexClient()
        with patch.object(
            client._session, "get", return_value=_mock_response(200, _payload([]))
        ) as get:
            client.search_works("x", limit=0)
        assert get.call_args.kwargs["params"]["per-page"] == "1"

    def test_429_retries_then_succeeds(self) -> None:
        client = OpenAlexClient()
        responses = [_mock_response(429), _mock_response(200, _payload([_work()]))]
        with (
            patch.object(client._session, "get", side_effect=responses),
            patch("paperread.openalex.time.sleep") as sleep,
        ):
            results = client.search_works("x")
        assert len(results) == 1
        sleep.assert_called()

    def test_409_raises_with_quota_message(self) -> None:
        client = OpenAlexClient()
        with (
            patch.object(client._session, "get", return_value=_mock_response(409)),
            pytest.raises(OpenAlexError, match="quota exhausted"),
        ):
            client.search_works("x")

    def test_other_4xx_raises(self) -> None:
        client = OpenAlexClient()
        with (
            patch.object(client._session, "get", return_value=_mock_response(400)),
            pytest.raises(OpenAlexError, match="HTTP 400"),
        ):
            client.search_works("x")

    def test_api_key_from_env_attached_as_param(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENALEX_API_KEY", "secret-key")
        client = OpenAlexClient()
        with patch.object(
            client._session, "get", return_value=_mock_response(200, _payload([]))
        ) as get:
            client.search_works("x")
        assert get.call_args.kwargs["params"]["api_key"] == "secret-key"


class TestParseYearRange:
    def test_empty_returns_none(self) -> None:
        assert parse_year_range("") == (None, None)

    def test_single_year(self) -> None:
        assert parse_year_range("2026") == ("2026-01-01", "2026-12-31")

    def test_range(self) -> None:
        assert parse_year_range("2024-2026") == ("2024-01-01", "2026-12-31")

    def test_open_start(self) -> None:
        assert parse_year_range("-2024") == (None, "2024-12-31")

    def test_open_end(self) -> None:
        assert parse_year_range("2024-") == ("2024-01-01", None)
