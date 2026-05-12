from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paperread.converter import (
    Backend,
    MarkdownBackend,
    MarkerBackend,
    MarkitdownBackend,
    make_backend,
)


class _FakeBackend(MarkdownBackend):
    def __init__(self, text: str) -> None:
        self._text = text

    def _to_text(self, pdf_path: Path) -> str:
        return self._text


class TestMarkdownBackendBase:
    def test_writes_text_to_output_path(self, tmp_path: Path) -> None:
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        out = tmp_path / "out" / "x.md"

        result = _FakeBackend("# hello\n\nworld").convert(pdf, out)

        assert result == out
        assert out.read_text(encoding="utf-8") == "# hello\n\nworld"

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        out = tmp_path / "deep" / "nested" / "x.md"

        _FakeBackend("ok").convert(pdf, out)

        assert out.is_file()

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _FakeBackend("x").convert(tmp_path / "missing.pdf", tmp_path / "out.md")


class TestMakeBackend:
    def test_marker(self) -> None:
        assert isinstance(make_backend(Backend.MARKER), MarkerBackend)

    def test_markitdown(self) -> None:
        # MarkitdownBackend eagerly constructs MarkItDown; we don't want that side effect here.
        with patch("markitdown.MarkItDown"):
            backend = make_backend(Backend.MARKITDOWN)
        assert isinstance(backend, MarkitdownBackend)


class TestMarkerBackend:
    def test_models_loaded_lazily(self) -> None:
        backend = MarkerBackend()
        assert backend._converter is None

    def test_to_text_invokes_marker(self, tmp_path: Path) -> None:
        backend = MarkerBackend()
        fake_converter = MagicMock(return_value="rendered_doc")
        backend._converter = fake_converter

        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with patch(
            "marker.output.text_from_rendered",
            return_value=("# converted", None, {}),
        ) as text_from_rendered:
            text = backend._to_text(pdf)

        assert text == "# converted"
        fake_converter.assert_called_once_with(str(pdf))
        text_from_rendered.assert_called_once_with("rendered_doc")


class TestMarkitdownBackend:
    def test_to_text_invokes_markitdown(self, tmp_path: Path) -> None:
        with patch("markitdown.MarkItDown") as cls:
            instance = cls.return_value
            instance.convert.return_value = MagicMock(text_content="# md output")
            backend = MarkitdownBackend()

        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        assert backend._to_text(pdf) == "# md output"
