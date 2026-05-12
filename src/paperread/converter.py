from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marker.converters.pdf import PdfConverter


class Backend(StrEnum):
    MARKER = "marker"
    MARKITDOWN = "markitdown"


class MarkdownBackend(ABC):
    """Convert a PDF on disk into a Markdown file.

    Concrete backends implement `_to_text`; the base class handles file I/O
    so backends only worry about the conversion itself.
    """

    @abstractmethod
    def _to_text(self, pdf_path: Path) -> str: ...

    def convert(self, pdf_path: Path, output_path: Path) -> Path:
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        text = self._to_text(pdf_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return output_path


class MarkitdownBackend(MarkdownBackend):
    """Fast, lightweight backend. Loses formulas and complex layout."""

    def __init__(self) -> None:
        from markitdown import MarkItDown

        self._md = MarkItDown()

    def _to_text(self, pdf_path: Path) -> str:
        return self._md.convert(str(pdf_path)).text_content


class MarkerBackend(MarkdownBackend):
    """High-fidelity backend with LaTeX equations. Loads ML models on first use."""

    def __init__(self) -> None:
        self._converter: PdfConverter | None = None

    def _ensure_loaded(self) -> PdfConverter:
        if self._converter is None:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            self._converter = PdfConverter(artifact_dict=create_model_dict())
        return self._converter

    def _to_text(self, pdf_path: Path) -> str:
        from marker.output import text_from_rendered

        converter = self._ensure_loaded()
        rendered = converter(str(pdf_path))
        text, _, _ = text_from_rendered(rendered)
        return text


def make_backend(name: Backend) -> MarkdownBackend:
    match name:
        case Backend.MARKER:
            return MarkerBackend()
        case Backend.MARKITDOWN:
            return MarkitdownBackend()
