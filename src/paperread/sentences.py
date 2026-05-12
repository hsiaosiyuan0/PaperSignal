from __future__ import annotations

import warnings
from dataclasses import dataclass

with warnings.catch_warnings():
    # pysbd 0.3.4 ships a couple of SyntaxWarnings under Python 3.12+.
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd

_segmenter_en = pysbd.Segmenter(language="en", clean=False)


@dataclass(frozen=True, slots=True)
class Sentence:
    index: int
    text: str
    leading_ws: str
    trailing_ws: str

    @property
    def core(self) -> str:
        """Text suitable for TTS / hashing (whitespace stripped)."""
        return self.text.strip()


def segment_english(text: str) -> list[Sentence]:
    """Split English text into sentences while preserving inter-sentence whitespace.

    pysbd's `clean=False` keeps trailing spaces on each segment but discards
    leading whitespace at the start of the text; we recover positional info
    by walking the original string in parallel.
    """
    raw_segments = _segmenter_en.segment(text)
    sentences: list[Sentence] = []
    cursor = 0
    for seg in raw_segments:
        core = seg.strip()
        if not core:
            continue
        # Drop markup leftovers (orphan `**`, single punctuation, etc.).
        if sum(1 for c in core if c.isalnum()) < 3:
            continue
        # Locate this segment's core in the source.
        try:
            start = text.index(core, cursor)
        except ValueError:
            # pysbd occasionally rewrites whitespace; fall back to no-context insert.
            sentences.append(
                Sentence(index=len(sentences), text=core, leading_ws="", trailing_ws=" ")
            )
            continue
        leading = text[cursor:start]
        end = start + len(core)
        trailing = ""
        # Consume the run of whitespace that follows so the next sentence's
        # `leading_ws` doesn't double up.
        while end < len(text) and text[end].isspace():
            trailing += text[end]
            end += 1
        sentences.append(
            Sentence(
                index=len(sentences),
                text=core,
                leading_ws=leading,
                trailing_ws=trailing or " ",
            )
        )
        cursor = end
    return sentences
