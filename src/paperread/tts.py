"""Pre-generate per-sentence audio with Microsoft Edge TTS.

The audio dir layout is `<report_dir>/audio/sNNNN.mp3` plus a `.manifest.json`
mapping sentence index → text hash → filename. Builds are incremental: a
sentence whose text hash matches the manifest skips synthesis, which matters
because edge-tts hits a remote service for every call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import edge_tts

from paperread.sentences import Sentence

DEFAULT_VOICE = "en-US-AvaMultilingualNeural"
MANIFEST_NAME = ".manifest.json"

# Inline math like `$h_t$` makes TTS read "dollar h underscore t dollar" — gibberish.
# Strip these (and any leftover markup) before synthesis.
_INLINE_MATH_RE = re.compile(r"\$[^$\n]+?\$")


def _clean_for_tts(text: str) -> str:
    """Remove inline math and collapse whitespace so edge-tts gets clean prose."""
    cleaned = _INLINE_MATH_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass(frozen=True, slots=True)
class AudioEntry:
    """One row of the audio manifest: maps a sentence index to its mp3 + hash."""

    index: int
    text: str
    hash: str
    filename: str  # relative to audio_dir, e.g. "s0001.mp3"


def synthesize_sentences(
    sentences: Iterable[Sentence],
    audio_dir: Path,
    *,
    voice: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[AudioEntry]:
    """Generate one mp3 per sentence, reusing existing files when text is unchanged."""
    voice = voice or os.environ.get("PAPERREAD_TTS_VOICE") or DEFAULT_VOICE
    audio_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_manifest(audio_dir)
    entries: list[AudioEntry] = []
    to_generate: list[tuple[Path, str]] = []

    sentences_list = list(sentences)
    for sentence in sentences_list:
        spoken = _clean_for_tts(sentence.core)
        # Skip sentences that become trivially short after cleaning (math-only,
        # punctuation residue, etc.); edge-tts rejects empty input.
        if sum(1 for c in spoken if c.isalnum()) < 3:
            continue
        digest = hashlib.sha1(spoken.encode("utf-8")).hexdigest()[:16]
        filename = f"s{len(entries) + len(to_generate) + 1:04d}.mp3"
        path = audio_dir / filename
        prior = existing.get(sentence.index)
        if (
            prior
            and prior.hash == digest
            and prior.filename == filename
            and path.is_file()
        ):
            entries.append(prior)
            continue
        entries.append(
            AudioEntry(index=sentence.index, text=spoken, hash=digest, filename=filename)
        )
        to_generate.append((path, spoken))

    failures: set[Path] = set()
    if to_generate:
        failed = asyncio.run(_synthesize_all(to_generate, voice=voice, on_progress=on_progress))
        failures = set(failed)

    # Drop manifest entries for sentences whose audio failed to generate, so
    # the HTML doesn't link to non-existent mp3s.
    entries = [e for e in entries if (audio_dir / e.filename) not in failures]

    # Purge files that are no longer referenced.
    kept = {e.filename for e in entries}
    for f in audio_dir.iterdir():
        if (
            f.is_file()
            and f.name.startswith("s")
            and (f.suffix == ".mp3" or f.name.endswith(".mp3.part"))
            and f.name not in kept
        ):
            f.unlink()

    _save_manifest(audio_dir, voice=voice, entries=entries)
    return entries


async def _synthesize_all(
    pairs: list[tuple[Path, str]],
    *,
    voice: str,
    on_progress: Callable[[str], None] | None,
) -> list[Path]:
    """Synthesize each (path, text) serially. Returns the list of paths that
    failed so the caller can skip them in the manifest.

    Serialized because Edge TTS uses one websocket per call and Microsoft
    throttles aggressive parallelism.
    """
    failures: list[Path] = []
    for path, text in pairs:
        try:
            await _synthesize_one(text, path, voice=voice)
        except Exception as exc:
            failures.append(path)
            # Clean up any partial file.
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.unlink(missing_ok=True)
            if on_progress is not None:
                on_progress(f"{path.name} FAILED ({type(exc).__name__})")
            continue
        if on_progress is not None:
            on_progress(path.name)
    return failures


async def _synthesize_one(text: str, output: Path, *, voice: str) -> None:
    communicate = edge_tts.Communicate(text, voice=voice)
    tmp = output.with_suffix(output.suffix + ".part")
    await communicate.save(str(tmp))
    tmp.replace(output)


def _load_manifest(audio_dir: Path) -> dict[int, AudioEntry]:
    path = audio_dir / MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[int, AudioEntry] = {}
    for item in data.get("sentences") or ():
        try:
            entry = AudioEntry(
                index=int(item["index"]),
                text=item["text"],
                hash=item["hash"],
                filename=item["filename"],
            )
        except (KeyError, ValueError, TypeError):
            continue
        out[entry.index] = entry
    return out


def _save_manifest(audio_dir: Path, *, voice: str, entries: list[AudioEntry]) -> None:
    payload = {
        "voice": voice,
        "sentences": [
            {"index": e.index, "text": e.text, "hash": e.hash, "filename": e.filename}
            for e in entries
        ],
    }
    (audio_dir / MANIFEST_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
