"""Transcript writers: txt, srt, vtt, json, md."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, Iterable, List

from .models import Segment, Transcript

FORMATS = ("txt", "srt", "vtt", "json", "md")


def format_timestamp(seconds: float, separator: str = ",") -> str:
    """``HH:MM:SS,mmm`` for SRT, ``HH:MM:SS.mmm`` for WebVTT."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return "{:02d}:{:02d}:{:02d}{}{:03d}".format(hours, minutes, secs, separator, millis)


def to_txt(transcript: Transcript, width: int = 100) -> str:
    if not transcript.text:
        return ""
    paragraphs = [
        textwrap.fill(part, width=width)
        for part in transcript.text.split("\n")
        if part.strip()
    ]
    return "\n\n".join(paragraphs) + "\n"


def _cue_lines(segments: Iterable[Segment]) -> List[Segment]:
    """Drop empty cues and repair zero/backwards durations."""
    cleaned: List[Segment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        start = max(0.0, segment.start)
        end = segment.end if segment.end > start else start + 1.0
        cleaned.append(Segment(start=start, end=end, text=text, speaker=segment.speaker))
    return cleaned


def to_srt(transcript: Transcript) -> str:
    lines: List[str] = []
    for index, segment in enumerate(_cue_lines(transcript.segments), start=1):
        text = segment.text
        if segment.speaker:
            text = "[{}] {}".format(segment.speaker, text)
        lines.append(str(index))
        lines.append(
            "{} --> {}".format(
                format_timestamp(segment.start), format_timestamp(segment.end)
            )
        )
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def to_vtt(transcript: Transcript) -> str:
    lines: List[str] = ["WEBVTT", ""]
    for segment in _cue_lines(transcript.segments):
        text = segment.text
        if segment.speaker:
            text = "<v {}>{}".format(segment.speaker, text)
        lines.append(
            "{} --> {}".format(
                format_timestamp(segment.start, "."), format_timestamp(segment.end, ".")
            )
        )
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def to_json(transcript: Transcript, include_raw: bool = False) -> str:
    return transcript.to_json(include_raw=include_raw) + "\n"


def to_markdown(transcript: Transcript) -> str:
    header = [
        "# Transcript",
        "",
        "| field | value |",
        "| --- | --- |",
        "| backend | `{}` |".format(transcript.backend),
        "| model | `{}` |".format(transcript.model),
        "| language | {} |".format(transcript.language or "unknown"),
    ]
    if transcript.audio_duration:
        header.append("| audio duration | {:.1f}s |".format(transcript.audio_duration))
    if transcript.processing_time:
        header.append("| processing time | {:.1f}s |".format(transcript.processing_time))
    rtf = transcript.realtime_factor
    if rtf:
        header.append("| realtime factor | {:.2f}x |".format(rtf))
    header.append("| words | {} |".format(transcript.word_count))
    header += ["", "## Segments", ""]

    body: List[str] = []
    for segment in _cue_lines(transcript.segments):
        label = "**[{} - {}]**".format(
            format_timestamp(segment.start, "."), format_timestamp(segment.end, ".")
        )
        if segment.speaker:
            label += " _{}_".format(segment.speaker)
        body.append("{} {}".format(label, segment.text))
        body.append("")

    if not body:
        body = [transcript.text, ""]
    return "\n".join(header + body)


_RENDERERS = {
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
    "json": to_json,
    "md": to_markdown,
}


def render(transcript: Transcript, fmt: str, include_raw: bool = False) -> str:
    fmt = fmt.lower()
    if fmt not in _RENDERERS:
        raise ValueError(
            "Unknown format '{}'. Available: {}".format(fmt, ", ".join(FORMATS))
        )
    if fmt == "json":
        return to_json(transcript, include_raw=include_raw)
    return _RENDERERS[fmt](transcript)


def write(
    transcript: Transcript,
    directory: Path,
    stem: str,
    formats: Iterable[str],
    include_raw: bool = False,
) -> Dict[str, Path]:
    """Write ``stem.<fmt>`` for each requested format; returns the paths."""
    directory.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for fmt in formats:
        path = directory / "{}.{}".format(stem, fmt)
        path.write_text(render(transcript, fmt, include_raw=include_raw), encoding="utf-8")
        written[fmt] = path
    return written
