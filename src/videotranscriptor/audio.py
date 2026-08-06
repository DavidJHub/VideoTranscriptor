"""ffmpeg-backed audio extraction, probing and chunk planning.

Both backends want the same thing from a video file: 16 kHz mono PCM WAV.
Deepgram accepts almost anything, but normalising up front means the two
approaches see byte-identical audio, which is the only way the comparison
report means anything.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

# Canary is trained on 16 kHz mono; Deepgram is happy with it too.
SAMPLE_RATE = 16000
CHANNELS = 1

VIDEO_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpg",
    ".mpeg", ".ts", ".3gp", ".ogv",
}
AUDIO_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff",
}


class FFmpegError(RuntimeError):
    """ffmpeg/ffprobe is missing or exited non-zero."""


def _which(binary: str) -> Optional[str]:
    return shutil.which(binary)


def ensure_ffmpeg() -> None:
    """Fail early with an actionable message rather than deep in a pipeline."""
    missing = [b for b in ("ffmpeg", "ffprobe") if _which(b) is None]
    if missing:
        raise FFmpegError(
            "Missing required binaries: {}. Install ffmpeg, e.g.\n"
            "  macOS:   brew install ffmpeg\n"
            "  Debian:  sudo apt-get install ffmpeg\n"
            "  Windows: winget install Gyan.FFmpeg".format(", ".join(missing))
        )


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    log.debug("running: %s", " ".join(cmd))
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise FFmpegError("`{}` failed (exit {}):\n{}".format(cmd[0], proc.returncode, tail))
    return proc


def probe(path: Path) -> dict:
    """Return the ffprobe JSON for a media file."""
    ensure_ffmpeg()
    proc = _run([
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ])
    return json.loads(proc.stdout or "{}")


def probe_duration(path: Path) -> Optional[float]:
    """Duration in seconds, or None when the container does not report one."""
    info = probe(path)
    raw = (info.get("format") or {}).get("duration")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio" and stream.get("duration"):
            try:
                return float(stream["duration"])
            except (TypeError, ValueError):
                continue
    return None


def has_audio_stream(path: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def extract_audio(
    source: Path,
    destination: Path,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    start: Optional[float] = None,
    duration: Optional[float] = None,
) -> Path:
    """Decode `source` to a mono PCM WAV at `destination`.

    Works for video and audio inputs alike; ffmpeg simply drops the video
    stream via ``-vn``.
    """
    ensure_ffmpeg()
    if not source.exists():
        raise FileNotFoundError("Input file not found: {}".format(source))
    if not has_audio_stream(source):
        raise FFmpegError("No audio stream found in {}".format(source.name))

    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    # Seeking before -i is the fast path.
    if start is not None:
        cmd += ["-ss", "{:.3f}".format(start)]
    cmd += ["-i", str(source)]
    if duration is not None:
        cmd += ["-t", "{:.3f}".format(duration)]
    cmd += [
        "-vn",
        "-map", "0:a:0",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(destination),
    ]
    _run(cmd)
    return destination


_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(
    path: Path,
    noise_db: float = -35.0,
    min_silence: float = 0.4,
) -> List[Tuple[float, float]]:
    """Return ``(start, end)`` spans of silence detected by ffmpeg."""
    ensure_ffmpeg()
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
            "-af", "silencedetect=noise={}dB:d={}".format(noise_db, min_silence),
            "-f", "null", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    return parse_silence_log(proc.stderr or "")


def parse_silence_log(stderr: str) -> List[Tuple[float, float]]:
    """Pull silence spans out of ffmpeg's ``silencedetect`` output.

    Split out from :func:`detect_silences` so the parsing is testable without
    ffmpeg present.
    """
    spans: List[Tuple[float, float]] = []
    pending: Optional[float] = None
    for line in stderr.splitlines():
        start_match = _SILENCE_START.search(line)
        if start_match:
            pending = max(0.0, float(start_match.group(1)))
            continue
        end_match = _SILENCE_END.search(line)
        if end_match and pending is not None:
            end = float(end_match.group(1))
            if end > pending:
                spans.append((pending, end))
            pending = None
    return spans


def plan_chunks(
    duration: float,
    silences: Sequence[Tuple[float, float]],
    max_chunk: float = 30.0,
    min_chunk: float = 5.0,
) -> List[Tuple[float, float]]:
    """Split ``[0, duration)`` into chunks no longer than ``max_chunk``.

    Canary processes a bounded window at a time, so long audio has to be cut.
    Cutting mid-word costs accuracy, so each boundary is pulled back to the
    middle of the last silence that falls inside the window. When a window has
    no silence at all we cut on the hard limit — a hard cut beats dropping
    audio.

    ``min_chunk`` keeps a boundary from landing immediately after the previous
    one, which would otherwise emit a stream of sub-second chunks on audio with
    dense silence markers.
    """
    if duration <= 0:
        return []
    if max_chunk <= 0:
        raise ValueError("max_chunk must be positive")
    if duration <= max_chunk:
        return [(0.0, duration)]

    # Candidate cut points: the midpoint of each silence span.
    cut_points = sorted((s + e) / 2.0 for s, e in silences if e > s)

    chunks: List[Tuple[float, float]] = []
    position = 0.0
    while position < duration - 1e-6:
        limit = position + max_chunk
        if limit >= duration:
            chunks.append((position, duration))
            break

        floor = position + min(min_chunk, max_chunk)
        candidates = [c for c in cut_points if floor < c <= limit]
        boundary = candidates[-1] if candidates else limit
        chunks.append((position, boundary))
        position = boundary

    # A sub-second tail decodes badly on its own. It cannot simply be folded
    # into its predecessor - the loop only continues while more than max_chunk
    # of audio remains, so the merged span would always overflow the window.
    # Splitting the last two chunks evenly keeps both usable.
    if len(chunks) >= 2 and (chunks[-1][1] - chunks[-1][0]) < 1.0:
        start = chunks[-2][0]
        end = chunks[-1][1]
        middle = (start + end) / 2.0
        chunks[-2:] = [(start, middle), (middle, end)]
    return chunks


def slice_audio(source: Path, destination: Path, start: float, end: float) -> Path:
    """Write ``[start, end)`` of an already-normalised WAV to a new file."""
    return extract_audio(source, destination, start=start, duration=max(0.0, end - start))


def looks_like_media(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES | AUDIO_SUFFIXES
