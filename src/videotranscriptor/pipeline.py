"""Orchestration: video in, transcripts (and a comparison) out.

Audio is extracted exactly once and handed to every backend, so the two
approaches are judged on identical input.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from . import audio as audio_utils
from . import formats
from .backends import TranscriptionBackend
from .backends.base import ProgressFn, _noop
from .compare import Comparison, compare
from .models import Transcript, TranscriptionError

log = logging.getLogger(__name__)


@dataclass
class BackendRun:
    """Outcome of one backend on one file. Exactly one of the two is set."""

    backend: str
    transcript: Optional[Transcript] = None
    error: Optional[str] = None
    outputs: Dict[str, Path] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.transcript is not None


@dataclass
class PipelineResult:
    source: Path
    audio_path: Optional[Path]
    audio_duration: Optional[float]
    runs: List[BackendRun] = field(default_factory=list)
    comparison: Optional[Comparison] = None
    comparison_path: Optional[Path] = None

    @property
    def successful(self) -> List[BackendRun]:
        return [run for run in self.runs if run.ok]

    @property
    def failed(self) -> List[BackendRun]:
        return [run for run in self.runs if not run.ok]


@contextmanager
def prepared_audio(
    source: Path,
    keep_audio_at: Optional[Path] = None,
    progress: ProgressFn = _noop,
) -> Iterator[Path]:
    """Yield a normalised 16 kHz mono WAV for ``source``.

    The temporary copy is removed on exit unless ``keep_audio_at`` is given, in
    which case it is also copied there.
    """
    audio_utils.ensure_ffmpeg()
    tmpdir = Path(tempfile.mkdtemp(prefix="vtx-audio-"))
    wav_path = tmpdir / "{}.wav".format(source.stem or "audio")
    try:
        progress("extracting audio from {}".format(source.name))
        audio_utils.extract_audio(source, wav_path)
        size_mb = wav_path.stat().st_size / (1024 * 1024)
        progress("audio ready: {} ({:.1f} MB, 16 kHz mono)".format(wav_path.name, size_mb))
        if keep_audio_at is not None:
            keep_audio_at.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_path, keep_audio_at)
            progress("saved extracted audio to {}".format(keep_audio_at))
        yield wav_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run(
    source: Path,
    backends: Sequence[TranscriptionBackend],
    output_dir: Optional[Path] = None,
    output_formats: Sequence[str] = ("txt", "srt", "json"),
    language: Optional[str] = None,
    keep_audio: bool = False,
    include_raw: bool = False,
    write_comparison: bool = True,
    progress: ProgressFn = _noop,
) -> PipelineResult:
    """Transcribe ``source`` with each backend and optionally compare them.

    A backend that fails does not stop the others — a missing Deepgram key
    should still leave you with the local transcript.
    """
    if not backends:
        raise ValueError("At least one backend is required")

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError("Input file not found: {}".format(source))

    output_dir = Path(output_dir) if output_dir else source.parent / "{}_transcripts".format(source.stem)
    keep_audio_at = output_dir / "{}.wav".format(source.stem) if keep_audio else None

    result = PipelineResult(source=source, audio_path=keep_audio_at, audio_duration=None)

    with prepared_audio(source, keep_audio_at=keep_audio_at, progress=progress) as wav_path:
        duration = audio_utils.probe_duration(wav_path)
        result.audio_duration = duration

        for backend in backends:
            label = backend.name
            status = backend.check()
            if not status.ok:
                progress("[{}] unavailable - {}".format(label, status.reason))
                result.runs.append(BackendRun(backend=label, error=status.reason))
                continue

            progress("[{}] starting ({})".format(label, backend.model_id))
            try:
                transcript = backend.transcribe(
                    wav_path,
                    language=language,
                    progress=lambda message, label=label: progress(
                        "[{}] {}".format(label, message)
                    ),
                )
            except (TranscriptionError, OSError) as exc:
                log.debug("%s failed", label, exc_info=True)
                progress("[{}] failed - {}".format(label, exc))
                result.runs.append(BackendRun(backend=label, error=str(exc)))
                continue

            if transcript.audio_duration is None:
                transcript.audio_duration = duration

            outputs = formats.write(
                transcript,
                directory=output_dir,
                stem="{}.{}".format(source.stem, label),
                formats=output_formats,
                include_raw=include_raw,
            )
            progress("[{}] wrote {}".format(label, ", ".join(sorted(outputs))))
            result.runs.append(
                BackendRun(backend=label, transcript=transcript, outputs=outputs)
            )

    successful = result.successful
    if write_comparison and len(successful) >= 2:
        first, second = successful[0].transcript, successful[1].transcript
        assert first is not None and second is not None  # narrowed by .successful
        result.comparison = compare(first, second)
        path = output_dir / "{}.comparison.md".format(source.stem)
        path.write_text(result.comparison.to_markdown(), encoding="utf-8")
        result.comparison_path = path
        progress("wrote comparison report to {}".format(path.name))

    return result
