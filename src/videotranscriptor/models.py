"""Backend-neutral data model for transcripts.

Every backend produces a :class:`Transcript`, so the output writers and the
comparison report never need to know where the words came from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Segment:
    """A contiguous stretch of speech with timestamps in seconds."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            text=data.get("text", ""),
            speaker=data.get("speaker"),
            confidence=data.get("confidence"),
        )


@dataclass
class Transcript:
    """The result of running one backend over one audio file."""

    text: str
    backend: str
    model: str
    segments: List[Segment] = field(default_factory=list)
    language: Optional[str] = None
    audio_duration: Optional[float] = None
    processing_time: Optional[float] = None
    # Whatever the backend gave us, kept for debugging. Not written to disk
    # unless --keep-raw is passed.
    raw: Optional[Dict[str, Any]] = None

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def realtime_factor(self) -> Optional[float]:
        """Processing seconds per second of audio. Lower is faster."""
        if not self.processing_time or not self.audio_duration:
            return None
        return self.processing_time / self.audio_duration

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "audio_duration": self.audio_duration,
            "processing_time": self.processing_time,
            "realtime_factor": self.realtime_factor,
            "word_count": self.word_count,
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
        }
        if include_raw and self.raw is not None:
            data["raw"] = self.raw
        return data

    def to_json(self, include_raw: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(include_raw), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transcript":
        return cls(
            text=data.get("text", ""),
            backend=data.get("backend", "unknown"),
            model=data.get("model", "unknown"),
            segments=[Segment.from_dict(s) for s in data.get("segments", [])],
            language=data.get("language"),
            audio_duration=data.get("audio_duration"),
            processing_time=data.get("processing_time"),
            raw=data.get("raw"),
        )


class TranscriptionError(RuntimeError):
    """Raised when a backend cannot produce a transcript."""


class BackendUnavailable(TranscriptionError):
    """Raised when a backend's dependencies or credentials are missing."""
