"""Common interface every transcription backend implements."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable, NamedTuple, Optional

from ..models import Transcript

ProgressFn = Callable[[str], None]


class Availability(NamedTuple):
    """Whether a backend can run right now, and why not if it cannot."""

    ok: bool
    reason: str = ""


def _noop(_message: str) -> None:
    pass


class TranscriptionBackend(abc.ABC):
    """Turn a 16 kHz mono WAV into a :class:`Transcript`."""

    #: short identifier used on the command line
    name: str = "base"
    #: human readable one-liner shown by `videotranscriptor backends`
    description: str = ""

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """Identifier of the concrete model being used."""

    @abc.abstractmethod
    def check(self) -> Availability:
        """Report whether credentials/dependencies/hardware are in place.

        Must not raise and must not do meaningful work — this is called to
        render the availability table.
        """

    @abc.abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        progress: ProgressFn = _noop,
    ) -> Transcript:
        """Transcribe a normalised WAV file.

        Raises :class:`~videotranscriptor.models.TranscriptionError` on failure.
        """

    def require_available(self) -> None:
        from ..models import BackendUnavailable

        status = self.check()
        if not status.ok:
            raise BackendUnavailable("{}: {}".format(self.name, status.reason))
