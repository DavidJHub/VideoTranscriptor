"""Backend registry.

Two approaches ship in the box: Deepgram over the network and NVIDIA Canary
locally. Adding a third means implementing
:class:`~videotranscriptor.backends.base.TranscriptionBackend` and listing it
in :data:`BACKENDS`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from .base import Availability, TranscriptionBackend
from .canary import CanaryBackend
from .deepgram import DeepgramBackend

BACKENDS: Dict[str, Type[TranscriptionBackend]] = {
    DeepgramBackend.name: DeepgramBackend,
    CanaryBackend.name: CanaryBackend,
}

#: order used by --backend both / auto
DEFAULT_ORDER: List[str] = [DeepgramBackend.name, CanaryBackend.name]


def build(name: str, **kwargs: Any) -> TranscriptionBackend:
    """Instantiate a backend by name, ignoring options it does not accept."""
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise ValueError(
            "Unknown backend '{}'. Available: {}".format(name, ", ".join(sorted(BACKENDS)))
        ) from None

    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{k: v for k, v in kwargs.items() if k in accepted and v is not None})


__all__ = [
    "Availability",
    "BACKENDS",
    "DEFAULT_ORDER",
    "CanaryBackend",
    "DeepgramBackend",
    "TranscriptionBackend",
    "build",
]
