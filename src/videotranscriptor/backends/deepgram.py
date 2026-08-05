"""Deepgram cloud backend (approach 1).

Talks to the pre-recorded ``/v1/listen`` endpoint over plain HTTP rather than
through the SDK: one dependency instead of a tree, and the response shape is
stable and well documented.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Segment, Transcript, TranscriptionError
from .base import Availability, ProgressFn, TranscriptionBackend, _noop

log = logging.getLogger(__name__)

API_URL = "https://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"
ENV_KEY = "DEEPGRAM_API_KEY"

# Deepgram retries: transient 5xx/429 only, everything else fails fast.
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


class DeepgramBackend(TranscriptionBackend):
    name = "deepgram"
    description = "Deepgram cloud API (fast, needs DEEPGRAM_API_KEY)"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        diarize: bool = False,
        smart_format: bool = True,
        timeout: float = 600.0,
    ) -> None:
        self.api_key = api_key or os.environ.get(ENV_KEY) or ""
        self.model = model
        self.diarize = diarize
        self.smart_format = smart_format
        self.timeout = timeout

    @property
    def model_id(self) -> str:
        return self.model

    def check(self) -> Availability:
        try:
            import requests  # noqa: F401
        except ImportError:
            return Availability(False, "requests is not installed (pip install requests)")
        if not self.api_key:
            return Availability(
                False,
                "{} is not set (get a key at https://console.deepgram.com)".format(ENV_KEY),
            )
        return Availability(True, "API key found, model {}".format(self.model))

    def _params(self, language: Optional[str]) -> Dict[str, str]:
        params = {
            "model": self.model,
            "smart_format": "true" if self.smart_format else "false",
            "punctuate": "true",
            "utterances": "true",
            "paragraphs": "true",
        }
        if self.diarize:
            params["diarize"] = "true"
        if language and language != "auto":
            params["language"] = language
        else:
            params["detect_language"] = "true"
        return params

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        progress: ProgressFn = _noop,
    ) -> Transcript:
        self.require_available()
        import requests

        size_mb = audio_path.stat().st_size / (1024 * 1024)
        progress("uploading {:.1f} MB to Deepgram ({})".format(size_mb, self.model))

        headers = {
            "Authorization": "Token {}".format(self.api_key),
            "Content-Type": "audio/wav",
        }
        params = self._params(language)

        started = time.monotonic()
        payload: Optional[Dict[str, Any]] = None
        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with audio_path.open("rb") as handle:
                    response = requests.post(
                        API_URL,
                        params=params,
                        headers=headers,
                        data=handle,
                        timeout=self.timeout,
                    )
            except requests.RequestException as exc:  # network-level failure
                last_error = "network error: {}".format(exc)
                if attempt == MAX_ATTEMPTS:
                    raise TranscriptionError(
                        "Deepgram request failed after {} attempts: {}".format(attempt, exc)
                    ) from exc
            else:
                if response.status_code == 200:
                    payload = response.json()
                    break
                last_error = "HTTP {}: {}".format(
                    response.status_code, response.text[:400].strip()
                )
                if response.status_code == 401:
                    raise TranscriptionError(
                        "Deepgram rejected the API key (HTTP 401). Check {}.".format(ENV_KEY)
                    )
                if response.status_code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS:
                    raise TranscriptionError("Deepgram request failed - {}".format(last_error))

            backoff = 2 ** attempt
            progress("retrying in {}s ({})".format(backoff, last_error))
            time.sleep(backoff)

        if payload is None:  # pragma: no cover - loop always raises or breaks
            raise TranscriptionError("Deepgram request failed - {}".format(last_error))

        elapsed = time.monotonic() - started
        transcript = parse_response(payload, model=self.model)
        transcript.processing_time = elapsed
        progress("received {} words in {:.1f}s".format(transcript.word_count, elapsed))
        return transcript


def parse_response(payload: Dict[str, Any], model: str = DEFAULT_MODEL) -> Transcript:
    """Turn a Deepgram ``/v1/listen`` response into a :class:`Transcript`.

    Kept module-level and pure so it can be tested against a fixture without
    touching the network.
    """
    results = payload.get("results") or {}
    channels = results.get("channels") or []
    if not channels:
        raise TranscriptionError("Deepgram response contained no channels")

    alternatives = channels[0].get("alternatives") or []
    if not alternatives:
        raise TranscriptionError("Deepgram response contained no alternatives")
    best = alternatives[0]

    segments: List[Segment] = []
    # `utterances` gives the cleanest segmentation; paragraphs and then the raw
    # word list are the fallbacks when the request did not ask for them.
    for utterance in results.get("utterances") or []:
        text = (utterance.get("transcript") or "").strip()
        if not text:
            continue
        speaker = utterance.get("speaker")
        segments.append(
            Segment(
                start=float(utterance.get("start", 0.0)),
                end=float(utterance.get("end", 0.0)),
                text=text,
                speaker="speaker {}".format(speaker) if speaker is not None else None,
                confidence=utterance.get("confidence"),
            )
        )

    if not segments:
        paragraphs = ((best.get("paragraphs") or {}).get("paragraphs")) or []
        for paragraph in paragraphs:
            sentences = paragraph.get("sentences") or []
            for sentence in sentences:
                text = (sentence.get("text") or "").strip()
                if text:
                    segments.append(
                        Segment(
                            start=float(sentence.get("start", 0.0)),
                            end=float(sentence.get("end", 0.0)),
                            text=text,
                        )
                    )

    if not segments:
        segments = _segments_from_words(best.get("words") or [])

    text = (best.get("transcript") or "").strip()
    if not text and segments:
        text = " ".join(s.text for s in segments)

    metadata = payload.get("metadata") or {}
    language = channels[0].get("detected_language") or best.get("language")

    return Transcript(
        text=text,
        backend=DeepgramBackend.name,
        model=(metadata.get("models") and model) or model,
        segments=segments,
        language=language,
        audio_duration=_as_float(metadata.get("duration")),
        raw=payload,
    )


def _segments_from_words(words: List[Dict[str, Any]], max_gap: float = 0.8) -> List[Segment]:
    """Group word timings into sentence-ish segments on pauses."""
    segments: List[Segment] = []
    current: List[Dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(
            (w.get("punctuated_word") or w.get("word") or "") for w in current
        ).strip()
        if text:
            segments.append(
                Segment(
                    start=float(current[0].get("start", 0.0)),
                    end=float(current[-1].get("end", 0.0)),
                    text=text,
                )
            )
        current.clear()

    for word in words:
        if current:
            gap = float(word.get("start", 0.0)) - float(current[-1].get("end", 0.0))
            written = (current[-1].get("punctuated_word") or "").endswith((".", "?", "!"))
            if gap > max_gap or written:
                flush()
        current.append(word)
    flush()
    return segments


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
