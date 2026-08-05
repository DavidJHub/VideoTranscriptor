"""Local NVIDIA Canary backend (approach 2).

Canary is an encoder-decoder speech model published by NVIDIA and served
through the NeMo toolkit. It runs entirely on the machine, so nothing leaves
the box and there is no per-minute cost — the trade is a multi-gigabyte
dependency tree, a model download, and a GPU if you want it to be quick.

Canary attends over a bounded audio window (roughly 40 s for ``canary-1b``),
so anything longer is cut into chunks on silence boundaries, transcribed as a
batch, and stitched back together with the chunk offsets restored.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import audio as audio_utils
from ..models import Segment, Transcript, TranscriptionError
from .base import Availability, ProgressFn, TranscriptionBackend, _noop

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/canary-1b"
# Stay comfortably inside the model's attention window.
DEFAULT_CHUNK_SECONDS = 30.0
DEFAULT_BATCH_SIZE = 4

# Canary 1B was trained on these four languages; later revisions add more, so
# this is a warning rather than a hard gate.
CANARY_1B_LANGUAGES = {"en", "de", "fr", "es"}


class CanaryBackend(TranscriptionBackend):
    name = "canary"
    description = "NVIDIA Canary via NeMo, running locally (no API key, needs torch)"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        punctuation: bool = True,
        keep_chunks: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.chunk_seconds = chunk_seconds
        self.batch_size = max(1, batch_size)
        self.punctuation = punctuation
        self.keep_chunks = keep_chunks
        self._loaded = None  # cached NeMo model

    @property
    def model_id(self) -> str:
        return self.model

    # ------------------------------------------------------------------
    # availability
    # ------------------------------------------------------------------
    def check(self) -> Availability:
        try:
            import torch  # noqa: F401
        except ImportError:
            return Availability(
                False,
                "torch is not installed (pip install 'videotranscriptor[local]')",
            )
        try:
            import nemo.collections.asr  # noqa: F401
        except ImportError:
            return Availability(
                False,
                "nemo_toolkit[asr] is not installed (pip install 'videotranscriptor[local]')",
            )
        return Availability(True, "{} on {}".format(self.model, self.resolve_device()))

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                # NeMo's ASR stack is not reliable on MPS; CPU is the safe path.
                return "cpu"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------
    def load(self, progress: ProgressFn = _noop):
        if self._loaded is not None:
            return self._loaded

        self.require_available()
        from nemo.collections.asr.models import EncDecMultiTaskModel

        device = self.resolve_device()
        progress("loading {} onto {} (first run downloads several GB)".format(self.model, device))
        started = time.monotonic()
        try:
            model = EncDecMultiTaskModel.from_pretrained(self.model)
        except Exception as exc:  # noqa: BLE001 - surface the real cause
            raise TranscriptionError(
                "Could not load NeMo model '{}': {}".format(self.model, exc)
            ) from exc

        model = model.to(device)
        model.eval()

        # Beam size 1 keeps decoding fast; Canary's greedy output is strong and
        # beam search on CPU is painful.
        try:
            decode_cfg = model.cfg.decoding
            decode_cfg.beam.beam_size = 1
            model.change_decoding_strategy(decode_cfg)
        except Exception as exc:  # noqa: BLE001 - optional tuning
            log.debug("could not adjust decoding strategy: %s", exc)

        progress("model ready in {:.1f}s".format(time.monotonic() - started))
        self._loaded = model
        return model

    # ------------------------------------------------------------------
    # transcription
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        progress: ProgressFn = _noop,
    ) -> Transcript:
        lang = (language or "en").lower()
        if lang == "auto":
            # Canary needs to be told the source language; it does not detect.
            lang = "en"
            progress("canary cannot auto-detect language, assuming en")
        if self.model == DEFAULT_MODEL and lang not in CANARY_1B_LANGUAGES:
            progress(
                "warning: {} officially supports {}; '{}' may be poor".format(
                    self.model, ", ".join(sorted(CANARY_1B_LANGUAGES)), lang
                )
            )

        model = self.load(progress)

        duration = audio_utils.probe_duration(audio_path) or 0.0
        started = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="vtx-canary-") as tmp:
            tmpdir = Path(tmp)
            chunks = self._plan(audio_path, duration, progress)
            chunk_paths = self._write_chunks(audio_path, chunks, tmpdir, progress)
            texts = self._run_model(model, chunk_paths, lang, progress)

        segments = build_segments(chunks, texts)
        elapsed = time.monotonic() - started
        text = " ".join(s.text for s in segments).strip()
        progress("decoded {} words in {:.1f}s".format(len(text.split()), elapsed))

        return Transcript(
            text=text,
            backend=self.name,
            model=self.model,
            segments=segments,
            language=lang,
            audio_duration=duration or None,
            processing_time=elapsed,
            raw={"chunks": [{"start": s, "end": e} for s, e in chunks], "texts": texts},
        )

    def _plan(
        self, audio_path: Path, duration: float, progress: ProgressFn
    ) -> List[Tuple[float, float]]:
        if duration <= 0:
            raise TranscriptionError("Could not determine audio duration for {}".format(audio_path))
        if duration <= self.chunk_seconds:
            return [(0.0, duration)]

        progress("scanning for silence to pick chunk boundaries")
        silences = audio_utils.detect_silences(audio_path)
        chunks = audio_utils.plan_chunks(duration, silences, max_chunk=self.chunk_seconds)
        progress(
            "split {:.1f}s of audio into {} chunks (<= {:.0f}s each)".format(
                duration, len(chunks), self.chunk_seconds
            )
        )
        return chunks

    def _write_chunks(
        self,
        audio_path: Path,
        chunks: Sequence[Tuple[float, float]],
        tmpdir: Path,
        progress: ProgressFn,
    ) -> List[Path]:
        if len(chunks) == 1 and chunks[0][0] == 0.0:
            return [audio_path]
        paths: List[Path] = []
        for index, (start, end) in enumerate(chunks):
            target = tmpdir / "chunk_{:04d}.wav".format(index)
            audio_utils.slice_audio(audio_path, target, start, end)
            paths.append(target)
        progress("wrote {} chunk files".format(len(paths)))
        return paths

    def _run_model(
        self,
        model: Any,
        chunk_paths: Sequence[Path],
        language: str,
        progress: ProgressFn,
    ) -> List[str]:
        paths = [str(p) for p in chunk_paths]
        pnc = "yes" if self.punctuation else "no"
        progress("transcribing {} chunk(s) on {}".format(len(paths), self.resolve_device()))

        try:
            hypotheses = model.transcribe(
                paths,
                batch_size=self.batch_size,
                task="asr",
                source_lang=language,
                target_lang=language,
                pnc=pnc,
                verbose=False,
            )
        except TypeError:
            # Older NeMo releases only accept a manifest describing the task.
            log.debug("falling back to manifest-based transcribe()")
            hypotheses = self._transcribe_via_manifest(model, paths, language, pnc)
        except Exception as exc:  # noqa: BLE001 - surface the real cause
            raise TranscriptionError("Canary inference failed: {}".format(exc)) from exc

        return [_hypothesis_text(h) for h in _flatten(hypotheses)]

    def _transcribe_via_manifest(
        self, model: Any, paths: Sequence[str], language: str, pnc: str
    ) -> Any:
        with tempfile.TemporaryDirectory(prefix="vtx-manifest-") as tmp:
            manifest = Path(tmp) / "manifest.json"
            with manifest.open("w", encoding="utf-8") as handle:
                for path in paths:
                    handle.write(
                        json.dumps(
                            {
                                "audio_filepath": path,
                                "duration": None,
                                "taskname": "asr",
                                "source_lang": language,
                                "target_lang": language,
                                "pnc": pnc,
                                "answer": "na",
                            }
                        )
                        + "\n"
                    )
            try:
                return model.transcribe(str(manifest), batch_size=self.batch_size)
            except Exception as exc:  # noqa: BLE001
                raise TranscriptionError("Canary inference failed: {}".format(exc)) from exc


def build_segments(
    chunks: Sequence[Tuple[float, float]], texts: Sequence[str]
) -> List[Segment]:
    """Pair chunk spans with their decoded text, dropping empty results.

    Canary returns one hypothesis per chunk with no internal timing, so the
    chunk boundary *is* the timestamp. That is coarse compared to Deepgram's
    word-level timings, which the comparison report calls out.
    """
    segments: List[Segment] = []
    for (start, end), text in zip(chunks, texts):
        cleaned = (text or "").strip()
        if cleaned:
            segments.append(Segment(start=round(start, 3), end=round(end, 3), text=cleaned))
    return segments


def _flatten(hypotheses: Any) -> List[Any]:
    """NeMo has returned bare lists and ``(best, all)`` tuples across versions."""
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
    if hypotheses is None:
        return []
    if isinstance(hypotheses, (str, bytes)):
        return [hypotheses]
    try:
        return list(hypotheses)
    except TypeError:
        return [hypotheses]


def _hypothesis_text(hypothesis: Any) -> str:
    if isinstance(hypothesis, str):
        return hypothesis
    text = getattr(hypothesis, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(hypothesis, dict):
        return str(hypothesis.get("text", ""))
    if isinstance(hypothesis, (list, tuple)) and hypothesis:
        return _hypothesis_text(hypothesis[0])
    return str(hypothesis or "")


def default_cache_dir() -> Path:
    """Where NeMo drops downloaded checkpoints, for the doctor command."""
    return Path(os.environ.get("NEMO_CACHE_DIR", Path.home() / ".cache" / "torch" / "NeMo"))
