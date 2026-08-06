"""Pipeline tests with stub backends - no ffmpeg, no network, no model."""

from pathlib import Path

import pytest

from videotranscriptor import pipeline
from videotranscriptor.backends.base import Availability, TranscriptionBackend
from videotranscriptor.models import Segment, Transcript, TranscriptionError


class StubBackend(TranscriptionBackend):
    def __init__(self, name, text="hello world", available=True, fail=False, **kwargs):
        self.name = name
        self.description = "stub"
        self._text = text
        self._available = available
        self._fail = fail
        self._extra = kwargs

    @property
    def model_id(self):
        return "stub-model"

    def check(self):
        return Availability(self._available, "ready" if self._available else "not configured")

    def transcribe(self, audio_path, language=None, progress=lambda m: None):
        progress("stub running")
        if self._fail:
            raise TranscriptionError("boom")
        return Transcript(
            text=self._text,
            backend=self.name,
            model="stub-model",
            segments=[Segment(start=0.0, end=1.0, text=self._text)],
            language=language or "en",
            processing_time=self._extra.get("processing_time", 1.0),
        )


@pytest.fixture
def fake_audio(monkeypatch, tmp_path):
    """Replace ffmpeg work with a stub WAV so the pipeline can be exercised."""

    def fake_prepared_audio(source, keep_audio_at=None, progress=lambda m: None):
        from contextlib import contextmanager

        @contextmanager
        def cm():
            wav = tmp_path / "stub.wav"
            wav.write_bytes(b"RIFF....WAVE")
            if keep_audio_at is not None:
                keep_audio_at.parent.mkdir(parents=True, exist_ok=True)
                keep_audio_at.write_bytes(wav.read_bytes())
            yield wav

        return cm()

    monkeypatch.setattr(pipeline, "prepared_audio", fake_prepared_audio)
    monkeypatch.setattr(pipeline.audio_utils, "probe_duration", lambda path: 60.0)

    video = tmp_path / "talk.mp4"
    video.write_bytes(b"not really a video")
    return video


def test_both_backends_run_and_are_compared(fake_audio, tmp_path):
    result = pipeline.run(
        fake_audio,
        backends=[StubBackend("deepgram"), StubBackend("canary", text="hello there")],
        output_dir=tmp_path / "out",
        output_formats=["txt", "json"],
    )

    assert [run.backend for run in result.successful] == ["deepgram", "canary"]
    assert result.comparison is not None
    assert result.comparison_path is not None and result.comparison_path.exists()
    assert "word agreement" in result.comparison_path.read_text(encoding="utf-8")


def test_outputs_are_named_per_backend(fake_audio, tmp_path):
    result = pipeline.run(
        fake_audio,
        backends=[StubBackend("deepgram")],
        output_dir=tmp_path / "out",
        output_formats=["txt", "srt"],
    )
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == ["talk.deepgram.srt", "talk.deepgram.txt"]
    assert set(result.successful[0].outputs) == {"txt", "srt"}


def test_a_failing_backend_does_not_stop_the_others(fake_audio, tmp_path):
    result = pipeline.run(
        fake_audio,
        backends=[StubBackend("deepgram", fail=True), StubBackend("canary")],
        output_dir=tmp_path / "out",
        output_formats=["txt"],
    )
    assert [run.backend for run in result.failed] == ["deepgram"]
    assert result.failed[0].error == "boom"
    assert [run.backend for run in result.successful] == ["canary"]
    # One transcript is not enough to compare.
    assert result.comparison is None


def test_unavailable_backend_is_skipped_with_its_reason(fake_audio, tmp_path):
    result = pipeline.run(
        fake_audio,
        backends=[StubBackend("deepgram", available=False), StubBackend("canary")],
        output_dir=tmp_path / "out",
        output_formats=["txt"],
    )
    assert result.failed[0].error == "not configured"
    assert len(result.successful) == 1


def test_comparison_can_be_disabled(fake_audio, tmp_path):
    result = pipeline.run(
        fake_audio,
        backends=[StubBackend("deepgram"), StubBackend("canary")],
        output_dir=tmp_path / "out",
        output_formats=["txt"],
        write_comparison=False,
    )
    assert result.comparison is None
    assert not (tmp_path / "out" / "talk.comparison.md").exists()


def test_audio_duration_is_backfilled_onto_transcripts(fake_audio, tmp_path):
    result = pipeline.run(
        fake_audio,
        backends=[StubBackend("deepgram")],
        output_dir=tmp_path / "out",
        output_formats=["txt"],
    )
    transcript = result.successful[0].transcript
    assert transcript.audio_duration == 60.0
    assert transcript.realtime_factor == pytest.approx(1 / 60)


def test_keep_audio_writes_the_wav_next_to_the_transcripts(fake_audio, tmp_path):
    pipeline.run(
        fake_audio,
        backends=[StubBackend("deepgram")],
        output_dir=tmp_path / "out",
        output_formats=["txt"],
        keep_audio=True,
    )
    assert (tmp_path / "out" / "talk.wav").exists()


def test_default_output_dir_sits_beside_the_input(fake_audio):
    result = pipeline.run(
        fake_audio, backends=[StubBackend("deepgram")], output_formats=["txt"]
    )
    expected = fake_audio.parent / "talk_transcripts"
    assert result.successful[0].outputs["txt"].parent == expected


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.run(Path(tmp_path / "absent.mp4"), backends=[StubBackend("deepgram")])


def test_no_backends_is_a_programming_error(fake_audio):
    with pytest.raises(ValueError, match="At least one backend"):
        pipeline.run(fake_audio, backends=[])
