import json

import pytest

from videotranscriptor import formats
from videotranscriptor.models import Segment, Transcript


@pytest.fixture
def transcript():
    return Transcript(
        text="Hello there. General Kenobi.",
        backend="deepgram",
        model="nova-3",
        language="en",
        audio_duration=10.0,
        processing_time=2.5,
        segments=[
            Segment(start=0.0, end=1.5, text="Hello there."),
            Segment(start=1.8, end=3.25, text="General Kenobi.", speaker="speaker 1"),
        ],
        raw={"results": "..."},
    )


@pytest.mark.parametrize(
    "seconds,separator,expected",
    [
        (0.0, ",", "00:00:00,000"),
        (1.5, ",", "00:00:01,500"),
        (61.25, ".", "00:01:01.250"),
        (3661.007, ",", "01:01:01,007"),
        (-2.0, ",", "00:00:00,000"),
        (59.9999, ",", "00:01:00,000"),  # rounds up across the minute
    ],
)
def test_format_timestamp(seconds, separator, expected):
    assert formats.format_timestamp(seconds, separator) == expected


def test_srt_numbering_and_timing(transcript):
    output = formats.to_srt(transcript)
    assert output.startswith("1\n00:00:00,000 --> 00:00:01,500\nHello there.\n")
    assert "2\n00:00:01,800 --> 00:00:03,250\n[speaker 1] General Kenobi." in output


def test_vtt_has_header_and_dotted_timestamps(transcript):
    output = formats.to_vtt(transcript)
    assert output.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.500" in output
    assert "<v speaker 1>General Kenobi." in output


def test_cues_skip_empty_text_and_repair_zero_duration():
    transcript = Transcript(
        text="ok",
        backend="canary",
        model="nvidia/canary-1b",
        segments=[
            Segment(start=0.0, end=0.0, text="   "),
            Segment(start=5.0, end=5.0, text="ok"),
        ],
    )
    output = formats.to_srt(transcript)
    assert output.count(" --> ") == 1
    assert "00:00:05,000 --> 00:00:06,000" in output


def test_json_omits_raw_unless_requested(transcript):
    without = json.loads(formats.to_json(transcript))
    assert "raw" not in without
    assert without["backend"] == "deepgram"
    assert without["realtime_factor"] == pytest.approx(0.25)
    assert len(without["segments"]) == 2

    assert "raw" in json.loads(formats.to_json(transcript, include_raw=True))


def test_txt_is_plain_wrapped_prose(transcript):
    assert formats.to_txt(transcript).strip() == "Hello there. General Kenobi."


def test_markdown_reports_speed_and_segments(transcript):
    output = formats.to_markdown(transcript)
    assert "| model | `nova-3` |" in output
    assert "| realtime factor | 0.25x |" in output
    assert "**[00:00:01.800 - 00:00:03.250]** _speaker 1_ General Kenobi." in output


def test_render_rejects_unknown_format(transcript):
    with pytest.raises(ValueError, match="Unknown format"):
        formats.render(transcript, "docx")


def test_write_creates_one_file_per_format(tmp_path, transcript):
    written = formats.write(transcript, tmp_path / "out", "talk.deepgram", ["txt", "srt", "json"])
    assert set(written) == {"txt", "srt", "json"}
    for fmt, path in written.items():
        assert path.name == "talk.deepgram.{}".format(fmt)
        assert path.read_text(encoding="utf-8")


def test_transcript_roundtrips_through_dict(transcript):
    restored = Transcript.from_dict(transcript.to_dict())
    assert restored.text == transcript.text
    assert [s.to_dict() for s in restored.segments] == [
        s.to_dict() for s in transcript.segments
    ]
