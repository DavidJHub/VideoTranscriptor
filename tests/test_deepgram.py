import pytest

from videotranscriptor.backends.deepgram import DeepgramBackend, parse_response
from videotranscriptor.models import TranscriptionError

RESPONSE = {
    "metadata": {"duration": 12.5, "models": ["nova-3"]},
    "results": {
        "channels": [
            {
                "detected_language": "en",
                "alternatives": [
                    {
                        "transcript": "Hello there. General Kenobi.",
                        "confidence": 0.99,
                        "words": [
                            {"word": "hello", "punctuated_word": "Hello", "start": 0.1, "end": 0.4},
                            {"word": "there", "punctuated_word": "there.", "start": 0.4, "end": 0.9},
                            {"word": "general", "punctuated_word": "General", "start": 2.0, "end": 2.4},
                            {"word": "kenobi", "punctuated_word": "Kenobi.", "start": 2.4, "end": 3.0},
                        ],
                    }
                ],
            }
        ],
        "utterances": [
            {"start": 0.1, "end": 0.9, "transcript": "Hello there.", "confidence": 0.98, "speaker": 0},
            {"start": 2.0, "end": 3.0, "transcript": "General Kenobi.", "confidence": 0.97, "speaker": 1},
        ],
    },
}


def test_parses_utterances_into_segments():
    transcript = parse_response(RESPONSE)
    assert transcript.backend == "deepgram"
    assert transcript.text == "Hello there. General Kenobi."
    assert transcript.language == "en"
    assert transcript.audio_duration == 12.5
    assert len(transcript.segments) == 2
    assert transcript.segments[1].start == 2.0
    assert transcript.segments[1].speaker == "speaker 1"
    assert transcript.segments[0].confidence == 0.98


def test_falls_back_to_paragraphs_when_utterances_are_absent():
    payload = {
        "metadata": {"duration": 5.0},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "One. Two.",
                            "paragraphs": {
                                "paragraphs": [
                                    {
                                        "sentences": [
                                            {"text": "One.", "start": 0.0, "end": 1.0},
                                            {"text": "Two.", "start": 1.2, "end": 2.0},
                                        ]
                                    }
                                ]
                            },
                        }
                    ]
                }
            ]
        },
    }
    transcript = parse_response(payload)
    assert [s.text for s in transcript.segments] == ["One.", "Two."]


def test_falls_back_to_word_timings_when_nothing_else_is_present():
    alternative = RESPONSE["results"]["channels"][0]["alternatives"][0]
    payload = {
        "metadata": {"duration": 5.0},
        "results": {"channels": [{"alternatives": [alternative]}]},
    }
    transcript = parse_response(payload)
    # Words are grouped on sentence-final punctuation and on long pauses.
    assert [s.text for s in transcript.segments] == ["Hello there.", "General Kenobi."]


def test_empty_channels_raise():
    with pytest.raises(TranscriptionError, match="no channels"):
        parse_response({"results": {"channels": []}})


def test_missing_alternatives_raise():
    with pytest.raises(TranscriptionError, match="no alternatives"):
        parse_response({"results": {"channels": [{"alternatives": []}]}})


def test_check_reports_missing_key():
    backend = DeepgramBackend(api_key="")
    status = backend.check()
    assert not status.ok
    assert "DEEPGRAM_API_KEY" in status.reason


def test_check_passes_with_a_key():
    assert DeepgramBackend(api_key="secret").check().ok


def test_language_auto_requests_detection():
    params = DeepgramBackend(api_key="k")._params(None)
    assert params["detect_language"] == "true"
    assert "language" not in params


def test_explicit_language_is_forwarded():
    params = DeepgramBackend(api_key="k")._params("de")
    assert params["language"] == "de"
    assert "detect_language" not in params


def test_diarization_is_opt_in():
    assert "diarize" not in DeepgramBackend(api_key="k")._params("en")
    assert DeepgramBackend(api_key="k", diarize=True)._params("en")["diarize"] == "true"
