from videotranscriptor.backends.canary import (
    CanaryBackend,
    _flatten,
    _hypothesis_text,
    build_segments,
)


class FakeHypothesis:
    """Stands in for NeMo's Hypothesis object."""

    def __init__(self, text):
        self.text = text


def test_segments_carry_the_chunk_offsets():
    chunks = [(0.0, 30.0), (30.0, 58.5)]
    segments = build_segments(chunks, ["first part", "second part"])
    assert [(s.start, s.end) for s in segments] == [(0.0, 30.0), (30.0, 58.5)]
    assert [s.text for s in segments] == ["first part", "second part"]


def test_empty_chunk_output_is_dropped():
    segments = build_segments([(0.0, 5.0), (5.0, 10.0)], ["  ", "words here"])
    assert len(segments) == 1
    assert segments[0].start == 5.0


def test_extra_chunks_without_text_are_ignored():
    segments = build_segments([(0.0, 5.0), (5.0, 10.0)], ["only one"])
    assert len(segments) == 1


def test_flatten_handles_the_shapes_nemo_has_returned():
    assert _flatten(["a", "b"]) == ["a", "b"]
    assert _flatten((["a"], ["ignored"])) == ["a"]
    assert _flatten(None) == []
    assert _flatten("solo") == ["solo"]


def test_hypothesis_text_extraction():
    assert _hypothesis_text(FakeHypothesis("hello")) == "hello"
    assert _hypothesis_text("hello") == "hello"
    assert _hypothesis_text({"text": "hello"}) == "hello"
    assert _hypothesis_text([FakeHypothesis("hello")]) == "hello"
    assert _hypothesis_text(None) == ""


def test_device_override_wins_over_detection():
    assert CanaryBackend(device="cpu").resolve_device() == "cpu"


def test_resolve_device_returns_something_usable():
    assert CanaryBackend().resolve_device() in {"cuda", "cpu"}


def test_check_explains_missing_dependencies_without_raising():
    status = CanaryBackend().check()
    assert isinstance(status.ok, bool)
    if not status.ok:
        assert "pip install" in status.reason


def test_model_id_reflects_the_configured_checkpoint():
    assert CanaryBackend(model="nvidia/canary-1b-flash").model_id == "nvidia/canary-1b-flash"
