import pytest

from videotranscriptor.audio import parse_silence_log, plan_chunks

SILENCE_LOG = """
[silencedetect @ 0x55] silence_start: 12.4
[silencedetect @ 0x55] silence_end: 13.1 | silence_duration: 0.7
[silencedetect @ 0x55] silence_start: 28.05
[silencedetect @ 0x55] silence_end: 29.0 | silence_duration: 0.95
size=N/A time=00:00:40.00 bitrate=N/A speed=112x
"""


def test_parse_silence_log_pairs_starts_and_ends():
    assert parse_silence_log(SILENCE_LOG) == [(12.4, 13.1), (28.05, 29.0)]


def test_parse_silence_log_ignores_unterminated_span():
    # ffmpeg omits the final silence_end when the file ends in silence.
    log = "silence_start: 5.0\n"
    assert parse_silence_log(log) == []


def test_parse_silence_log_clamps_negative_start():
    log = "silence_start: -0.02\nsilence_end: 1.0\n"
    assert parse_silence_log(log) == [(0.0, 1.0)]


def test_short_audio_is_a_single_chunk():
    assert plan_chunks(20.0, [], max_chunk=30.0) == [(0.0, 20.0)]


def test_empty_audio_produces_no_chunks():
    assert plan_chunks(0.0, []) == []


def test_chunks_cover_the_whole_timeline_without_gaps():
    chunks = plan_chunks(125.0, [(29.0, 30.0), (58.0, 59.0), (95.0, 96.0)], max_chunk=30.0)
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == pytest.approx(125.0)
    for (_, end), (next_start, _) in zip(chunks, chunks[1:]):
        assert end == pytest.approx(next_start)


def test_boundaries_snap_to_silence_midpoints():
    chunks = plan_chunks(60.0, [(28.0, 30.0)], max_chunk=30.0)
    assert chunks[0][1] == pytest.approx(29.0)


def test_hard_cut_when_a_window_has_no_silence():
    chunks = plan_chunks(75.0, [], max_chunk=30.0)
    assert chunks == [(0.0, 30.0), (30.0, 60.0), (60.0, 75.0)]


def test_no_chunk_exceeds_the_maximum():
    silences = [(float(t), float(t) + 0.5) for t in range(5, 300, 7)]
    for start, end in plan_chunks(300.0, silences, max_chunk=30.0):
        assert end - start <= 30.0 + 1e-6


def test_dense_silence_does_not_produce_tiny_chunks():
    # A silence every second must not turn into one chunk per second.
    silences = [(float(t), float(t) + 0.2) for t in range(1, 90)]
    chunks = plan_chunks(90.0, silences, max_chunk=30.0, min_chunk=5.0)
    assert all(end - start >= 5.0 for start, end in chunks[:-1])


def test_trailing_sliver_is_rebalanced_into_two_usable_chunks():
    chunks = plan_chunks(60.2, [], max_chunk=30.0)
    assert chunks == pytest.approx([(0.0, 30.0), (30.0, 45.1), (45.1, 60.2)])
    assert all(end - start <= 30.0 + 1e-6 for start, end in chunks)
    assert all(end - start >= 1.0 for start, end in chunks)


def test_dense_silence_never_overflows_the_window_after_rebalancing():
    silences = [(float(t), float(t) + 0.2) for t in range(1, 90)]
    for start, end in plan_chunks(90.0, silences, max_chunk=30.0):
        assert end - start <= 30.0 + 1e-6


def test_rejects_non_positive_max_chunk():
    with pytest.raises(ValueError):
        plan_chunks(60.0, [], max_chunk=0.0)
