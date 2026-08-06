import pytest

from videotranscriptor.compare import (
    compare,
    find_divergences,
    normalize,
    tokenize,
    word_edit_counts,
)
from videotranscriptor.models import Transcript


def make(text, backend="deepgram", **kwargs):
    return Transcript(text=text, backend=backend, model="m", **kwargs)


def test_normalize_folds_case_and_punctuation():
    assert normalize("Hello, World!") == "hello world"


def test_normalize_keeps_word_internal_apostrophes_and_hyphens():
    assert normalize("It's a well-known fact.") == "it's a well-known fact"


def test_normalize_unifies_smart_quotes_and_dashes():
    assert normalize("it’s — fine") == normalize("it's - fine")


def test_tokenize_strips_edge_punctuation():
    assert tokenize("'quoted' word-") == ["quoted", "word"]


def test_identical_token_streams_have_no_errors():
    tokens = ["the", "quick", "brown", "fox"]
    counts = word_edit_counts(tokens, tokens)
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 0)
    assert counts.hits == 4
    assert counts.error_rate == 0.0
    assert counts.agreement == 1.0


def test_counts_substitution_deletion_and_insertion():
    reference = ["a", "b", "c", "d", "e", "f"]
    hypothesis = ["a", "x", "c", "e", "f", "g"]  # b->x, d dropped, g added
    counts = word_edit_counts(reference, hypothesis)
    assert counts.substitutions == 1
    assert counts.deletions == 1
    assert counts.insertions == 1
    assert counts.hits == 4
    assert counts.error_rate == pytest.approx(0.5)


def test_ties_are_resolved_toward_substitutions():
    # "a b c d" -> "a x d e" is reachable in three edits either way; the
    # aligner takes the substitution path rather than mixing in a
    # deletion and an insertion of the same cost.
    counts = word_edit_counts(["a", "b", "c", "d"], ["a", "x", "d", "e"])
    assert counts.substitutions == 3
    assert counts.deletions == counts.insertions == 0
    assert counts.error_rate == pytest.approx(0.75)


def test_empty_hypothesis_is_all_deletions():
    counts = word_edit_counts(["a", "b"], [])
    assert counts.deletions == 2
    assert counts.error_rate == 1.0


def test_empty_reference_is_all_insertions():
    counts = word_edit_counts([], ["a", "b"])
    assert counts.insertions == 2
    assert counts.error_rate == 1.0


def test_two_empty_transcripts_agree():
    counts = word_edit_counts([], [])
    assert counts.error_rate == 0.0
    assert counts.agreement == 1.0


def test_error_rate_above_one_is_clamped_for_agreement():
    counts = word_edit_counts(["a"], ["x", "y", "z", "w"])
    assert counts.error_rate > 1.0
    assert counts.agreement == 0.0


def test_edit_counts_are_consistent_with_the_distance():
    reference = "the quick brown fox jumps over the lazy dog".split()
    hypothesis = "the quick brown cat jumped over a lazy dog today".split()
    counts = word_edit_counts(reference, hypothesis)
    assert counts.hits + counts.substitutions + counts.deletions == len(reference)
    assert counts.hits + counts.substitutions + counts.insertions == len(hypothesis)


def test_divergences_are_ordered_by_size():
    reference = "one two three four five six".split()
    hypothesis = "one XX YY ZZ five six".split()
    divergences = find_divergences(reference, hypothesis)
    assert divergences[0].kind == "replace"
    assert divergences[0].reference == "two three four"
    assert divergences[0].hypothesis == "XX YY ZZ"


def test_divergence_limit_is_respected():
    reference = ["word{}".format(i) for i in range(40)]
    hypothesis = ["other{}".format(i) if i % 2 else reference[i] for i in range(40)]
    assert len(find_divergences(reference, hypothesis, limit=5)) == 5


def test_compare_ignores_formatting_differences():
    result = compare(
        make("Hello, World — it’s fine."),
        make("hello world it's fine", backend="canary"),
    )
    assert result.counts.agreement == 1.0
    assert result.total_divergences == 0


def test_compare_reports_totals_beyond_the_shown_limit():
    reference = " ".join("word{}".format(i) for i in range(60))
    hypothesis = " ".join(
        "word{}".format(i) if i % 3 else "other{}".format(i) for i in range(60)
    )
    result = compare(make(reference), make(hypothesis, backend="canary"), max_divergences=3)
    assert len(result.divergences) == 3
    assert result.total_divergences > 3


def test_markdown_report_names_both_backends_and_metrics():
    result = compare(
        make("hello world", audio_duration=10.0, processing_time=1.0),
        make("hello there", backend="canary", audio_duration=10.0, processing_time=40.0),
    )
    report = result.to_markdown()
    assert "`deepgram` (reference) vs `canary`" in report
    assert "word agreement" in report
    assert "0.10x" in report and "4.00x" in report
    assert "substitutions" in report


def test_text_report_is_compact():
    result = compare(make("hello world"), make("hello world", backend="canary"))
    text = result.to_text()
    assert "deepgram vs canary" in text
    assert "100.0%" in text
