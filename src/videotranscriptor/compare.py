"""Compare the two approaches on the same audio.

Neither backend is ground truth, so nothing here is an accuracy score. What it
measures is *agreement*: where the cloud model and the local model produce the
same words, you can be fairly confident; where they diverge is where a human
should look.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .models import Transcript

_QUOTES = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
    "—": "-",
}
# Keep letters, digits, and word-internal apostrophes/hyphens.
_STRIP = re.compile(r"[^\w'\-\s]", re.UNICODE)
_EDGE_PUNCT = re.compile(r"^['\-]+|['\-]+$")


def normalize(text: str) -> str:
    """Case-fold and drop punctuation so cosmetic differences do not count.

    Deepgram's smart formatting writes "Dr. Smith, 20%"; Canary writes it
    differently. Comparing raw strings would report disagreement on formatting
    rather than on what was heard.
    """
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _QUOTES.items():
        text = text.replace(src, dst)
    text = _STRIP.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    tokens = []
    for token in normalize(text).split():
        token = _EDGE_PUNCT.sub("", token)
        if token:
            tokens.append(token)
    return tokens


@dataclass
class EditCounts:
    """Word-level edit operations turning `reference` into `hypothesis`."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def reference_length(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def error_rate(self) -> float:
        """Word error rate. 0.0 means identical; can exceed 1.0."""
        length = self.reference_length
        if length == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return (self.substitutions + self.deletions + self.insertions) / length

    @property
    def agreement(self) -> float:
        """Share of reference words both backends agree on, clamped to [0, 1]."""
        return max(0.0, min(1.0, 1.0 - self.error_rate))


def word_edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Levenshtein alignment over word tokens, carrying the op breakdown.

    Two rows of state keeps memory linear, so hour-long transcripts do not
    allocate an N*M matrix.
    """
    ref_len, hyp_len = len(reference), len(hypothesis)
    if ref_len == 0:
        return EditCounts(insertions=hyp_len)
    if hyp_len == 0:
        return EditCounts(deletions=ref_len)

    # Each cell: (distance, substitutions, deletions, insertions, hits)
    previous: List[Tuple[int, int, int, int, int]] = [
        (j, 0, 0, j, 0) for j in range(hyp_len + 1)
    ]

    for i in range(1, ref_len + 1):
        current = [(i, 0, i, 0, 0)]
        ref_word = reference[i - 1]
        for j in range(1, hyp_len + 1):
            match = ref_word == hypothesis[j - 1]
            diag = previous[j - 1]
            up = previous[j]
            left = current[j - 1]

            sub_cost = diag[0] + (0 if match else 1)
            del_cost = up[0] + 1
            ins_cost = left[0] + 1
            best = min(sub_cost, del_cost, ins_cost)

            if best == sub_cost:
                if match:
                    cell = (best, diag[1], diag[2], diag[3], diag[4] + 1)
                else:
                    cell = (best, diag[1] + 1, diag[2], diag[3], diag[4])
            elif best == del_cost:
                cell = (best, up[1], up[2] + 1, up[3], up[4])
            else:
                cell = (best, left[1], left[2], left[3] + 1, left[4])
            current.append(cell)
        previous = current

    _, subs, dels, ins, hits = previous[hyp_len]
    return EditCounts(substitutions=subs, deletions=dels, insertions=ins, hits=hits)


@dataclass
class Divergence:
    """One stretch where the two transcripts disagree."""

    kind: str  # replace | delete | insert
    reference: str
    hypothesis: str
    position: int  # word index in the reference

    def describe(self, ref_name: str, hyp_name: str) -> str:
        if self.kind == "delete":
            return "only in {}: '{}'".format(ref_name, self.reference)
        if self.kind == "insert":
            return "only in {}: '{}'".format(hyp_name, self.hypothesis)
        return "{}: '{}'  |  {}: '{}'".format(
            ref_name, self.reference, hyp_name, self.hypothesis
        )


def find_divergences(
    reference: Sequence[str], hypothesis: Sequence[str], limit: int = 25
) -> List[Divergence]:
    """Human-readable disagreement spans, longest first."""
    matcher = difflib.SequenceMatcher(a=list(reference), b=list(hypothesis), autojunk=False)
    found: List[Divergence] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        found.append(
            Divergence(
                kind=tag,
                reference=" ".join(reference[i1:i2]),
                hypothesis=" ".join(hypothesis[j1:j2]),
                position=i1,
            )
        )
    found.sort(key=lambda d: -(len(d.reference.split()) + len(d.hypothesis.split())))
    return found[:limit]


@dataclass
class Comparison:
    """Side-by-side result for two backends on the same audio."""

    reference: Transcript
    hypothesis: Transcript
    counts: EditCounts
    divergences: List[Divergence] = field(default_factory=list)
    total_divergences: int = 0

    @property
    def names(self) -> Tuple[str, str]:
        return self.reference.backend, self.hypothesis.backend

    def speed_summary(self) -> Dict[str, Optional[float]]:
        return {
            self.reference.backend: self.reference.realtime_factor,
            self.hypothesis.backend: self.hypothesis.realtime_factor,
        }

    def to_markdown(self) -> str:
        ref_name, hyp_name = self.names
        lines = [
            "# Backend comparison",
            "",
            "`{}` (reference) vs `{}`".format(ref_name, hyp_name),
            "",
            "| metric | {} | {} |".format(ref_name, hyp_name),
            "| --- | --- | --- |",
            "| model | `{}` | `{}` |".format(self.reference.model, self.hypothesis.model),
            "| words | {} | {} |".format(
                self.reference.word_count, self.hypothesis.word_count
            ),
            "| segments | {} | {} |".format(
                len(self.reference.segments), len(self.hypothesis.segments)
            ),
            "| processing time | {} | {} |".format(
                _fmt_seconds(self.reference.processing_time),
                _fmt_seconds(self.hypothesis.processing_time),
            ),
            "| realtime factor | {} | {} |".format(
                _fmt_rtf(self.reference.realtime_factor),
                _fmt_rtf(self.hypothesis.realtime_factor),
            ),
            "",
            "## Agreement",
            "",
            "| metric | value |",
            "| --- | --- |",
            "| word agreement | {:.1%} |".format(self.counts.agreement),
            "| word difference rate | {:.1%} |".format(self.counts.error_rate),
            "| matching words | {} |".format(self.counts.hits),
            "| substitutions | {} |".format(self.counts.substitutions),
            "| deletions (missing from {}) | {} |".format(hyp_name, self.counts.deletions),
            "| insertions (extra in {}) | {} |".format(hyp_name, self.counts.insertions),
            "",
            "Neither transcript is ground truth. These numbers measure how much "
            "the two approaches agree, and treat `{}` as the reference purely as "
            "a convention.".format(ref_name),
            "",
        ]

        if self.divergences:
            lines += [
                "## Largest disagreements ({} of {})".format(
                    len(self.divergences), self.total_divergences
                ),
                "",
            ]
            for divergence in self.divergences:
                lines.append("- word ~{}: {}".format(divergence.position, divergence.describe(ref_name, hyp_name)))
            lines.append("")
        else:
            lines += ["The two transcripts are identical after normalisation.", ""]
        return "\n".join(lines)

    def to_text(self) -> str:
        ref_name, hyp_name = self.names
        return "\n".join(
            [
                "{} vs {}".format(ref_name, hyp_name),
                "  word agreement : {:.1%}".format(self.counts.agreement),
                "  substitutions  : {}".format(self.counts.substitutions),
                "  only in {:<7}: {}".format(ref_name[:7], self.counts.deletions),
                "  only in {:<7}: {}".format(hyp_name[:7], self.counts.insertions),
                "  speed          : {} {} / {} {}".format(
                    ref_name,
                    _fmt_rtf(self.reference.realtime_factor),
                    hyp_name,
                    _fmt_rtf(self.hypothesis.realtime_factor),
                ),
            ]
        )


def compare(
    reference: Transcript, hypothesis: Transcript, max_divergences: int = 25
) -> Comparison:
    ref_tokens = tokenize(reference.text)
    hyp_tokens = tokenize(hypothesis.text)
    counts = word_edit_counts(ref_tokens, hyp_tokens)

    all_divergences = find_divergences(ref_tokens, hyp_tokens, limit=10**9)
    return Comparison(
        reference=reference,
        hypothesis=hypothesis,
        counts=counts,
        divergences=all_divergences[:max_divergences],
        total_divergences=len(all_divergences),
    )


def _fmt_seconds(value: Optional[float]) -> str:
    return "{:.1f}s".format(value) if value else "n/a"


def _fmt_rtf(value: Optional[float]) -> str:
    return "{:.2f}x".format(value) if value else "n/a"
