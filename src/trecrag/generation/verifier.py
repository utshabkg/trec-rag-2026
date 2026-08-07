"""Check a generated answer against the evidence it was given.

Two failure modes, measured on this system rather than assumed:

* **Misattribution, not fabrication.** Of 1,135 checkable sentences in a real
  run, 309 matched a cited document verbatim, 33 matched a document that was
  retrieved but *not* cited, and the rest were paraphrase (`WORK_LOG.md` §5.5).
  So the realistic hallucination here is a sentence whose content is not in the
  passage it points at -- which makes every citation in the submission suspect
  even when the fact itself was retrieved.

* **Omission.** Answers hold at ~373 words against a 1024-word budget and ~9.5
  sentences whether a narrative asks 3 things or 18, and the missing material
  sits in documents the answer already cites (§5.2). Evidence handed to the
  generator and then never used is the observable form of that.

Both checks produce *instructions*, not just verdicts: a rejected draft is
regenerated with the specific gap named, since "try again" reliably produces the
same answer.

`LexicalGroundingVerifier` is the default because it is free. The LLM judge is a
second full generation-sized call on a stage that is already 67% of runtime, so
it is opt-in and worth enabling only when the budget is there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from trecrag.clients.llm_client import LLMClient
from trecrag.generation.prompts import (
    VERIFIER_PASSAGE_CHAR_CAP,
    Revision,
    render_verification_prompt,
)
from trecrag.logging_utils import get_logger
from trecrag.schema import AnswerSentence, Candidate, RagAnswer, Topic
from trecrag.text_utils import tokenize

logger = get_logger("verifier")

# Words carry no topical content, so their overlap says nothing about grounding.
_STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does for from had has have
    how i if in into is it its may might more most must no not of on or should
    so such than that the their them then there these they this those to was
    were what when where which who will with would you your""".split()
)


class VerifierUnavailableError(RuntimeError):
    """The verifier could not judge answers at all, repeatedly."""


@dataclass
class VerificationResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    # False when the verifier could not actually reach a judgement. Distinct
    # from `ok=True`: "I checked and it is fine" and "I could not check" are the
    # same value to the generator, which is how a broken verifier turns into a
    # run of unverified answers that looks like a run of clean ones.
    checked: bool = True

    def as_revision(self) -> Revision:
        return Revision(issues=list(self.issues), instructions=list(self.instructions))


class AnswerVerifier(ABC):
    name: str = "verifier"

    @abstractmethod
    def verify(
        self, topic: Topic, evidence: list[Candidate], answer: RagAnswer
    ) -> VerificationResult: ...


def _content_words(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 2}


class LexicalGroundingVerifier(AnswerVerifier):
    """Deterministic, no API call.

    A sentence is *ungrounded* when too few of its content words appear in any
    passage it cites. This cannot detect a fluent paraphrase that changes a
    number, and it is not meant to: it catches the measured failure -- a sentence
    pointed at a passage that does not carry it -- at zero cost.

    The threshold is a floor, not a similarity score. Paraphrase legitimately
    drops overlap, so `min_overlap` is set low enough that only sentences with
    almost nothing in common with their citation are flagged. Raising it starts
    rejecting good paraphrase, which costs a regeneration and buys nothing.
    """

    name = "lexical"

    def __init__(
        self,
        min_overlap: float = 0.25,
        max_ungrounded_ratio: float = 0.34,
        min_evidence_used: float = 0.5,
        check_hallucination: bool = True,
        check_omission: bool = True,
    ):
        self.min_overlap = min_overlap
        self.max_ungrounded_ratio = max_ungrounded_ratio
        self.min_evidence_used = min_evidence_used
        self.check_hallucination = check_hallucination
        self.check_omission = check_omission

    def verify(
        self, topic: Topic, evidence: list[Candidate], answer: RagAnswer
    ) -> VerificationResult:
        issues: list[str] = []
        instructions: list[str] = []
        if not answer.sentences or not evidence:
            return VerificationResult(ok=True)

        passage_words = [_content_words(c.text or "") for c in evidence]

        if self.check_hallucination:
            ungrounded = self._ungrounded(answer.sentences, passage_words)
            ratio = len(ungrounded) / len(answer.sentences)
            if ratio > self.max_ungrounded_ratio:
                shown = "; ".join(f'"{s[:70]}…"' for s in ungrounded[:3])
                issues.append(
                    f"{len(ungrounded)} of {len(answer.sentences)} sentences state "
                    f"things their cited passages do not support: {shown}"
                )
                instructions.append(
                    "Every sentence must be supported by the passage it cites. If a "
                    "claim is not in any passage, drop the claim -- do not re-cite it "
                    "against a different passage."
                )

        if self.check_omission:
            issues_omission, instructions_omission = self._omissions(evidence, answer)
            issues.extend(issues_omission)
            instructions.extend(instructions_omission)

        return VerificationResult(ok=not issues, issues=issues, instructions=instructions)

    def _ungrounded(
        self, sentences: list[AnswerSentence], passage_words: list[set[str]]
    ) -> list[str]:
        ungrounded: list[str] = []
        for sentence in sentences:
            words = _content_words(sentence.text)
            if not words:
                continue
            cited = [
                passage_words[i] for i in sentence.citations if 0 <= i < len(passage_words)
            ]
            if not cited:
                ungrounded.append(sentence.text)
                continue
            best = max((len(words & p) / len(words) for p in cited), default=0.0)
            if best < self.min_overlap:
                ungrounded.append(sentence.text)
        return ungrounded

    def _omissions(
        self, evidence: list[Candidate], answer: RagAnswer
    ) -> tuple[list[str], list[str]]:
        issues: list[str] = []
        instructions: list[str] = []

        used = {i for s in answer.sentences for i in s.citations}
        unused = [i for i in range(len(evidence)) if i not in used]
        if len(used) < len(evidence) * self.min_evidence_used:
            issues.append(
                f"only {len(used)} of {len(evidence)} evidence passages were cited; "
                f"passages {unused[:6]} contribute nothing to the answer"
            )
            instructions.append(
                "Every passage was retrieved because it answers part of this request. "
                "Use the uncited ones: add the material they carry, or say why that "
                "part of the request is answered differently."
            )

        # Sub-question coverage, available only when the decomposed arm stamped
        # provenance onto the candidates. A sub-question that has evidence in the
        # pool but no cited passage is the coverage failure in its exact form.
        uncovered = self._uncovered_sub_questions(evidence, used)
        if uncovered:
            issues.append(
                "the answer cites nothing for sub-question(s) "
                + ", ".join(sorted(uncovered))
                + ", although the evidence includes passages for them"
            )
            instructions.append(
                "Address every sub-question. Each one has at least one supporting "
                "passage in the evidence -- find it and cite it."
            )
        return issues, instructions

    @staticmethod
    def _uncovered_sub_questions(evidence: list[Candidate], used: set[int]) -> set[str]:
        available: set[str] = set()
        covered: set[str] = set()
        for i, candidate in enumerate(evidence):
            for key, value in candidate.features.items():
                if not key.startswith("ain_hit_sq") or not value:
                    continue
                sub_question_id = key[len("ain_hit_sq") :]
                available.add(sub_question_id)
                if i in used:
                    covered.add(sub_question_id)
        return available - covered


class LLMVerifier(AnswerVerifier):
    """One judging call per answer. Catches paraphrase-level fabrication that
    lexical overlap cannot, at the price of a second generation-sized call on
    the stage that already dominates runtime (34.1s of a 50.6s narrative).

    An unparsable judgment passes the answer rather than failing it: a verifier
    that rejects on its own parse errors would regenerate good answers and, at
    `max_attempts`, ship the last draft anyway -- pure cost, no signal.
    """

    name = "llm"

    # The judgment itself is two lines. A think block is spent before either of
    # them, so asking for one without raising the ceiling truncates it: the
    # block loses its closing tag, `strip_reasoning` drops everything, and an
    # empty judgment is read as "answer passed" -- verification would quietly
    # switch itself off instead of failing.
    BASE_MAX_TOKENS = 512
    THINKING_TOKEN_HEADROOM = 1024

    def __init__(
        self,
        llm: LLMClient,
        thinking: bool = False,
        max_passage_chars: int = VERIFIER_PASSAGE_CHAR_CAP,
        max_tokens: int | None = None,
        max_consecutive_unparsable: int = 3,
        strict: bool = False,
    ):
        self.llm = llm
        self.thinking = thinking
        self.max_passage_chars = max_passage_chars
        self.max_tokens = max_tokens or (
            self.BASE_MAX_TOKENS + (self.THINKING_TOKEN_HEADROOM if thinking else 0)
        )
        self.max_consecutive_unparsable = max_consecutive_unparsable
        self.strict = strict
        self.degraded = False
        self.stats = {
            "judged": 0,
            "parsed": 0,
            "unparsable": 0,
            "empty_responses": 0,
            "consecutive_unparsable": 0,
        }

    def verify(
        self, topic: Topic, evidence: list[Candidate], answer: RagAnswer
    ) -> VerificationResult:
        if not answer.sentences or not evidence:
            return VerificationResult(ok=True)
        if self.degraded:
            # Already established this verifier cannot judge. Continuing to call
            # it would spend a generation-sized request per narrative to learn
            # the same thing 119 more times.
            return VerificationResult(ok=True, checked=False)

        prompt = render_verification_prompt(
            question=topic.text,
            passage_texts=[c.text or "" for c in evidence],
            sentences=[(s.text, list(s.citations)) for s in answer.sentences],
            thinking=self.thinking,
            passage_char_cap=self.max_passage_chars,
        )
        response = self.llm.complete(prompt, max_tokens=self.max_tokens)
        result = self._parse(response, len(answer.sentences))
        self._record(result, response, topic)
        return result

    def _record(self, result: VerificationResult, response: str, topic: Topic) -> None:
        self.stats["judged"] += 1
        if result.checked:
            self.stats["parsed"] += 1
            self.stats["consecutive_unparsable"] = 0
            return

        self.stats["unparsable"] += 1
        self.stats["consecutive_unparsable"] += 1

        if not response.strip():
            # An empty response after `strip_reasoning` is the signature of a
            # think block that consumed the whole budget: the block loses its
            # closing tag, so everything is stripped. Name the fix rather than
            # reporting a generic parse failure.
            self.stats["empty_responses"] += 1
            cause = (
                f"usually the think block exhausting max_tokens ({self.max_tokens}) -- "
                f"raise verification.max_tokens or set verification.thinking: false"
                if self.thinking
                else f"the judge model returned an empty completion -- check the model "
                f"and max_tokens ({self.max_tokens})"
            )
            logger.warning("Topic %s: verifier returned nothing. This is %s.",
                           topic.topic_id, cause)
        else:
            logger.warning(
                "Topic %s: verifier reply had no UNSUPPORTED:/MISSING: lines; "
                "answer passed unchecked.", topic.topic_id,
            )

        if self.stats["consecutive_unparsable"] >= self.max_consecutive_unparsable:
            message = (
                f"Answer verification is NOT running: {self.stats['consecutive_unparsable']} "
                f"consecutive unreadable judgements "
                f"({self.stats['empty_responses']} empty). Every answer since has shipped "
                f"unverified. Check verification.max_tokens ({self.max_tokens}), "
                f"verification.thinking ({self.thinking}), and the judge model."
            )
            if self.strict:
                raise VerifierUnavailableError(message)
            self.degraded = True
            logger.error("%s Disabling the verifier for the rest of this run.", message)

    @staticmethod
    def _parse(response: str, n_sentences: int) -> VerificationResult:
        issues: list[str] = []
        instructions: list[str] = []
        unsupported_line = ""
        missing_line = ""
        for line in response.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("UNSUPPORTED:"):
                unsupported_line = stripped.split(":", 1)[1].strip()
            elif upper.startswith("MISSING:"):
                missing_line = stripped.split(":", 1)[1].strip()

        if not unsupported_line and not missing_line:
            # Passed through rather than rejected: a verifier that failed the
            # answer on its own parse errors would regenerate good drafts and,
            # at max_attempts, ship the last one anyway. `checked=False` is what
            # keeps this from being indistinguishable from a clean judgement.
            return VerificationResult(ok=True, checked=False)

        if unsupported_line and unsupported_line.upper() != "NONE":
            numbers = [
                int(tok) for tok in unsupported_line.replace(",", " ").split() if tok.isdigit()
            ]
            numbers = [n for n in numbers if 1 <= n <= n_sentences]
            if numbers:
                issues.append(
                    "sentences "
                    + ", ".join(str(n) for n in numbers)
                    + " state things their cited passages do not support"
                )
                instructions.append(
                    "Rewrite or drop those sentences. Keep only claims the cited "
                    "passage actually carries."
                )

        if missing_line and missing_line.upper() != "NONE":
            issues.append(f"the draft leaves out material the evidence covers: {missing_line}")
            instructions.append(
                "Work that material into the answer and cite the passage carrying it."
            )

        return VerificationResult(ok=not issues, issues=issues, instructions=instructions)


class CompositeVerifier(AnswerVerifier):
    """Runs several verifiers, merging their findings. Ordered cheapest-first so
    a draft the free check already rejects never pays for the LLM judge."""

    name = "composite"

    def __init__(self, verifiers: list[AnswerVerifier]):
        self.verifiers = verifiers

    def verify(
        self, topic: Topic, evidence: list[Candidate], answer: RagAnswer
    ) -> VerificationResult:
        issues: list[str] = []
        instructions: list[str] = []
        # Checked if *any* member reached a judgement. The lexical verifier
        # always does, so a composite containing it never reports the answer as
        # wholly unchecked -- which is accurate: it was checked, just not by the
        # member that failed.
        checked = False
        for verifier in self.verifiers:
            result = verifier.verify(topic, evidence, answer)
            checked = checked or result.checked
            if not result.ok:
                issues.extend(result.issues)
                instructions.extend(result.instructions)
                break
        return VerificationResult(
            ok=not issues, issues=issues, instructions=instructions, checked=checked
        )
