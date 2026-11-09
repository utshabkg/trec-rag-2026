"""Final stage: turn the top-ranked evidence into a sentence-cited RAG answer.

`ExtractiveGenerator` is the zero-dependency default — for each evidence
passage it selects the sentence with the highest lexical overlap with the
query and cites that passage. It is always grounded by construction, which
makes it a safe fallback for `LLMGenerator` when the LLM response is missing
or unparsable.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from trecrag.clients.llm_client import LLMClient
from trecrag.generation.postprocess import postprocess_rag_answer
from trecrag.generation.prompts import Revision, render_answer_prompt
from trecrag.generation.verifier import AnswerVerifier
from trecrag.logging_utils import get_logger
from trecrag.schema import AnswerSentence, Candidate, RagAnswer, Topic
from trecrag.text_utils import split_sentences, tokenize

logger = get_logger("generator")

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class Generator(ABC):
    @abstractmethod
    def generate(
        self, topic: Topic, evidence: list[Candidate], max_words: int, max_citations: int
    ) -> RagAnswer: ...


class ExtractiveGenerator(Generator):
    def generate(
        self, topic: Topic, evidence: list[Candidate], max_words: int, max_citations: int
    ) -> RagAnswer:
        usable = [c for c in evidence if c.text]
        references = [c.docid for c in usable]
        query_tokens = set(tokenize(topic.text))

        sentences = []
        for i, candidate in enumerate(usable):
            best = self._best_sentence(candidate.text, query_tokens)
            if best:
                sentences.append(AnswerSentence(text=best, citations=[i]))

        return postprocess_rag_answer(references, sentences, max_words, max_citations)

    @staticmethod
    def _best_sentence(text: str, query_tokens: set[str]) -> str | None:
        sentences = split_sentences(text) or [text.strip()]
        best_sentence = None
        best_score = -1.0
        for sentence in sentences:
            tokens = tokenize(sentence)
            if not tokens:
                continue
            score = sum(1 for token in tokens if token in query_tokens) / (len(tokens) ** 0.5)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        return best_sentence


class LLMGenerator(Generator):
    # A 1024-word cited answer is ~1900 tokens of JSON; 2048 truncated the array
    # often enough that the parse failed and silently dropped to the extractive
    # fallback. A think block is spent before any of that is emitted, so asking
    # for one without raising the ceiling would reintroduce that truncation.
    BASE_MAX_TOKENS = 4096
    THINKING_TOKEN_HEADROOM = 2048

    def __init__(
        self,
        llm: LLMClient,
        fallback: Generator | None = None,
        thinking: bool = False,
        verifier: AnswerVerifier | None = None,
        max_attempts: int = 1,
        max_tokens: int | None = None,
    ):
        self.llm = llm
        self.fallback = fallback or ExtractiveGenerator()
        self.thinking = thinking
        self.verifier = verifier
        self.max_attempts = max(1, max_attempts)
        self.max_tokens = max_tokens or (
            self.BASE_MAX_TOKENS + (self.THINKING_TOKEN_HEADROOM if thinking else 0)
        )

    def generate(
        self, topic: Topic, evidence: list[Candidate], max_words: int, max_citations: int
    ) -> RagAnswer:
        usable = [c for c in evidence if c.text]
        if not usable:
            return RagAnswer(references=[], sentences=[])

        references = [c.docid for c in usable]
        revision: Revision | None = None
        last_answer: RagAnswer | None = None

        for attempt in range(1, self.max_attempts + 1):
            answer = self._attempt(topic, usable, references, max_words, max_citations, revision)
            if answer is None or not answer.sentences:
                # Unparsable, or nothing survived post-processing (e.g. every
                # citation was unusable), which would submit an empty answer and
                # score zero for this narrative. An extractive answer beats none.
                return self.fallback.generate(topic, evidence, max_words, max_citations)

            last_answer = answer
            if self.verifier is None or attempt == self.max_attempts:
                return answer

            result = self.verifier.verify(topic, usable, answer)
            if result.ok:
                if not result.checked:
                    # Shipping unverified. Logged per topic so a run that
                    # believes it verified 119 answers cannot quietly have
                    # verified none.
                    logger.warning(
                        "Topic %s: answer shipped WITHOUT verification (%s could not judge)",
                        topic.topic_id, self.verifier.name,
                    )
                return answer
            logger.info(
                "Topic %s: draft %d rejected by %s (%s); regenerating",
                topic.topic_id, attempt, self.verifier.name, "; ".join(result.issues)[:200],
            )
            revision = result.as_revision()

        # Unreachable while max_attempts >= 1, but returning the last draft is
        # the correct exhaustion behaviour: a rejected answer still beats none.
        return last_answer or self.fallback.generate(topic, evidence, max_words, max_citations)

    def _attempt(
        self,
        topic: Topic,
        usable: list[Candidate],
        references: list[str],
        max_words: int,
        max_citations: int,
        revision: Revision | None,
    ) -> RagAnswer | None:
        prompt = render_answer_prompt(
            question=topic.text,
            passage_texts=[c.text or "" for c in usable],
            max_words=max_words,
            max_citations=max_citations,
            thinking=self.thinking,
            revision=revision,
        )
        # `<think>` blocks are removed by the client's `strip_reasoning`, which
        # also drops an unterminated one to end-of-string -- without that, a
        # think block cut off by the token budget would leave its digits and
        # brackets for `_JSON_ARRAY_RE` to parse as the answer.
        response = self.llm.complete(prompt, max_tokens=self.max_tokens)
        parsed = self._parse_response(response, len(usable))
        if parsed is None:
            return None
        sentences = [AnswerSentence(text=s["text"], citations=s["citations"]) for s in parsed]
        return postprocess_rag_answer(references, sentences, max_words, max_citations)

    @staticmethod
    def _parse_response(response: str, num_evidence: int) -> list[dict] | None:
        match = _JSON_ARRAY_RE.search(response)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list):
            return None

        parsed = []
        for item in data:
            if not isinstance(item, dict) or "text" not in item:
                continue
            citations = item.get("citations", [])
            if not isinstance(citations, list):
                citations = []
            # Models routinely emit "0" instead of 0; dropping those would strip a
            # sentence of all citations and get it deleted in post-processing.
            coerced = []
            for c in citations:
                if isinstance(c, bool):
                    continue
                if isinstance(c, int):
                    coerced.append(c)
                elif isinstance(c, str) and c.strip().lstrip("-").isdigit():
                    coerced.append(int(c.strip()))
            citations = [c for c in coerced if 0 <= c < num_evidence]
            parsed.append({"text": str(item["text"]), "citations": citations})
        return parsed or None
