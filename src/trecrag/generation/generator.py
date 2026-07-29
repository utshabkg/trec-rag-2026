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
from trecrag.schema import AnswerSentence, Candidate, RagAnswer, Topic
from trecrag.text_utils import split_sentences, tokenize

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
    def __init__(self, llm: LLMClient, fallback: Generator | None = None):
        self.llm = llm
        self.fallback = fallback or ExtractiveGenerator()

    def generate(
        self, topic: Topic, evidence: list[Candidate], max_words: int, max_citations: int
    ) -> RagAnswer:
        usable = [c for c in evidence if c.text]
        if not usable:
            return RagAnswer(references=[], sentences=[])

        prompt = self._build_prompt(topic, usable, max_words, max_citations)
        # A 1024-word cited answer is ~1900 tokens of JSON; 2048 truncated the
        # array often enough that the parse failed and silently dropped to the
        # extractive fallback.
        response = self.llm.complete(prompt, max_tokens=4096)
        parsed = self._parse_response(response, len(usable))
        if parsed is None:
            return self.fallback.generate(topic, evidence, max_words, max_citations)

        references = [c.docid for c in usable]
        sentences = [AnswerSentence(text=s["text"], citations=s["citations"]) for s in parsed]
        answer = postprocess_rag_answer(references, sentences, max_words, max_citations)
        if not answer.sentences:
            # Parsed fine but nothing survived post-processing (e.g. every
            # citation was unusable), which would submit an empty answer and
            # score zero for this narrative. An extractive answer beats none.
            return self.fallback.generate(topic, evidence, max_words, max_citations)
        return answer

    @staticmethod
    def _build_prompt(
        topic: Topic, usable: list[Candidate], max_words: int, max_citations: int
    ) -> str:
        lines = [
            "You are writing a grounded, cited answer to a research question using only "
            "the evidence passages below.",
            f"Question: {topic.text}",
            "",
            "Evidence passages:",
        ]
        for i, candidate in enumerate(usable):
            lines.append(f"[{i}] {(candidate.text or '')[:2500]}")
        lines += [
            "",
            # These narratives bundle several distinct sub-questions and are scored
            # on how much of the expected content the answer covers, so an explicit
            # length target matters: "at most N words" alone reliably produced
            # ~250-word answers against a 1024-word budget.
            f"Write a thorough answer of {int(max_words * 0.6)}-{int(max_words * 0.9)} "
            f"words (hard maximum {max_words}), as a JSON array of objects with "
            '"text" and "citations".',
            "Identify every distinct sub-question the request asks and address each "
            "one explicitly; breadth of coverage matters more than depth on any "
            "single point. If the evidence does not cover a sub-question, say so "
            "rather than inventing an answer.",
            f'"citations" must be a list of at most {max_citations} zero-based indices '
            "into the evidence passages above, ordered strongest support first.",
            "Every sentence must be supported by at least one cited passage. Do not "
            "invent facts absent from the evidence.",
            'Respond with only the JSON array, e.g. [{"text": "...", "citations": [0, 2]}].',
        ]
        return "\n".join(lines)

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
