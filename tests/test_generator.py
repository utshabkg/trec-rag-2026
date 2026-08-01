import unittest

from trecrag.generation.generator import ExtractiveGenerator, LLMGenerator
from trecrag.clients.llm_client import LLMClient
from trecrag.schema import Candidate, Topic


class TestExtractiveGenerator(unittest.TestCase):
    def test_every_sentence_cites_the_source_it_was_drawn_from(self):
        topic = Topic(topic_id="t1", text="congestion pricing transit funding")
        evidence = [
            Candidate(
                docid="doc0",
                score=1.0,
                rank=1,
                text="Irrelevant filler sentence. Congestion pricing can fund transit reliably.",
            ),
            Candidate(
                docid="doc1",
                score=0.9,
                rank=2,
                text="Another filler sentence. Transit funding requires accountability and oversight.",
            ),
        ]
        answer = ExtractiveGenerator().generate(topic, evidence, max_words=1024, max_citations=3)
        self.assertEqual(answer.references, ["doc0", "doc1"])
        self.assertEqual(len(answer.sentences), 2)
        for i, sentence in enumerate(answer.sentences):
            self.assertEqual(sentence.citations, [i])

    def test_skips_evidence_without_text(self):
        topic = Topic(topic_id="t1", text="q")
        evidence = [Candidate(docid="notext", score=1.0, rank=1, text=None)]
        answer = ExtractiveGenerator().generate(topic, evidence, max_words=1024, max_citations=3)
        self.assertEqual(answer.references, [])
        self.assertEqual(answer.sentences, [])

    def test_respects_word_budget(self):
        topic = Topic(topic_id="t1", text="funding")
        evidence = [
            Candidate(docid=f"d{i}", score=1.0, rank=i, text=f"funding sentence number {i} here.")
            for i in range(1, 4)
        ]
        answer = ExtractiveGenerator().generate(topic, evidence, max_words=6, max_citations=3)
        total_words = sum(len(s.text.split()) for s in answer.sentences)
        self.assertLessEqual(total_words, 6)


class _JsonLLMClient(LLMClient):
    def __init__(self, response: str):
        self._response = response

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        return self._response


class TestLLMGenerator(unittest.TestCase):
    def test_parses_well_formed_json_response(self):
        topic = Topic(topic_id="t1", text="q")
        evidence = [Candidate(docid="d0", score=1.0, rank=1, text="some evidence text")]
        response = '[{"text": "A grounded claim.", "citations": [0]}]'
        generator = LLMGenerator(_JsonLLMClient(response))
        answer = generator.generate(topic, evidence, max_words=1024, max_citations=3)
        self.assertEqual(answer.references, ["d0"])
        self.assertEqual(answer.sentences[0].text, "A grounded claim.")

    def test_falls_back_to_extractive_on_malformed_response(self):
        topic = Topic(topic_id="t1", text="funding evidence")
        evidence = [Candidate(docid="d0", score=1.0, rank=1, text="funding evidence sentence here.")]
        generator = LLMGenerator(_JsonLLMClient("not json at all"))
        answer = generator.generate(topic, evidence, max_words=1024, max_citations=3)
        self.assertEqual(answer.references, ["d0"])
        self.assertTrue(len(answer.sentences) >= 1)

    def test_no_usable_evidence_returns_empty_answer(self):
        generator = LLMGenerator(_JsonLLMClient("[]"))
        answer = generator.generate(Topic("t1", "q"), [], max_words=1024, max_citations=3)
        self.assertEqual(answer.references, [])
        self.assertEqual(answer.sentences, [])

    def test_bool_citations_are_rejected_not_treated_as_ints(self):
        response = '[{"text": "A claim.", "citations": [true, 2]}]'
        parsed = LLMGenerator._parse_response(response, num_evidence=3)
        self.assertEqual(parsed[0]["citations"], [2])

    def test_coerces_digit_strings_to_int_citations(self):
        """Models routinely emit "0" instead of 0; dropping those would strip
        a sentence of all citations and get it deleted in post-processing."""
        response = '[{"text": "A claim.", "citations": ["0", "-1", "abc"]}]'
        parsed = LLMGenerator._parse_response(response, num_evidence=3)
        self.assertEqual(parsed[0]["citations"], [0])

    def test_falls_back_to_extractive_when_postprocessing_empties_every_citation(self):
        """Well-formed JSON whose citations are all out of range parses fine but
        leaves zero sentences after post-processing -- that must still trigger
        the extractive fallback, not a schema-valid empty answer."""
        topic = Topic(topic_id="t1", text="funding evidence")
        evidence = [Candidate(docid="d0", score=1.0, rank=1, text="funding evidence sentence here.")]
        response = '[{"text": "A claim.", "citations": [99]}]'
        generator = LLMGenerator(_JsonLLMClient(response))
        answer = generator.generate(topic, evidence, max_words=1024, max_citations=3)
        self.assertEqual(answer.references, ["d0"])
        self.assertTrue(len(answer.sentences) >= 1)

    def test_parses_json_array_with_trailing_bracketed_text(self):
        topic = Topic(topic_id="t1", text="q")
        evidence = [
            Candidate(docid="d0", score=1.0, rank=1, text="evidence zero"),
            Candidate(docid="d1", score=1.0, rank=2, text="evidence one"),
        ]
        response = '[{"text": "A grounded claim.", "citations": [0]}]\n\nSources: [0], [2]'
        generator = LLMGenerator(_JsonLLMClient(response))
        answer = generator.generate(topic, evidence, max_words=1024, max_citations=3)
        self.assertEqual(answer.references, ["d0"])
        self.assertEqual(answer.sentences[0].text, "A grounded claim.")


if __name__ == "__main__":
    unittest.main()
