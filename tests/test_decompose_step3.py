"""Tests for step-3 query conditioning, guarded expansion, and client-side RM3.

Two failure modes drive these tests, and neither is visible downstream:

  * semantic drift -- a conditioned or expanded query that retrieves perfectly well
    for a topic nobody asked about. Hence rule 6 in parse_conditioned and its
    independent re-derivation in the write gate;
  * a PRF stage that quietly degrades to its input. An API failure must be visible
    as a status on the record, never as a silently unexpanded query.
"""

import copy
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "decompose"))

import preprocess_step3_decompose as step3  # noqa: E402
from preprocess_step3_decompose import (  # noqa: E402
    PRFResult,
    QueryExpansionParseError,
    build_jobs,
    expand_queries,
    parse_conditioned,
    parse_expansions,
    rm3_expansion,
    topic_tokens,
    validate_final_record,
)

AIN = "Evidence regarding leadership accountability for nursing DEI."
COND = "<conditioned>leadership accountability for nursing DEI initiatives</conditioned>"
EXP = (
    "<expansions>"
    '<expansion id="1">equitable hiring</expansion>'
    '<expansion id="2">inclusive recruitment</expansion>'
    "</expansions>"
)


class TestParseConditioned(unittest.TestCase):
    def test_accepts_and_normalises_whitespace(self):
        raw = "<conditioned>\n  leadership accountability   for nursing DEI\n</conditioned>"
        self.assertEqual(parse_conditioned(raw, AIN), "leadership accountability for nursing DEI")

    def test_fences_are_tolerated(self):
        self.assertTrue(parse_conditioned(f"```\n{COND}\n```", AIN))

    def _assert_rejected(self, raw, rule, ain=AIN):
        with self.assertRaises(QueryExpansionParseError) as ctx:
            parse_conditioned(raw, ain)
        self.assertIn(rule, str(ctx.exception))

    def test_empty_generation(self):
        for raw in ("", "   ", None):
            with self.assertRaises(QueryExpansionParseError):
                parse_conditioned(raw, AIN)

    def test_rule1_requires_exactly_one_tag(self):
        self._assert_rejected("leadership accountability", "rule1")
        self._assert_rejected(COND + COND, "rule1")

    def test_rule2_rejects_surrounding_prose(self):
        self._assert_rejected(f"Here is the query:\n{COND}", "rule2")

    def test_rule3_rejects_an_empty_query(self):
        self._assert_rejected("<conditioned>   </conditioned>", "rule3")

    def test_rule4_rejects_a_truncated_tag(self):
        self._assert_rejected("<conditioned>leadership <condi</conditioned>", "rule4")

    def test_rule5_rejects_a_query_that_grew_into_a_paragraph(self):
        """Conditioning specialises. Past this length it has started adding scope,
        which is the one thing this stage must not do."""
        long_query = " ".join(["leadership"] + ["padding"] * 60)
        self._assert_rejected(f"<conditioned>{long_query}</conditioned>", "rule5")

    def test_rule6_rejects_a_query_that_lost_the_topic(self):
        """The dangerous case: a fluent query for a different need. Nothing
        downstream could detect it -- it would retrieve, just wrongly."""
        self._assert_rejected("<conditioned>hospital budget planning cycles</conditioned>", "rule6")

    def test_rule6_ignores_the_evidence_regarding_scaffolding(self):
        """Step 2 writes every AIN as 'Evidence regarding X' and 3A is told to drop
        that prefix, so it must not count as topic overlap."""
        self.assertEqual(topic_tokens(AIN) & {"evidence", "regarding"}, set())
        self._assert_rejected("<conditioned>evidence regarding budgets</conditioned>", "rule6")


class TestParseExpansions(unittest.TestCase):
    def test_accepts_a_variant_list(self):
        self.assertEqual(parse_expansions(EXP), ["equitable hiring", "inclusive recruitment"])

    def _assert_rejected(self, raw, rule):
        with self.assertRaises(QueryExpansionParseError) as ctx:
            parse_expansions(raw)
        self.assertIn(rule, str(ctx.exception))

    def test_rule1_requires_exactly_one_wrapper(self):
        self._assert_rejected('<expansion id="1">a</expansion>', "rule1")
        self._assert_rejected(EXP + EXP, "rule1")

    def test_rule2_rejects_prose_outside_or_between_tags(self):
        self._assert_rejected(f"Sure!\n{EXP}", "rule2")
        self._assert_rejected(
            '<expansions><expansion id="1">a</expansion> and also b </expansions>', "rule2"
        )

    def test_rule3_requires_at_least_one_non_empty_variant(self):
        self._assert_rejected("<expansions></expansions>", "rule3")
        self._assert_rejected('<expansions><expansion id="1">  </expansion></expansions>', "rule3")

    def test_rule4_rejects_id_gaps_and_disorder(self):
        self._assert_rejected('<expansions><expansion id="2">a</expansion></expansions>', "rule4")
        self._assert_rejected(
            '<expansions><expansion id="1">a</expansion><expansion id="3">b</expansion></expansions>',
            "rule4",
        )

    def test_rule5_rejects_a_truncated_tag(self):
        self._assert_rejected('<expansions><expansion id="1">a <exp</expansion></expansions>', "rule5")

    def test_rule6_rejects_duplicates_case_insensitively(self):
        self._assert_rejected(
            "<expansions>"
            '<expansion id="1">fair hiring</expansion><expansion id="2">Fair Hiring</expansion>'
            "</expansions>",
            "rule6",
        )

    def test_rule7_rejects_a_list_that_became_a_brainstorm(self):
        raw = "<expansions>" + "".join(
            f'<expansion id="{i}">variant {i}</expansion>' for i in range(1, 14)
        ) + "</expansions>"
        self._assert_rejected(raw, "rule7")


@dataclass
class FakeCandidate:
    docid: str
    score: float
    text: str
    rank: int = 1


class TestRM3(unittest.TestCase):
    def test_promotes_terms_that_recur_across_feedback_documents(self):
        pool = [
            FakeCandidate("d1", 10.0, "inclusive recruitment inclusive recruitment nurses"),
            FakeCandidate("d2", 9.0, "inclusive recruitment for nurses in hospitals"),
            FakeCandidate("d3", 8.0, "inclusive recruitment nurses hospitals"),
        ]
        result = rm3_expansion("equitable hiring", pool, fb_terms=5)
        self.assertEqual(result.status, "ok")
        terms = [t for t, _ in result.terms]
        self.assertIn("inclusive", terms)
        self.assertIn("recruitment", terms)

    def test_query_terms_are_never_returned_as_expansion(self):
        """They are already in the query being appended to; re-adding them would
        double their term frequency rather than widen coverage."""
        pool = [FakeCandidate(f"d{i}", 5.0, "hiring hiring nursing policy") for i in range(3)]
        result = rm3_expansion("hiring nursing", pool)
        self.assertNotIn("hiring", [t for t, _ in result.terms])
        self.assertNotIn("nursing", [t for t, _ in result.terms])

    def test_stopwords_and_short_tokens_never_survive(self):
        pool = [FakeCandidate(f"d{i}", 5.0, "the and of a it to policy") for i in range(3)]
        terms = [t for t, _ in rm3_expansion("hiring", pool).terms]
        self.assertEqual(set(terms) & {"the", "and", "of", "a", "it", "to"}, set())

    def test_a_single_outlier_document_cannot_supply_the_expansion(self):
        pool = [
            FakeCandidate("d1", 10.0, "unicorn unicorn unicorn unicorn"),
            FakeCandidate("d2", 9.0, "policy documents"),
            FakeCandidate("d3", 8.0, "policy documents"),
        ]
        terms = [t for t, _ in rm3_expansion("hiring", pool, min_doc_freq=2).terms]
        self.assertNotIn("unicorn", terms)
        self.assertIn("policy", terms)

    def test_weights_are_rm3_scaled_and_ordered(self):
        pool = [FakeCandidate(f"d{i}", 5.0, "alpha alpha beta gamma") for i in range(3)]
        result = rm3_expansion("hiring", pool, alpha=0.5, fb_terms=3)
        weights = [w for _, w in result.terms]
        self.assertEqual(weights, sorted(weights, reverse=True))
        # (1 - alpha) * a normalised distribution sums to (1 - alpha).
        self.assertAlmostEqual(sum(weights), 0.5, places=4)

    def test_ties_break_deterministically(self):
        pool = [FakeCandidate(f"d{i}", 5.0, "beta alpha gamma") for i in range(3)]
        first = rm3_expansion("hiring", pool, fb_terms=2).terms
        second = rm3_expansion("hiring", pool, fb_terms=2).terms
        self.assertEqual(first, second)
        self.assertEqual([t for t, _ in first], ["alpha", "beta"])   # equal mass -> alphabetical

    def test_no_usable_feedback_is_a_status_not_an_exception(self):
        self.assertEqual(rm3_expansion("hiring", []).status, "no-feedback")
        self.assertEqual(
            rm3_expansion("hiring", [FakeCandidate("d1", 1.0, "")]).status, "no-feedback"
        )

    def test_negative_scores_do_not_produce_negative_document_weights(self):
        pool = [
            FakeCandidate("d1", -2.0, "policy documents nursing"),
            FakeCandidate("d2", -5.0, "policy documents nursing"),
        ]
        result = rm3_expansion("hiring", pool)
        self.assertEqual(result.status, "ok")
        self.assertTrue(all(w >= 0 for _, w in result.terms))


class FakeClient:
    """Stands in for PyseriniClient: same `stats` contract, scripted failures."""

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.stats = {"cache_hits": 0, "cache_misses": 0}
        self.calls = 0

    def search_adaptive(self, query, hits, min_hits=5):
        behaviour = self.behaviours[min(self.calls, len(self.behaviours) - 1)]
        self.calls += 1
        self.stats["cache_misses"] += 1
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour, hits


class TestRunPRF(unittest.TestCase):
    """The PRF loop talks to rate-limited shared infrastructure, so its contract is
    that it never raises and never keeps hammering a service that is refusing."""

    def _pool(self):
        return [FakeCandidate(f"d{i}", 5.0, "policy documents nursing recruitment") for i in range(3)]

    def test_a_retry_error_degrades_instead_of_killing_the_run(self):
        """urllib3 raises RetryError -- a bare RequestException, not a
        ConnectionError -- once a 429 ladder is exhausted, so it walks past the
        client's own transport handling and would otherwise discard every
        generation already paid for on the GPU."""
        import requests

        client = FakeClient([requests.exceptions.RetryError("too many 429 error responses")])
        results = step3.run_prf(client, ["q1", "q2"], hits=10, delay=0)
        self.assertEqual([r.status for r in results], ["api-error", "api-error"])

    def test_a_transient_failure_does_not_poison_later_queries(self):
        import requests

        client = FakeClient([requests.exceptions.ConnectionError("boom"), self._pool()])
        results = step3.run_prf(client, ["q1", "q2"], hits=10, delay=0)
        self.assertEqual([r.status for r in results], ["api-error", "ok"])

    def test_it_stops_asking_after_sustained_failure(self):
        import requests

        client = FakeClient([requests.exceptions.RetryError("429")])
        results = step3.run_prf(
            client, [f"q{i}" for i in range(20)], hits=10, delay=0, max_consecutive_failures=3
        )
        self.assertEqual(client.calls, 3)          # stopped asking, did not run to 20
        self.assertEqual(len(results), 20)         # every query still accounted for
        self.assertTrue(all(r.status == "api-error" for r in results))
        self.assertIn("consecutive failures", results[-1].detail)

    def test_an_auth_failure_stops_immediately(self):
        from trecrag.clients.pyserini_client import PyseriniAuthError

        client = FakeClient([PyseriniAuthError("bad token")])
        results = step3.run_prf(client, ["q1", "q2", "q3"], hits=10, delay=0)
        self.assertEqual(client.calls, 1)          # a rejected token will not fix itself
        self.assertEqual(len(results), 3)

    def test_successful_searches_produce_rm3_terms(self):
        client = FakeClient([self._pool()])
        results = step3.run_prf(client, ["hiring"], hits=10, delay=0)
        self.assertEqual(results[0].status, "ok")
        self.assertIn("recruitment", [t for t, _ in results[0].terms])


def _record(query_id="q1", ains=(("1", 2),), **extra):
    """ains: ((sub_question_id, n_ains), ...)"""
    record = {
        "query_id": query_id,
        "comp_query": f"comp {query_id}",
        "query_metadata": "hospital nursing council",
        "sub_questions": [
            {"sub_question_id": sq_id, "sub_question_text": f"{query_id}-q{sq_id}?"}
            for sq_id, _ in ains
        ],
        "dependencies": [],
        "ain": [
            {
                "sub_question_id": sq_id,
                "ain_array": [
                    {"ain_id": i, "ain_text": f"Evidence regarding topic{sq_id}{i} accountability."}
                    for i in range(1, n + 1)
                ],
            }
            for sq_id, n in ains
        ],
    }
    record.update(extra)
    return record


def _with_expansions(record):
    record = copy.deepcopy(record)
    record["expanded_queries"] = [
        {
            "sub_question_id": entry["sub_question_id"],
            "eq_array": [
                {
                    "eq_id": i,
                    "ain_id": item["ain_id"],
                    "step3a_conditioned_query": (
                        f"topic{entry['sub_question_id']}{item['ain_id']} accountability in nursing"
                    ),
                    "step3b_expansion_terms": ["a term", "another term"],
                    "step3b_expanded_query": "conditioned a term another term",
                    "step3c_prf_terms": [{"term": "recruitment", "weight": 0.25}],
                    "step3c_expanded_query": "conditioned a term another term recruitment",
                    "step3c_status": "ok",
                }
                for i, item in enumerate(entry["ain_array"], start=1)
            ],
        }
        for entry in record["ain"]
    ]
    return record


class TestFinalGate(unittest.TestCase):
    def test_accepts_a_valid_record(self):
        validate_final_record(_with_expansions(_record(ains=(("1", 2), ("2", 1)))))

    def test_upstream_fields_ride_through(self):
        validate_final_record(_with_expansions(_record(some_future_field=[1, 2])))

    def _assert_rejected(self, mutate, record=None):
        record = _with_expansions(record or _record())
        mutate(record)
        with self.assertRaises(QueryExpansionParseError):
            validate_final_record(record)

    def test_rejects_missing_keys(self):
        self._assert_rejected(lambda r: r.pop("expanded_queries"))

    def test_rejects_coverage_gaps(self):
        """One expanded query per AIN, or the needs missing from it are needs
        nothing will ever retrieve for."""
        self._assert_rejected(lambda r: r["expanded_queries"][0]["eq_array"].pop())
        self._assert_rejected(lambda r: r["expanded_queries"].pop())

    def test_rejects_a_broken_cross_reference(self):
        self._assert_rejected(lambda r: r["expanded_queries"][0]["eq_array"][0].update({"ain_id": 9}))
        self._assert_rejected(lambda r: r["expanded_queries"][0]["eq_array"][0].update({"eq_id": 7}))
        self._assert_rejected(
            lambda r: r["expanded_queries"][0].update({"sub_question_id": "9"})
        )

    def test_rejects_a_conditioned_query_that_lost_the_topic(self):
        """Re-derived here rather than trusted from the parser -- the same
        independence the earlier steps' gates have."""
        self._assert_rejected(
            lambda r: r["expanded_queries"][0]["eq_array"][0].update(
                {"step3a_conditioned_query": "entirely unrelated hospital budgeting"}
            )
        )

    def test_rejects_empty_queries_or_terms(self):
        for key in ("step3a_conditioned_query", "step3b_expanded_query", "step3c_expanded_query"):
            self._assert_rejected(
                lambda r, k=key: r["expanded_queries"][0]["eq_array"][0].update({k: "  "})
            )
        self._assert_rejected(
            lambda r: r["expanded_queries"][0]["eq_array"][0].update({"step3b_expansion_terms": []})
        )

    def test_rejects_malformed_prf_terms(self):
        eq = lambda r: r["expanded_queries"][0]["eq_array"][0]
        self._assert_rejected(lambda r: eq(r).update({"step3c_prf_terms": [{"term": "x"}]}))
        self._assert_rejected(
            lambda r: eq(r).update({"step3c_prf_terms": [{"term": "x", "weight": "0.2"}]})
        )
        self._assert_rejected(lambda r: eq(r).update({"step3c_status": "maybe"}))

    def test_rejects_prf_terms_under_a_non_ok_status(self):
        """A degraded PRF stage must be visibly degraded: terms and status cannot
        disagree about whether feedback happened."""
        self._assert_rejected(
            lambda r: r["expanded_queries"][0]["eq_array"][0].update({"step3c_status": "api-error"})
        )

    def test_gate_is_independent_of_the_parsers(self):
        with self.assertRaises(QueryExpansionParseError):
            validate_final_record({"query_id": "a"})


class TestBuildJobs(unittest.TestCase):
    def test_one_job_per_atomic_need_carrying_its_context(self):
        jobs = build_jobs([_record("a", ains=(("1", 2), ("2", 1)))])
        self.assertEqual(
            [(j.sub_question_id, j.ain_id) for j in jobs], [("1", 1), ("1", 2), ("2", 1)]
        )
        self.assertEqual(jobs[0].metadata, "hospital nursing council")
        self.assertEqual(jobs[2].sub_question_text, "a-q2?")

    def test_empty_metadata_is_labelled(self):
        jobs = build_jobs([_record("a", query_metadata="")])
        self.assertEqual(jobs[0].metadata, "(none given)")


class TestExpandQueries(unittest.TestCase):
    """End-to-end over the stage's own logic with generation and search replaced by
    scripted doubles -- no GPU, no API."""

    def setUp(self):
        self._real_inference = step3.run_inference
        self._real_prf = step3.run_prf
        self.addCleanup(setattr, step3, "run_inference", self._real_inference)
        self.addCleanup(setattr, step3, "run_prf", self._real_prf)

    def _scripted(self, conditioned_by_topic=None, expansions=None):
        """Replies keyed by the topic token that appears in the prompt."""
        conditioned_by_topic = conditioned_by_topic or {}

        def fake(llm, tokenizer, conversations, batch_size, max_tokens):
            outputs = []
            for conversation in conversations:
                body = conversation[-1]["content"]
                if "Information need:" in body:
                    topic = next(t for t in conditioned_by_topic if t in body)
                    outputs.append(conditioned_by_topic[topic])
                else:
                    outputs.append(expansions or EXP)
            return outputs

        step3.run_inference = fake

    def _no_prf(self):
        step3.run_prf = lambda client, queries, hits, *a, **kw: [
            PRFResult(terms=[("recruitment", 0.3)], status="ok") for _ in queries
        ]

    def test_assembles_one_expanded_query_per_ain_with_cross_references(self):
        self._scripted({"topic11": "<conditioned>topic11 accountability in nursing</conditioned>"})
        self._no_prf()
        good, flagged, stats = expand_queries(
            None, None, [_record("a", ains=(("1", 1),))], 0, prf_client=object(), prf_hits=10
        )
        self.assertEqual(flagged, [])
        eq = good[0]["expanded_queries"][0]["eq_array"][0]
        self.assertEqual((eq["eq_id"], eq["ain_id"]), (1, 1))
        self.assertEqual(eq["step3b_expansion_terms"], ["equitable hiring", "inclusive recruitment"])
        self.assertIn("equitable hiring", eq["step3b_expanded_query"])
        self.assertTrue(eq["step3c_expanded_query"].endswith("recruitment"))
        self.assertEqual(stats["needs"], 1)

    def test_upstream_fields_are_carried_through_with_the_new_field_last(self):
        self._scripted({"topic11": "<conditioned>topic11 accountability in nursing</conditioned>"})
        self._no_prf()
        source = _record("a", ains=(("1", 1),))
        good, _, _ = expand_queries(
            None, None, [copy.deepcopy(source)], 0, prf_client=object(), prf_hits=10
        )
        self.assertEqual(list(good[0])[-1], "expanded_queries")
        for key, value in source.items():
            self.assertEqual(good[0][key], value)

    def test_prf_disabled_still_produces_a_usable_query(self):
        self._scripted({"topic11": "<conditioned>topic11 accountability in nursing</conditioned>"})
        good, flagged, _ = expand_queries(
            None, None, [_record("a", ains=(("1", 1),))], 0, prf_client=None, prf_hits=10
        )
        eq = good[0]["expanded_queries"][0]["eq_array"][0]
        self.assertEqual(eq["step3c_status"], "disabled")
        self.assertEqual(eq["step3c_prf_terms"], [])
        self.assertEqual(eq["step3c_expanded_query"], eq["step3b_expanded_query"])
        self.assertEqual(flagged, [])

    def test_an_api_failure_degrades_one_query_and_never_flags_the_record(self):
        """Network weather must not decide what is in the output file."""
        self._scripted({"topic11": "<conditioned>topic11 accountability in nursing</conditioned>"})
        step3.run_prf = lambda client, queries, hits, *a, **kw: [
            PRFResult(terms=[], status="api-error", detail="boom") for _ in queries
        ]
        good, flagged, stats = expand_queries(
            None, None, [_record("a", ains=(("1", 1),))], 0, prf_client=object(), prf_hits=10
        )
        self.assertEqual(flagged, [])
        eq = good[0]["expanded_queries"][0]["eq_array"][0]
        self.assertEqual(eq["step3c_status"], "api-error")
        self.assertEqual(eq["step3c_expanded_query"], eq["step3b_expanded_query"])
        self.assertEqual(stats["prf_status"], {"api-error": 1})

    def test_a_3a_failure_flags_the_record_and_names_the_stage(self):
        self._scripted({"topic11": "I cannot do that."})
        self._no_prf()
        good, flagged, _ = expand_queries(
            None, None, [_record("a", ains=(("1", 1),))], 0, prf_client=object(), prf_hits=10
        )
        self.assertEqual(good, [])
        self.assertEqual(flagged[0]["failure_stage"], "3a-conditioning")
        self.assertIn("AIN 1", flagged[0]["failure_reason"])
        self.assertNotIn("expanded_queries", flagged[0])

    def test_a_3b_failure_flags_the_record_and_names_the_stage(self):
        self._scripted(
            {"topic11": "<conditioned>topic11 accountability in nursing</conditioned>"},
            expansions="no tags here",
        )
        self._no_prf()
        good, flagged, _ = expand_queries(
            None, None, [_record("a", ains=(("1", 1),))], 0, prf_client=object(), prf_hits=10
        )
        self.assertEqual(good, [])
        self.assertEqual(flagged[0]["failure_stage"], "3b-expansion")

    def test_one_failed_need_flags_the_whole_record(self):
        """A partial expanded_queries block looks complete."""
        self._scripted(
            {
                "topic11": "<conditioned>topic11 accountability in nursing</conditioned>",
                "topic12": "nope",
            }
        )
        self._no_prf()
        good, flagged, _ = expand_queries(
            None, None, [_record("a", ains=(("1", 2),))], 0, prf_client=object(), prf_hits=10
        )
        self.assertEqual(good, [])
        self.assertEqual(len(flagged), 1)

    def test_no_records_is_not_an_error(self):
        step3.run_inference = lambda *a, **kw: []
        good, flagged, stats = expand_queries(None, None, [], 0, prf_client=None, prf_hits=10)
        self.assertEqual((good, flagged, stats["needs"]), ([], [], 0))


if __name__ == "__main__":
    unittest.main()
