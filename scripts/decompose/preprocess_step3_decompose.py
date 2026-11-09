#!/usr/bin/env python3
"""Preprocessing step 3: turn each Atomic Information Need into a retrieval query.

Steps 1 and 2 take a composite narrative apart. That is what makes them useful and
also what breaks them: by the time a sub-question has been split into atomic needs,
"leadership accountability" no longer says whether it means corporate governance,
political leadership, or a hospital nursing council. The context that disambiguated
it is sitting in `query_metadata`, one field away and no longer attached to anything.

Step 3 puts a query back together in three passes, each one narrowing rather than
widening the need:

  3A  METADATA CONDITIONING (LLM). Re-attach the persistent context the asker
      supplied. "leadership accountability" becomes "leadership accountability for
      nursing diversity, equity and inclusion initiatives supporting nurse
      recruitment and retention within hospital systems". Same information need,
      much less ambiguity. Conditioning may only specialise: every concept in the
      output must already be present in the atomic need, its sub-question, or the
      metadata.

  3B  GUARDED LEXICAL EXPANSION (LLM). BM25 matches words, not meanings, so
      "equitable hiring" misses documents that say "inclusive recruitment" or
      "structured interviews". This pass adds meaning-preserving surface forms
      only -- paraphrase, synonym, acronym expansion, terminology normalisation --
      and is explicitly forbidden to add interventions, stakeholders, objectives,
      causal claims, or recommendations. Unconstrained expansion drifts, and drift
      is retrieval noise.

  3C  PSEUDO-RELEVANCE FEEDBACK (RM3, no LLM). The model expands from parametric
      knowledge; the corpus has its own vocabulary. RM3 reads it off the top
      documents the 3B query actually retrieves and folds those terms back in.

    preprocessed_queries_step2.jsonl      (ain arrays + everything upstream)
        -> 3A: one vLLM call per AIN  -> <conditioned> tag
        -> 3B: one vLLM call per AIN  -> <expansions>/<expansion id="N"> tags
        -> 3C: one BM25 search per AIN -> client-side RM3 relevance model
        -> preprocessed_queries_step3.jsonl

RM3 NOTE (read before trusting the numbers)
    Anserini/Pyserini implement RM3 inside the index: they build the relevance model
    from Lucene term statistics and re-issue a *weighted* query vector. ClimbMix is
    reachable only through the Pyserini REST API, whose search endpoint accepts a
    plain `query` string and a `hits` count -- there is no rm3 parameter and no way
    to submit term weights. So the relevance model is computed here, client-side,
    from the returned document text (Lavrenko & Croft RM1, interpolated with the
    original query at `PRF_ALPHA`), and the weighted query is realised as term
    append, which is the closest a text-only endpoint allows. The per-term RM3
    weights are written to the output so a future stage with a weighted-query
    retriever can use them properly rather than re-deriving them.

    Feedback is taken from query-focused windows rather than whole documents:
    ClimbMix documents are whole web pages and books (mean 66k chars), and a
    relevance model built over all of that models the book, not the answer.

Every step-2 field is carried through unchanged, with one new `expanded_queries`
field appended. LLM failures that survive both attempts flag the whole record;
`preprocessed_queries_step3_flagged.jsonl` and `_errors.log` are the audit trail, so
every input record appears in exactly one of the two outputs.

QUICKSTART
    conda activate trec-rag-2026-vllm
    python scripts/decompose/preprocess_step3_decompose.py

    # 3A and 3B only -- no Pyserini API traffic at all:
    python scripts/decompose/preprocess_step3_decompose.py --no-prf

    # PRF needs PYSERINI_API_TOKEN in .env.local. Searches are cached by
    # (index, query, hits) in the shared cache from configs/pipeline.gpu.yaml, so a
    # re-run costs no API calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_HERE = Path(__file__).resolve().parent

# vLLM loading rules (auto-sizing to free VRAM, quantization detection, chat-template
# fallback) are step 1's, kept in one place so a fix there is a fix here.
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "src"))

from preprocess_step1_decompose import (  # noqa: E402
    _render,
    resolve_gpu_memory_utilization,
    resolve_quantization,
)
from trecrag.chunking import select_passage  # noqa: E402
from trecrag.config import get_env, load_dotenv_local, load_pipeline_config  # noqa: E402
from trecrag.text_utils import tokenize  # noqa: E402

# --------------------------------------------------------------------------------
# 1. CONFIGURATION -- edit these, or override any of them on the command line.
# --------------------------------------------------------------------------------

INPUT_PATH = "output_decomp/preprocessed_queries_step2.jsonl"

LLM_PATH = (
    "/media/20TB/shared/models/qwen/Qwen2.5-72B-Instruct-AWQ/"
    "models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/"
    "698703eae6604af048a3d2f509995dc302088217"
)

OUTPUT_PATH = "output_decomp/preprocessed_queries_step3.jsonl"
FLAGGED_PATH = "output_decomp/preprocessed_queries_step3_flagged.jsonl"
ERROR_LOG_PATH = "output_decomp/preprocessed_queries_step3_errors.log"

# Generation / runtime knobs.
QUANTIZATION: str | None = None
TENSOR_PARALLEL_SIZE = 2
GPU_MEMORY_UTILIZATION = "auto"
MAX_MODEL_LEN = 8192
CONDITION_MAX_TOKENS = 256        # 3A emits one line
EXPANSION_MAX_TOKENS = 512        # 3B emits a short list of phrases
TEMPERATURE = 0.0                 # deterministic, and reproducible next year
TOP_P = 1.0
SEED = 0
BATCH_SIZE = 0                    # 0 = submit every prompt in one generate() call

# 3A guards. Conditioning must specialise, not expand: a conditioned query that has
# grown past a couple of clauses has started adding scope rather than context.
MAX_CONDITIONED_TOKENS = 60
# 3B guards. The worked example runs to seven variants; the cap catches a list that
# has stopped paraphrasing and started brainstorming.
MAX_EXPANSION_TERMS = 12

# --- 3C: RM3 pseudo-relevance feedback -------------------------------------------
PRF_ENABLED = True
PRF_CONFIG_PATH = "configs/pipeline.gpu.yaml"   # supplies base_url/index/token/cache
PRF_HITS = 10                     # search depth; also the feedback pool ceiling
PRF_FB_DOCS = 10                  # Anserini's default fbDocs
PRF_FB_TERMS = 10                 # Anserini's default fbTerms
PRF_ALPHA = 0.5                   # Anserini's default originalQueryWeight
PRF_MIN_DOC_FREQ = 2              # a term must appear in >= 2 feedback documents,
                                  # so one outlier document cannot supply the whole
                                  # expansion
PRF_MIN_TERM_LEN = 3
PRF_USE_PASSAGE_WINDOWS = True    # feed RM1 the query-relevant span, not the book
PRF_WINDOW_CHARS = 2000
PRF_STRIDE_CHARS = 1000
PRF_WINDOWS_PER_DOC = 2

# This stage issues one search per atomic information need -- roughly ten times what
# a full pipeline run asks of the API, against rate-limited shared track
# infrastructure. MEASURED: an unpaced sequential loop at ~1 request/second drew HTTP
# 429s within the first thirty queries, after which every request spent the retry
# ladder (2s, 4s, 8s, ...) before failing. A search itself takes ~1.4s at hits=10, so
# pacing costs little and is the difference between cooperating with the service and
# being throttled by it. Only network calls are paced; a cache hit sleeps for nothing.
PRF_REQUEST_DELAY_SECONDS = 1.5
# If the service is refusing sustained traffic, continuing to ask 1200 more times is
# neither polite nor productive. Stop issuing requests and mark the remainder.
PRF_MAX_CONSECUTIVE_FAILURES = 5

# Stop set for RM1 term SELECTION, which is a harder filter than the one an index
# needs. Lucene's default English set is 33 words, and removing exactly it leaves RM1
# ranking terms by raw probability mass -- which function words win. MEASURED on four
# real queries with the 33-word set: "can" was the top-weighted feedback term for two
# of them, and `can/also/more/from/when/against` took 3-4 of the 10 term slots in
# every list. Those terms are near-harmless in BM25 (their IDF is tiny) but each one
# spends a slot that a corpus term should have had, which is the entire point of this
# stage.
#
# So: the standard NLTK English list (>=3 characters; shorter tokens never survive
# PRF_MIN_TERM_LEN anyway), plus a supplement of high-frequency discourse words that
# the standard list omits and that this corpus produces constantly. The supplement is
# a judgement call and is kept visibly separate for that reason -- every word in it is
# a connective or hedge, never a topic.
_NLTK_STOPWORDS = """
    about above after again against all and any are aren been before being below
    between both but can did didn does doesn doing don down during each few for from
    further had hadn has hasn have haven having her hers herself him himself his how
    into isn its itself just ma mightn more most mustn myself needn nor not now off
    once only other our ours ourselves out over own same shan she should shouldn some
    such than that the their theirs them themselves then there these they this those
    through too under until very was wasn were weren what when where which while who
    whom why will with won wouldn you your yours yourself yourselves
"""
_DISCOURSE_STOPWORDS = """
    also among another any because been could even every first however including
    like made make many may might much must often one two three several since still
    therefore thus toward towards upon use used using via well within without would
"""
_STOPWORDS = frozenset(_NLTK_STOPWORDS.split()) | frozenset(_DISCOURSE_STOPWORDS.split())

logger = logging.getLogger("preprocess_step3")


# --------------------------------------------------------------------------------
# 2. PROMPTS -- isolated so they can be read and iterated on independently.
# --------------------------------------------------------------------------------

# Both prompts are written against the same failure mode from opposite sides. 3A can
# fail by drifting (adding scope while "adding context"); 3B can fail by drifting
# (adding concepts while "adding vocabulary"). Neither failure is visible downstream
# -- a drifted query still retrieves, just the wrong thing -- so both prompts spend
# most of their length on what must NOT change.

CONDITION_SYSTEM_PROMPT = """\
You rewrite a single search topic so that it carries the context it was separated \
from. You never answer anything, and you never change what is being asked for.

You are given three things: the asker's BACKGROUND CONTEXT, the SUB-QUESTION the \
topic was taken from, and the INFORMATION NEED itself. The information need was \
split out of a longer question, and that split stripped away the setting that made \
it unambiguous. "Leadership accountability" on its own could be corporate \
governance, political leadership, or school administration. Your job is to put the \
setting back.

CONDITIONING SPECIALISES; IT NEVER EXPANDS. The conditioned query must ask for \
exactly what the information need asked for, in a narrower and less ambiguous form:

- Every concept you add must already appear in the background context, the \
sub-question, or the information need itself. Add nothing else - no new topics, no \
new stakeholders, no new interventions, no objectives, no recommendations, no \
assumptions about causes or outcomes.
- Do not broaden. Never turn one topic into several, never add "and ..." clauses \
that bring in a second subject, and never restate the whole sub-question.
- Keep the original topic words. The conditioned query must still contain the words \
that name the topic, so it can be recognised as the same need.
- Add only the context that actually disambiguates: the domain, the setting, the \
population, the purpose the asker stated. If the background context adds nothing \
useful for this particular topic, leave the topic almost as it is. Padding a query \
with context it does not need makes it worse, not better.
- Write it as a search query: a noun phrase, lower case, no trailing full stop, no \
question mark. Do NOT begin with "Evidence regarding" - write the topic itself.
- One line. Roughly 5 to 30 words.

Output format - emit ONLY this tag, nothing else. No preamble, no trailing \
commentary, no markdown code fences, no explanation:

<conditioned>the conditioned query</conditioned>

Example A - the background context resolves a genuinely ambiguous topic:

Background context: I'm on a hospital nursing DEI council that has to recommend a \
three-year plan with limited money. Our chief nursing officer wants something that \
helps recruitment and retention, not just a one-time training.
Sub-question: What should we prioritize across nursing school pipeline partnerships, \
hiring and promotion practices, reporting systems for racism, leadership \
accountability, and curriculum/continuing education?
Information need: Evidence regarding leadership accountability for nursing DEI.

<conditioned>leadership accountability for nursing diversity, equity and inclusion \
initiatives supporting nurse recruitment and retention within hospital \
systems</conditioned>

Example B - no background context, and a topic that is already unambiguous. Nothing \
is invented to fill the gap; the topic is simply written as a query:

Background context: (none given)
Sub-question: What causes the northern lights?
Information need: Evidence regarding the physical causes of the aurora borealis.

<conditioned>physical causes of the aurora borealis</conditioned>
"""

CONDITION_USER_TEMPLATE = """\
Background context: {metadata}
Sub-question: {sub_question}
Information need: {ain}"""

CONDITION_FALLBACK_PROMPT = """\
Rewrite the information need below as one search query, with the setting from the \
background context folded in. Output the tag and NOTHING else.

Required shape, exactly:

<conditioned>the conditioned query</conditioned>

Hard rules:
- Start your reply with the character < and end it with the character >.
- No code fences. No "Here is". No notes before or after the tag.
- Exactly one <conditioned> tag, on one line.
- Keep the topic words from the information need. Do not drop them or swap them out.
- Add ONLY context that already appears in the background context, the sub-question, \
or the information need. Invent nothing.
- Do not add a second topic, a recommendation, or an assumption.
- Do not start with "Evidence regarding". Write a lower-case noun phrase with no \
full stop and no question mark, roughly 5 to 30 words.

Background context: {metadata}
Sub-question: {sub_question}
Information need: {ain}
"""

EXPANSION_SYSTEM_PROMPT = """\
You widen the VOCABULARY of a search query without changing what it asks for. You \
never answer anything.

A keyword search engine matches words, not meanings. A query about "equitable \
hiring" will miss a document that says "inclusive recruitment", "fair hiring", or \
"structured interviews", however relevant it is. You produce the alternative wordings \
that close that gap.

You are allowed exactly these transformations, and nothing else:

- lexical paraphrase of the same idea;
- synonyms and near-synonyms of the terms already present;
- acronym expansion, and the acronym for an expanded term;
- terminology normalisation (the standard name for the thing described);
- domain-specific terminology for the same concept;
- contextual clarification using words from the background context.

You must NOT:

- introduce a new intervention, programme, method, or tool;
- invent a stakeholder, population, or organisation that was not mentioned;
- add an objective, outcome, or goal that was not asked for;
- assume a cause, effect, or mechanism;
- supply a recommendation or a policy position;
- speculate beyond the query and the background context in any other way.

The test for every variant: would a document matching this phrase be answering the \
SAME question? If it would answer a related but different question, drop it. When in \
doubt, leave it out - a missing paraphrase costs a little recall, while a drifted one \
costs precision on every document it pulls in.

Produce between 3 and 8 variants. Each is a short phrase, not a sentence and not a \
question, lower case, no trailing full stop. Do not repeat a variant, and do not \
merely reorder the words of another one. Including the query's own wording as one \
variant is fine and usually right.

Output format - emit ONLY these tags, nothing else. No preamble, no trailing \
commentary, no markdown code fences, no explanation:

<expansions>
<expansion id="1">first variant</expansion>
<expansion id="2">second variant</expansion>
</expansions>

The <expansions> wrapper is always present and always holds at least one \
<expansion>. ids start at 1 and increase by 1 with no gaps.

Example A - synonyms, domain terminology, and the query's own wording, all pointing \
at the same need:

Background context: I'm on a hospital nursing DEI council that has to recommend a \
three-year plan with limited money.
Query: equitable hiring practices for nurses in hospital systems

<expansions>
<expansion id="1">equitable hiring</expansion>
<expansion id="2">inclusive recruitment</expansion>
<expansion id="3">fair hiring</expansion>
<expansion id="4">bias-aware recruitment</expansion>
<expansion id="5">structured hiring</expansion>
<expansion id="6">standardized interview practices</expansion>
<expansion id="7">nursing workforce recruitment</expansion>
</expansions>

Example B - an acronym expanded and normalised. Note what is NOT here: no new \
programme types, no claims about what works, no populations the query did not name:

Background context: Our city council is deciding what to fund to reduce shootings.
Query: CVI programs for reducing community gun violence

<expansions>
<expansion id="1">community violence intervention</expansion>
<expansion id="2">CVI programs</expansion>
<expansion id="3">violence interruption programs</expansion>
<expansion id="4">street outreach violence prevention</expansion>
<expansion id="5">community-based gun violence reduction</expansion>
</expansions>
"""

EXPANSION_USER_TEMPLATE = """\
Background context: {metadata}
Query: {query}"""

EXPANSION_FALLBACK_PROMPT = """\
List alternative wordings for the query below. Output the tags and NOTHING else.

Required shape, exactly:

<expansions>
<expansion id="1">first wording</expansion>
<expansion id="2">second wording</expansion>
</expansions>

Hard rules:
- Start your reply with the character < and end it with the character >.
- No code fences. No "Here is". No notes before or after the tags.
- Exactly one <expansions> wrapper, holding 3 to 8 <expansion> tags. ids are 1, 2, \
3, ... in order with no gaps.
- Each tag is a short lower-case phrase: no sentence, no question, no full stop.
- Every wording must mean the SAME thing as the query. Synonyms, paraphrases, \
acronym expansions, and standard terminology only.
- Add no new programme, method, group, goal, cause, or recommendation. If a wording \
would answer a different question, leave it out.
- No duplicates, and no variant that is just another one with the words reordered.

Background context: {metadata}
Query: {query}
"""


# --------------------------------------------------------------------------------
# 3. INPUT
# --------------------------------------------------------------------------------


class QueryExpansionParseError(Exception):
    """Raised with a human-readable reason naming the validation rule that failed."""


def load_step2_records(path: str | Path) -> list[dict[str, Any]]:
    """Read step 2's JSONL output.

    Machine-generated input: a malformed line is an upstream bug, but it must not
    take a GPU run down halfway. Bad lines are warned about and skipped, so the
    effect is a visibly smaller record count rather than a crash. Only the fields
    step 3 consumes are checked; everything else rides through untouched.
    """
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d: skipping unparseable JSON (%s)", path, line_no, exc)
                continue
            if not isinstance(record, dict) or not _has_step2_shape(record):
                logger.warning("%s:%d: skipping record with unexpected shape", path, line_no)
                continue
            if record["query_id"] in seen:
                logger.warning(
                    "%s:%d: duplicate query_id %r, skipping", path, line_no, record["query_id"]
                )
                continue
            seen.add(record["query_id"])
            records.append(record)
    return records


def _has_step2_shape(record: dict[str, Any]) -> bool:
    if not isinstance(record.get("query_id"), str) or not record["query_id"]:
        return False
    if not isinstance(record.get("query_metadata"), str):
        return False
    sub_questions = record.get("sub_questions")
    ain = record.get("ain")
    if not isinstance(sub_questions, list) or not sub_questions:
        return False
    if not isinstance(ain, list) or len(ain) != len(sub_questions):
        return False
    for entry, sub_question in zip(ain, sub_questions):
        if not isinstance(entry, dict) or not isinstance(entry.get("ain_array"), list):
            return False
        if not entry["ain_array"]:
            return False
        if entry.get("sub_question_id") != sub_question.get("sub_question_id"):
            return False
        for item in entry["ain_array"]:
            if not isinstance(item, dict):
                return False
            if not isinstance(item.get("ain_id"), int) or isinstance(item.get("ain_id"), bool):
                return False
            if not isinstance(item.get("ain_text"), str) or not item["ain_text"].strip():
                return False
    return True


# --------------------------------------------------------------------------------
# 4. PARSING -- deterministic, all-or-nothing.
# --------------------------------------------------------------------------------

_CONDITIONED_RE = re.compile(r"<conditioned\s*>(.*?)</conditioned\s*>", re.DOTALL | re.IGNORECASE)
_EXPANSIONS_BLOCK_RE = re.compile(
    r"<expansions\s*>(.*?)</expansions\s*>", re.DOTALL | re.IGNORECASE
)
# `<expansion` requires whitespace and an id= before the bracket, so the
# <expansions> wrapper itself can never match this.
_EXPANSION_RE = re.compile(
    r"""<expansion\s+id\s*=\s*["'](\d+)["']\s*>(.*?)</expansion\s*>""",
    re.DOTALL | re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
# Step 2 writes every AIN as "Evidence regarding X"; that prefix is scaffolding, not
# topic, and 3A is told to drop it. Strip it before comparing topic words.
_AIN_PREFIX_RE = re.compile(r"^\s*evidence\s+regarding\s+", re.IGNORECASE)


def topic_tokens(text: str) -> set[str]:
    """Content words of a topic: what has to survive conditioning."""
    return {
        token
        for token in tokenize(_AIN_PREFIX_RE.sub("", text))
        if token not in _STOPWORDS and len(token) >= PRF_MIN_TERM_LEN
    }


def _strip_outside(text: str, start: int, end: int) -> str:
    return (text[:start] + text[end:]).strip()


def parse_conditioned(raw_output: str, ain_text: str) -> str:
    """Parse a <conditioned> block into the 3A query string.

    All-or-nothing, like every parser in this pipeline. The rule that matters most
    is rule 6: a conditioned query that shares no content word with the atomic need
    it came from is not a conditioned query, it is a different one -- and nothing
    downstream could tell, because it would retrieve perfectly well for the wrong
    topic.
    """
    if raw_output is None:
        raise QueryExpansionParseError("empty generation")
    text = _FENCE_RE.sub("", raw_output.strip())
    if not text:
        raise QueryExpansionParseError("empty generation")

    # Rule 1: exactly one <conditioned> block.
    matches = list(_CONDITIONED_RE.finditer(text))
    if len(matches) != 1:
        raise QueryExpansionParseError(
            f"rule1: expected exactly 1 <conditioned> block, found {len(matches)}"
        )
    match = matches[0]

    # Rule 2: nothing but whitespace outside it.
    stray = _strip_outside(text, *match.span())
    if stray:
        raise QueryExpansionParseError(f"rule2: stray text outside <conditioned>: {stray[:80]!r}")

    query = " ".join(match.group(1).split())

    # Rule 3: non-empty.
    if not query:
        raise QueryExpansionParseError("rule3: <conditioned> block was empty")

    # Rule 4: no raw '<', which would mean a truncated or unclosed tag.
    if "<" in query:
        raise QueryExpansionParseError(f"rule4: '<' inside tag content: {query[:80]!r}")

    # Rule 5: conditioning specialises. Past this length it has started adding scope.
    words = query.split()
    if len(words) > MAX_CONDITIONED_TOKENS:
        raise QueryExpansionParseError(
            f"rule5: conditioned query is {len(words)} words, over "
            f"MAX_CONDITIONED_TOKENS={MAX_CONDITIONED_TOKENS}"
        )

    # Rule 6: the topic survived.
    wanted = topic_tokens(ain_text)
    if wanted and not (wanted & set(tokenize(query))):
        raise QueryExpansionParseError(
            f"rule6: conditioned query shares no topic word with the information need "
            f"({sorted(wanted)[:6]}): {query[:80]!r}"
        )
    return query


def parse_expansions(raw_output: str) -> list[str]:
    """Parse an <expansions> block into the 3B variant list."""
    if raw_output is None:
        raise QueryExpansionParseError("empty generation")
    text = _FENCE_RE.sub("", raw_output.strip())
    if not text:
        raise QueryExpansionParseError("empty generation")

    # Rule 1: exactly one <expansions> wrapper.
    blocks = list(_EXPANSIONS_BLOCK_RE.finditer(text))
    if len(blocks) != 1:
        raise QueryExpansionParseError(
            f"rule1: expected exactly 1 <expansions> block, found {len(blocks)}"
        )
    block = blocks[0]

    # Rule 2a: nothing but whitespace outside the wrapper.
    stray = _strip_outside(text, *block.span())
    if stray:
        raise QueryExpansionParseError(f"rule2: stray text outside <expansions>: {stray[:80]!r}")

    inner = block.group(1)
    matches = list(_EXPANSION_RE.finditer(inner))

    # Rule 3: at least one <expansion>.
    if not matches:
        raise QueryExpansionParseError("rule3: no <expansion> blocks found")

    # Rule 4: ids are exactly 1..N in order.
    ids = [m.group(1) for m in matches]
    if ids != [str(i) for i in range(1, len(ids) + 1)]:
        raise QueryExpansionParseError(f"rule4: expansion ids are {ids}, expected 1..N in order")

    # Rule 2b: inside the wrapper, only <expansion> tags and whitespace. Without
    # this a prose list between tags would parse as a valid shorter answer.
    remainder, cursor = [], 0
    for match in matches:
        remainder.append(inner[cursor : match.start()])
        cursor = match.end()
    remainder.append(inner[cursor:])
    stray = "".join(remainder).strip()
    if stray:
        raise QueryExpansionParseError(f"rule2: stray text inside <expansions>: {stray[:80]!r}")

    terms = [" ".join(m.group(2).split()) for m in matches]

    # Rule 3 (cont.): an empty variant is a query term that matches everything.
    if any(not term for term in terms):
        raise QueryExpansionParseError("rule3: an <expansion> block was empty")

    # Rule 5: no raw '<' inside content.
    for term in terms:
        if "<" in term:
            raise QueryExpansionParseError(f"rule5: '<' inside tag content: {term[:80]!r}")

    # Rule 6: no duplicates. A repeated variant adds term frequency, not coverage.
    lowered = [term.casefold() for term in terms]
    if len(set(lowered)) != len(lowered):
        duplicate = next(t for t in lowered if lowered.count(t) > 1)
        raise QueryExpansionParseError(f"rule6: duplicate <expansion> text: {duplicate[:80]!r}")

    # Rule 7: a list this long has stopped paraphrasing and started brainstorming.
    if len(terms) > MAX_EXPANSION_TERMS:
        raise QueryExpansionParseError(
            f"rule7: {len(terms)} <expansion> blocks exceeds "
            f"MAX_EXPANSION_TERMS={MAX_EXPANSION_TERMS}"
        )
    return terms


# --------------------------------------------------------------------------------
# 4b. FINAL PRE-WRITE GATE -- independent of the parsers above.
# --------------------------------------------------------------------------------

_REQUIRED_KEYS = ("query_id", "comp_query", "query_metadata", "sub_questions", "ain",
                  "expanded_queries")
_EQ_KEYS = {
    "eq_id",
    "ain_id",
    "step3a_conditioned_query",
    "step3b_expansion_terms",
    "step3b_expanded_query",
    "step3c_prf_terms",
    "step3c_expanded_query",
    "step3c_status",
}
PRF_STATUSES = ("ok", "no-feedback", "api-error", "disabled")


def validate_final_record(record: Any) -> None:
    """Last line of defence before serialization.

    Re-derives every invariant rather than trusting the parsers, so a future edit to
    one of them still cannot put a malformed record in the output. Extra keys are
    allowed on purpose: `dependencies`, `ain`, and whatever an earlier step grows
    next must ride through. What is checked hard is the alignment -- one expanded
    query per AIN, cross-referenced by (sub_question_id, ain_id) to an AIN that
    actually exists, with the topic word still in it.
    """
    if not isinstance(record, dict):
        raise QueryExpansionParseError("final-gate: record is not a dict")
    missing = [key for key in _REQUIRED_KEYS if key not in record]
    if missing:
        raise QueryExpansionParseError(f"final-gate: missing key(s) {missing}")

    ain = record["ain"]
    expanded = record["expanded_queries"]
    if not isinstance(ain, list) or not isinstance(expanded, list):
        raise QueryExpansionParseError("final-gate: ain and expanded_queries must be lists")
    if len(expanded) != len(ain):
        raise QueryExpansionParseError(
            f"final-gate: expanded_queries has {len(expanded)} entries for {len(ain)} ain entries"
        )

    for i, (entry, ain_entry) in enumerate(zip(expanded, ain)):
        if not isinstance(entry, dict) or set(entry) != {"sub_question_id", "eq_array"}:
            raise QueryExpansionParseError(f"final-gate: expanded_queries[{i}] keys are wrong")
        if entry["sub_question_id"] != ain_entry.get("sub_question_id"):
            raise QueryExpansionParseError(
                f"final-gate: expanded_queries[{i}] is for sub_question "
                f"{entry['sub_question_id']!r}, expected {ain_entry.get('sub_question_id')!r}"
            )

        ain_by_id = {item["ain_id"]: item["ain_text"] for item in ain_entry["ain_array"]}
        eq_array = entry["eq_array"]
        if not isinstance(eq_array, list) or len(eq_array) != len(ain_entry["ain_array"]):
            raise QueryExpansionParseError(
                f"final-gate: expanded_queries[{i}].eq_array does not cover every AIN"
            )

        for j, eq in enumerate(eq_array, start=1):
            where = f"expanded_queries[{i}].eq_array[{j-1}]"
            if not isinstance(eq, dict) or set(eq) != _EQ_KEYS:
                raise QueryExpansionParseError(f"final-gate: {where} keys are wrong")
            # eq_id is this stage's own handle; ain_id is the cross-reference back
            # into `ain`. They run 1:1 today, and are kept separate so a future
            # stage may emit more than one query per AIN without a schema change.
            if eq["eq_id"] != j:
                raise QueryExpansionParseError(f"final-gate: {where} eq_id is {eq['eq_id']!r}")
            if eq["ain_id"] not in ain_by_id:
                raise QueryExpansionParseError(
                    f"final-gate: {where} ain_id {eq['ain_id']!r} is not in this sub-question"
                )

            for key in ("step3a_conditioned_query", "step3b_expanded_query",
                        "step3c_expanded_query"):
                if not isinstance(eq[key], str) or not eq[key].strip():
                    raise QueryExpansionParseError(f"final-gate: {where} {key} is empty")

            terms = eq["step3b_expansion_terms"]
            if not isinstance(terms, list) or not terms:
                raise QueryExpansionParseError(f"final-gate: {where} has no expansion terms")
            if any(not isinstance(t, str) or not t.strip() for t in terms):
                raise QueryExpansionParseError(f"final-gate: {where} has an empty expansion term")
            if len({t.casefold() for t in terms}) != len(terms):
                raise QueryExpansionParseError(f"final-gate: {where} has duplicate expansion terms")

            # The topic must have survived conditioning, re-derived here rather than
            # trusted from parse_conditioned.
            wanted = topic_tokens(ain_by_id[eq["ain_id"]])
            if wanted and not (wanted & set(tokenize(eq["step3a_conditioned_query"]))):
                raise QueryExpansionParseError(
                    f"final-gate: {where} conditioned query lost the topic entirely"
                )

            if eq["step3c_status"] not in PRF_STATUSES:
                raise QueryExpansionParseError(
                    f"final-gate: {where} step3c_status {eq['step3c_status']!r} is unknown"
                )
            prf_terms = eq["step3c_prf_terms"]
            if not isinstance(prf_terms, list):
                raise QueryExpansionParseError(f"final-gate: {where} step3c_prf_terms is not a list")
            if eq["step3c_status"] != "ok" and prf_terms:
                raise QueryExpansionParseError(
                    f"final-gate: {where} has PRF terms but status {eq['step3c_status']!r}"
                )
            seen_terms: set[str] = set()
            for term in prf_terms:
                if not isinstance(term, dict) or set(term) != {"term", "weight"}:
                    raise QueryExpansionParseError(f"final-gate: {where} PRF term keys are wrong")
                if not isinstance(term["term"], str) or not term["term"].strip():
                    raise QueryExpansionParseError(f"final-gate: {where} has an empty PRF term")
                if not isinstance(term["weight"], (int, float)) or isinstance(term["weight"], bool):
                    raise QueryExpansionParseError(f"final-gate: {where} PRF weight is not a number")
                if term["term"] in seen_terms:
                    raise QueryExpansionParseError(f"final-gate: {where} repeats a PRF term")
                seen_terms.add(term["term"])


# --------------------------------------------------------------------------------
# 5. MODEL / INFERENCE
# --------------------------------------------------------------------------------


@dataclass
class ExpansionJob:
    """One AIN's journey through 3A and 3B.

    One job per atomic need, carried through both LLM passes, so the record can be
    reassembled by (query_id, sub_question_id, ain_id) without any positional
    bookkeeping.
    """

    query_id: str
    sub_question_id: str
    ain_id: int
    ain_text: str
    metadata: str
    sub_question_text: str

    conditioned: str = ""
    expansion_terms: list[str] = field(default_factory=list)

    def label(self) -> str:
        return f"sub-question {self.sub_question_id} / AIN {self.ain_id}"


def _context(metadata: str) -> str:
    """Metadata is legitimately empty for a bare question; say so rather than leaving
    a dangling 'Background context:' line for the model to interpret."""
    return metadata.strip() or "(none given)"


def build_jobs(records: Iterable[dict[str, Any]]) -> list[ExpansionJob]:
    jobs: list[ExpansionJob] = []
    for record in records:
        text_by_sub_question = {
            sq["sub_question_id"]: sq["sub_question_text"] for sq in record["sub_questions"]
        }
        for entry in record["ain"]:
            for item in entry["ain_array"]:
                jobs.append(
                    ExpansionJob(
                        query_id=record["query_id"],
                        sub_question_id=entry["sub_question_id"],
                        ain_id=item["ain_id"],
                        ain_text=item["ain_text"],
                        metadata=_context(record["query_metadata"]),
                        sub_question_text=text_by_sub_question.get(
                            entry["sub_question_id"], ""
                        ).strip(),
                    )
                )
    return jobs


def condition_prompts(job: ExpansionJob) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    fields = {
        "metadata": job.metadata,
        "sub_question": job.sub_question_text,
        "ain": job.ain_text,
    }
    return (
        [
            {"role": "system", "content": CONDITION_SYSTEM_PROMPT},
            {"role": "user", "content": CONDITION_USER_TEMPLATE.format(**fields)},
        ],
        [{"role": "user", "content": CONDITION_FALLBACK_PROMPT.format(**fields)}],
    )


def expansion_prompts(job: ExpansionJob) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    fields = {"metadata": job.metadata, "query": job.conditioned}
    return (
        [
            {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
            {"role": "user", "content": EXPANSION_USER_TEMPLATE.format(**fields)},
        ],
        [{"role": "user", "content": EXPANSION_FALLBACK_PROMPT.format(**fields)}],
    )


def run_inference(
    llm,
    tokenizer,
    conversations: list[list[dict[str, str]]],
    batch_size: int,
    max_tokens: int,
) -> list[str]:
    """Batch every prompt through vLLM's offline API.

    One generate() call over the whole list is the point of using vLLM here: the
    scheduler batches continuously, so per-prompt calls would waste most of the GPU.
    """
    from vllm import SamplingParams

    if not conversations:
        return []
    sampling = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=max_tokens, seed=SEED
    )
    prompts = [_render(tokenizer, c) for c in conversations]
    chunk = batch_size if batch_size and batch_size > 0 else len(prompts)

    outputs: list[str] = []
    for start in range(0, len(prompts), chunk):
        batch = prompts[start : start + chunk]
        logger.info("generating %d/%d prompts", start + len(batch), len(prompts))
        for result in llm.generate(batch, sampling):
            outputs.append(result.outputs[0].text if result.outputs else "")
    return outputs


def run_llm_pass(
    llm,
    tokenizer,
    jobs: list[ExpansionJob],
    indices: list[int],
    build_prompts,
    parse,
    max_tokens: int,
    batch_size: int,
    stage: str,
) -> tuple[dict[int, Any], dict[int, str], dict[int, str], dict[int, str], int]:
    """Two attempts over one stage, shared by 3A and 3B.

    Both stages have the identical shape -- batch, parse, retry the failures with a
    more mechanical prompt, give up -- so they share one implementation and differ
    only in their prompts and parser. Returns (parsed, reasons, raw_primary,
    raw_fallback, recovered).
    """
    parsed: dict[int, Any] = {}
    reasons: dict[int, str] = {}
    raw_primary: dict[int, str] = {}
    raw_fallback: dict[int, str] = {}

    prompts = [build_prompts(jobs[i]) for i in indices]
    outputs = run_inference(llm, tokenizer, [p[0] for p in prompts], batch_size, max_tokens)
    retry: list[int] = []
    for position, (i, raw) in enumerate(zip(indices, outputs)):
        raw_primary[i] = raw
        try:
            parsed[i] = parse(raw, jobs[i])
        except QueryExpansionParseError as exc:
            reasons[i] = f"primary: {exc}"
            retry.append(position)
    logger.info("%s pass 1: %d parsed, %d need retry", stage, len(parsed), len(retry))

    recovered = 0
    if retry:
        outputs = run_inference(
            llm, tokenizer, [prompts[p][1] for p in retry], batch_size, max_tokens
        )
        for position, raw in zip(retry, outputs):
            i = indices[position]
            raw_fallback[i] = raw
            try:
                parsed[i] = parse(raw, jobs[i])
                recovered += 1
                # Name the primary failure here: recoveries never reach the flagged
                # file, and the rule that tripped is the signal for prompt work.
                logger.info(
                    "%s: %s (%s) recovered via fallback prompt (%s)",
                    stage, jobs[i].query_id, jobs[i].label(), reasons[i],
                )
            except QueryExpansionParseError as exc:
                reasons[i] = f"{reasons[i]} | fallback: {exc}"
    return parsed, reasons, raw_primary, raw_fallback, recovered


# --------------------------------------------------------------------------------
# 6. STEP 3C -- RM3 pseudo-relevance feedback, computed client-side.
# --------------------------------------------------------------------------------


@dataclass
class PRFResult:
    terms: list[tuple[str, float]]
    status: str
    feedback_docs: int = 0
    detail: str = ""


def _feedback_text(candidate, query_tokens: set[str]) -> str:
    """The span of a feedback document that the query is actually about.

    ClimbMix documents are whole web pages and books (mean 66k chars). A relevance
    model estimated over all of that is a model of the document, not of the answer,
    and the terms it promotes are the ones the book happens to use most. Selecting
    the query-relevant windows first is what keeps RM1 in the regime it was designed
    for -- passage-length feedback -- and the repo already measures that the leading
    window is almost never the relevant one.
    """
    text = candidate.text or ""
    if not text or not PRF_USE_PASSAGE_WINDOWS:
        return text
    return select_passage(
        text,
        query_tokens,
        window_chars=PRF_WINDOW_CHARS,
        stride_chars=PRF_STRIDE_CHARS,
        windows_per_doc=PRF_WINDOWS_PER_DOC,
    )


def rm3_expansion(
    query: str,
    candidates: list[Any],
    *,
    fb_docs: int = PRF_FB_DOCS,
    fb_terms: int = PRF_FB_TERMS,
    alpha: float = PRF_ALPHA,
    min_doc_freq: int = PRF_MIN_DOC_FREQ,
) -> PRFResult:
    """Lavrenko & Croft RM1, interpolated with the original query (RM3).

        RM1:  p(w | R)  = SUM_d  p(d | Q) * p(w | d)
        RM3:  p(w | Q') = alpha * p(w | Q) + (1 - alpha) * p(w | R)

    `p(d | Q)` is the retrieval score normalised over the feedback set -- the only
    document evidence a REST search endpoint returns. `p(w | d)` is maximum
    likelihood over the feedback span. Only terms NOT already in the query are
    returned, because the query side is already carried by the text being appended
    to; their weight is therefore `(1 - alpha) * p(w | R)`, which is exactly their
    RM3 weight.
    """
    query_tokens = {
        token for token in tokenize(query)
        if token not in _STOPWORDS and len(token) >= PRF_MIN_TERM_LEN
    }
    pool = [c for c in candidates[:fb_docs] if (c.text or "").strip()]
    if not pool:
        return PRFResult(terms=[], status="no-feedback", detail="no feedback document had text")

    # p(d | Q) from the retrieval scores. BM25 scores are unbounded and not
    # probabilities; normalising over the feedback set is the standard client-side
    # stand-in. Shifting by the minimum keeps the weakest feedback document from
    # getting a negative weight if the API ever returns one.
    scores = [float(c.score) for c in pool]
    floor = min(scores)
    shifted = [s - floor + 1e-9 for s in scores]
    total = sum(shifted)
    doc_weights = [s / total for s in shifted] if total > 0 else [1.0 / len(pool)] * len(pool)

    rm1: Counter[str] = Counter()
    doc_freq: Counter[str] = Counter()
    used = 0
    for candidate, weight in zip(pool, doc_weights):
        tokens = [
            token for token in tokenize(_feedback_text(candidate, query_tokens))
            if len(token) >= PRF_MIN_TERM_LEN and token not in _STOPWORDS and not token.isdigit()
        ]
        if not tokens:
            continue
        used += 1
        counts = Counter(tokens)
        length = len(tokens)
        for term, tf in counts.items():
            rm1[term] += weight * (tf / length)
            doc_freq[term] += 1

    if not used:
        return PRFResult(terms=[], status="no-feedback", detail="feedback documents had no terms")

    candidates_terms = {
        term: mass
        for term, mass in rm1.items()
        if term not in query_tokens and doc_freq[term] >= min_doc_freq
    }
    if not candidates_terms:
        return PRFResult(
            terms=[], status="no-feedback", feedback_docs=used,
            detail=f"no term outside the query appeared in >= {min_doc_freq} documents",
        )

    # Ties broken by term so the output is reproducible across runs and machines.
    ranked = sorted(candidates_terms.items(), key=lambda kv: (-kv[1], kv[0]))[:fb_terms]
    mass = sum(weight for _, weight in ranked)
    terms = [
        (term, round((1.0 - alpha) * (weight / mass), 6)) for term, weight in ranked
    ] if mass > 0 else []
    return PRFResult(terms=terms, status="ok", feedback_docs=used)


def build_prf_client(config_path: str, index_override: str | None = None):
    """Construct the Pyserini client from the pipeline config.

    Deliberately the same config, index, token convention, and on-disk response
    cache as the retrieval pipeline: PRF feedback must come from the index that will
    answer these queries, and a cached search is one the project has already paid
    for.
    """
    from trecrag.clients.pyserini_client import PyseriniClient

    load_dotenv_local()
    config = load_pipeline_config(config_path).get("pyserini", {})
    token = get_env(config.get("token_env", "PYSERINI_API_TOKEN"))
    if not token:
        raise RuntimeError(
            f"PRF needs a Pyserini token: set {config.get('token_env', 'PYSERINI_API_TOKEN')!r} "
            f"in .env.local, or pass --no-prf to run 3A and 3B only."
        )
    return PyseriniClient(
        base_url=config["base_url"],
        index=index_override or config["index"],
        token=token,
        timeout_seconds=float(config.get("timeout_seconds", 60)),
        max_retries=int(config.get("max_retries", 3)),
        cache_dir=config.get("cache_dir"),
        backoff_factor=float(config.get("backoff_factor", 0.5)),
    )


def run_prf(
    client,
    queries: list[str],
    hits: int,
    delay: float = PRF_REQUEST_DELAY_SECONDS,
    max_consecutive_failures: int = PRF_MAX_CONSECUTIVE_FAILURES,
) -> list[PRFResult]:
    """One paced search per query, then RM3 over what comes back.

    Nothing here raises. A search failure degrades that one query to its 3B form and
    is counted; it does not flag the record and it does not end the run. The
    distinction is deliberate: a malformed generation means there is no query, while
    a flaky shared API means there is a perfectly good query that merely missed its
    corpus-vocabulary pass. Failing the record for the second would make the output
    depend on network weather -- and losing the run to it would throw away every
    3A and 3B generation that has already been paid for on the GPU.

    That is not hypothetical: urllib3 raises RetryError (a bare RequestException, not
    a ConnectionError) once a 429 ladder is exhausted, which walks straight past the
    client's own transport-error handling. Hence the broad except.
    """
    import requests
    from trecrag.clients.pyserini_client import PyseriniAuthError, PyseriniRequestError

    results: list[PRFResult] = []
    consecutive = 0
    stopped = ""
    for i, query in enumerate(queries, start=1):
        if i % 50 == 0 or i == len(queries):
            logger.info("PRF: %d/%d queries", i, len(queries))
        if stopped:
            results.append(PRFResult(terms=[], status="api-error", detail=stopped))
            continue

        before = client.stats["cache_misses"]
        try:
            candidates, _ = client.search_adaptive(query, hits=hits, min_hits=5)
        except PyseriniAuthError as exc:
            # A rejected token will not fix itself, and 1200 more attempts will not
            # help. Stop asking, keep everything already generated.
            stopped = f"authentication failed: {exc}"
            logger.error("PRF stopping: %s", stopped)
            results.append(PRFResult(terms=[], status="api-error", detail=stopped))
            continue
        except (PyseriniRequestError, requests.exceptions.RequestException) as exc:
            consecutive += 1
            logger.warning(
                "PRF search failed (%s, %d in a row): %s", type(exc).__name__, consecutive, exc
            )
            if consecutive >= max_consecutive_failures:
                stopped = (
                    f"stopped after {consecutive} consecutive failures "
                    f"({type(exc).__name__}); the remaining queries keep their 3B form"
                )
                logger.error("PRF stopping: %s", stopped)
            results.append(PRFResult(terms=[], status="api-error", detail=str(exc)[:200]))
            continue

        consecutive = 0
        results.append(rm3_expansion(query, candidates))
        # Pace only what actually touched the network: a cached search costs the
        # service nothing, so a re-run should not crawl.
        if delay > 0 and client.stats["cache_misses"] > before and i < len(queries):
            time.sleep(delay)
    return results


# --------------------------------------------------------------------------------
# 7. OUTPUT
# --------------------------------------------------------------------------------


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_error_log(path: str | Path, flagged: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if not flagged:
            fh.write("No queries failed conditioning or expansion.\n")
            return
        by_stage = Counter(record.get("failure_stage", "?") for record in flagged)
        breakdown = ", ".join(f"{count} {stage}" for stage, count in sorted(by_stage.items()))
        fh.write(f"{len(flagged)} query/queries failed step 3 ({breakdown})\n\n")
        for record in flagged:
            fh.write(
                f"{record['query_id']}\t{record.get('failure_stage','?')}\t"
                f"{record.get('failure_reason','')}\n"
            )


# --------------------------------------------------------------------------------
# ORCHESTRATION
# --------------------------------------------------------------------------------


def _join_raw(indices: list[int], raw_by_index: dict[int, str], jobs: list[ExpansionJob]) -> str:
    parts = [(i, raw_by_index[i]) for i in indices if i in raw_by_index]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0][1]
    return "\n---\n".join(f"[{jobs[i].label()}] {raw}" for i, raw in parts)


def _query_text(*parts: str) -> str:
    """Join query fragments for a text-only search endpoint.

    Terms repeated across fragments raise their term frequency, which is the only
    weighting a plain-string API can express -- and the repeated terms are the core
    ones, so the effect points the right way.
    """
    return " ".join(part.strip() for part in parts if part and part.strip())


def expand_queries(
    llm, tokenizer, records: list[dict[str, Any]], batch_size: int, prf_client,
    prf_hits: int, prf_delay: float = PRF_REQUEST_DELAY_SECONDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run 3A, 3B and 3C over every AIN and assemble `expanded_queries`.

    Kept out of main() so the whole stage is exercisable without a GPU or the API.
    Returns (good, flagged, stats).
    """
    jobs = build_jobs(records)
    logger.info("step 3: %d record(s) -> %d atomic information need(s)", len(records), len(jobs))

    # --- 3A: metadata conditioning ------------------------------------------
    all_indices = list(range(len(jobs)))
    conditioned, reasons_a, raw_a1, raw_a2, recovered_a = run_llm_pass(
        llm, tokenizer, jobs, all_indices,
        build_prompts=condition_prompts,
        parse=lambda raw, job: parse_conditioned(raw, job.ain_text),
        max_tokens=CONDITION_MAX_TOKENS, batch_size=batch_size, stage="3A",
    )
    for i, query in conditioned.items():
        jobs[i].conditioned = query

    # --- 3B: guarded lexical expansion --------------------------------------
    # Only needs to run for needs that have a conditioned query to expand.
    expandable = [i for i in all_indices if i in conditioned]
    expansions, reasons_b, raw_b1, raw_b2, recovered_b = run_llm_pass(
        llm, tokenizer, jobs, expandable,
        build_prompts=expansion_prompts,
        parse=lambda raw, job: parse_expansions(raw),
        max_tokens=EXPANSION_MAX_TOKENS, batch_size=batch_size, stage="3B",
    )
    for i, terms in expansions.items():
        jobs[i].expansion_terms = terms

    # --- 3C: RM3 -------------------------------------------------------------
    # Only for needs that got all the way through 3B: PRF adapts an expanded query,
    # and there is nothing to adapt without one.
    ready = [i for i in all_indices if i in expansions]
    step3b_text = {
        i: _query_text(jobs[i].conditioned, " ".join(jobs[i].expansion_terms)) for i in ready
    }
    if prf_client is None:
        prf = {i: PRFResult(terms=[], status="disabled") for i in ready}
    else:
        logger.info("3C: RM3 over %d query/queries (hits=%d)", len(ready), prf_hits)
        prf = dict(
            zip(ready, run_prf(prf_client, [step3b_text[i] for i in ready], prf_hits, prf_delay))
        )

    # --- assemble per record, gate, and route --------------------------------
    jobs_by_query: dict[str, list[int]] = {}
    for i, job in enumerate(jobs):
        jobs_by_query.setdefault(job.query_id, []).append(i)

    good: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    prf_status = Counter()
    prf_term_count = 0
    for record in records:
        indices = jobs_by_query.get(record["query_id"], [])
        failed_a = [i for i in indices if i not in conditioned]
        failed_b = [i for i in indices if i in conditioned and i not in expansions]
        stage, reason = "", ""

        if failed_a or failed_b:
            stage = "3a-conditioning" if failed_a else "3b-expansion"
            failed = failed_a or failed_b
            source = reasons_a if failed_a else reasons_b
            reason = " || ".join(
                f"[{jobs[i].label()}] {source.get(i, 'unknown')}" for i in failed
            )
        else:
            by_sub_question: dict[str, list[dict[str, Any]]] = {}
            for i in indices:
                job = jobs[i]
                result = prf[i]
                array = by_sub_question.setdefault(job.sub_question_id, [])
                array.append(
                    {
                        "eq_id": len(array) + 1,
                        "ain_id": job.ain_id,
                        "step3a_conditioned_query": job.conditioned,
                        "step3b_expansion_terms": job.expansion_terms,
                        "step3b_expanded_query": step3b_text[i],
                        "step3c_prf_terms": [
                            {"term": term, "weight": weight} for term, weight in result.terms
                        ],
                        "step3c_expanded_query": _query_text(
                            step3b_text[i], " ".join(term for term, _ in result.terms)
                        ),
                        "step3c_status": result.status,
                    }
                )
            candidate = dict(record)
            candidate["expanded_queries"] = [
                {"sub_question_id": entry["sub_question_id"],
                 "eq_array": by_sub_question.get(entry["sub_question_id"], [])}
                for entry in record["ain"]
            ]
            try:
                validate_final_record(candidate)
            except QueryExpansionParseError as exc:
                stage, reason = "final-gate", str(exc)
            else:
                # Counted only once the record is certain to be written, so the
                # PRF tallies describe the output file rather than the attempt.
                for i in indices:
                    prf_status[prf[i].status] += 1
                    prf_term_count += len(prf[i].terms)
                good.append(candidate)
                continue

        # One unresolved need fails the whole record: a partial expanded_queries
        # block looks complete, and the needs missing from it are exactly the ones
        # nothing would ever retrieve for.
        flagged.append(
            {
                **record,
                "first_attempt_raw_output": _join_raw(
                    indices, raw_b1 if stage == "3b-expansion" else raw_a1, jobs
                ),
                "fallback_attempt_raw_output": _join_raw(
                    indices, raw_b2 if stage == "3b-expansion" else raw_a2, jobs
                ),
                "failure_stage": stage,
                "failure_reason": reason or "unknown",
            }
        )

    stats = {
        "needs": len(jobs),
        "conditioned": len(conditioned),
        "recovered_3a": recovered_a,
        "expanded": len(expansions),
        "recovered_3b": recovered_b,
        "prf_status": dict(prf_status),
        "prf_terms": prf_term_count,
    }
    return good, flagged, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default=INPUT_PATH, help="step 2's JSONL output")
    parser.add_argument("--model", default=LLM_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--flagged", default=FLAGGED_PATH)
    parser.add_argument("--error-log", default=ERROR_LOG_PATH)
    parser.add_argument("--quantization", default=QUANTIZATION)
    parser.add_argument("--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE)
    parser.add_argument("--gpu-memory-utilization", default=GPU_MEMORY_UTILIZATION,
                        help='float, or "auto" to fit alongside other GPU jobs')
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N records")
    parser.add_argument("--prf", dest="prf", action=argparse.BooleanOptionalAction,
                        default=PRF_ENABLED,
                        help="run step 3C (RM3 over the Pyserini API); --no-prf stops after 3B")
    parser.add_argument("--prf-config", default=PRF_CONFIG_PATH)
    parser.add_argument("--prf-index", default=None, help="override the index from the config")
    parser.add_argument("--prf-hits", type=int, default=PRF_HITS)
    parser.add_argument("--prf-delay", type=float, default=PRF_REQUEST_DELAY_SECONDS,
                        help="seconds to wait between PRF searches that hit the network")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    records = load_step2_records(args.input)
    if args.limit:
        records = records[: args.limit]
    if not records:
        logger.error("no usable records loaded from %s", args.input)
        raise SystemExit(1)
    logger.info("loaded %d record(s) from %s", len(records), args.input)

    # Built before the model loads: a missing token should fail in a second, not
    # after a five-minute 72B load.
    prf_client = build_prf_client(args.prf_config, args.prf_index) if args.prf else None
    if prf_client is None:
        logger.warning("step 3C disabled: writing 3A/3B queries with step3c_status='disabled'")

    from transformers import AutoTokenizer
    from vllm import LLM

    gpu_util = resolve_gpu_memory_utilization(
        args.model, args.tensor_parallel_size, args.gpu_memory_utilization
    )
    quantization = resolve_quantization(args.model, args.quantization)
    logger.info(
        "loading %s (quantization=%s, tp=%d)", args.model, quantization, args.tensor_parallel_size
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        quantization=quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=gpu_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        seed=SEED,
    )

    good, flagged, stats = expand_queries(
        llm, tokenizer, records, args.batch_size, prf_client, args.prf_hits, args.prf_delay
    )

    write_jsonl(args.output, good)
    write_jsonl(args.flagged, flagged)
    write_error_log(args.error_log, flagged)

    # Invariant: every input record lands in exactly one output file.
    accounted = {r["query_id"] for r in good} | {r["query_id"] for r in flagged}
    missing = {r["query_id"] for r in records} - accounted
    if missing:
        logger.error("BUG: %d query_id(s) unaccounted for: %s", len(missing), sorted(missing)[:10])

    written_needs = sum(len(e["eq_array"]) for r in good for e in r["expanded_queries"])
    prf_ok = stats["prf_status"].get("ok", 0)

    print()
    print(f"  records read               : {len(records)}")
    print(f"  atomic needs / LLM calls   : {stats['needs']} x2 (3A + 3B)")
    print(f"  3A conditioned             : {stats['conditioned']}"
          f"  ({stats['recovered_3a']} via fallback)")
    print(f"  3B expanded                : {stats['expanded']}"
          f"  ({stats['recovered_3b']} via fallback)")
    if prf_client is not None:
        mean_terms = f"  (mean {stats['prf_terms'] / prf_ok:.1f} per query)" if prf_ok else ""
        print(f"  3C RM3 searches            : {sum(stats['prf_status'].values())}"
              f"  ({prf_ok} yielded feedback terms)")
        print(f"  3C terms added             : {stats['prf_terms']}{mean_terms}")
        degraded = {k: v for k, v in stats["prf_status"].items() if k != "ok" and v}
        if degraded:
            # Never silent: a query that quietly kept its 3B form is a query that
            # never got its corpus-vocabulary pass.
            print(f"  3C degraded                : {degraded}")
    else:
        print("  3C RM3                     : disabled (--no-prf)")
    print(f"  records written            : {len(good)}")
    print(f"  records flagged            : {len(flagged)}")
    print(f"  expanded queries written   : {written_needs}")
    print(f"  main output                : {args.output}")
    if flagged:
        print(f"  flagged output             : {args.flagged}")
        print(f"  error log                  : {args.error_log}")
    if missing:
        print(f"  *** UNACCOUNTED query_ids  : {len(missing)} ***")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
