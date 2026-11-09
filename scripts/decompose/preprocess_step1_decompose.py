#!/usr/bin/env python3
"""Preprocessing step 1: decompose composite TREC topics into metadata + sub-questions.

Many TREC RAG 2026 narratives bundle situational context with two or more distinct
sub-questions. Handing the whole blob to a first-stage retriever dilutes it, so this
script splits each topic into a `query_metadata` string and an ordered list of
`sub_questions` *before* any retrieval happens. Retrieval is out of scope here.

    query_id  query_text                    (TSV, no header)
        -> vLLM offline batch generation
        -> <metadata>/<sub_question> XML tags
        -> deterministic regex parse + validation
        -> step 1b: <dependencies> XML tags, same run, same model session
        -> preprocessed_queries_step1.jsonl

Step 1b asks a second question of the same loaded model: among a query's
sub-questions, does answering one require knowing the answer to another first? The
verdict lands in a `dependencies` field on the same record, so a single run of this
script produces the finished file -- there is no second script and nothing reads the
main output back in. Most pairs have no dependency, and that is the expected answer.

Anything that fails twice is written to the flagged file instead, never dropped:
every input query_id appears in exactly one of the two outputs. That includes a topic
that decomposed cleanly but whose dependency judgement could not be pinned down --
it is redirected to the flagged file (`failure_stage: "dependency-detection"`) rather
than written with a guessed `dependencies` value, which keeps the main output's
invariant absolute: every record in it has a fully validated `dependencies` field.

QUICKSTART
    # Use the dedicated vLLM env, NOT trec-rag-2026. vLLM 0.13 pins torch 2.9,
    # and the main pipeline runs torch 2.11 under sentence-transformers --
    # installing vLLM alongside it would downgrade torch beneath a validated run.
    conda activate trec-rag-2026-vllm
    python scripts/decompose/preprocess_step1_decompose.py

    # or override the constants below:
    python scripts/decompose/preprocess_step1_decompose.py \
        --topics trec_rag_2026_test_query_set.tsv \
        --model /media/20TB/shared/models/qwen/Qwen2.5-72B-Instruct-AWQ/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217 \
        --output output_decomp/preprocessed_queries_step1.jsonl

    # GPU_MEMORY_UTILIZATION defaults to "auto": it measures free VRAM at launch and
    # sizes the allocation to fit beside whatever else is running, rather than
    # assuming the cards are idle. This box is shared, so it refuses to start (with
    # the numbers and a clear message) when too little would be left for a usable KV
    # cache -- instead of loading and then batching one request at a time.

MODEL NOTES (verified on this machine)
  Qwen2.5-72B-Instruct-AWQ  awq 4-bit/group 128, 11 shards, ~39GB (19.4 GiB/GPU at
                            tp=2), HAS a chat template. This is the model to use.
                            It needs both cards essentially idle: it does not fit
                            beside another large resident model, and "auto" sizing
                            will say so rather than OOM.
  gemma-4-31B-it            Gemma4ForConditionalGeneration, NOT quantized (~59GB),
                            and its tokenizer_config.json has NO chat_template.
                            CHAT_TEMPLATE_FALLBACK below handles that; verify the
                            output format on a few topics before trusting a full run.
Both are HF-cache layouts, so LLM_PATH must point at the snapshots/<hash>/ directory
that actually contains config.json, not at the top-level model directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------------
# 1. CONFIGURATION -- edit these, or override any of them on the command line.
# --------------------------------------------------------------------------------

TEST_TOPIC_PATH = "trec_rag_2026_test_query_set.tsv"

# Must be the directory containing config.json. For an HF-cache layout that is the
# snapshots/<hash>/ subdirectory, not the repo root.
LLM_PATH = (
    "/media/20TB/shared/models/qwen/Qwen2.5-72B-Instruct-AWQ/"
    "models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/"
    "698703eae6604af048a3d2f509995dc302088217"
)

OUTPUT_PATH = "output_decomp/preprocessed_queries_step1.jsonl"
FLAGGED_PATH = "output_decomp/preprocessed_queries_step1_flagged.jsonl"
ERROR_LOG_PATH = "output_decomp/preprocessed_queries_step1_errors.log"

# Generation / runtime knobs.
QUANTIZATION: str | None = None   # None = auto-detect from config.json (see resolve_quantization)
TENSOR_PARALLEL_SIZE = 2          # 2x RTX 6000 Ada; a 72B AWQ checkpoint will not fit on one
GPU_MEMORY_UTILIZATION = "auto"   # "auto" = size to whatever VRAM is free now
MAX_MODEL_LEN = 8192              # topics are short; this keeps the KV cache small
MAX_TOKENS = 1024                 # a decomposition is far shorter, but truncation fails the parse
TEMPERATURE = 0.0                 # deterministic: this is extraction, not generation
TOP_P = 1.0
SEED = 0
BATCH_SIZE = 0                    # 0 = submit every prompt in one generate() call

# Step 1b (dependency detection) knobs.
PAIRWISE_API = False   # False = judge all of a query's sub-question pairs in one
                       # LLM call (default). True = one LLM call per individual
                       # pair. Only matters once queries have 3+ sub-questions;
                       # with the current at-most-2 dataset the two modes are
                       # functionally identical (both do exactly one call per
                       # query, since there's at most one pair). Flip to True if
                       # a future dataset has enough sub-questions per query that
                       # judging every pair implicitly within a single prompt
                       # risks the model losing precision.
DEPENDENCY_MAX_TOKENS = 512       # an empty verdict is ~10 tokens and one dependency
                                  # ~25, so this is generous even for a query with
                                  # many sub-questions -- but far below MAX_TOKENS,
                                  # so a runaway generation is cut off early instead
                                  # of burning the batch.

# Used only when the tokenizer has no chat_template (e.g. the local gemma-4-31B-it).
CHAT_TEMPLATE_FALLBACK = (
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}{{ m['content'] + '\n\n' }}"
    "{% elif m['role'] == 'user' %}{{ 'User: ' + m['content'] + '\n\nAssistant: ' }}"
    "{% else %}{{ m['content'] }}{% endif %}"
    "{% endfor %}"
)

logger = logging.getLogger("preprocess_step1")


# --------------------------------------------------------------------------------
# 4. PROMPTS -- isolated so they can be read and iterated on independently.
# --------------------------------------------------------------------------------

DECOMPOSITION_SYSTEM_PROMPT = """\
You split composite information-seeking queries into their parts. You extract; you \
never invent, summarise, or answer.

A composite query contains two kinds of material:

1. METADATA - situational or background framing about the asker: who they are, their \
role, constraints, budget, deadlines, what someone else wants, why they are asking. \
Metadata sets the scene. It is not itself a question.
2. SUB-QUESTIONS - the discrete things the asker actually wants answered. Each should \
stand on its own as an independently answerable question.

Rules:

- The number of sub-questions is VARIABLE and unknown in advance. It may be 1, 2, 3, \
or more. Identify exactly as many distinct questions as are genuinely present. Never \
force a particular count, and never merge two clearly separate questions into one or \
split a single question into pieces.
- Preserve the asker's ORIGINAL WORDING for each sub-question. Copy the text \
verbatim wherever possible; at most, fix a dangling connective so the question reads \
standalone. Do not paraphrase, compress, or rewrite. Do not answer anything.
- A single sentence containing several coordinated asks ("what should we measure, and \
what mistakes should we avoid?") is ONE sub-question if the asker posed it as one \
sentence. Split on the asker's own sentence boundaries, not on every "and".
- If the query has NO situational framing and is just a bare question, emit an empty \
metadata tag. Never invent background that is not in the text.
- If the query contains only one question, emit exactly one sub_question tag.

Output format - emit ONLY these tags, nothing else. No preamble, no trailing \
commentary, no markdown code fences, no explanation:

<metadata>...</metadata>
<sub_question id="1">...</sub_question>
<sub_question id="2">...</sub_question>

The metadata tag is always present, even when empty: <metadata></metadata>. \
sub_question ids start at 1 and increase by 1 with no gaps, in the order the \
questions appear in the query.

Example A - background context plus two questions:

Query: I run a small bakery and my flour costs have doubled since last year. I want to \
keep my regulars but I can't absorb this. Should I raise prices across the board or \
just on the specialty items? And how do I tell customers without it feeling like a \
bait and switch?

<metadata>I run a small bakery and my flour costs have doubled since last year. I want \
to keep my regulars but I can't absorb this.</metadata>
<sub_question id="1">Should I raise prices across the board or just on the specialty \
items?</sub_question>
<sub_question id="2">And how do I tell customers without it feeling like a bait and \
switch?</sub_question>

Example B - a bare question with no framing:

Query: What causes the northern lights?

<metadata></metadata>
<sub_question id="1">What causes the northern lights?</sub_question>
"""

DECOMPOSITION_USER_TEMPLATE = """Query: {query_text}"""

# Shorter and far more mechanical: when the first attempt fails the parser it is
# almost always because the model wrapped the tags in prose or a code fence, so the
# retry drops the conceptual framing and makes the output shape the whole message.
# Re-sending the identical prompt against the same weights at temperature 0 would
# reproduce the same failure, so this must be a genuinely different prompt.
DECOMPOSITION_FALLBACK_PROMPT = """\
Rewrite the query below as XML tags. Output the tags and NOTHING else.

Required shape, exactly:

<metadata>background context here, or leave empty</metadata>
<sub_question id="1">first question here</sub_question>
<sub_question id="2">second question here, if there is one</sub_question>

Hard rules:
- Start your reply with the character < and end it with the character >.
- No code fences. No "Here is". No notes before or after the tags.
- Exactly one <metadata> tag. Keep it even if empty: <metadata></metadata>
- One <sub_question> tag per question actually asked. ids are 1, 2, 3, ... in order.
- Copy question text verbatim from the query. Do not answer, shorten, or reword.
- Background/context goes in metadata. Questions go in sub_question tags.

Query: {query_text}
"""


# --- step 1b: dependency detection -----------------------------------------------
# Two prompt pairs, one per PAIRWISE_API mode. Both emit the same <dependencies>
# wrapper so everything downstream -- parser, gate, retry, flagging -- is shared.
# The hard part of this judgement is not finding dependencies but NOT finding them:
# every pair shares a topic and a metadata blob, which reads as relatedness, and a
# model that mistakes relatedness for an ordering constraint will tag nearly
# everything. Hence "no dependency is the default" is stated in all four prompts and
# the worked examples all use the SAME metadata, so the only thing distinguishing them
# is whether one answer actually feeds the other.
#
# Measured on the 119-topic test set (Qwen2.5-72B-AWQ, temperature 0, 275 pairs): an
# isolated yes/no judgement biases hard toward yes. An earlier pairwise prompt with
# the same two examples as the batched one tagged 57% of pairs against the batched
# prompt's 36%, and 72 of their 94 disagreements were the pairwise call adding an edge
# the batched call did not. Adding a third worked example -- a second independent pair,
# both questions about the same plan -- brought pairwise to 33% against batched's 38%,
# and mode agreement from 66% to 75%. The batched prompt does not carry that third
# example: judging every pair in one pass exposes the base rate on its own.

DEPENDENCY_SYSTEM_PROMPT = """\
You judge whether one sub-question has to be answered before another one can be \
answered well. You judge the ordering only. You never answer the sub-questions, and \
you never rewrite them.

You are given the asker's METADATA (their situation, constraints, and framing) and \
the numbered SUB-QUESTIONS taken from a single query. Consider every pair of \
sub-questions.

Sub-question B DEPENDS ON sub-question A when a good answer to B cannot be written \
until the answer to A is settled -- B is about the choice, plan, set, or outcome that \
A produces, so answering B first would mean guessing at A's answer.

Apply this test to each pair, in both directions: could a well-informed writer answer \
B completely without first stating an answer to A? If yes, there is no dependency.

NO DEPENDENCY IS THE DEFAULT, AND THE COMMON CASE. Most pairs are simply two facets \
of one topic that can be answered in either order. Do NOT tag a dependency merely \
because:

- the two sub-questions are about the same subject, or draw on the same metadata;
- one is asked before the other in the query;
- answering both together would read more naturally;
- one is broad and the other is narrow, or one is "what" and the other is "how";
- one states a requirement on the other's answer (sources to use, length, tone). A \
constraint shapes an answer; it is not something that has to be answered first.

Tag a dependency ONLY when there is a genuine answer-order requirement. When in \
doubt, tag nothing.

Use the metadata as evidence: stated priorities, constraints, deadlines, or a \
sequence the asker already implies can turn an ambiguous ordering into a real one, \
and can equally make clear that two questions are independent.

Output format - emit ONLY these tags, nothing else. No preamble, no trailing \
commentary, no markdown code fences, no explanation:

<dependencies>
<dependency from="1" to="2"/>
</dependencies>

<dependency from="X" to="Y"/> means sub-question Y depends on sub-question X, so X \
must be answered first. Rules for the tags:

- Use the sub-question ids exactly as they are given to you, and no others.
- The two ids in a tag must differ: a sub-question never depends on itself.
- One tag per pair at most. Never emit the same pair twice, and never emit both \
directions for one pair - choose the single stronger direction, or neither.
- When you find no dependency at all, still emit the wrapper, empty:

<dependencies></dependencies>

Never reply with an empty message. The empty wrapper is how you say "I considered \
the pairs and there is no dependency", and it is a perfectly good answer.

Example A - a genuine dependency. The migration asked about in sub-question 2 is a \
property of whichever tool sub-question 1 settles on, so 1 has to be answered first:

Metadata: I manage a 12-person support team and we are moving off shared email onto a \
ticketing tool next quarter.
Sub-questions:
1. Which ticketing tool should we choose for our volume and budget?
2. How do we migrate three years of existing email threads into it without losing \
history?

<dependencies>
<dependency from="1" to="2"/>
</dependencies>

Example B - same asker, same topic, but independent. Both questions are about \
running the team, yet neither answer needs the other one first, so nothing is tagged:

Metadata: I manage a 12-person support team and we are moving off shared email onto a \
ticketing tool next quarter.
Sub-questions:
1. What first-response time is realistic for a team our size?
2. How do other teams this size handle weekend coverage?

<dependencies></dependencies>
"""

DEPENDENCY_USER_TEMPLATE = """\
Metadata: {metadata}
Sub-questions:
{sub_questions}"""

# Same relationship to DEPENDENCY_SYSTEM_PROMPT that DECOMPOSITION_FALLBACK_PROMPT
# has to DECOMPOSITION_SYSTEM_PROMPT: drop the conceptual framing, make the output
# shape the whole message. Re-sending the identical prompt to the same weights at
# temperature 0 would reproduce the same failure, so the retry has to differ.
DEPENDENCY_FALLBACK_PROMPT = """\
Decide which of the sub-questions below must be answered before which. Output XML \
tags and NOTHING else.

Required shape, exactly:

<dependencies>
<dependency from="X" to="Y"/>
</dependencies>

Hard rules:
- Start your reply with the character < and end it with the character >.
- No code fences. No "Here is". No notes before or after the tags.
- Exactly one <dependencies> wrapper. Keep it even when nothing goes inside it: \
<dependencies></dependencies>
- <dependency from="X" to="Y"/> means Y cannot be answered until X is answered.
- X and Y must both be taken from this id list, and must differ: {ids}
- At most one tag per pair, and never both from="X" to="Y" and from="Y" to="X".
- Emit a tag ONLY if answering one question really requires the other's answer \
first. Test each pair: if one could be answered completely without first answering \
the other, emit nothing for it. Two questions on the same topic are NOT a dependency, \
and emitting no tags at all is the normal, expected answer.

Metadata: {metadata}
Sub-questions:
{sub_questions}
"""

DEPENDENCY_PAIRWISE_SYSTEM_PROMPT = """\
You are given the asker's METADATA (their situation, constraints, and framing) and \
exactly TWO sub-questions taken from a single query. Decide whether one of them has \
to be answered before the other. You judge the ordering only. You never answer the \
sub-questions, and you never rewrite them.

Sub-question B DEPENDS ON sub-question A when a good answer to B cannot be written \
until the answer to A is settled -- B is about the choice, plan, set, or outcome that \
A produces, so answering B first would mean guessing at A's answer.

There are exactly three possible verdicts, and the third is the most common:

1. the second depends on the first;
2. the first depends on the second;
3. neither - they can be answered in either order, independently.

Apply this test in both directions: could a well-informed writer answer one of them \
completely without first stating an answer to the other? If yes either way round, the \
verdict is 3.

VERDICT 3 IS THE DEFAULT, AND MOST PAIRS ARE VERDICT 3. The two sub-questions in \
front of you were taken from one query, so they will always look related; that is not \
what you are judging. Do NOT report a dependency merely because:

- both are about the same subject, or draw on the same metadata;
- one is asked before the other in the query;
- answering both together would read more naturally;
- one is broad and the other is narrow, or one is "what" and the other is "how";
- one states a requirement on the other's answer (sources to use, length, tone). A \
constraint shapes an answer; it is not something that has to be answered first.

Report a dependency ONLY when there is a genuine answer-order requirement. When in \
doubt, report none.

Use the metadata as evidence: stated priorities, constraints, deadlines, or a \
sequence the asker already implies can turn an ambiguous ordering into a real one, \
and can equally make clear that the two questions are independent.

Output format - emit ONLY these tags, nothing else. No preamble, no trailing \
commentary, no markdown code fences, no explanation:

<dependencies>
<dependency from="X" to="Y"/>
</dependencies>

<dependency from="X" to="Y"/> means sub-question Y depends on sub-question X, so X \
must be answered first. X and Y must be the two ids you were given, and must differ. \
Emit at most one tag: only one pair is under consideration.

For verdict 3, still emit the wrapper, empty:

<dependencies></dependencies>

Never reply with an empty message. The empty wrapper is how you say "I considered \
this pair and there is no dependency", and it is a perfectly good answer.

Example A - a genuine dependency. The migration asked about in sub-question 2 is a \
property of whichever tool sub-question 1 settles on, so 1 has to be answered first:

Metadata: I manage a 12-person support team and we are moving off shared email onto a \
ticketing tool next quarter.
Sub-question 1: Which ticketing tool should we choose for our volume and budget?
Sub-question 2: How do we migrate three years of existing email threads into it \
without losing history?

<dependencies>
<dependency from="1" to="2"/>
</dependencies>

Example B - same asker, same topic, but independent. Both questions are about \
running the team, yet neither answer needs the other one first, so nothing is tagged:

Metadata: I manage a 12-person support team and we are moving off shared email onto a \
ticketing tool next quarter.
Sub-question 1: What first-response time is realistic for a team our size?
Sub-question 2: How do other teams this size handle weekend coverage?

<dependencies></dependencies>

Example C - both questions are about the very same migration, and still independent: \
the training plan and the customer announcement can each be written without the \
other:

Metadata: I manage a 12-person support team and we are moving off shared email onto a \
ticketing tool next quarter.
Sub-question 1: How should we train the team on the new tool?
Sub-question 2: What should we tell customers about the change?

<dependencies></dependencies>
"""

DEPENDENCY_PAIRWISE_USER_TEMPLATE = """\
Metadata: {metadata}
Sub-question {id_a}: {text_a}
Sub-question {id_b}: {text_b}"""

# The pairwise retry can be blunter than the batched one: with only two ids in play
# there are exactly three legal replies, so the prompt can simply list them.
DEPENDENCY_PAIRWISE_FALLBACK_PROMPT = """\
Two sub-questions are below. Decide whether one must be answered before the other. \
Output XML and NOTHING else.

Your reply must be exactly one of these three lines:

<dependencies><dependency from="{id_a}" to="{id_b}"/></dependencies>
<dependencies><dependency from="{id_b}" to="{id_a}"/></dependencies>
<dependencies></dependencies>

- Choose the first if sub-question {id_b} cannot be answered until sub-question \
{id_a} is answered.
- Choose the second if sub-question {id_a} cannot be answered until sub-question \
{id_b} is answered.
- Choose the third if neither needs the other's answer first. This is the normal \
case: being on the same topic is NOT a dependency. Test it -- if one could be \
answered completely without first answering the other, choose the third.

Start your reply with the character < and end it with the character >. No code \
fences, no explanation, no ids other than {id_a} and {id_b}.

Metadata: {metadata}
Sub-question {id_a}: {text_a}
Sub-question {id_b}: {text_b}
"""


# --------------------------------------------------------------------------------
# 5. INPUT
# --------------------------------------------------------------------------------


@dataclass
class Topic:
    query_id: str
    query_text: str


class DecompositionParseError(Exception):
    """Raised with a human-readable reason naming the validation rule that failed."""


def load_topics(path: str | Path) -> list[Topic]:
    """Read a headerless TSV of `query_id \\t query_text`.

    Split once only: narratives legitimately contain tabs, and splitting on every
    tab would silently truncate a query. IDs stay strings -- they can be
    alphanumeric (`rag2026-0`, `1rag2026-0`) and int coercion would corrupt them.
    """
    topics: list[Topic] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue  # blank lines are messiness, not an error
            parts = line.split("\t", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                logger.warning("%s:%d: skipping malformed row", path, line_no)
                continue
            query_id, query_text = parts[0].strip(), parts[1].strip()
            if query_id in seen:
                logger.warning("%s:%d: duplicate query_id %r, skipping", path, line_no, query_id)
                continue
            seen.add(query_id)
            topics.append(Topic(query_id=query_id, query_text=query_text))
    return topics


# --------------------------------------------------------------------------------
# 3. PARSING -- deterministic, all-or-nothing.
# --------------------------------------------------------------------------------

_METADATA_RE = re.compile(r"<metadata>(.*?)</metadata>", re.DOTALL | re.IGNORECASE)
_SUBQ_RE = re.compile(
    r'<sub_question\s+id\s*=\s*"(\d+)"\s*>(.*?)</sub_question>', re.DOTALL | re.IGNORECASE
)
# Models sometimes wrap correct tags in ``` fences; that is a formatting artefact
# rather than a decomposition failure, so it is stripped before validation.
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def parse_decomposition(raw_output: str, query_id: str, comp_query: str) -> dict[str, Any]:
    """Parse the tagged LLM output into the target schema.

    Raises DecompositionParseError if ANY validation rule fails. Deliberately no
    partial extraction: a half-understood decomposition is worse than a retry,
    because it would silently ship a wrong split.
    """
    if raw_output is None:
        raise DecompositionParseError("empty generation")

    text = _FENCE_RE.sub("", raw_output.strip())

    # Rule 1: exactly one metadata block.
    metadata_matches = list(_METADATA_RE.finditer(text))
    if len(metadata_matches) != 1:
        raise DecompositionParseError(
            f"rule1: expected exactly 1 <metadata> block, found {len(metadata_matches)}"
        )

    # Rule 2: at least one sub_question block.
    subq_matches = list(_SUBQ_RE.finditer(text))
    if not subq_matches:
        raise DecompositionParseError("rule2: no <sub_question> blocks found")

    # Rule 3: ids are exactly 1..N in order, no gaps or duplicates.
    ids = [m.group(1) for m in subq_matches]
    if ids != [str(i) for i in range(1, len(ids) + 1)]:
        raise DecompositionParseError(f"rule3: sub_question ids are {ids}, expected 1..N in order")

    # Rule 4: nothing but whitespace outside the matched tag spans.
    spans = [m.span() for m in metadata_matches] + [m.span() for m in subq_matches]
    spans.sort()
    remainder, cursor = [], 0
    for start, end in spans:
        remainder.append(text[cursor:start])
        cursor = end
    remainder.append(text[cursor:])
    stray = "".join(remainder).strip()
    if stray:
        raise DecompositionParseError(f"rule4: stray text outside tags: {stray[:80]!r}")

    # Rule 5: no raw '<' inside content, which would mean a truncated/unclosed tag.
    metadata_text = metadata_matches[0].group(1).strip()
    contents = [metadata_text] + [m.group(2).strip() for m in subq_matches]
    for content in contents:
        if "<" in content:
            raise DecompositionParseError(f"rule5: '<' inside tag content: {content[:80]!r}")

    sub_questions = [
        {"sub_question_id": str(i), "sub_question_text": m.group(2).strip()}
        for i, m in enumerate(subq_matches, start=1)
    ]
    if any(not sq["sub_question_text"] for sq in sub_questions):
        raise DecompositionParseError("rule2: a <sub_question> block was empty")

    return {
        "query_id": query_id,
        "comp_query": comp_query,
        "query_metadata": metadata_text,
        "sub_questions": sub_questions,
    }


# --------------------------------------------------------------------------------
# 3b. FINAL PRE-WRITE GATE -- independent of the parser above.
# --------------------------------------------------------------------------------

_EXPECTED_KEYS = {"query_id", "comp_query", "query_metadata", "sub_questions"}


def validate_final_record(record: Any) -> None:
    """Last line of defence before serialization.

    Intentionally re-derives every invariant rather than trusting the parser, so a
    future edit that breaks parse_decomposition still cannot put a malformed record
    in the output file. Raises DecompositionParseError describing the first failure.
    """
    if not isinstance(record, dict):
        raise DecompositionParseError("final-gate: record is not a dict")
    if set(record) != _EXPECTED_KEYS:
        raise DecompositionParseError(
            f"final-gate: keys are {sorted(record)}, expected {sorted(_EXPECTED_KEYS)}"
        )
    for key in ("query_id", "comp_query", "query_metadata"):
        if not isinstance(record[key], str):
            raise DecompositionParseError(f"final-gate: {key!r} is not a string")
    if not record["query_id"]:
        raise DecompositionParseError("final-gate: empty query_id")
    if not record["comp_query"]:
        raise DecompositionParseError("final-gate: empty comp_query")

    sub_questions = record["sub_questions"]
    if not isinstance(sub_questions, list) or not sub_questions:
        raise DecompositionParseError("final-gate: sub_questions must be a non-empty list")
    for i, sq in enumerate(sub_questions, start=1):
        if not isinstance(sq, dict) or set(sq) != {"sub_question_id", "sub_question_text"}:
            raise DecompositionParseError(f"final-gate: sub_questions[{i-1}] has wrong keys")
        if not isinstance(sq["sub_question_id"], str) or not isinstance(
            sq["sub_question_text"], str
        ):
            raise DecompositionParseError(f"final-gate: sub_questions[{i-1}] values must be strings")
        if sq["sub_question_id"] != str(i):
            raise DecompositionParseError(
                f"final-gate: sub_question_id {sq['sub_question_id']!r} at position {i}"
            )
        if not sq["sub_question_text"].strip():
            raise DecompositionParseError(f"final-gate: sub_questions[{i-1}] text is empty")


# --------------------------------------------------------------------------------
# 3c. DEPENDENCY DETECTION (step 1b) -- runs in-process, in the same invocation,
#     on records that have already cleared the decomposition final gate.
# --------------------------------------------------------------------------------

# The wrapper is matched in two forms: the required <dependencies></dependencies> and
# the self-closed <dependencies/>. Both are an explicit "I considered the pairs and
# found none"; only a blank reply is ambiguous, and that still fails. Tolerating the
# self-closed spelling is the same call as stripping ``` fences above -- a formatting
# artefact, not a failed judgement.
_DEPS_BLOCK_RE = re.compile(
    r"<dependencies\s*>(.*?)</dependencies\s*>|<dependencies\s*/>", re.DOTALL | re.IGNORECASE
)
_DEP_TAG_RE = re.compile(r"<dependency\b([^<>]*?)/>", re.DOTALL | re.IGNORECASE)
_DEP_ATTR_RE = re.compile(r"""(\w+)\s*=\s*["']([^"']*)["']""")


def parse_dependencies(
    raw_output: str,
    valid_ids: Iterable[str],
    expected_pair: tuple[str, str] | None = None,
) -> list[dict[str, str]]:
    """Parse a <dependencies> block into the final `dependencies` schema.

    All-or-nothing, exactly like parse_decomposition: a hallucinated id or a stray
    sentence fails the whole judgement rather than being dropped, because silently
    discarding half a verdict is indistinguishable in the output from the model
    having answered "no dependency".

    `expected_pair` is supplied only in PAIRWISE_API mode, where a single call judges
    one specific pair and any tag must reference exactly those two ids.
    """
    if raw_output is None:
        raise DecompositionParseError("empty generation")

    text = _FENCE_RE.sub("", raw_output.strip())
    if not text:
        raise DecompositionParseError("empty generation")

    # Rule 1: exactly one <dependencies> wrapper.
    blocks = list(_DEPS_BLOCK_RE.finditer(text))
    if len(blocks) != 1:
        raise DecompositionParseError(
            f"dep-rule1: expected exactly 1 <dependencies> block, found {len(blocks)}"
        )
    block = blocks[0]

    # Rule 5a: nothing but whitespace outside the wrapper.
    outside = (text[: block.start()] + text[block.end() :]).strip()
    if outside:
        raise DecompositionParseError(
            f"dep-rule5: stray text outside <dependencies>: {outside[:80]!r}"
        )

    inner = block.group(1) or ""   # group(1) is None for the self-closed spelling
    tags = list(_DEP_TAG_RE.finditer(inner))

    # Rule 5b: inside the wrapper, only self-closing <dependency/> tags and
    # whitespace. Without this, "<dependencies>1 must come first</dependencies>"
    # would parse as an empty list, i.e. as the opposite verdict.
    remainder, cursor = [], 0
    for match in tags:
        remainder.append(inner[cursor : match.start()])
        cursor = match.end()
    remainder.append(inner[cursor:])
    stray = "".join(remainder).strip()
    if stray:
        raise DecompositionParseError(
            f"dep-rule5: stray text inside <dependencies>: {stray[:80]!r}"
        )

    # Rule 7a (pairwise only): one pair was judged, so at most one tag can come back.
    if expected_pair is not None and len(tags) > 1:
        raise DecompositionParseError(
            f"dep-rule7: {len(tags)} <dependency> tags for a single pair, expected at most 1"
        )

    valid = set(valid_ids)
    pairs: list[tuple[str, str]] = []
    for match in tags:
        attrs = dict(_DEP_ATTR_RE.findall(match.group(1)))
        if set(attrs) != {"from", "to"}:
            raise DecompositionParseError(
                f"dep-rule1: <dependency> needs exactly from= and to=, got {sorted(attrs)}"
            )
        source, target = attrs["from"].strip(), attrs["to"].strip()

        # Rule 2: both ids must exist in this query.
        if source not in valid or target not in valid:
            raise DecompositionParseError(
                f"dep-rule2: unknown sub_question id in {source!r} -> {target!r}; "
                f"valid ids are {sorted(valid)}"
            )
        # Rule 3: no self-dependency.
        if source == target:
            raise DecompositionParseError(f"dep-rule3: self-dependency on id {source!r}")
        # Rule 7b (pairwise only): nothing else was under consideration in this call.
        if expected_pair is not None and {source, target} != set(expected_pair):
            raise DecompositionParseError(
                f"dep-rule7: tag {source!r} -> {target!r} is not the judged pair "
                f"{tuple(expected_pair)}"
            )
        pairs.append((source, target))

    # Rule 4: no duplicate pair.
    if len(set(pairs)) != len(pairs):
        raise DecompositionParseError(f"dep-rule4: duplicate <dependency> pair in {pairs}")
    # Rule 6: no pair asserted in both directions. Resolving that heuristically would
    # mean inventing a verdict the model did not give.
    for source, target in pairs:
        if (target, source) in pairs:
            raise DecompositionParseError(
                f"dep-rule6: contradictory pair {source}->{target} and {target}->{source}"
            )

    return [{"depends_on": source, "dependent": target} for source, target in pairs]


def validate_dependencies(dependencies: Any, sub_questions: Any) -> None:
    """Final gate for the `dependencies` field, independent of parse_dependencies.

    Stands to parse_dependencies exactly as validate_final_record stands to
    parse_decomposition: it re-derives every invariant, so a future edit to the
    parser still cannot put an unanchored or contradictory dependency in the output.
    An empty list is always valid -- that is the common verdict.
    """
    if not isinstance(dependencies, list):
        raise DecompositionParseError("dep-gate: dependencies is not a list")
    if not isinstance(sub_questions, list) or not sub_questions:
        raise DecompositionParseError("dep-gate: sub_questions must be a non-empty list")

    valid = {
        sq["sub_question_id"]
        for sq in sub_questions
        if isinstance(sq, dict) and isinstance(sq.get("sub_question_id"), str)
    }
    seen: set[tuple[str, str]] = set()
    for i, dep in enumerate(dependencies):
        if not isinstance(dep, dict) or set(dep) != {"depends_on", "dependent"}:
            found = sorted(dep) if isinstance(dep, dict) else type(dep).__name__
            raise DecompositionParseError(
                f"dep-gate: dependencies[{i}] is {found}, expected keys "
                f"['dependent', 'depends_on']"
            )
        source, target = dep["depends_on"], dep["dependent"]
        if not isinstance(source, str) or not isinstance(target, str):
            raise DecompositionParseError(f"dep-gate: dependencies[{i}] values must be strings")
        if source not in valid or target not in valid:
            raise DecompositionParseError(
                f"dep-gate: dependencies[{i}] references an id outside {sorted(valid)}"
            )
        if source == target:
            raise DecompositionParseError(f"dep-gate: dependencies[{i}] is a self-dependency")
        if (source, target) in seen:
            raise DecompositionParseError(f"dep-gate: duplicate dependency {source}->{target}")
        if (target, source) in seen:
            raise DecompositionParseError(
                f"dep-gate: contradictory dependency {source}->{target}"
            )
        seen.add((source, target))


@dataclass
class DependencyJob:
    """One dependency judgement to ask the model for.

    In batched mode that is a whole query (every pair at once); in pairwise mode it
    is one pair of a query. Everything downstream works on jobs, so the two modes
    differ only in how many jobs a record produces and what is in their prompts.
    """

    query_id: str
    valid_ids: list[str]                    # every sub_question_id in the record
    expected_pair: tuple[str, str] | None    # None in batched mode
    primary: list[dict[str, str]]
    fallback: list[dict[str, str]]

    def label(self) -> str:
        return "all pairs" if self.expected_pair is None else "pair %s->%s" % self.expected_pair


def _dependency_metadata(record: dict[str, Any]) -> str:
    """Metadata is legitimately empty for a bare question; say so rather than
    leaving a dangling 'Metadata:' line the model has to interpret."""
    return record["query_metadata"].strip() or "(none given)"


def _build_batched_job(record: dict[str, Any]) -> DependencyJob:
    ids = [sq["sub_question_id"] for sq in record["sub_questions"]]
    fields = {
        "metadata": _dependency_metadata(record),
        "sub_questions": "\n".join(
            f"{sq['sub_question_id']}. {sq['sub_question_text']}" for sq in record["sub_questions"]
        ),
        "ids": ", ".join(ids),
    }
    return DependencyJob(
        query_id=record["query_id"],
        valid_ids=ids,
        expected_pair=None,
        primary=[
            {"role": "system", "content": DEPENDENCY_SYSTEM_PROMPT},
            {"role": "user", "content": DEPENDENCY_USER_TEMPLATE.format(**fields)},
        ],
        fallback=[{"role": "user", "content": DEPENDENCY_FALLBACK_PROMPT.format(**fields)}],
    )


def _build_pairwise_jobs(record: dict[str, Any]) -> list[DependencyJob]:
    ids = [sq["sub_question_id"] for sq in record["sub_questions"]]
    text_by_id = {sq["sub_question_id"]: sq["sub_question_text"] for sq in record["sub_questions"]}
    jobs: list[DependencyJob] = []
    # combinations() rather than a hardcoded (ids[0], ids[1]): for the current
    # at-most-2 dataset this yields the single pair, and it does not fall over if a
    # future TSV produces a 3-sub-question query.
    for id_a, id_b in combinations(ids, 2):
        fields = {
            "metadata": _dependency_metadata(record),
            "id_a": id_a,
            "text_a": text_by_id[id_a],
            "id_b": id_b,
            "text_b": text_by_id[id_b],
        }
        jobs.append(
            DependencyJob(
                query_id=record["query_id"],
                valid_ids=ids,
                expected_pair=(id_a, id_b),
                primary=[
                    {"role": "system", "content": DEPENDENCY_PAIRWISE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": DEPENDENCY_PAIRWISE_USER_TEMPLATE.format(**fields),
                    },
                ],
                fallback=[
                    {"role": "user", "content": DEPENDENCY_PAIRWISE_FALLBACK_PROMPT.format(**fields)}
                ],
            )
        )
    return jobs


def build_dependency_prompts(
    records: Iterable[dict[str, Any]], pairwise_api: bool
) -> list[DependencyJob]:
    """The one and only branch point between the two PAIRWISE_API modes.

    A record with fewer than two sub-questions has no pair to judge, so it produces
    no job at all and gets `dependencies: []` without costing a generation.
    """
    jobs: list[DependencyJob] = []
    for record in records:
        if len(record["sub_questions"]) < 2:
            continue
        if pairwise_api:
            jobs.extend(_build_pairwise_jobs(record))
        else:
            jobs.append(_build_batched_job(record))
    return jobs


def _join_raw(indices: list[int], raw_by_index: dict[int, str], jobs: list[DependencyJob]) -> str:
    """Flatten a query's raw dependency outputs into one field for the flagged file.

    A batched query has exactly one; a pairwise query has one per pair, so each is
    labelled with the pair it came from.
    """
    parts = [(i, raw_by_index[i]) for i in indices if i in raw_by_index]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0][1]
    return "\n---\n".join(f"[{jobs[i].label()}] {raw}" for i, raw in parts)


def detect_dependencies(
    llm,
    tokenizer,
    records: list[dict[str, Any]],
    batch_size: int,
    pairwise_api: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Attach a validated `dependencies` field to every record, in this same run.

    Reuses the loaded model rather than a second session -- same weights, same
    tokenizer, different prompt. Returns (good, flagged, stats): `good` records have
    gained a `dependencies` key and are ready to write; `flagged` records failed both
    attempts and carry the existing flagged-file shape with
    failure_stage="dependency-detection", their decomposition intact so nothing that
    already succeeded is thrown away.
    """
    jobs = build_dependency_prompts(records, pairwise_api)
    logger.info(
        "dependency detection: %d record(s) -> %d LLM call(s) (%s mode)",
        len(records), len(jobs), "pairwise" if pairwise_api else "batched",
    )

    parsed: dict[int, list[dict[str, str]]] = {}
    reasons: dict[int, str] = {}
    raw_primary: dict[int, str] = {}
    raw_fallback: dict[int, str] = {}

    # --- attempt 1 ----------------------------------------------------------
    outputs = run_inference(
        llm, tokenizer, [job.primary for job in jobs], batch_size,
        max_tokens=DEPENDENCY_MAX_TOKENS,
    )
    retry: list[int] = []
    for i, (job, raw) in enumerate(zip(jobs, outputs)):
        raw_primary[i] = raw
        try:
            parsed[i] = parse_dependencies(raw, job.valid_ids, job.expected_pair)
        except DecompositionParseError as exc:
            reasons[i] = f"primary: {exc}"
            retry.append(i)
    logger.info("dependency pass 1: %d parsed, %d need retry", len(parsed), len(retry))

    # --- attempt 2: fallback prompt for the failures ------------------------
    fallback_ok: set[str] = set()
    if retry:
        outputs = run_inference(
            llm, tokenizer, [jobs[i].fallback for i in retry], batch_size,
            max_tokens=DEPENDENCY_MAX_TOKENS,
        )
        for i, raw in zip(retry, outputs):
            raw_fallback[i] = raw
            job = jobs[i]
            try:
                parsed[i] = parse_dependencies(raw, job.valid_ids, job.expected_pair)
                fallback_ok.add(job.query_id)
                logger.info(
                    "query %s (%s): dependencies recovered via fallback prompt",
                    job.query_id, job.label(),
                )
            except DecompositionParseError as exc:
                reasons[i] = f"{reasons[i]} | fallback: {exc}"

    # --- merge per query, gate, and route -----------------------------------
    jobs_by_query: dict[str, list[int]] = {}
    for i, job in enumerate(jobs):
        jobs_by_query.setdefault(job.query_id, []).append(i)

    good: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for record in records:
        indices = jobs_by_query.get(record["query_id"], [])
        failed = [i for i in indices if i not in parsed]
        dependencies: list[dict[str, str]] | None = None
        reason = ""

        if failed:
            reason = " || ".join(
                f"[{jobs[i].label()}] {reasons.get(i, 'unknown')}" for i in failed
            )
        else:
            # In pairwise mode this concatenates one verdict per pair; in batched
            # mode there is a single verdict. Either way the gate re-checks the
            # merged list, which is where a cross-pair contradiction would surface.
            merged = [dep for i in indices for dep in parsed[i]]
            try:
                validate_dependencies(merged, record["sub_questions"])
                dependencies = merged
            except DecompositionParseError as exc:
                reason = f"dependency final-gate: {exc}"

        if dependencies is not None:
            record["dependencies"] = dependencies
            good.append(record)
            continue

        fallback_ok.discard(record["query_id"])   # it was not, in the end, recovered
        flagged.append(
            {
                "query_id": record["query_id"],
                "comp_query": record["comp_query"],
                "query_metadata": record["query_metadata"],
                "sub_questions": record["sub_questions"],
                # Same key names as a decomposition failure -- failure_stage says
                # which stage these raw outputs came from.
                "first_attempt_raw_output": _join_raw(indices, raw_primary, jobs),
                "fallback_attempt_raw_output": _join_raw(indices, raw_fallback, jobs),
                "failure_stage": "dependency-detection",
                "failure_reason": reason or "unknown",
            }
        )

    stats = {
        "mode": "pairwise" if pairwise_api else "batched",
        "calls": len(jobs),
        "fallback_recovered": len(fallback_ok),
        "flagged": len(flagged),
        "with_dependency": sum(1 for record in good if record["dependencies"]),
    }
    return good, flagged, stats


# --------------------------------------------------------------------------------
# 2. MODEL / INFERENCE
# --------------------------------------------------------------------------------


def _gpu_memory_mib() -> list[tuple[int, int]]:
    """(free, total) MiB per visible GPU, or [] if nvidia-smi is unavailable."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        free, total = (int(x.strip()) for x in line.split(","))
        rows.append((free, total))
    return rows


def _weights_gib(model_path: str) -> float:
    from glob import glob

    files = glob(str(Path(model_path) / "*.safetensors")) or glob(
        str(Path(model_path) / "*.bin")
    )
    return sum(Path(f).stat().st_size for f in files) / 1024**3


def resolve_gpu_memory_utilization(
    model_path: str,
    tensor_parallel_size: int,
    requested: float | str,
    headroom_gib: float = 4.0,
    use_fraction_of_free: float = 0.90,
) -> float:
    """Size the vLLM allocation to what is actually free right now.

    vLLM takes `gpu_memory_utilization` as a fraction of TOTAL memory and
    pre-allocates it at startup -- there is no grow/shrink. On a shared box where
    another job already holds most of a card, the default 0.90 simply fails. This
    computes the largest fraction that fits beside the current occupant.

    It also refuses to start when the leftover headroom (budget minus weights)
    is too small to hold a useful KV cache: vLLM would load and then batch one
    request at a time, which is slower than not using vLLM at all, and reads as a
    mysterious hang rather than an error.
    """
    if requested != "auto":
        return float(requested)

    gpus = _gpu_memory_mib()
    if not gpus:
        logger.warning("could not read GPU memory; falling back to 0.90")
        return 0.90

    gpus = gpus[:tensor_parallel_size]
    free_gib = min(f for f, _ in gpus) / 1024
    total_gib = min(t for _, t in gpus) / 1024
    budget_gib = free_gib * use_fraction_of_free
    util = round(budget_gib / total_gib, 3)

    weights_per_gpu = _weights_gib(model_path) / max(1, tensor_parallel_size)
    spare = budget_gib - weights_per_gpu
    logger.info(
        "GPU sizing: %.1f GiB free/GPU -> budget %.1f GiB (util=%.3f); "
        "weights %.1f GiB/GPU; %.1f GiB left for KV cache",
        free_gib, budget_gib, util, weights_per_gpu, spare,
    )
    if spare < headroom_gib:
        raise SystemExit(
            f"\nNot enough free GPU memory to run this model usefully.\n"
            f"  free/GPU        : {free_gib:.1f} GiB\n"
            f"  usable budget   : {budget_gib:.1f} GiB\n"
            f"  weights need    : {weights_per_gpu:.1f} GiB/GPU "
            f"(tensor_parallel_size={tensor_parallel_size})\n"
            f"  left for KV     : {spare:.1f} GiB  (need >= {headroom_gib} GiB to batch)\n\n"
            f"Check `nvidia-smi` for what else is running and wait for it, or pass an\n"
            f"explicit --gpu-memory-utilization to override this check.\n"
        )
    return util


def resolve_quantization(model_path: str, override: str | None) -> str | None:
    """Read quantization off the checkpoint's config.json.

    vLLM can usually infer this, but an explicit value keeps a silent fallback to
    an unquantized load (which would OOM on a 72B) from looking like a vLLM bug.
    """
    if override:
        return override
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        logger.warning("no config.json at %s; letting vLLM auto-detect quantization", model_path)
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    method = (config.get("quantization_config") or {}).get("quant_method")
    if method:
        logger.info("detected quantization from config.json: %s", method)
    return method


def build_primary_prompts(topics: Iterable[Topic]) -> list[list[dict[str, str]]]:
    return [
        [
            {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
            {"role": "user", "content": DECOMPOSITION_USER_TEMPLATE.format(query_text=t.query_text)},
        ]
        for t in topics
    ]


def build_fallback_prompts(topics: Iterable[Topic]) -> list[list[dict[str, str]]]:
    return [
        [{"role": "user", "content": DECOMPOSITION_FALLBACK_PROMPT.format(query_text=t.query_text)}]
        for t in topics
    ]


def _render(tokenizer, conversation: list[dict[str, str]]) -> str:
    """Apply the model's own chat template.

    gemma-4-31B-it on this machine ships no chat_template, so fall back to a plain
    role-prefixed rendering rather than crashing -- but say so loudly, because a
    mis-rendered prompt degrades quality silently.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
    logger.warning(
        "tokenizer has no chat_template; using CHAT_TEMPLATE_FALLBACK. "
        "Verify output format on a few topics before trusting a full run."
    )
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=CHAT_TEMPLATE_FALLBACK,
    )


def run_inference(
    llm,
    tokenizer,
    conversations: list[list[dict[str, str]]],
    batch_size: int,
    max_tokens: int = MAX_TOKENS,
) -> list[str]:
    """Batch every prompt through vLLM's offline API.

    One generate() call over the whole list is the point of using vLLM here: the
    scheduler batches continuously, so per-prompt calls would waste most of the GPU.
    `max_tokens` defaults to the decomposition budget; the dependency stage passes
    its own, much smaller one.
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


# --------------------------------------------------------------------------------
# 5. OUTPUT
# --------------------------------------------------------------------------------


def write_output(path: str | Path, records: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_flagged(path: str | Path, flagged: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in flagged:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_error_log(path: str | Path, flagged: list[dict[str, Any]]) -> None:
    """One log for the whole script. failure_stage separates a decomposition failure
    ("parser" / "final-gate") from a dependency one ("dependency-detection"), which
    matters when scanning: the latter still has a usable decomposition behind it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if not flagged:
            fh.write("No queries failed decomposition or dependency detection.\n")
            return
        by_stage = Counter(record.get("failure_stage", "?") for record in flagged)
        breakdown = ", ".join(f"{count} {stage}" for stage, count in sorted(by_stage.items()))
        fh.write(f"{len(flagged)} query/queries failed ({breakdown})\n\n")
        for record in flagged:
            fh.write(
                f"{record['query_id']}\t{record.get('failure_stage','?')}\t"
                f"{record.get('failure_reason','')}\n"
            )


# --------------------------------------------------------------------------------
# ORCHESTRATION
# --------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topics", default=TEST_TOPIC_PATH)
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
    parser.add_argument(
        "--pairwise-api", dest="pairwise_api", action=argparse.BooleanOptionalAction,
        default=PAIRWISE_API,
        help="judge dependencies one sub-question pair per LLM call instead of all "
             "of a query's pairs in one call (identical while queries have <= 2 "
             "sub-questions)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only process the first N topics")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    topics = load_topics(args.topics)
    if args.limit:
        topics = topics[: args.limit]
    if not topics:
        logger.error("no topics loaded from %s", args.topics)
        raise SystemExit(1)
    logger.info("loaded %d topics from %s", len(topics), args.topics)

    from transformers import AutoTokenizer
    from vllm import LLM

    gpu_util = resolve_gpu_memory_utilization(
        args.model, args.tensor_parallel_size, args.gpu_memory_utilization
    )
    quantization = resolve_quantization(args.model, args.quantization)
    logger.info("loading %s (quantization=%s, tp=%d)", args.model, quantization, args.tensor_parallel_size)
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

    # --- pass 1 -------------------------------------------------------------
    raw_first = run_inference(llm, tokenizer, build_primary_prompts(topics), args.batch_size)

    records: dict[str, dict[str, Any]] = {}
    first_reason: dict[str, str] = {}
    retry: list[Topic] = []
    for topic, raw in zip(topics, raw_first):
        try:
            records[topic.query_id] = parse_decomposition(raw, topic.query_id, topic.query_text)
        except DecompositionParseError as exc:
            first_reason[topic.query_id] = str(exc)
            retry.append(topic)
    logger.info("pass 1: %d parsed, %d need retry", len(records), len(retry))

    # --- pass 2: fallback prompt for first-pass failures ---------------------
    raw_first_by_id = dict(zip((t.query_id for t in topics), raw_first))
    raw_fallback_by_id: dict[str, str] = {}
    fallback_ok: list[str] = []
    if retry:
        raw_second = run_inference(llm, tokenizer, build_fallback_prompts(retry), args.batch_size)
        raw_fallback_by_id = dict(zip((t.query_id for t in retry), raw_second))
        for topic, raw in zip(retry, raw_second):
            try:
                records[topic.query_id] = parse_decomposition(raw, topic.query_id, topic.query_text)
                fallback_ok.append(topic.query_id)
                logger.info("query %s recovered via fallback prompt", topic.query_id)
            except DecompositionParseError as exc:
                first_reason[topic.query_id] = (
                    f"primary: {first_reason.get(topic.query_id, '?')} | fallback: {exc}"
                )

    # --- final gate ---------------------------------------------------------
    passed: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for topic in topics:
        record = records.get(topic.query_id)
        stage = "parser"
        if record is not None:
            try:
                validate_final_record(record)
                passed.append(record)
                continue
            except DecompositionParseError as exc:
                stage = "final-gate"
                first_reason[topic.query_id] = str(exc)
        flagged.append(
            {
                "query_id": topic.query_id,
                "comp_query": topic.query_text,
                "first_attempt_raw_output": raw_first_by_id.get(topic.query_id, ""),
                "fallback_attempt_raw_output": raw_fallback_by_id.get(topic.query_id, ""),
                "failure_stage": stage,
                "failure_reason": first_reason.get(topic.query_id, "unknown"),
            }
        )

    # --- step 1b: dependency detection, same run, same model session ---------
    # Every record that cleared the gate above is judged here and gains its
    # `dependencies` field before anything is written, so the file below is complete
    # on its first write. Records that fail both attempts move to `flagged` instead.
    good, dependency_flagged, dep_stats = detect_dependencies(
        llm, tokenizer, passed, args.batch_size, args.pairwise_api
    )
    flagged.extend(dependency_flagged)
    # Dependency failures are appended after the decomposition ones; re-sort so the
    # flagged file still reads in topic order.
    topic_order = {topic.query_id: i for i, topic in enumerate(topics)}
    flagged.sort(key=lambda record: topic_order.get(record["query_id"], len(topic_order)))

    write_output(args.output, good)
    write_flagged(args.flagged, flagged)
    write_error_log(args.error_log, flagged)

    # Invariant: every input id lands in exactly one output file.
    accounted = {r["query_id"] for r in good} | {r["query_id"] for r in flagged}
    missing = {t.query_id for t in topics} - accounted
    if missing:
        logger.error("BUG: %d query_id(s) unaccounted for: %s", len(missing), sorted(missing)[:10])

    # Decomposition counts are taken from `passed`, not `good`: a record can clear
    # decomposition and still be diverted by the dependency stage.
    passed_ids = {record["query_id"] for record in passed}
    recovered = sum(1 for query_id in fallback_ok if query_id in passed_ids)

    print()
    print(f"  topics read                : {len(topics)}")
    print(f"  succeeded (first attempt)  : {len(passed) - recovered}")
    print(f"  succeeded (after fallback) : {recovered}")
    print(f"  dependency mode            : {dep_stats['mode']} ({dep_stats['calls']} LLM calls)")
    print(f"  dependency fallbacks used  : {dep_stats['fallback_recovered']}")
    print(f"  records with a dependency  : {dep_stats['with_dependency']}")
    print(f"  flagged (dependency stage) : {dep_stats['flagged']}")
    print(f"  flagged / unresolved       : {len(flagged)}")
    print(f"  written to main output     : {len(good)}")
    print(f"  main output                : {args.output}")
    if flagged:
        print(f"  flagged output             : {args.flagged}")
        print(f"  error log                  : {args.error_log}")
    if missing:
        print(f"  *** UNACCOUNTED query_ids  : {len(missing)} ***")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
