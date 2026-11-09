#!/usr/bin/env python3
"""Preprocessing step 2: split each step-1 sub-question into Atomic Information Needs.

A step-1 sub-question is one *question*, but it is often not one *retrieval objective*.
"What should we prioritize across nursing school pipeline partnerships, hiring and
promotion practices, reporting systems for racism, leadership accountability, and
curriculum/continuing education?" asks a single thing of a reader and five things of a
retriever.

That is a problem specifically for lexical retrieval. BM25 scores a document by the
query terms it contains, so when one query spans several unrelated concepts the
evidence fragments: a document that covers pipeline partnerships thoroughly and says
nothing about promotion practices, reporting systems, or curriculum is missing most of
the query's terms and scores far below its actual usefulness. Issuing one query per
concept removes that competition, and recall for each concept is maximised on its own
terms.

An ATOMIC INFORMATION NEED (AIN) is a semantically indivisible retrieval objective:
evidence about ONE topic under ONE information-seeking intent. It carries one primary
topic, one objective, never needs two unrelated concepts at once, and can be issued to
a stateless retriever on its own.

    preprocessed_queries_step1.jsonl      (query_id, sub_questions, ...)
        -> one vLLM call per sub-question
        -> <ains>/<ain id="N"> XML tags
        -> deterministic regex parse + validation
        -> preprocessed_queries_step2.jsonl

Higher-order reasoning is deliberately dropped here. Prioritising, ranking, comparing,
and weighing tradeoffs are performed later, over the retrieved evidence; this stage
only separates the topics to collect it for. Dependencies between AINs are likewise
out of scope -- retrieval is stateless, so every AIN is treated as independent.

Every step-1 record is carried through unchanged (including `dependencies`) with one
new `ain` field appended. Anything whose AINs fail to parse twice goes to the flagged
file instead, never dropped: every input query_id appears in exactly one of the two
outputs.

QUICKSTART
    # Same dedicated vLLM env as step 1, and the same model -- see that script's
    # header for why torch cannot be shared with the main pipeline env.
    conda activate trec-rag-2026-vllm
    python scripts/decompose/preprocess_step2_decompose.py

    # or override the constants below:
    python scripts/decompose/preprocess_step2_decompose.py \
        --input output_decomp/preprocessed_queries_step1.jsonl \
        --output output_decomp/preprocessed_queries_step2.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# The vLLM loading rules were worked out and validated against this box in step 1
# (auto-sizing the allocation to free VRAM, reading quantization off config.json,
# falling back to a plain chat template for models that ship none). Importing them
# keeps one copy: a fix there is a fix here. Everything step 2 decides for itself --
# prompts, parser, gate, sampling -- stays local, so the two scripts share
# infrastructure without sharing behaviour.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_step1_decompose import (  # noqa: E402
    _render,
    resolve_gpu_memory_utilization,
    resolve_quantization,
)

# --------------------------------------------------------------------------------
# 1. CONFIGURATION -- edit these, or override any of them on the command line.
# --------------------------------------------------------------------------------

INPUT_PATH = "output_decomp/preprocessed_queries_step1.jsonl"

# Same checkpoint as step 1: the directory containing config.json, which for an
# HF-cache layout is the snapshots/<hash>/ subdirectory.
LLM_PATH = (
    "/media/20TB/shared/models/qwen/Qwen2.5-72B-Instruct-AWQ/"
    "models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/"
    "698703eae6604af048a3d2f509995dc302088217"
)

OUTPUT_PATH = "output_decomp/preprocessed_queries_step2.jsonl"
FLAGGED_PATH = "output_decomp/preprocessed_queries_step2_flagged.jsonl"
ERROR_LOG_PATH = "output_decomp/preprocessed_queries_step2_errors.log"

# Generation / runtime knobs.
QUANTIZATION: str | None = None   # None = auto-detect from config.json
TENSOR_PARALLEL_SIZE = 2          # 2x RTX 6000 Ada; a 72B AWQ checkpoint will not fit on one
GPU_MEMORY_UTILIZATION = "auto"   # "auto" = size to whatever VRAM is free now
MAX_MODEL_LEN = 8192              # one sub-question plus its context is short
MAX_TOKENS = 768                  # ~25 tokens per AIN; generous even for a long list,
                                  # while still cutting off a runaway generation
TEMPERATURE = 0.0                 # deterministic: this is extraction, not generation
TOP_P = 1.0
SEED = 0
BATCH_SIZE = 0                    # 0 = submit every prompt in one generate() call

# A runaway guard, not a style rule: it exists to catch a generation that has stopped
# splitting and started enumerating (one atomic need per word, or a list that never
# terminates), not to cap how wide a genuinely wide question may be.
#
# Set from the measured distribution over the 119-topic set. Most sub-questions land
# at 1-8. The tail is real: "compare the real costs, safety, cancellation risk, space,
# location, and day-to-day practicality of these options" crosses six aspects with
# four options and legitimately yields 24. At 12 those records were rejected outright
# even though every tag in them was well formed, so the cap sat below the data rather
# than above the failure mode. Lowering it again is a deliberate choice to trade
# per-concept recall -- the whole point of this stage -- for a smaller retrieval
# budget downstream.
MAX_AINS_PER_SUB_QUESTION = 30

logger = logging.getLogger("preprocess_step2")


# --------------------------------------------------------------------------------
# 2. PROMPTS -- isolated so they can be read and iterated on independently.
# --------------------------------------------------------------------------------

# The two failure modes pull in opposite directions. Under-splitting leaves the
# lexical competition this stage exists to remove; over-splitting invents topics the
# asker never raised and pads the retrieval budget with queries that match nothing.
# The prompt therefore states the count is variable in BOTH directions -- "exactly one
# AIN" is a correct answer -- and the examples cover both a six-way split and a
# single-topic question.
AIN_SYSTEM_PROMPT = """\
You split one information-seeking sub-question into ATOMIC INFORMATION NEEDS. You \
separate retrieval topics; you never answer anything.

An ATOMIC INFORMATION NEED is a semantically indivisible retrieval objective: it seeks \
evidence about ONE topic under ONE information-seeking intent. Each one must:

- cover exactly one primary topic;
- carry exactly one retrieval objective;
- never require evidence about two unrelated concepts at the same time;
- stand on its own as a search query, understandable without the other atomic needs \
and without the original sub-question.

Why this matters: each atomic need is sent to a keyword search engine by itself. When \
one query mixes several unrelated concepts, a document that covers only one of them \
matches few of the query's words and is lost, however useful it is. One atomic need \
per concept removes that competition.

Rules:

- The number of atomic needs is VARIABLE and unknown in advance. It may be 1, 2, 5, or \
more. Produce exactly as many as the sub-question genuinely contains. A sub-question \
that already asks about a single topic yields exactly ONE atomic need: do not invent \
extra ones to pad the list, and never merge two distinct topics to shorten it.
- When the sub-question lists items, categories, or options, each one becomes its own \
atomic need. If a listed item itself bundles two topics ("hiring and promotion \
practices"), split it further.
- SPLIT ONLY WHAT THE SUB-QUESTION ITSELF NAMES. A broad request that names one thing \
("help me build a retirement plan") is ONE atomic need, however much it implies. Do \
not break a broad topic into the sub-topics you believe it covers, and do not add the \
aspects an expert would want to consider. That is expansion, not splitting, and it \
invents information needs the asker never expressed.
- If the sub-question names no topic of its own - it only states a requirement about \
how something else should be answered ("use authoritative sources", "keep it short") - \
emit exactly ONE atomic need that names that requirement as literally as the wording \
allows. Never restate the topics of a different sub-question.
- LEAVE OUT the higher-order reasoning. Prioritising, ranking, comparing, weighing \
tradeoffs, and choosing between options are done later, over the evidence retrieved \
here. "What should we prioritise across A, B, and C?" becomes evidence about A, \
evidence about B, and evidence about C - there is no atomic need about prioritising, \
and none about the tradeoffs between them. Keep everything that evidence can be found \
about, including outcome measures, risks, and common pitfalls; those are topics, not \
reasoning.
- Use the BACKGROUND CONTEXT to make each atomic need self-contained: fold in the \
domain, population, or setting it needs to be interpretable alone ("in nursing", "for \
small businesses"). Never write "this", "the plan", "they", or anything else that \
points back at the sub-question.
- Do NOT introduce topics the sub-question does not raise. Context may be added to \
disambiguate; new subject matter may not.
- Write each atomic need as a short noun phrase beginning "Evidence regarding ". Never \
a question, never a sentence of explanation.

Output format - emit ONLY these tags, nothing else. No preamble, no trailing \
commentary, no markdown code fences, no explanation:

<ains>
<ain id="1">Evidence regarding ...</ain>
<ain id="2">Evidence regarding ...</ain>
</ains>

The <ains> wrapper is always present and always holds at least one <ain>. ids start at \
1 and increase by 1 with no gaps, in the order the topics appear in the sub-question.

Example A - a list of five categories, one of which bundles two topics, and a \
prioritisation frame that is dropped:

Background context: I'm on a hospital nursing DEI council that has to recommend a \
three-year plan with limited money.
Sub-question: What should we prioritize across nursing school pipeline partnerships, \
hiring and promotion practices, reporting systems for racism, leadership \
accountability, and curriculum/continuing education?

<ains>
<ain id="1">Evidence regarding nursing school pipeline partnerships for workforce \
diversity.</ain>
<ain id="2">Evidence regarding equitable hiring practices in nursing.</ain>
<ain id="3">Evidence regarding equitable promotion practices in nursing.</ain>
<ain id="4">Evidence regarding workplace racism reporting systems.</ain>
<ain id="5">Evidence regarding leadership accountability for nursing DEI.</ain>
<ain id="6">Evidence regarding diversity-related nursing curriculum and continuing \
education.</ain>
</ains>

Example B - already a single topic, so exactly one atomic need:

Background context: (none given)
Sub-question: What causes the northern lights?

<ains>
<ain id="1">Evidence regarding the physical causes of the aurora borealis.</ain>
</ains>

Example C - three coordinated asks in one sentence. The tradeoffs framing is \
reasoning and is dropped; measurement and pitfalls are topics and are kept, each made \
self-contained from the background context:

Background context: I manage a 12-person support team and we are moving off shared \
email onto a ticketing tool next quarter.
Sub-question: How should we think about the tradeoffs, what outcomes should we \
measure, and what mistakes should we avoid?

<ains>
<ain id="1">Evidence regarding outcome metrics for customer support ticketing \
systems.</ain>
<ain id="2">Evidence regarding common pitfalls when migrating a support team to a \
ticketing tool.</ain>
</ains>

Example D - one broad request that names a single thing. It implies budgeting, \
interest rates, and repayment schedules, but it does not ask about them, so it stays \
ONE atomic need:

Background context: I'm 22 and just started my first job.
Sub-question: Can you help me put together a plan for paying off my student loans?

<ains>
<ain id="1">Evidence regarding student loan repayment strategies for early-career \
borrowers.</ain>
</ains>

Example E - a sub-question that names no topic of its own, only a requirement on how \
another answer should be produced. One atomic need, taken literally, and none of the \
other sub-question's topics:

Background context: I'm 22 and just started my first job.
Sub-question: This plan should be based on academic and authoritative sources.

<ains>
<ain id="1">Evidence regarding authoritative and academic sources on personal finance \
planning.</ain>
</ains>
"""

AIN_USER_TEMPLATE = """\
Background context: {metadata}
Sub-question: {sub_question}"""

# Same relationship to AIN_SYSTEM_PROMPT that DECOMPOSITION_FALLBACK_PROMPT has to
# DECOMPOSITION_SYSTEM_PROMPT in step 1: when the first attempt fails the parser it is
# almost always shape, not judgement, so the retry drops the conceptual framing and
# makes the output format the whole message. Re-sending the identical prompt to the
# same weights at temperature 0 would reproduce the same failure.
AIN_FALLBACK_PROMPT = """\
Rewrite the sub-question below as a list of atomic search topics in XML. Output the \
tags and NOTHING else.

Required shape, exactly:

<ains>
<ain id="1">Evidence regarding the first topic.</ain>
<ain id="2">Evidence regarding the second topic, if there is one.</ain>
</ains>

Hard rules:
- Start your reply with the character < and end it with the character >.
- No code fences. No "Here is". No notes before or after the tags.
- Exactly one <ains> wrapper, holding at least one <ain>. ids are 1, 2, 3, ... in \
order with no gaps.
- One topic per <ain>, one <ain> per topic. Split every item in a list, including \
items joined by "and".
- Split only what the sub-question names. Do not break one broad topic into the \
sub-topics you think it implies, and do not add aspects it does not mention.
- Each <ain> is a short noun phrase beginning "Evidence regarding ". Never a question.
- Each <ain> must work alone as a search query: no "this", "the plan", "they". Add the \
setting from the background context when it is needed to make sense.
- Leave out prioritising, ranking, comparing, and weighing tradeoffs. Keep only topics \
that evidence can be found about.
- If the sub-question covers a single topic, emit exactly one <ain>.

Background context: {metadata}
Sub-question: {sub_question}
"""


# --------------------------------------------------------------------------------
# 3. INPUT
# --------------------------------------------------------------------------------


class AINParseError(Exception):
    """Raised with a human-readable reason naming the validation rule that failed."""


def load_step1_records(path: str | Path) -> list[dict[str, Any]]:
    """Read step 1's JSONL output.

    Machine-generated input, so a malformed line is a bug upstream rather than
    messiness to absorb -- but it must not take the run down either. Bad lines are
    warned about and skipped, and the count is reported at the end, so a truncated
    or half-written step-1 file is visible instead of silently shrinking the run.

    Only the fields step 2 depends on are checked here. Everything else in the
    record (`dependencies`, and whatever step 1 grows next) is carried through
    untouched.
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
            if not isinstance(record, dict) or not _has_step1_shape(record):
                logger.warning("%s:%d: skipping record with unexpected shape", path, line_no)
                continue
            query_id = record["query_id"]
            if query_id in seen:
                logger.warning("%s:%d: duplicate query_id %r, skipping", path, line_no, query_id)
                continue
            seen.add(query_id)
            records.append(record)
    return records


def _has_step1_shape(record: dict[str, Any]) -> bool:
    if not all(isinstance(record.get(k), str) and record[k] for k in ("query_id", "comp_query")):
        return False
    if not isinstance(record.get("query_metadata"), str):
        return False
    sub_questions = record.get("sub_questions")
    if not isinstance(sub_questions, list) or not sub_questions:
        return False
    return all(
        isinstance(sq, dict)
        and isinstance(sq.get("sub_question_id"), str)
        and isinstance(sq.get("sub_question_text"), str)
        and sq["sub_question_text"].strip()
        for sq in sub_questions
    )


# --------------------------------------------------------------------------------
# 4. PARSING -- deterministic, all-or-nothing.
# --------------------------------------------------------------------------------

_AINS_BLOCK_RE = re.compile(r"<ains\s*>(.*?)</ains\s*>", re.DOTALL | re.IGNORECASE)
# `<ain` requires whitespace and an id= before the closing bracket, so the <ains>
# wrapper itself can never match this.
_AIN_RE = re.compile(
    r"""<ain\s+id\s*=\s*["'](\d+)["']\s*>(.*?)</ain\s*>""", re.DOTALL | re.IGNORECASE
)
# Models sometimes wrap correct tags in ``` fences; that is a formatting artefact
# rather than a failed split, so it is stripped before validation.
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def parse_ains(raw_output: str) -> list[dict[str, Any]]:
    """Parse the tagged LLM output into the `ain_array` schema.

    Raises AINParseError if ANY validation rule fails. Deliberately no partial
    extraction: a half-parsed list is not a visibly broken record, it is a plausible
    shorter one, and the topics it dropped are exactly the ones that would never be
    retrieved for. A retry is cheap; a silently narrowed information need is not.
    """
    if raw_output is None:
        raise AINParseError("empty generation")

    text = _FENCE_RE.sub("", raw_output.strip())
    if not text:
        raise AINParseError("empty generation")

    # Rule 1: exactly one <ains> wrapper.
    blocks = list(_AINS_BLOCK_RE.finditer(text))
    if len(blocks) != 1:
        raise AINParseError(f"rule1: expected exactly 1 <ains> block, found {len(blocks)}")
    block = blocks[0]

    # Rule 4a: nothing but whitespace outside the wrapper.
    outside = (text[: block.start()] + text[block.end() :]).strip()
    if outside:
        raise AINParseError(f"rule4: stray text outside <ains>: {outside[:80]!r}")

    inner = block.group(1)
    matches = list(_AIN_RE.finditer(inner))

    # Rule 2: at least one <ain>.
    if not matches:
        raise AINParseError("rule2: no <ain> blocks found")

    # Rule 3: ids are exactly 1..N in order, no gaps or duplicates.
    ids = [m.group(1) for m in matches]
    if ids != [str(i) for i in range(1, len(ids) + 1)]:
        raise AINParseError(f"rule3: ain ids are {ids}, expected 1..N in order")

    # Rule 4b: inside the wrapper, only <ain> tags and whitespace. Without this, a
    # numbered prose list between the tags would parse as a shorter valid answer.
    remainder, cursor = [], 0
    for match in matches:
        remainder.append(inner[cursor : match.start()])
        cursor = match.end()
    remainder.append(inner[cursor:])
    stray = "".join(remainder).strip()
    if stray:
        raise AINParseError(f"rule4: stray text inside <ains>: {stray[:80]!r}")

    texts = [m.group(2).strip() for m in matches]

    # Rule 2 (cont.): an empty <ain> is a retrieval query that matches everything.
    if any(not t for t in texts):
        raise AINParseError("rule2: an <ain> block was empty")

    # Rule 5: no raw '<' inside content, which would mean a truncated/unclosed tag.
    for ain_text in texts:
        if "<" in ain_text:
            raise AINParseError(f"rule5: '<' inside tag content: {ain_text[:80]!r}")

    # Rule 6: no duplicate topics. Two identical AINs are two identical retrievals,
    # and their presence means the model padded rather than split.
    lowered = [t.casefold() for t in texts]
    if len(set(lowered)) != len(lowered):
        duplicate = next(t for t in lowered if lowered.count(t) > 1)
        raise AINParseError(f"rule6: duplicate <ain> text: {duplicate[:80]!r}")

    # Rule 7: a degenerate over-split, caught here rather than shipped as dozens of
    # near-empty retrieval queries.
    if len(texts) > MAX_AINS_PER_SUB_QUESTION:
        raise AINParseError(
            f"rule7: {len(texts)} <ain> blocks exceeds MAX_AINS_PER_SUB_QUESTION="
            f"{MAX_AINS_PER_SUB_QUESTION}"
        )

    return [{"ain_id": i, "ain_text": ain_text} for i, ain_text in enumerate(texts, start=1)]


# --------------------------------------------------------------------------------
# 4b. FINAL PRE-WRITE GATE -- independent of the parser above.
# --------------------------------------------------------------------------------

_REQUIRED_KEYS = ("query_id", "comp_query", "query_metadata", "sub_questions", "ain")


def validate_final_record(record: Any) -> None:
    """Last line of defence before serialization.

    Intentionally re-derives every invariant rather than trusting the parser, so a
    future edit to parse_ains still cannot put a malformed record in the output file.
    Extra keys are allowed on purpose -- `dependencies` and anything else step 1 adds
    later rides through this stage untouched -- but the `ain` block must line up with
    `sub_questions` exactly: same count, same ids, same order.

    Raises AINParseError describing the first failure.
    """
    if not isinstance(record, dict):
        raise AINParseError("final-gate: record is not a dict")
    missing = [key for key in _REQUIRED_KEYS if key not in record]
    if missing:
        raise AINParseError(f"final-gate: missing key(s) {missing}")
    for key in ("query_id", "comp_query", "query_metadata"):
        if not isinstance(record[key], str):
            raise AINParseError(f"final-gate: {key!r} is not a string")
    if not record["query_id"]:
        raise AINParseError("final-gate: empty query_id")
    if not record["comp_query"]:
        raise AINParseError("final-gate: empty comp_query")

    sub_questions = record["sub_questions"]
    if not isinstance(sub_questions, list) or not sub_questions:
        raise AINParseError("final-gate: sub_questions must be a non-empty list")

    ain = record["ain"]
    if not isinstance(ain, list):
        raise AINParseError("final-gate: ain must be a list")
    if len(ain) != len(sub_questions):
        raise AINParseError(
            f"final-gate: ain has {len(ain)} entries for {len(sub_questions)} sub_questions"
        )

    for i, (entry, sub_question) in enumerate(zip(ain, sub_questions)):
        if not isinstance(sub_question, dict) or not isinstance(
            sub_question.get("sub_question_id"), str
        ):
            raise AINParseError(f"final-gate: sub_questions[{i}] is malformed")
        if not isinstance(entry, dict) or set(entry) != {"sub_question_id", "ain_array"}:
            raise AINParseError(f"final-gate: ain[{i}] keys are wrong")
        if entry["sub_question_id"] != sub_question["sub_question_id"]:
            raise AINParseError(
                f"final-gate: ain[{i}] is for sub_question "
                f"{entry['sub_question_id']!r}, expected {sub_question['sub_question_id']!r}"
            )

        ain_array = entry["ain_array"]
        if not isinstance(ain_array, list) or not ain_array:
            raise AINParseError(f"final-gate: ain[{i}] ain_array must be a non-empty list")
        seen: set[str] = set()
        for j, item in enumerate(ain_array, start=1):
            if not isinstance(item, dict) or set(item) != {"ain_id", "ain_text"}:
                raise AINParseError(f"final-gate: ain[{i}].ain_array[{j-1}] has wrong keys")
            # ain_id is an int here while sub_question_id is a string: that is the
            # agreed downstream schema, so the gate enforces the types separately
            # rather than normalising one to the other.
            if not isinstance(item["ain_id"], int) or isinstance(item["ain_id"], bool):
                raise AINParseError(f"final-gate: ain[{i}].ain_array[{j-1}] ain_id is not an int")
            if item["ain_id"] != j:
                raise AINParseError(
                    f"final-gate: ain[{i}].ain_array ain_id {item['ain_id']!r} at position {j}"
                )
            if not isinstance(item["ain_text"], str) or not item["ain_text"].strip():
                raise AINParseError(f"final-gate: ain[{i}].ain_array[{j-1}] text is empty")
            key = item["ain_text"].casefold()
            if key in seen:
                raise AINParseError(f"final-gate: ain[{i}].ain_array has a duplicate text")
            seen.add(key)


# --------------------------------------------------------------------------------
# 5. MODEL / INFERENCE
# --------------------------------------------------------------------------------


@dataclass
class AINJob:
    """One AIN extraction: a single sub-question of a single query.

    Per sub-question rather than per query, because that is the unit the definition
    is about -- and it keeps ain ids restarting at 1 in each sub-question's array
    without any post-hoc renumbering.
    """

    query_id: str
    sub_question_id: str
    primary: list[dict[str, str]]
    fallback: list[dict[str, str]]

    def label(self) -> str:
        return f"sub-question {self.sub_question_id}"


def _context(record: dict[str, Any]) -> str:
    """Metadata is legitimately empty for a bare question (16 of 119 topics); say so
    rather than leaving a dangling 'Background context:' line to interpret."""
    return record["query_metadata"].strip() or "(none given)"


def build_ain_prompts(records: Iterable[dict[str, Any]]) -> list[AINJob]:
    jobs: list[AINJob] = []
    for record in records:
        for sub_question in record["sub_questions"]:
            fields = {
                "metadata": _context(record),
                "sub_question": sub_question["sub_question_text"].strip(),
            }
            jobs.append(
                AINJob(
                    query_id=record["query_id"],
                    sub_question_id=sub_question["sub_question_id"],
                    primary=[
                        {"role": "system", "content": AIN_SYSTEM_PROMPT},
                        {"role": "user", "content": AIN_USER_TEMPLATE.format(**fields)},
                    ],
                    fallback=[
                        {"role": "user", "content": AIN_FALLBACK_PROMPT.format(**fields)}
                    ],
                )
            )
    return jobs


def run_inference(
    llm, tokenizer, conversations: list[list[dict[str, str]]], batch_size: int
) -> list[str]:
    """Batch every prompt through vLLM's offline API.

    One generate() call over the whole list is the point of using vLLM here: the
    scheduler batches continuously, so per-prompt calls would waste most of the GPU.
    Sampling params are step 2's own rather than imported, so retuning this stage
    cannot silently retune step 1.
    """
    from vllm import SamplingParams

    if not conversations:
        return []
    sampling = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, seed=SEED
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
# 6. OUTPUT
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
            fh.write("No queries failed AIN extraction.\n")
            return
        by_stage = Counter(record.get("failure_stage", "?") for record in flagged)
        breakdown = ", ".join(f"{count} {stage}" for stage, count in sorted(by_stage.items()))
        fh.write(f"{len(flagged)} query/queries failed AIN extraction ({breakdown})\n\n")
        for record in flagged:
            fh.write(
                f"{record['query_id']}\t{record.get('failure_stage','?')}\t"
                f"{record.get('failure_reason','')}\n"
            )


# --------------------------------------------------------------------------------
# ORCHESTRATION
# --------------------------------------------------------------------------------


def _join_raw(indices: list[int], raw_by_index: dict[int, str], jobs: list[AINJob]) -> str:
    """Flatten a query's raw outputs into one field for the flagged file, labelled by
    sub-question so a multi-sub-question failure can still be read."""
    parts = [(i, raw_by_index[i]) for i in indices if i in raw_by_index]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0][1]
    return "\n---\n".join(f"[{jobs[i].label()}] {raw}" for i, raw in parts)


def extract_ains(
    llm, tokenizer, records: list[dict[str, Any]], batch_size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run both attempts over every sub-question and assemble the `ain` field.

    Kept out of main() so the whole stage -- retry, per-record assembly, gate,
    flagging -- can be exercised without a GPU. Returns (good, flagged, stats).
    """
    jobs = build_ain_prompts(records)
    logger.info("AIN extraction: %d record(s) -> %d LLM call(s)", len(records), len(jobs))

    parsed: dict[int, list[dict[str, Any]]] = {}
    reasons: dict[int, str] = {}
    raw_primary: dict[int, str] = {}
    raw_fallback: dict[int, str] = {}

    # --- attempt 1 ----------------------------------------------------------
    outputs = run_inference(llm, tokenizer, [job.primary for job in jobs], batch_size)
    retry: list[int] = []
    for i, raw in enumerate(outputs):
        raw_primary[i] = raw
        try:
            parsed[i] = parse_ains(raw)
        except AINParseError as exc:
            reasons[i] = f"primary: {exc}"
            retry.append(i)
    logger.info("pass 1: %d parsed, %d need retry", len(parsed), len(retry))

    # --- attempt 2: fallback prompt for the failures -------------------------
    recovered = 0
    if retry:
        outputs = run_inference(llm, tokenizer, [jobs[i].fallback for i in retry], batch_size)
        for i, raw in zip(retry, outputs):
            raw_fallback[i] = raw
            try:
                parsed[i] = parse_ains(raw)
                recovered += 1
                # The primary failure is named here rather than only in the flagged
                # file: recoveries never reach that file, and the rule that tripped
                # is the signal for whether the main prompt needs work.
                logger.info(
                    "query %s (%s) recovered via fallback prompt (%s)",
                    jobs[i].query_id, jobs[i].label(), reasons[i],
                )
            except AINParseError as exc:
                reasons[i] = f"{reasons[i]} | fallback: {exc}"

    # --- assemble per record, gate, and route --------------------------------
    jobs_by_query: dict[str, list[int]] = {}
    for i, job in enumerate(jobs):
        jobs_by_query.setdefault(job.query_id, []).append(i)

    good: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for record in records:
        indices = jobs_by_query[record["query_id"]]
        failed = [i for i in indices if i not in parsed]
        stage, reason = "parser", ""

        if not failed:
            # Every step-1 field rides through untouched; `ain` is appended last so
            # the record reads in pipeline order.
            candidate = dict(record)
            candidate["ain"] = [
                {"sub_question_id": jobs[i].sub_question_id, "ain_array": parsed[i]}
                for i in indices
            ]
            try:
                validate_final_record(candidate)
                good.append(candidate)
                continue
            except AINParseError as exc:
                stage, reason = "final-gate", str(exc)
        else:
            # One unresolved sub-question fails the whole record: a partial `ain`
            # block would look like a complete one and quietly retrieve nothing for
            # the topics it is missing.
            reason = " || ".join(f"[{jobs[i].label()}] {reasons.get(i, 'unknown')}" for i in failed)

        flagged.append(
            {
                **record,
                "first_attempt_raw_output": _join_raw(indices, raw_primary, jobs),
                "fallback_attempt_raw_output": _join_raw(indices, raw_fallback, jobs),
                "failure_stage": stage,
                "failure_reason": reason or "unknown",
            }
        )

    ain_counts = [len(entry["ain_array"]) for r in good for entry in r["ain"]]
    stats = {
        "calls": len(jobs),
        "first_attempt": len(jobs) - len(retry),
        "recovered": recovered,
        "unresolved": len(retry) - recovered,
        "ains": sum(ain_counts),
        "ain_counts": ain_counts,
    }
    return good, flagged, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default=INPUT_PATH, help="step 1's JSONL output")
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
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    records = load_step1_records(args.input)
    if args.limit:
        records = records[: args.limit]
    if not records:
        logger.error("no usable records loaded from %s", args.input)
        raise SystemExit(1)
    logger.info(
        "loaded %d record(s) / %d sub-question(s) from %s",
        len(records), sum(len(r["sub_questions"]) for r in records), args.input,
    )

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

    good, flagged, stats = extract_ains(llm, tokenizer, records, args.batch_size)

    write_jsonl(args.output, good)
    write_jsonl(args.flagged, flagged)
    write_error_log(args.error_log, flagged)

    # Invariant: every input record lands in exactly one output file.
    accounted = {r["query_id"] for r in good} | {r["query_id"] for r in flagged}
    missing = {r["query_id"] for r in records} - accounted
    if missing:
        logger.error("BUG: %d query_id(s) unaccounted for: %s", len(missing), sorted(missing)[:10])

    ain_counts = stats["ain_counts"]
    distribution = ", ".join(
        f"{size}:{count}" for size, count in sorted(Counter(ain_counts).items())
    )

    # Counts are labelled by unit: the LLM works per sub-question, but a record is
    # written or flagged as a whole.
    print()
    print(f"  records read               : {len(records)}")
    print(f"  sub-questions / LLM calls  : {stats['calls']}")
    print(f"  parsed on first attempt    : {stats['first_attempt']}")
    print(f"  recovered via fallback     : {stats['recovered']}")
    print(f"  unresolved sub-questions   : {stats['unresolved']}")
    print(f"  records written            : {len(good)}")
    print(f"  records flagged            : {len(flagged)}")
    print(f"  AINs extracted             : {stats['ains']}")
    if ain_counts:
        print(f"  AINs per sub-question      : mean {stats['ains'] / len(ain_counts):.2f}"
              f"  (distribution {distribution})")
    print(f"  main output                : {args.output}")
    if flagged:
        print(f"  flagged output             : {args.flagged}")
        print(f"  error log                  : {args.error_log}")
    if missing:
        print(f"  *** UNACCOUNTED query_ids  : {len(missing)} ***")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
