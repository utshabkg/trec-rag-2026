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
        -> preprocessed_queries_step1.jsonl

Anything that fails twice is written to the flagged file instead, never dropped:
every input query_id appears in exactly one of the two outputs.

QUICKSTART
    # vLLM is not in the trec-rag-2026 env; use one that has it:
    conda activate agentic-search
    python scripts/preprocess_step1_decompose.py

    # or override the constants below:
    python scripts/preprocess_step1_decompose.py \
        --topics trec_rag_2026_test_query_set.tsv \
        --model /media/20TB/shared/models/qwen/Qwen2.5-72B-Instruct-AWQ/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217 \
        --output output_decomp/preprocessed_queries_step1.jsonl

MODEL NOTES (verified on this machine)
  Qwen2.5-72B-Instruct-AWQ  awq 4-bit/group 128, 11 shards, ~39GB, HAS a chat
                            template. Needs TENSOR_PARALLEL_SIZE=2 on 2x48GB, and
                            the GPUs must be free (stop any Ollama model first).
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
from dataclasses import dataclass
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
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 8192              # topics are short; this keeps the KV cache small
MAX_TOKENS = 1024                 # a decomposition is far shorter, but truncation fails the parse
TEMPERATURE = 0.0                 # deterministic: this is extraction, not generation
TOP_P = 1.0
SEED = 0
BATCH_SIZE = 0                    # 0 = submit every prompt in one generate() call

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
# 2. MODEL / INFERENCE
# --------------------------------------------------------------------------------


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


def run_inference(llm, tokenizer, conversations: list[list[dict[str, str]]], batch_size: int) -> list[str]:
    """Batch every prompt through vLLM's offline API.

    One generate() call over the whole list is the point of using vLLM here: the
    scheduler batches continuously, so per-prompt calls would waste most of the GPU.
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if not flagged:
            fh.write("No queries failed decomposition.\n")
            return
        fh.write(f"{len(flagged)} query/queries failed decomposition\n\n")
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
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

    quantization = resolve_quantization(args.model, args.quantization)
    logger.info("loading %s (quantization=%s, tp=%d)", args.model, quantization, args.tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        quantization=quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
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
    good: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for topic in topics:
        record = records.get(topic.query_id)
        stage = "parser"
        if record is not None:
            try:
                validate_final_record(record)
                good.append(record)
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

    write_output(args.output, good)
    write_flagged(args.flagged, flagged)
    write_error_log(args.error_log, flagged)

    # Invariant: every input id lands in exactly one output file.
    accounted = {r["query_id"] for r in good} | {r["query_id"] for r in flagged}
    missing = {t.query_id for t in topics} - accounted
    if missing:
        logger.error("BUG: %d query_id(s) unaccounted for: %s", len(missing), sorted(missing)[:10])

    print()
    print(f"  topics read                : {len(topics)}")
    print(f"  succeeded (first attempt)  : {len(good) - len(fallback_ok)}")
    print(f"  succeeded (after fallback) : {len(fallback_ok)}")
    print(f"  flagged / unresolved       : {len(flagged)}")
    print(f"  main output                : {args.output}")
    if flagged:
        print(f"  flagged output             : {args.flagged}")
        print(f"  error log                  : {args.error_log}")
    if missing:
        print(f"  *** UNACCOUNTED query_ids  : {len(missing)} ***")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
