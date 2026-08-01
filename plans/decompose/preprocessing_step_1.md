# Task: Build a Query Decomposition Preprocessing Script (TREC-style Composite Queries)

## Context

I have a TREC-style test topics file (`.tsv` format) where each row is a query/topic.
Column 1 = `query_id`, Column 2 = `query_text`. There is no header row.

Many of these queries are **composite queries**: they bundle together (a) background
context/metadata about the asker's situation, and (b) one or more distinct
sub-questions the asker actually wants answered. Off-the-shelf first-stage retrievers
and re-rankers perform poorly on these because they aren't designed to handle
multi-part questions with embedded scene-setting context.

The fix: use a locally-hosted LLM (served via vLLM for throughput) to decompose each
composite query into structured parts before any retrieval happens. This script is
**preprocessing step 1** of a larger pipeline — its only job is decomposition, not
retrieval.

## Example of the Decomposition

**Raw input query:**
> I'm on a hospital nursing DEI council that has to recommend a three-year plan with
> limited money. Our chief nursing officer wants something that helps recruitment and
> retention, not just a one-time training. What should we prioritize across nursing
> school pipeline partnerships, hiring and promotion practices, reporting systems for
> racism, leadership accountability, and curriculum/continuing education? How should
> we think about tradeoffs, what outcomes should we measure, and what mistakes would
> make the plan feel performative to nurses of color?

**This decomposes into:**
- **Metadata** (background/situational context, not itself a question): "I'm on a
  hospital nursing DEI council that has to recommend a three-year plan with limited
  money. Our chief nursing officer wants something that helps recruitment and
  retention, not just a one-time training."
- **Sub-question 1**: "What should we prioritize across nursing school pipeline
  partnerships, hiring and promotion practices, reporting systems for racism,
  leadership accountability, and curriculum/continuing education?"
- **Sub-question 2**: "How should we think about tradeoffs, what outcomes should we
  measure, and what mistakes would make the plan feel performative to nurses of
  color?"

Note: **the number of sub-questions is variable** — a query might have 1, 2, 3, or
more sub-questions. The extraction logic and output schema must not assume exactly 2.

## Required Final Output Schema (JSON object per query, written to the .jsonl file)

Note: this is the schema for the **final output file** after parsing. It is *not*
the raw format the LLM itself produces — see the "Structured output strategy"
section below for the intermediate XML tag format the LLM is prompted to emit and
the deterministic parser that converts it into this JSON schema.

```json
{
  "query_id": "1rag2026-0",
  "comp_query": "<the full original, unmodified query text>",
  "query_metadata": "<the extracted background/context portion>",
  "sub_questions": [
    {"sub_question_id": "1", "sub_question_text": "<first sub-question, verbatim or lightly cleaned>"},
    {"sub_question_id": "2", "sub_question_text": "<second sub-question, verbatim or lightly cleaned>"}
  ]
}
```

Notes on the schema:
- `sub_question_id` values are strings, sequential starting at `"1"`, in the order the
  sub-questions appear in the original query.
- `sub_questions` is always a JSON array, even if it only ends up containing one
  element. Never collapse it to a bare object.
- `comp_query` must preserve the original query text exactly as it appeared in the TSV
  (do not paraphrase or normalize whitespace beyond what's needed for valid JSON).
- If `query_metadata` genuinely doesn't exist for a given query (i.e., the query is
  just a bare question with no situational framing), it should be an empty string
  `""`, not null and not omitted.

The full set of these JSON objects, one per line, becomes the output file:
**`preprocessed_queries_step1.jsonl`**

## Build Requirements

### 1. Configuration variables (top of script, easy to edit)
Define these as clearly-labeled constants near the top of the script:
- `TEST_TOPIC_PATH` — path to the input `.tsv` file (I will fill this in myself)
- `LLM_PATH` — local filesystem path to the model weights to load into vLLM (I will
  point this at either my local `Qwen-2.5-72B-Instruct-AWQ` or `Gemma-4-31b-it`
  checkpoint depending on which I want to run; the script should not hardcode
  assumptions about which model is in use beyond what's needed for correct vLLM
  loading — e.g., quantization method, if it can be auto-detected or needs a flag)
- `OUTPUT_PATH` — where `preprocessed_queries_step1.jsonl` gets written
- Any generation/runtime parameters that should be tunable (batch size, max tokens,
  temperature, tensor parallel size, etc.) should also be pulled up as named constants
  or CLI args rather than buried in function calls.

### 2. Model serving via vLLM
- Use vLLM's **offline batch inference API** (`vllm.LLM` + `.generate()` /
  `.chat()`), not the OpenAI-compatible server, unless there's a strong reason to
  prefer the server route — offline batching is simpler for a one-shot preprocessing
  job over a static file and avoids extra infrastructure. Use your judgment here, but
  default to offline batch mode.
- Load the model from `LLM_PATH`. Handle AWQ-quantized checkpoints correctly (set the
  appropriate `quantization` argument for vLLM when loading an AWQ model).
- Batch all queries and submit them together (or in chunks) to vLLM's `generate()`
  call rather than looping one-by-one with separate calls — this is the whole point of
  using vLLM for throughput.
- Use the model's native chat template (via tokenizer `apply_chat_template` or vLLM's
  built-in chat handling) so the prompt is formatted correctly for whichever
  instruct-tuned model is loaded.

### 3. Structured output strategy: XML tags + deterministic regex parsing (NOT guided JSON)

We are deliberately **not** using vLLM's guided-JSON decoding for this step. The
reasoning: sub-question boundary detection (deciding where one question ends and the
next begins in a run-on composite query) is a judgment call, and we want that judgment
made once by the LLM and then extracted **deterministically** — not re-interpreted by
a second layer of schema-constrained sampling. XML-style tags let us parse with a
plain regex, which is transparent, debuggable, and has no hidden failure modes the way
guided-decoding grammars sometimes do with certain quantized/AWQ models.

**Required raw LLM output shape** (the model should emit *only* this, no
surrounding prose):

```
<metadata>I'm on a hospital nursing DEI council that has to recommend a three-year plan with limited money. Our chief nursing officer wants something that helps recruitment and retention, not just a one-time training.</metadata>
<sub_question id="1">What should we prioritize across nursing school pipeline partnerships, hiring and promotion practices, reporting systems for racism, leadership accountability, and curriculum/continuing education?</sub_question>
<sub_question id="2">How should we think about tradeoffs, what outcomes should we measure, and what mistakes would make the plan feel performative to nurses of color?</sub_question>
```

Schema rules for this tag format (bake these into the prompt design in section 4
below, and into the parser in this section):
- Exactly one `<metadata>...</metadata>` block. If there is no separable background
  context, emit `<metadata></metadata>` (empty, but the tag must still be present —
  this keeps the parser's "did the model even attempt the format" check simple: tag
  presence, not tag content, is what's checked first).
- One or more `<sub_question id="N">...</sub_question>` blocks, `N` starting at `1`
  and incrementing by 1 with no gaps, in document order.
- No nesting, no attributes other than `id`, no content outside these tags (no
  preamble like "Here is the decomposition:", no trailing commentary).
- Tags are matched by pattern, not by line splitting, since some vLLM outputs may
  collapse or add whitespace unpredictably.

**Deterministic parser** (write this as a standalone function, e.g.
`parse_decomposition(raw_output: str) -> dict | None`):
- Use `re` to extract the `<metadata>` block and all `<sub_question id="...">` blocks.
- Validation rules, ALL of which must pass for the parse to be considered successful:
  1. Exactly one `<metadata>` tag pair found.
  2. At least one `<sub_question>` tag pair found.
  3. Extracted `id` attributes are exactly `"1", "2", ..., "N"` with no gaps, no
     duplicates, no out-of-order ids (i.e. the sequence of ids encountered must equal
     `[1, 2, ..., N]` in order).
  4. No non-whitespace content in `raw_output` falls outside the matched tag spans
     (guards against the model appending stray commentary before/after — decide on a
     reasonable tolerance, e.g. strip whitespace/newlines before checking, but don't
     silently accept arbitrary extra prose).
  5. Neither the metadata content nor any sub-question content is allowed to itself
     contain a raw `<` character that would suggest a malformed/unclosed tag leaking
     into content (basic sanity guard against truncated generations).
- If **all** checks pass: return the parsed dict in the target JSON schema
  (`query_id`, `comp_query`, `query_metadata`, `sub_questions` list with string
  `sub_question_id`s).
- If **any** check fails: return `None` (or raise a specific `DecompositionParseError`
  — your choice, just be consistent) so the caller knows to proceed to the retry
  step below. Do not attempt partial/best-effort extraction on failure — a failed
  parse should be all-or-nothing.

### 3a. Retry-on-failure with a fallback prompt

- For any query whose **first-attempt** output fails the parser above, retry
  **once** using a second, distinct fallback prompt (not a re-send of the identical
  first prompt — re-running the exact same prompt against the same weights is
  unlikely to change the outcome). Design the fallback prompt to be shorter and more
  mechanically directive: restate the exact tag format as the entire focus of the
  prompt (less framing/explanation, more "output must look exactly like this"), and
  include the original query again. You have latitude on exact wording (see section 4
  for both prompts' design requirements).
- Run the fallback prompt through the same generation call and the same
  `parse_decomposition` parser.
- If the fallback attempt **succeeds**: use that result, and record in the run log
  that this `query_id` required a fallback (useful signal for later prompt tuning,
  even though it doesn't block the pipeline).
- If the fallback attempt **also fails**: do NOT write anything for this
  `query_id` to the main output file. Instead:
  - Append a record to a separate flagged-queries file (e.g.
    `preprocessed_queries_step1_flagged.jsonl`) containing at minimum: `query_id`,
    `comp_query` (original text), `first_attempt_raw_output`,
    `fallback_attempt_raw_output`, and a short failure reason if you can
    distinguish which validation rule(s) failed.
  - Also log a one-line summary to `preprocessed_queries_step1_errors.log` for quick
    scanning.
  - The script must continue processing the rest of the queries — a flagged query
    is never a fatal error for the run.

### 3b. Final pre-write validation gate

- Immediately before any record is serialized into
  `preprocessed_queries_step1.jsonl`, run it through one last check: confirm the
  in-memory dict has exactly the expected keys, `sub_questions` is a non-empty list,
  every element has both `sub_question_id` and `sub_question_text` as non-null
  strings, and `sub_question_id`s are the expected `"1"..."N"` sequence. This is a
  belt-and-suspenders check independent of the regex parser above — its only job is
  to guarantee that nothing malformed ever reaches the output file, even if a future
  edit to the parser introduces a bug.
- Anything that fails this final gate gets routed to the same flagged-queries file
  described in 3a rather than written to the main output, with a failure reason
  noting it was caught at the final-gate stage specifically (distinguishing it from
  parser-stage failures is useful for debugging later).

### 4. The decomposition prompt itself
**Leave the exact wording of the prompt sent to the local LLM up to your own
judgment/design** — you have full latitude here. But it must satisfy these design
requirements:

- Clearly instructs the model to separate **situational/background metadata** from
  **discrete, independently-answerable sub-questions**.
- Makes explicit that the number of sub-questions is variable and unknown in
  advance — the model should identify however many distinct questions are actually
  present (could be 1, could be more than 2), not force a fixed count.
- Instructs the model to preserve the original wording of each sub-question as
  closely as possible (extraction, not paraphrasing/summarizing) — we want the
  sub-questions to remain faithful to what the user actually asked, not the model's
  rephrasing of it.
- Handles the edge case where a query has **no separable metadata** (i.e., it's just
  a direct question or list of questions with no framing/context) — in that case
  `query_metadata` should be an empty string, and the model shouldn't invent
  background context that isn't there.
- Handles the edge case where a query has **only one implicit question** (not
  obviously split into multiple sub-questions by punctuation/structure) — in that
  case `sub_questions` should contain a single entry.
- Includes the **exact required XML tag output format** directly in the prompt
  (spec it out explicitly: `<metadata>...</metadata>` followed by one or more
  `<sub_question id="N">...</sub_question>` tags — see section 3 above for the full
  schema this must match), and is explicit that the model must emit **only** these
  tags with no preamble, no trailing commentary, and no markdown code fences around
  the output.
- Includes 1-2 short worked examples in the prompt (few-shot) showing raw query in,
  correctly-tagged output out, to anchor the model's behavior — using a different
  example than the nursing DEI one above is fine, or you may construct a minimal
  illustrative one. At least one example should demonstrate the empty-metadata case
  (`<metadata></metadata>`) and at least one should have more than one sub-question,
  so both edge cases are represented in-context.

Please write out the full text of **both** prompts you design — the primary
decomposition prompt and the shorter fallback/retry prompt described in section 3a —
as clearly isolated string/template constants in the script (e.g.
`DECOMPOSITION_SYSTEM_PROMPT` and `DECOMPOSITION_FALLBACK_PROMPT`) so it's easy for me
to read, review, and iterate on each independently of the rest of the code.

### 5. I/O handling
- Read the `.tsv` at `TEST_TOPIC_PATH` with no assumed header row; column 1 is
  `query_id` (string — don't coerce to int, some IDs may be alphanumeric like
  `1rag2026-0`), column 2 is `query_text`.
- Handle basic TSV messiness defensively (e.g., stray tabs within a field, blank
  lines) without crashing.
- Write output as JSONL to `OUTPUT_PATH` (default filename
  `preprocessed_queries_step1.jsonl`) — one compact JSON object per line, UTF-8
  encoded, `ensure_ascii=False` so text isn't escaped unnecessarily.
- Print a short run summary at the end: total queries read; total succeeded on
  first attempt; total succeeded only after fallback; total flagged/unresolved
  (written to `preprocessed_queries_step1_flagged.jsonl` instead of the main
  output); pointer to the error log for anyone who wants to scan failures quickly.

### 6. Code quality / structure expectations
- Single Python script is fine (no need to over-engineer into a package) — organize
  into clear functions (e.g., `load_topics`, `build_primary_prompts`,
  `build_fallback_prompts`, `run_inference`, `parse_decomposition`,
  `validate_final_record`, `write_output`, `write_flagged`) rather than one long
  procedural block.
- Add a `if __name__ == "__main__":` entry point.
- Add minimal but clear docstrings/comments explaining each stage, since I'll be
  extending this pipeline with further preprocessing steps later.
- Don't add a CLI argument framework unless it's lightweight (e.g., `argparse` is
  fine if you want to allow overriding the top-of-file constants from the command
  line, but the top-of-file constants should work standalone with no args passed).

## Out of scope for this task
- Do not implement any retrieval, re-ranking, or downstream pipeline steps — this
  script's sole responsibility is producing `preprocessed_queries_step1.jsonl`.
- Do not assume which of the two models (Qwen vs Gemma) is being used — the script
  should work for either once `LLM_PATH` is set, modulo whatever
  quantization/chat-template handling is needed for correctness.

## Deliverable
A single runnable Python script (plus a short README or top-of-file comment block
explaining how to set the config variables and run it) that, from the TSV at
`TEST_TOPIC_PATH` using the local vLLM-served model at `LLM_PATH`, produces:

1. **`preprocessed_queries_step1.jsonl`** — the main output: one valid,
   schema-conformant JSON object per line for every query that was successfully
   decomposed (first attempt or fallback).
2. **`preprocessed_queries_step1_flagged.jsonl`** — queries that failed both the
   primary and fallback decomposition attempts (or failed the final validation
   gate), preserved with their raw LLM outputs for manual human review.
3. **`preprocessed_queries_step1_errors.log`** — a human-scannable plain-text log
   summarizing what failed and why, for quick triage without opening the flagged
   JSONL.

No query should ever silently disappear — every `query_id` from the input TSV should
be accounted for in either the main output or the flagged output.
