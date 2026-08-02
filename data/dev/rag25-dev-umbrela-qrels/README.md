# RAG25 Development Umbrela Qrels

This directory contains Umbrela-style qrels for the RAG25 development candidate pools projected onto ClimbMix.

These are LLM-generated judgments over the pooled candidate documents, not exhaustive corpus-wide judgments.
Each qrels file is in standard TREC qrels format:

```text
<topic_id> 0 <docid> <score>
```

Scores are on a 0-4 scale:

- `0`: not relevant to the narrative
- `1`: related to the narrative, but does not answer a sub-narrative
- `2`: answers 1 sub-narrative in sufficient detail
- `3`: answers 2-3 sub-narratives in sufficient detail
- `4`: answers 4 or more sub-narratives in sufficient detail

All current files contain 26,341 judgments over the 22 RAG25 development topics present in the candidate pool.

## Current qrels

These are the qrels we recommend using for new experiments:

```text
rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels
rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels
rag25-climbmix-umbrela-ministral-3-14b-instruct-2512-v2.qrels
```

The Qwen3.5-9B v2 and Ministral v2 qrels use the current atomic `sub_narratives` from:

```text
https://trec.nist.gov/data/rag/trec25_narratives_final_w_questions_w_sub_narratives_edit_20250822.json
```

The Codex GPT-5.5 medium reasoning qrels were also run with the corrected narrative/sub-narrative formulation. The `v1` suffix here just means this is the first released Codex qrels file for this setup; it is not part of the older Qwen/Ministral formulation.

## How we created these qrels

First, we created a candidate document pool for each of the 22 RAG25 development topics. To do this, we generated 15 projection queries per topic, using the ClimbMix Pyserini REST API:

```text
https://github.com/TREC-RAG/trec-rag-skills/tree/main/skills/pyserini-rest-api
```

For each topic, the 15 projection queries were:

- the original query
- 4 Codex query rewrites
- 10 relevant passages from the original qrels, used as additional retrieval queries

These 15 queries were then sent to BM25, which returned a ranked list of documents for each query. We pooled and deduplicated those results to form the candidate document set that Umbrela judged.

Once this was done, we ran Umbrela over the pooled documents with three LLM judges:

- Codex GPT-5.5 medium reasoning
- Qwen3.5-9B
- Ministral-3-14B-Instruct-2512

For all three LLMs, we used the narrative/sub-narrative prompt below. The narratives and sub-narratives come from the TREC RAG 2025 narrative file:

```text
https://trec.nist.gov/data/rag/trec25_narratives_final_w_questions_w_sub_narratives_edit_20250822.json
```

```text
Given a user first person narrative and a passage, you must provide a score on an integer scale of 0 to 4
with the following meaning:
0 - represents that the passage has nothing to do with the narrative,
1 - represents that the passage seems related to the query but does not contain any sub-narrative of an
answer to it,
2 - represents that the passage contains detailed answer for the 1 sub-narrative of the narrative with enough
description,
3 - represents that the passage contains detailed answer for the 2 to 3 sub-narratives of the narrative with
enough description and
4 - represents that the passage contains detailed answer for the 4+ sub-narratives of the narrative with
enough description.
Narrative: {narrative}
Sub-narratives: {sub_narratives}
Passage: {passage}
Instructions: Determine whether each sub-narrative is fully and properly explained or only briefly mentioned. Assign a final integer score (0-4) based on the number of sub-narratives addressed.
Rule: A passage must cover a sub-narrative in a detailed, proper way to count; mere mentions or vague
references do not qualify. If there is some extra information not relevant to the narrative at all, downgrade
the level by 1. If lot of extra information then downgrade by 2. Do not provide reasoning after listing
answered parts; only give the score in the format: ##final score: X
```

## Older versions

Earlier Qwen3.5-9B and Ministral qrels are retained in:

```text
older-versions/
```

These older versions were generated with a different sub-narrative formulation. They are kept for reproducibility and comparison, but for new experiments, if you plan to use the Qwen3.5-9B or Ministral files, we recommend the v2 versions in the parent directory.

```text
older-versions/rag25-climbmix-umbrela-qwen3.5-9b-v1.qrels
older-versions/rag25-climbmix-umbrela-ministral-3-14b-instruct-2512-v1.qrels
```
