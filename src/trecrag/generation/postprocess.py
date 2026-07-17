"""Shared enforcement of the RAG task's hard submission rules, applied after
either generator backend produces raw sentences:

  - at most `max_citations` citations per sentence, citing only valid indices
  - every kept sentence must cite at least one reference (drop otherwise)
  - the full answer must not exceed `max_words` words (drop trailing sentences)
  - every remaining reference must be cited by at least one sentence
    (uncited references are dropped and citation indices are remapped)
"""

from __future__ import annotations

from trecrag.schema import AnswerSentence, RagAnswer


def postprocess_rag_answer(
    references: list[str],
    sentences: list[AnswerSentence],
    max_words: int,
    max_citations: int,
) -> RagAnswer:
    cleaned: list[AnswerSentence] = []
    for sentence in sentences:
        text = sentence.text.strip()
        if not text:
            continue
        valid_citations = [i for i in sentence.citations if 0 <= i < len(references)][:max_citations]
        if not valid_citations:
            continue
        cleaned.append(AnswerSentence(text=text, citations=valid_citations))

    budget = max_words
    kept: list[AnswerSentence] = []
    for sentence in cleaned:
        word_count = len(sentence.text.split())
        if word_count > budget:
            break
        kept.append(sentence)
        budget -= word_count

    used_old_indices = sorted({i for s in kept for i in s.citations})
    remap = {old: new for new, old in enumerate(used_old_indices)}
    new_references = [references[i] for i in used_old_indices]
    final_sentences = [
        AnswerSentence(text=s.text, citations=[remap[i] for i in s.citations]) for s in kept
    ]
    return RagAnswer(references=new_references, sentences=final_sentences)
