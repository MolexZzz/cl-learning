"""Prompt construction for zero-shot QA, memory-supported QA, and labelling."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

MEMORY_HEADER = "[Memory]"
QUESTION_HEADER = "[Question]"
ANSWER_HEADER = "Answer:"

_MEM_BLOCK = re.compile(r"\[Memory\](.*?)\[Question\]", re.DOTALL)
_QUESTION_BLOCK = re.compile(r"\[Question\](.*?)Answer:", re.DOTALL)


def format_memory(memories: Iterable[str], max_chars: int = 2000) -> str:
    lines: list[str] = []
    used = 0
    for mem in memories:
        mem = " ".join(str(mem).split())
        if not mem:
            continue
        line = f"- {mem}"
        if used + len(line) > max_chars and lines:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def build_qa_prompt(
    question: str,
    memories: Sequence[str] | None = None,
    instruction: str | None = None,
) -> str:
    instruction = instruction or (
        "Answer the question with a short factual phrase (a few words). "
        "Do not explain. If the memory contains the answer, rely on it."
    )
    if memories:
        mem = format_memory(memories)
        return (
            f"{instruction}\n"
            f"{MEMORY_HEADER}\n{mem}\n"
            f"{QUESTION_HEADER}\n{question.strip()}\n"
            f"{ANSWER_HEADER}\n"
        )
    return (
        f"{instruction}\n"
        f"{QUESTION_HEADER}\n{question.strip()}\n"
        f"{ANSWER_HEADER}\n"
    )


def build_qa_prompt_no_memory(question: str) -> str:
    return build_qa_prompt(question, memories=None)


def extract_memory_block(prompt: str) -> str:
    m = _MEM_BLOCK.search(prompt)
    if not m:
        return ""
    return m.group(1).strip()


def extract_question(prompt: str) -> str:
    m = _QUESTION_BLOCK.search(prompt)
    if m:
        return m.group(1).strip()
    # fallback: last non-empty line before the answer header
    body = prompt.split(ANSWER_HEADER)[0]
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def memory_lines(prompt: str) -> list[str]:
    block = extract_memory_block(prompt)
    return [ln[2:].strip() for ln in block.splitlines() if ln.startswith("- ")]


def build_sft_query(question: str) -> str:
    return f"{QUESTION_HEADER}\n{question.strip()}\n{ANSWER_HEADER}\n"


def build_sft_sentence(fact_text: str) -> str:
    return f"State the following fact.\n{QUESTION_HEADER}\nWhat is the fact?\n{ANSWER_HEADER}\n"
