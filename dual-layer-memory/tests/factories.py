"""Small in-memory fact builder for unit tests (not an experiment backend)."""

from __future__ import annotations

from dlmem.data.schema import Fact, Probe


def make_fact(
    i: int,
    m: int = 30,
    label: str = "write-new",
    text: str | None = None,
    subject: str | None = None,
    obj: str | None = None,
) -> Fact:
    subject = subject or f"Entity{i}"
    obj = obj or f"Answer{i}"
    return Fact(
        fact_id=f"f{i:04d}",
        subject=subject,
        relation="is associated with",
        obj=obj,
        text=text or f"{subject} is associated with {obj}.",
        probes=[
            Probe(question=f"What is {subject} associated with? (variant {j})", answer=obj)
            for j in range(m)
        ],
        label=label,
        split="test",
    )


def make_facts(n: int, m: int = 30, labels=("non-write", "write-new", "write-update")) -> list[Fact]:
    return [make_fact(i, m=m, label=labels[i % len(labels)]) for i in range(n)]
