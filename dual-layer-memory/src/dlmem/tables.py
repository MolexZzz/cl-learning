"""Markdown/CSV table rendering for the paper's Tables 1 and 2."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from .utils import ensure_dir


def write_csv(rows: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{100.0 * float(x):.2f}%"


def _num(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{float(x):.{digits}f}"


def table1_markdown(rows: Sequence[dict]) -> str:
    header = (
        "| Policy | QA EM | Tok. F1 | Ref. Rate | Store (↓) | Store Prec. | "
        "Store Recall | Store F1 | Routing Cost† |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        label = r.get("label") or r.get("policy", "")
        if r.get("section"):
            lines.append(f"| **{r['section']}** | | | | | | | | |")
            continue
        lines.append(
            "| {label} | {em} | {f1} | {ref} | {store} | {prec} | {rec} | {sf1} | {cost} |".format(
                label=label,
                em=_pct(r.get("em")),
                f1=_num(r.get("f1"), 4),
                ref=_pct(r.get("refusal_rate")),
                store=_pct(r.get("storage")),
                prec=_pct(r.get("store_precision")),
                rec=_pct(r.get("store_recall")),
                sf1=_pct(r.get("store_f1")),
                cost=r.get("cost_label", ""),
            )
        )
    footnote = (
        "\n†Routing cost measures inference overhead. Gen8B = autoregressive generation "
        "with the 8B model; Fwd8B / Fwd1.7B = one forward pass. The cascade filters every "
        "input with the 1.7B model and escalates a subset to the 8B model.\n"
    )
    return header + "\n".join(lines) + "\n" + footnote


def table2_markdown(result: dict) -> str:
    order = result["order"]
    matrix = result["matrix"]
    header = "| Base \\ SFT | " + " | ".join(order) + " | Total (Base) |\n"
    header += "|---|" + "---|" * (len(order) + 1) + "\n"
    lines = []
    for b in order:
        total = sum(matrix[b].values())
        cells = " | ".join(f"{matrix[b][a]:,}" for a in order)
        lines.append(f"| {b} | {cells} | {total:,} |")
    totals = [sum(matrix[b][a] for b in order) for a in order]
    lines.append(
        "| Total (SFT) | " + " | ".join(f"{t:,}" for t in totals) + f" | {sum(totals):,} |"
    )
    summary = (
        f"\nChanged labels: {result['changed']:,}/{result['n']:,} "
        f"({100.0 * result['changed_rate']:.2f}%); internalised: {result['internalized']:,}; "
        f"partial absorption: {result['partial_absorption']:,}; "
        f"interference: {result['interference']:,}.\n"
    )
    return header + "\n".join(lines) + summary


def table3_markdown(rows: Sequence[dict]) -> str:
    header = "| Metric | Small router | Large router | Cascade (cap) |\n|---|---|---|---|\n"
    lines = []
    for r in rows:
        lines.append(
            f"| {r['metric']} | {r.get('small', '-')} | {r.get('large', '-')} | "
            f"{r.get('cascade', '-')} |"
        )
    return header + "\n".join(lines)
