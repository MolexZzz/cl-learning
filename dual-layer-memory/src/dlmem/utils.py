"""Shared utilities: seeding, IO, logging, normalisation, QA metrics, refusals.

Everything in this module is dependency-free (numpy optional) so that data
preparation and analysis can run on machines without a deep-learning stack.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Paths / IO
# --------------------------------------------------------------------------- #


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | os.PathLike[str]) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike[str], obj: Any) -> None:
    ensure_dir(Path(path).parent)

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serialisable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


def chunked(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), max(1, size)):
        yield seq[i : i + size]


# --------------------------------------------------------------------------- #
# Seeding / logging
# --------------------------------------------------------------------------- #


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch absent
        pass


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "dlmem", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


class Timer:
    """``with Timer() as t: ...; t.elapsed``."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._t0 = 0.0

    def start(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self._t0
        return self.elapsed

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0


# --------------------------------------------------------------------------- #
# Hashing / text normalisation
# --------------------------------------------------------------------------- #


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_uint(text: str, mod: int = 2**31 - 1) -> int:
    return int(sha1(text)[:16], 16) % mod


_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """SQuAD-style answer normalisation."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


def tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


# --------------------------------------------------------------------------- #
# QA metrics
# --------------------------------------------------------------------------- #


def exact_match(pred: str, gold: str | Sequence[str]) -> float:
    """Containment-based EM, the standard ZsRE / SQuAD-style evaluation.

    A prediction counts as correct when the normalised gold answer appears as a
    contiguous token span of the normalised prediction (so "The answer is Paris"
    is correct for gold "Paris").  Empty predictions are never correct.
    """
    pred_n = normalize_answer(pred)
    if not pred_n:
        return 0.0
    golds = [gold] if isinstance(gold, str) else list(gold)
    gold_toks = [tokens(g) for g in golds]
    pred_toks = pred_n.split()
    for gt in gold_toks:
        if not gt:
            continue
        if " ".join(gt) == pred_n:
            return 1.0
        n = len(gt)
        if any(pred_toks[i : i + n] == gt for i in range(len(pred_toks) - n + 1)):
            return 1.0
    return 0.0


def exact_match_strict(pred: str, gold: str | Sequence[str]) -> float:
    golds = [gold] if isinstance(gold, str) else list(gold)
    return float(normalize_answer(pred) in {normalize_answer(g) for g in golds})


def token_f1(pred: str, gold: str | Sequence[str]) -> float:
    golds = [gold] if isinstance(gold, str) else list(gold)
    pred_toks = tokens(pred)
    best = 0.0
    for g in golds:
        gold_toks = tokens(g)
        if not pred_toks or not gold_toks:
            best = max(best, float(pred_toks == gold_toks))
            continue
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi (?:do not|don't|dont) know\b",
        r"\bi(?:'m| am) not sure\b",
        r"\bi (?:cannot|can't|can not|am unable to) (?:answer|determine|provide|say|confirm)\b",
        r"\bi (?:don't|do not|dont) have (?:enough|sufficient|any) information\b",
        r"\bnot enough information\b",
        r"\bno (?:information|answer) (?:is )?(?:available|provided)\b",
        r"\bunknown\b",
        r"\bunanswerable\b",
        r"\bnot mentioned\b",
        r"\bn/?a\b",
        r"\bi (?:apologize|am sorry|'m sorry)\b",
        r"\bi (?:cannot|can't) (?:provide|give|offer|help with) (?:that|this|an answer)\b",
        r"\bi (?:do not|don't) have (?:access to )?(?:that|this|the) information\b",
        r"\bnot (?:specified|provided|stated)\b",
        r"\bas an ai\b",
        r"\bi (?:could not|couldn't) find\b",
        r"\bthe (?:context|text|passage|memory) (?:does not|doesn't) (?:mention|contain|say|specify)\b",
        r"^(?:unknown|n/?a|none)$",
    ]
)


def is_refusal(text: str) -> bool:
    if not (text or "").strip():
        return True
    return any(p.search(text) for p in REFUSAL_PATTERNS)


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #


def mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0


def format_pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f}%"


@contextlib.contextmanager
def suppress_warnings() -> Iterator[None]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def git_revision(cwd: str | os.PathLike[str] | None = None) -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def human_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"
