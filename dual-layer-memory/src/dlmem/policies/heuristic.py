"""Heuristic write policies from Appendix B.

* ``HeuristicStore``   -- self-probe: turn the fact into a question, answer it
  zero-shot, and store when the model is wrong or under-confident.
* ``PPLCconditional``  -- threshold the model's per-token cross-entropy on the
  incoming fact (surprisal as a familiarity proxy).

Both select their operating threshold on the validation split.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.schema import Fact
from ..memory.qa import build_qa_prompt
from ..utils import exact_match
from .base import PolicyOutput, WritePolicy, threshold_for_storage


class HeuristicStorePolicy(WritePolicy):
    name = "heuristic"
    cost_label = "2 x Gen8B"

    def __init__(
        self,
        backbone,
        target_storage: float = 0.59,
        batch_size: int = 64,
    ) -> None:
        self.backbone = backbone
        self.target_storage = float(target_storage)
        self.batch_size = batch_size
        self.threshold: float | None = None

    # ------------------------------------------------------------------ #
    def scores(self, facts: Sequence[Fact]) -> np.ndarray:
        """Higher score => stronger evidence that the fact is worth storing."""
        prompts = [build_qa_prompt(f.probes[0].question) for f in facts]
        answers: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            answers.extend(self.backbone.generate(prompts[start : start + self.batch_size]))
        em = np.asarray([exact_match(a, f.probes[0].answer) for a, f in zip(answers, facts)])
        confidence = self.backbone.answer_confidence(prompts, answers)
        # wrong answers dominate; confidence only breaks ties
        return (1.0 - em) + 1e-3 * np.asarray(confidence, dtype=np.float64)

    def calibrate(self, facts: Sequence[Fact], target_storage: float | None = None) -> float:
        target = self.target_storage if target_storage is None else target_storage
        s = self.scores(facts)
        self.threshold = threshold_for_storage(s, target)
        return float(self.threshold)

    def decide_all(self, facts: Sequence[Fact]) -> PolicyOutput:
        s = self.scores(facts)
        threshold = self.threshold
        if threshold is None:
            threshold = threshold_for_storage(s, self.target_storage)
        write = s >= threshold
        return PolicyOutput(
            self.name,
            write,
            score=s,
            cost_label=self.cost_label,
            meta={"threshold": float(threshold), "target_storage": self.target_storage},
        )


class PPLConditionalPolicy(WritePolicy):
    name = "ppl_conditional"
    cost_label = "1 x Fwd8B"

    def __init__(self, backbone, target_storage: float = 0.717) -> None:
        self.backbone = backbone
        self.target_storage = float(target_storage)
        self.threshold: float | None = None

    def scores(self, facts: Sequence[Fact]) -> np.ndarray:
        return np.asarray(self.backbone.mean_nll([f.text for f in facts]), dtype=np.float64)

    def calibrate(self, facts: Sequence[Fact], target_storage: float | None = None) -> float:
        target = self.target_storage if target_storage is None else target_storage
        s = self.scores(facts)
        self.threshold = threshold_for_storage(s, target)
        return float(self.threshold)

    def decide_all(self, facts: Sequence[Fact]) -> PolicyOutput:
        s = self.scores(facts)
        threshold = self.threshold
        if threshold is None:
            threshold = threshold_for_storage(s, self.target_storage)
        write = s >= threshold
        return PolicyOutput(
            self.name,
            write,
            score=s,
            cost_label=self.cost_label,
            meta={"threshold": float(threshold), "target_storage": self.target_storage},
        )
