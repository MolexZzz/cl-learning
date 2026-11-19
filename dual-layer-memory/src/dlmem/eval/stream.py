"""Online streaming evaluation (paper Sec. 4.1, Appendix A.2).

Facts are injected over time; the policy decides write/discard *before* any
query about the fact is seen.  Queries are answered from the current external
memory + parametric model.  Episodes are processed in lockstep so that
generation batches span episodes (important for real backbones).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..data.schema import Fact
from ..memory.qa import build_qa_prompt
from ..memory.store import ExternalMemory
from ..utils import Timer, exact_match, get_logger, is_refusal, token_f1
from .metrics import expected_em, storage_counts

log = get_logger("dlmem.eval.stream")


@dataclass
class StreamingResult:
    policy: str
    metrics: dict
    per_episode: list[dict] = field(default_factory=list)
    predictions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "metrics": self.metrics,
            "per_episode": self.per_episode,
        }


def _write_map(policy_output, fact_ids: Sequence[str]) -> dict[str, bool]:
    if len(policy_output.write) != len(fact_ids):
        raise ValueError(
            f"policy produced {len(policy_output.write)} decisions for {len(fact_ids)} facts"
        )
    return {fid: bool(w) for fid, w in zip(fact_ids, policy_output.write)}


def run_streaming(
    episodes,
    facts: Sequence[Fact],
    policy_output,
    fact_ids: Sequence[str],
    backbone,
    top_k: int = 3,
    batch_episodes: int = 8,
    keep_predictions: bool = False,
    max_episodes: int = 0,
) -> StreamingResult:
    episodes = list(episodes[:max_episodes]) if max_episodes else list(episodes)
    write_by_id = _write_map(policy_output, fact_ids)
    timer = Timer()
    per_episode: list[dict] = []
    predictions: list[dict] = []

    with timer:
        for start in range(0, len(episodes), max(1, batch_episodes)):
            chunk = episodes[start : start + batch_episodes]
            memories = [ExternalMemory(top_k=top_k) for _ in chunk]
            stats = [
                {
                    "em": 0.0,
                    "f1": 0.0,
                    "refusal": 0,
                    "n": 0,
                    "hits": 0,
                    "stored": 0,
                    "facts": len(ep.fact_indices),
                }
                for ep in chunk
            ]
            turns = chunk[0].turns
            for t in range(turns):
                prompts: list[str] = []
                slots: list[int] = []
                query_info: list[tuple[int, int, int]] = []
                for i, ep in enumerate(chunk):
                    inj = ep.injections.get(t)
                    if inj is not None:
                        fact = facts[ep.fact_indices[inj]]
                        if write_by_id.get(fact.fact_id, False):
                            if memories[i].add(fact):
                                stats[i]["stored"] += 1
                    q = ep.queries.get(t)
                    if q is None:
                        continue
                    slot, probe_idx = q
                    fact = facts[ep.fact_indices[slot]]
                    probe = fact.probes[probe_idx]
                    retrieved = memories[i].retrieve(probe.question)
                    if any(r.fact_id == fact.fact_id for r in retrieved):
                        stats[i]["hits"] += 1
                    prompts.append(
                        build_qa_prompt(probe.question, [r.text for r in retrieved] if retrieved else None)
                    )
                    slots.append(i)
                    query_info.append((slot, probe_idx, i))
                if not prompts:
                    continue
                answers = backbone.generate(prompts)
                for (slot, probe_idx, i), answer in zip(query_info, answers):
                    fact = facts[chunk[i].fact_indices[slot]]
                    probe = fact.probes[probe_idx]
                    stats[i]["em"] += exact_match(answer, probe.answer)
                    stats[i]["f1"] += token_f1(answer, probe.answer)
                    stats[i]["refusal"] += int(is_refusal(answer))
                    stats[i]["n"] += 1
                    if keep_predictions:
                        predictions.append(
                            {
                                "episode": chunk[i].episode_id,
                                "turn": t,
                                "fact_id": fact.fact_id,
                                "question": probe.question,
                                "gold": probe.answer,
                                "prediction": answer,
                                "memory_size": len(memories[i]),
                            }
                        )
            for i, ep in enumerate(chunk):
                s = stats[i]
                n = max(1, s["n"])
                per_episode.append(
                    {
                        "episode_id": ep.episode_id,
                        "em": s["em"] / n,
                        "f1": s["f1"] / n,
                        "refusal_rate": s["refusal"] / n,
                        "n_queries": s["n"],
                        "retrieval_hit_rate": s["hits"] / n,
                        "memory_size": len(memories[i]),
                        "storage_ratio": s["stored"] / max(1, s["facts"]),
                    }
                )

    # ------------------------------------------------------------------ #
    eval_fact_indices = sorted({idx for ep in episodes for idx in ep.fact_indices})
    eval_facts = [facts[i] for i in eval_fact_indices]
    eval_write = [write_by_id.get(f.fact_id, False) for f in eval_facts]
    counts = storage_counts(eval_facts, eval_write)
    offline = expected_em(eval_facts, eval_write)

    n_queries = sum(e["n_queries"] for e in per_episode)
    metrics = {
        "em": float(np.mean([e["em"] for e in per_episode])) if per_episode else 0.0,
        "f1": float(np.mean([e["f1"] for e in per_episode])) if per_episode else 0.0,
        "refusal_rate": float(np.mean([e["refusal_rate"] for e in per_episode]))
        if per_episode
        else 0.0,
        "n_queries": n_queries,
        "n_facts": len(eval_facts),
        "storage": counts.storage,
        "store_precision": counts.precision,
        "store_recall": counts.recall,
        "store_f1": counts.f1,
        "offline_em": offline,
        "retrieval_hit_rate": float(np.mean([e["retrieval_hit_rate"] for e in per_episode]))
        if per_episode
        else 0.0,
        "escalation_rate": float(np.mean(policy_output.escalate))
        if len(policy_output.escalate)
        else 0.0,
        "routing_cost": float(np.mean(policy_output.cost)) if len(policy_output.cost) else 0.0,
        "cost_label": policy_output.cost_label,
        "eval_seconds": timer.elapsed,
    }
    log.info(
        "%s | EM %.4f F1 %.4f store %.2f (P %.2f R %.2f) cost %.2f",
        policy_output.name,
        metrics["em"],
        metrics["f1"],
        metrics["storage"],
        metrics["store_precision"],
        metrics["store_recall"],
        metrics["routing_cost"],
    )
    return StreamingResult(policy_output.name, metrics, per_episode, predictions)
