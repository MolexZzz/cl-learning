"""Dual-Layer Agentic Memory: fast write routing + slow parametric consolidation.

Reference implementation of

    W. Li, D. Nie, T. Lyu, R. Lan, P. Wang, L. Hong, W. Pan, B. Lin, B. Pan, Y. Hu,
    "Dual-Layer Agentic Memory with Fast Write Routing and Slow Consolidation",
    arXiv:2608.22215v2, 2026.

The package is split along the paper's data flow:

    data/      ZsRE facts -> behavioural labels -> features -> streaming episodes
    models/    frozen LLM backbones, reward MLP heads, escalation gate
    policies/  static / heuristic / supervised / cascaded write policies
    memory/    external memory store + retrieval + QA prompting
    train/     stage-wise training (labels, routers, gate, write-back SFT)
    eval/      streaming benchmark, metrics, Pareto sweep, ablations

Experiments require ``torch`` and ``transformers`` (plus ``peft`` for LoRA
write-back); ``numpy``/``matplotlib`` cover data analysis and figures. There is
no simulated backend: every run loads real Qwen3 checkpoints.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
