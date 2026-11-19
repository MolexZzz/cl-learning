"""Frozen LLM backbones: interface, encoding containers, and device helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Encoding:
    """Per-text features extracted from a frozen backbone."""

    embedding: np.ndarray  # (hidden,)
    nll_curve: np.ndarray  # (length,)
    mean_nll: float
    length: int


class Backbone:
    """Minimal interface shared by all backbones (HF Qwen3 in this repo)."""

    name: str = "backbone"
    hidden_size: int = 0

    def encode_batch(self, texts: Sequence[str], pooling: str = "mean") -> list[Encoding]:
        raise NotImplementedError

    def generate(self, prompts: Sequence[str], max_new_tokens: int | None = None) -> list[str]:
        raise NotImplementedError

    def mean_nll(self, texts: Sequence[str]) -> np.ndarray:
        encs = self.encode_batch(texts)
        return np.asarray([e.mean_nll for e in encs], dtype=np.float64)

    def generate_one(self, prompt: str, max_new_tokens: int | None = None) -> str:
        return self.generate([prompt], max_new_tokens=max_new_tokens)[0]

    def answer_confidence(self, prompts: Sequence[str], answers: Sequence[str] | None = None) -> np.ndarray:
        """Mean NLL of prompt+answer; used by the heuristic self-probing baseline."""
        if answers is None:
            answers = self.generate(prompts)
        texts = [f"{p} {a}" for p, a in zip(prompts, answers)]
        return self.mean_nll(texts)


def resolve_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str):
    import torch

    if device == "cpu":
        return torch.float32
    if device == "mps" and dtype in {"bfloat16", "bf16"}:
        # older MPS stacks do not implement bf16 matmuls
        return torch.float16
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }[dtype]
