"""Optional real-model integration test.

Run on a GPU node with:

    DLMEM_HF_MODEL=Qwen/Qwen3-0.6B pytest tests/test_hf_integration.py -q -s

It is skipped when the variable is unset, so the default suite stays CPU-only
and never fabricates model behaviour.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

MODEL = os.environ.get("DLMEM_HF_MODEL", "")

pytestmark = pytest.mark.skipif(not MODEL, reason="set DLMEM_HF_MODEL to run real-model checks")


def test_real_backbone_encode_and_generate():
    from dlmem.config import load_config
    from dlmem.models.hf import load_backbone

    cfg = load_config(
        "configs/gpu_smoke.yaml",
        overrides=[
            f"model.small={MODEL}",
            f"model.large={MODEL}",
            "model.device_map=null",
            "model.prompt_batch_size=2",
            "model.generate_batch_size=2",
            "model.max_new_tokens=8",
        ],
    )
    backbone = load_backbone(cfg, "small")
    encs = backbone.encode_batch(["The capital of France is Paris."], pooling="mean")
    assert len(encs) == 1
    assert encs[0].embedding.ndim == 1 and encs[0].embedding.size > 0
    assert encs[0].nll_curve.size >= 1
    assert np.isfinite(encs[0].mean_nll)

    answers = backbone.generate(["Answer with one word. What is the capital of France?"])
    assert len(answers) == 1 and isinstance(answers[0], str)
