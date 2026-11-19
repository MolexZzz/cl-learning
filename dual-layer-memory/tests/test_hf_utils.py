from __future__ import annotations

import numpy as np

from dlmem.models.backbone import resolve_device, resolve_dtype
from dlmem.models.hf import _token_nll


def test_resolve_device_and_dtype():
    assert resolve_device("cpu") == "cpu"
    import torch

    assert resolve_dtype("float32", "cpu") is torch.float32
    assert resolve_dtype("bfloat16", "mps") is torch.float16


def test_token_nll_matches_manual_log_softmax():
    import torch

    torch.manual_seed(0)
    length, vocab = 5, 11
    logits = torch.randn(length, vocab)
    ids = torch.randint(0, vocab, (length,))
    nll = _token_nll(logits, ids, length, "left")
    expected = -torch.log_softmax(logits[:-1], dim=-1).gather(
        -1, ids[1:].unsqueeze(-1)
    ).squeeze(-1)
    assert nll.shape == (length - 1,)
    assert np.allclose(nll, expected.numpy(), atol=1e-5)

    # a single token has no next-token prediction
    assert _token_nll(logits[:1], ids[:1], 1, "left").shape == (1,)
