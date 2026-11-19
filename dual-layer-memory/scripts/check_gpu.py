#!/usr/bin/env python3
"""GPU preflight: verify CUDA, load Qwen3, extract features and generate once.

Usage
-----
    python scripts/check_gpu.py                       # Qwen/Qwen3-1.7B
    python scripts/check_gpu.py --model Qwen/Qwen3-8B
    python scripts/check_gpu.py --config configs/paper_full.yaml --role large

This is the fastest way to catch a broken CUDA/PyTorch/Transformers stack before
starting the multi-hour labelling job.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--config", default=None)
    parser.add_argument("--role", default="small", choices=["small", "large"])
    parser.add_argument("--no-generate", action="store_true")
    args = parser.parse_args()

    import torch

    print(f"torch      : {torch.__version__}")
    print(f"cuda built : {torch.version.cuda}")
    print(f"cuda avail : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available; the paper-scale run needs a CUDA node.")
        return 2
    n = torch.cuda.device_count()
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"  gpu[{i}]   : {props.name}, {props.total_memory / 2**30:.1f} GiB")

    try:
        import transformers

        print(f"transformers: {transformers.__version__}")
    except Exception as exc:
        print(f"ERROR: transformers missing: {exc}")
        return 2

    from dlmem.config import load_config
    from dlmem.models.hf import HFBackbone

    cfg = load_config(args.config) if args.config else load_config(
        "configs/paper_full.yaml"
    )
    name = cfg.get_path(f"model.{args.role}") if args.config else args.model
    print(f"\nloading {name} ...")
    t0 = time.perf_counter()
    backbone = HFBackbone(
        name,
        dtype=cfg.get_path("model.dtype", "bfloat16"),
        device=cfg.get_path("model.device", "auto"),
        device_map=cfg.get_path("model.device_map", "auto"),
        feature_max_length=int(cfg.get_path("model.feature_max_length", 256)),
        qa_max_length=int(cfg.get_path("model.qa_max_length", 768)),
        batch_size=2,
        generate_batch_size=2,
        max_new_tokens=8,
    )
    encs = backbone.encode_batch(
        ["Kurt Vonnegut was born in Indianapolis.", "The capital of France is Paris."],
        pooling="mean",
    )
    print(f"encode ok in {time.perf_counter() - t0:.1f}s")
    for e in encs:
        print(f"  embedding {e.embedding.shape}, nll {e.nll_curve.shape}, mean {e.mean_nll:.3f}")

    if torch.cuda.is_available():
        used = sum(torch.cuda.memory_allocated(i) for i in range(n)) / 2**30
        print(f"gpu memory allocated: {used:.1f} GiB")

    if not args.no_generate:
        t1 = time.perf_counter()
        out = backbone.generate(
            ["Answer with one short phrase. Question: Where was Kurt Vonnegut born? Answer:"]
        )
        print(f"generate ok in {time.perf_counter() - t1:.1f}s -> {out[0]!r}")
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
