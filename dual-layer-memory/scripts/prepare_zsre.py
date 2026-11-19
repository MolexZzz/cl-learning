#!/usr/bin/env python3
"""Fetch / validate the ZsRE corpus used for behavioural labelling.

The paper's dataset is the Zero-Shot Relation Extraction benchmark (Levy et al.,
2017) filtered to facts with exactly m=30 paraphrase probes:
    N = 201,352 facts  ->  N * m = 6,040,560 probes.

Usage
-----
    # 1) download from the HuggingFace hub (needs `datasets` + network)
    python scripts/prepare_zsre.py --hf-name zjunlp/zsre --hf-config ZsRE --out data/zsre

    # 2) or point at an existing dump and validate it
    python scripts/prepare_zsre.py --input /path/to/zsre --out data/zsre

    # 3) validate only
    python scripts/prepare_zsre.py --input data/zsre --validate-only

Supported record layouts (auto-detected by ``dlmem.data.zsre``):

    {"input": "Kurt Vonnegut was born in Indianapolis.",
     "output": [{"question": "Where was Kurt Vonnegut born?", "answer": "Indianapolis"}, ...]}

    {"text": "...", "probes": [{"question": "...", "answer": "..."}, ...],
     "subject": "...", "relation": "...", "object": "..."}

    {"src": "...", "subject": "...", "relation": "...", "object": "...",
     "queries": [{"query": "...", "answer": "..."}, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dlmem.data.zsre import load_zsre_facts  # noqa: E402
from dlmem.utils import write_jsonl  # noqa: E402

DEFAULT_HF_CANDIDATES = ["zjunlp/zsre", "davidgur/zsre"]


def try_hf(name: str | None, config: str | None = None) -> list[dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "`datasets` is required for --hf-name; install requirements-gpu.txt "
            "or download the corpus manually"
        ) from exc
    names = [name] if name else DEFAULT_HF_CANDIDATES
    for candidate in names:
        try:
            ds = load_dataset(candidate, config, split="train") if config else load_dataset(candidate, split="train")
            print(f"loaded {candidate}: {len(ds)} rows")
            return [dict(row) for row in ds]
        except Exception as exc:  # pragma: no cover
            print(f"  {candidate} failed: {exc}")
    raise SystemExit("could not load any candidate ZsRE dataset; pass --input instead")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-name", default=None, help="HuggingFace dataset id")
    parser.add_argument("--hf-config", default=None, help="HuggingFace dataset config name")
    parser.add_argument("--input", default=None, help="existing file/directory of records")
    parser.add_argument("--out", default="data/zsre", help="output directory")
    parser.add_argument("--m", type=int, default=30, help="probes per fact")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        facts = load_zsre_facts(args.input or args.out, m=args.m)
        print(f"OK: {len(facts)} facts with exactly {args.m} probes")
        print(f"    total probes: {sum(len(f.probes) for f in facts):,}")
        return 0

    if args.input:
        facts = load_zsre_facts(args.input, m=args.m)
        print(f"normalised {len(facts)} facts from {args.input}")
    else:
        rows = try_hf(args.hf_name, args.hf_config)
        facts = load_zsre_facts_from_rows(rows, m=args.m)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "facts.jsonl", [f.to_dict() for f in facts])
    meta = {
        "n_facts": len(facts),
        "m": args.m,
        "n_probes": sum(len(f.probes) for f in facts),
        "note": "filtered to exactly m probes per fact",
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'facts.jsonl'}: {meta['n_facts']} facts / {meta['n_probes']:,} probes")
    return 0


def load_zsre_facts_from_rows(rows, m: int = 30):
    """Normalise in-memory HF rows through the same code path as files."""
    from dlmem.data.zsre import record_to_fact

    facts = []
    for i, row in enumerate(rows):
        fact = record_to_fact(row, i, m)
        if fact is not None:
            facts.append(fact)
    if not facts:
        raise SystemExit(f"no rows survived the exactly-{m}-probe filter")
    return facts


if __name__ == "__main__":
    raise SystemExit(main())
