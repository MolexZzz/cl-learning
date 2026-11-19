"""Command-line interface.

Examples
--------
    python -m dlmem.cli prepare  --config configs/paper_full.yaml
    python -m dlmem.cli train    --config configs/paper_full.yaml
    python -m dlmem.cli evaluate --config configs/paper_full.yaml
    python -m dlmem.cli run-all  --config configs/gpu_smoke.yaml
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .utils import get_logger

log = get_logger("dlmem.cli")

STAGES = [
    "prepare",
    "train",
    "evaluate",
    "sweep",
    "ablate",
    "sft",
    "tables",
    "run-all",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlmem",
        description="Dual-Layer Agentic Memory: fast write routing + slow consolidation",
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in STAGES:
        p = sub.add_parser(stage, help=f"run the {stage} stage")
        p.add_argument("--config", default=None, help="YAML config path")
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="dotted config override, e.g. --set eval.episodes=4",
        )
        p.add_argument("--force", action="store_true", help="ignore cached artifacts")
        if stage in {"sft", "evaluate", "sweep", "ablate", "tables"}:
            p.add_argument("--prepare", action="store_true", help="run the prepare stage first")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    from .pipeline import Experiment

    exp = Experiment(cfg)
    stage = args.stage
    if stage == "prepare":
        exp.prepare(force=args.force)
    elif stage == "train":
        if not exp.facts:
            exp.prepare(force=False)
        exp.cascade = exp.fit_cascade(exp.path("routers"))
    elif stage == "evaluate":
        if args.prepare or not exp.facts:
            exp.prepare()
        if exp.cascade is None:
            exp.load_cascade()
        exp.evaluate()
        exp.export_tables()
    elif stage == "sweep":
        if args.prepare or not exp.facts:
            exp.prepare()
        if exp.cascade is None:
            exp.load_cascade()
        exp.sweep()
    elif stage == "ablate":
        if args.prepare or not exp.facts:
            exp.prepare()
        if exp.cascade is None:
            exp.load_cascade()
        exp.ablate()
    elif stage == "sft":
        if args.prepare or not exp.facts:
            exp.prepare()
        if exp.cascade is None:
            exp.load_cascade()
        exp.run_sft()
        exp.evaluate(prefix="features_sft", tag="sft")
        exp.export_tables()
        exp._plot_transition()
    elif stage == "tables":
        exp.export_tables()
    elif stage == "run-all":
        exp.run_all()
    else:  # pragma: no cover
        raise SystemExit(f"unknown stage {stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
