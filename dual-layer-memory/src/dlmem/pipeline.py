"""End-to-end experiment orchestration.

Stages mirror the paper:

    label  -> behavioural taxonomy on ZsRE probes          (Appendix A.1)
    data   -> streaming episodes                           (Appendix A.2)
    feat   -> semantic + uncertainty features              (Sec. 3.3.3)
    train  -> small/large routers + escalation gate        (Sec. 3.3, C)
    eval   -> online benchmark + baselines                 (Sec. 4.1-4.3)
    sft    -> write-back consolidation + re-labelling      (Sec. 3.4, 4.5)
    sweep  -> Pareto frontiers + ablations                 (Sec. 4.4, App. D)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Config
from .data.build import facts_by_split, prepare_facts, split_size
from .data.episodes import build_episodes, load_episodes, save_episodes
from .data.features import FeatureSet, extract_features, load_features, save_features
from .data.labeling import label_statistics, run_labeling
from .data.schema import Fact
from .eval.stream import StreamingResult, run_streaming
from .eval.sweep import (
    cascade_write,
    interp_to_storage_grid,
    sweep_cascade,
    sweep_single,
)
from .models.backbone import Backbone
from .models.router import CascadeRouter, Gate, Router
from .policies.base import PolicyOutput, RandomStorePolicy, FullStorePolicy, NoStorePolicy
from .policies.supervised import ArrayPolicy
from .policies.heuristic import HeuristicStorePolicy, PPLConditionalPolicy
from .policies.supervised import LogisticPolicy, MLPClassifierPolicy
from .tables import table1_markdown, table2_markdown, write_csv
from .train.sft import (
    SFTResult,
    build_sft_examples,
    evaluate_consolidation,
    run_hf_sft,
    select_write_facts,
    transition_matrix,
    transition_to_rows,
)
from .utils import (
    Timer,
    ensure_dir,
    get_logger,
    git_revision,
    human_duration,
    seed_everything,
    write_json,
    write_jsonl,
)

log = get_logger("dlmem.pipeline")


class Experiment:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        seed_everything(int(cfg["seed"]))
        self.run_dir = ensure_dir(cfg.out_dir)
        cfg.dump(self.run_dir / "config.json")
        self.facts: list[Fact] = []
        self.episodes = []
        self.small: Backbone | None = None
        self.large: Backbone | None = None
        self.label_backbone: Backbone | None = None
        self.sft_small: Backbone | None = None
        self.sft_large: Backbone | None = None
        self.features: dict[tuple[str, str, str], FeatureSet] = {}
        self.cascade: CascadeRouter | None = None
        self.sft_cascade: CascadeRouter | None = None
        self.stream_results: dict[str, StreamingResult] = {}
        self.sft_stream_results: dict[str, StreamingResult] = {}
        self._manifest: dict[str, Any] = {
            "config": str(self.run_dir / "config.json"),
            "seed": int(cfg["seed"]),
            "git": git_revision(),
            "stages": {},
        }
        self._save_manifest()

    # ------------------------------------------------------------------ #
    # infrastructure
    # ------------------------------------------------------------------ #
    def _save_manifest(self) -> None:
        write_json(self.run_dir / "manifest.json", self._manifest)

    def _mark(self, stage: str, **info: Any) -> None:
        self._manifest["stages"][stage] = {"time": info.pop("time", None), **info}
        self._save_manifest()

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    # ------------------------------------------------------------------ #
    # stage 1: corpus, labels, episodes, features
    # ------------------------------------------------------------------ #
    def prepare(self, force: bool = False) -> None:
        timer = Timer()
        with timer:
            self.facts = prepare_facts(self.cfg, force=force)
            self._build_backbones()
            self._label_splits(force=force)
            self._build_episodes(force=force)
            self._extract_all(force=force, pooling="mean", prefix="features")
        stats = {
            "n_facts": len(self.facts),
            "splits": {s: split_size(self.facts, s) for s in ("train", "val", "test")},
            "seconds": timer.elapsed,
        }
        self._mark("prepare", time=timer.elapsed, **stats)
        log.info("prepare done in %s", human_duration(timer.elapsed))

    def _build_backbones(self) -> None:
        from .models.hf import load_backbone

        self.small = load_backbone(self.cfg, "small")
        self.large = load_backbone(self.cfg, "large")
        # labelling / QA use the 8B frozen baseline (Appendix A.1)
        self.label_backbone = self.large

    def split_facts(self, split: str) -> list[Fact]:
        return facts_by_split(self.facts, split)

    def _label_splits(self, force: bool = False) -> None:
        labels_dir = ensure_dir(self.path("labels"))
        for split in ("train", "val", "test"):
            facts = self.split_facts(split)
            if not facts:
                continue
            run_labeling(
                facts,
                self.label_backbone,
                self.cfg,
                out_path=labels_dir / f"{split}.jsonl",
                force=force,
            )
        stats = {
            split: label_statistics(self.split_facts(split))
            for split in ("train", "val", "test")
            if self.split_facts(split)
        }
        write_json(labels_dir / "stats.json", stats)
        log.info(
            "label distribution (test): %s",
            stats.get("test", {}).get("fact_fraction", {}),
        )

    def _build_episodes(self, force: bool = False) -> None:
        path = self.path("episodes", "test.json")
        if path.exists() and not force:
            self.episodes = load_episodes(path)
            log.info("loaded %d episodes from %s", len(self.episodes), path)
            return
        ev = self.cfg["eval"]
        self.episodes = build_episodes(
            self.facts,
            episodes=int(ev["episodes"]),
            turns=int(ev["turns"]),
            facts_per_episode=int(ev["facts_per_episode"]),
            seed=int(self.cfg["seed"]),
            split=str(ev.get("split", "test")),
            m=int(self.cfg["data"]["m_probes"]),
        )
        save_episodes(path, self.episodes)
        self._mark("episodes", n=len(self.episodes))

    def _extract_all(self, force: bool = False, pooling: str = "mean", prefix: str = "features") -> None:
        if prefix == "features_sft" and self.sft_small is not None:
            backbones = (("small", self.sft_small), ("large", self.sft_large))
        else:
            backbones = (("small", self.small), ("large", self.large))
        for split in ("train", "val", "test"):
            facts = self.split_facts(split)
            if not facts:
                continue
            for source, backbone in backbones:
                path = self.path(prefix, f"{source}-{split}.npz")
                if path.exists() and not force:
                    fs = load_features(path)
                else:
                    fs = extract_features(
                        facts,
                        backbone,
                        pooling=pooling,
                        interp_len=int(self.cfg["features"]["interp_len"]),
                        batch_size=int(self.cfg["features"]["batch_size"]),
                    )
                    save_features(
                        path,
                        fs,
                        meta={
                            "source": source,
                            "split": split,
                            "pooling": pooling,
                            "backbone": backbone.name,
                            "n": len(facts),
                        },
                    )
                self.features[(source, split, prefix)] = fs
        self._mark("features", pooling=pooling, prefix=prefix)

    def feature(self, source: str, split: str, prefix: str = "features") -> FeatureSet:
        key = (source, split, prefix)
        if key not in self.features:
            path = self.path(prefix, f"{source}-{split}.npz")
            if not path.exists():
                raise FileNotFoundError(f"features missing: {path}; run the feature stage first")
            self.features[key] = load_features(path)
        return self.features[key]

    # ------------------------------------------------------------------ #
    # stage 2: routers + gate
    # ------------------------------------------------------------------ #
    def fit_cascade(
        self,
        out_dir: str | Path,
        feature_prefix: str = "features",
        feature_modes: dict[str, str] | None = None,
        loss: str | None = None,
        pooling_note: str = "",
    ) -> CascadeRouter:
        router_cfg = dict(self.cfg["router"])
        if loss:
            router_cfg["loss"] = loss
        modes = {
            "small": str(router_cfg.get("feature_mode", "semantic_uncertainty")),
            "large": str(router_cfg.get("large_feature_mode", "semantic")),
        }
        if feature_modes:
            modes.update(feature_modes)
        small_router = Router.fit(
            [self.feature("small", "train", feature_prefix)],
            [self.feature("small", "val", feature_prefix)],
            router_cfg,
            lambda_s=float(router_cfg["lambda_s"]),
            feature_mode=modes["small"],
            source="small",
            seed=int(self.cfg["seed"]),
        )
        large_router = Router.fit(
            [self.feature("large", "train", feature_prefix)],
            [self.feature("large", "val", feature_prefix)],
            router_cfg,
            lambda_s=float(router_cfg["lambda_s"]),
            feature_mode=modes["large"],
            source="large",
            seed=int(self.cfg["seed"]) + 1,
        )
        gate = Gate.fit(
            [self.feature("small", "val", feature_prefix)],
            small_router,
            large_router,
            [self.feature("large", "val", feature_prefix)],
            dict(self.cfg["gate"]),
            lambda_e=float(self.cfg["gate"]["lambda_e"]),
            feature_mode=str(self.cfg["gate"].get("feature_mode", "full")),
            seed=int(self.cfg["seed"]) + 2,
        )
        cascade = CascadeRouter(
            small_router,
            large_router,
            gate,
            small_cost=float(self.cfg["compute"]["small_cost"]),
            large_cost=float(self.cfg["compute"]["large_cost"]),
        )
        out_dir = Path(out_dir)
        cascade.save(out_dir)
        write_json(
            out_dir / "training_summary.json",
            {
                "small_history": small_router.history,
                "large_history": large_router.history,
                "gate_history": gate.history,
                "feature_modes": modes,
                "loss": router_cfg.get("loss"),
                "pooling_note": pooling_note,
            },
        )
        return cascade

    def load_cascade(self, which: str = "routers") -> CascadeRouter:
        path = self.path(which)
        if not (path / "cascade.json").exists():
            raise FileNotFoundError(
                f"no cascade at {path}; run `train` first"
            )
        cascade = CascadeRouter.load(path, dict(self.cfg["router"]))
        if which == "routers_sft":
            self.sft_cascade = cascade
        else:
            self.cascade = cascade
        return cascade

    def train(self) -> CascadeRouter:
        timer = Timer()
        with timer:
            self.cascade = self.fit_cascade(self.path("routers"))
        self._mark("train", time=timer.elapsed, dir=str(self.path("routers")))
        return self.cascade

    # ------------------------------------------------------------------ #
    # stage 3: streaming evaluation + baselines
    # ------------------------------------------------------------------ #
    def _test_context(self):
        facts = self.split_facts("test")
        return (
            facts,
            self.feature("small", "test"),
            self.feature("large", "test"),
        )

    def cascade_output(
        self,
        cascade: CascadeRouter,
        lambda_s: float,
        lambda_e: float,
        name: str,
        escalation_cap: float | None = None,
        prefix: str = "features",
        split: str = "test",
    ) -> ArrayPolicy:
        if escalation_cap is None:
            escalation_cap = float(self.cfg["cascade"].get("escalation_cap", 0.0)) or None
        small_fs = self.feature("small", split, prefix)
        large_fs = self.feature("large", split, prefix)
        sp = cascade.small.predict_sets([small_fs])
        lp = cascade.large.predict_sets([large_fs])
        if cascade.gate is not None:
            g = cascade.gate.predict([small_fs], sp)
            train_lambda_e = cascade.gate.lambda_e
        else:
            g = np.zeros(len(sp), dtype=np.float32)
            train_lambda_e = lambda_e
        write, escalate = cascade_write(
            sp,
            lp,
            g,
            lambda_s,
            lambda_e,
            cascade.small.train_lambda_s,
            train_lambda_e,
            escalation_cap,
        )
        cost = float(self.cfg["compute"]["small_cost"]) + escalate.astype(
            np.float32
        ) * float(self.cfg["compute"]["large_cost"])
        return ArrayPolicy(
            name,
            write,
            escalate,
            cost,
            score=g,
            cost_label=f"Fwd1.7B + {100 * escalate.mean():.1f}% Fwd8B",
            meta={"lambda_s": lambda_s, "lambda_e": lambda_e, "cap": escalation_cap},
        )

    def build_policies(self, prefix: str = "features") -> dict[str, PolicyOutput]:
        facts = self.split_facts("test")
        train_facts = self.split_facts("train")
        val_facts = self.split_facts("val")
        fs_train_small = self.feature("small", "train", prefix)
        fs_test_small = self.feature("small", "test", prefix)
        fs_val_small = self.feature("small", "val", prefix)
        fs_train_large = self.feature("large", "train", prefix)
        fs_test_large = self.feature("large", "test", prefix)
        fs_val_large = self.feature("large", "val", prefix)
        backbone = self.sft_large if prefix == "features_sft" and self.sft_large else self.large

        policies: dict[str, PolicyOutput] = {}
        policies["no_store"] = NoStorePolicy().decide_all(facts)
        policies["random_0.5"] = RandomStorePolicy(0.5, int(self.cfg["seed"])).decide_all(facts)
        policies["random_0.8"] = RandomStorePolicy(0.8, int(self.cfg["seed"])).decide_all(facts)
        policies["full_store"] = FullStorePolicy().decide_all(facts)

        heuristic = HeuristicStorePolicy(
            backbone,
            target_storage=float(self.cfg["eval"]["targets"]["heuristic"]),
            batch_size=int(self.cfg["labeling"]["batch_size"]),
        )
        heuristic.calibrate(val_facts)
        policies["heuristic"] = heuristic.decide_all(facts)

        ppl = PPLConditionalPolicy(
            backbone, target_storage=float(self.cfg["eval"]["targets"]["ppl"])
        )
        ppl.calibrate(val_facts)
        policies["ppl_conditional"] = ppl.decide_all(facts)

        logistic = LogisticPolicy(feature_mode="semantic").fit(fs_train_small)
        out = logistic.decide_all(facts, fs_test_small)
        out.cost = np.full(len(facts), float(self.cfg["compute"]["large_cost"]))
        out.cost_label = "1 x Fwd8B"
        policies["logistic"] = out

        mlp = MLPClassifierPolicy(
            feature_mode="semantic_uncertainty", cfg=dict(self.cfg["router"])
        ).fit(fs_train_small)
        out = mlp.decide_all(facts, fs_test_small)
        out.cost = np.full(len(facts), float(self.cfg["compute"]["large_cost"]))
        out.cost_label = "1 x Fwd8B"
        policies["mlp"] = out

        cascade = self.sft_cascade if prefix == "features_sft" and self.sft_cascade else self.cascade
        if cascade is not None:
            for op in self.cfg["eval"]["operating_points"]:
                name = f"write_router_{op['lambda_s']:g}"
                policies[name] = self.cascade_output(
                    cascade,
                    float(op["lambda_s"]),
                    float(op["lambda_e"]),
                    name,
                    prefix=prefix,
                )
        _ = (train_facts, fs_train_large, fs_val_small, fs_val_large, fs_test_large)
        return policies

    def evaluate(self, prefix: str = "features", tag: str = "base") -> dict[str, StreamingResult]:
        policies = self.build_policies(prefix=prefix)
        backbone = self.sft_large if tag == "sft" and self.sft_large else self.large
        facts = self.split_facts("test")
        fact_ids = [f.fact_id for f in facts]
        results: dict[str, StreamingResult] = {}
        for name, output in policies.items():
            if hasattr(output, "decide_all"):
                output = output.decide_all(facts)
            result = run_streaming(
                self.episodes,
                facts,
                output,
                fact_ids,
                backbone,
                top_k=int(self.cfg["eval"]["top_k"]),
                batch_episodes=int(self.cfg["eval"]["batch_episodes"]),
                max_episodes=int(self.cfg["eval"].get("max_episodes", 0)),
            )
            results[name] = result
            write_json(self.path("results", f"{tag}_stream_{name}.json"), result.to_dict())
        if tag == "sft":
            self.sft_stream_results = results
        else:
            self.stream_results = results
        return results

    def table1(self) -> list[dict]:
        rows: list[dict] = []

        def add_section(title: str) -> None:
            rows.append({"section": title})

        def add(label: str, result: StreamingResult) -> None:
            m = result.metrics
            rows.append(
                {
                    "label": label,
                    "policy": result.policy,
                    "em": m["em"],
                    "f1": m["f1"],
                    "refusal_rate": m["refusal_rate"],
                    "storage": m["storage"],
                    "store_precision": m["store_precision"],
                    "store_recall": m["store_recall"],
                    "store_f1": m["store_f1"],
                    "cost_label": result.metrics.get("cost_label", ""),
                }
            )

        res = self.stream_results
        if res:
            add_section("Base Model Framework (qwen3-8B)")
            add("No Store", res["no_store"])
            add("Random Store (p = 0.5)", res["random_0.5"])
            add("Random Store (p = 0.8)", res["random_0.8"])
            add("Full Store", res["full_store"])
            add("Heuristic Store", res["heuristic"])
            add("PPL-Cond. Store", res["ppl_conditional"])
            add("Logistic Regression", res["logistic"])
            add("MLP Classifier", res["mlp"])
            for op in self.cfg["eval"]["operating_points"]:
                name = f"write_router_{op['lambda_s']:g}"
                if name in res:
                    add(
                        f"Write Router (λs={op['lambda_s']:g}, λe={op['lambda_e']:g})",
                        res[name],
                    )
        sres = self.sft_stream_results
        if sres:
            add_section("SFT Model Framework (Knowledge-adapted qwen3-8B)")
            add("No Store", sres["no_store"])
            add("Full Store", sres["full_store"])
            for op in self.cfg["eval"]["operating_points"]:
                name = f"write_router_{op['lambda_s']:g}"
                if name in sres:
                    add(
                        f"Write Router SFT (λs={op['lambda_s']:g}, λe={op['lambda_e']:g})",
                        sres[name],
                    )
        return rows

    def export_tables(self) -> None:
        rows = self.table1()
        if not rows:
            return
        results_dir = ensure_dir(self.path("results"))
        write_csv(rows, results_dir / "table1.csv")
        (results_dir / "table1.md").write_text(table1_markdown(rows), encoding="utf-8")
        self._mark("table1", rows=len(rows))

    # ------------------------------------------------------------------ #
    # stage 4: slow consolidation (write-back)
    # ------------------------------------------------------------------ #
    def run_sft(self) -> dict:
        timer = Timer()
        base_labels = {f.fact_id: f.label for f in self.facts}
        test_facts = self.split_facts("test")
        source_split = str(self.cfg["sft"].get("source", "test"))
        sft_facts = self.split_facts(source_split)

        assert self.cascade is not None, "train routers before write-back"
        # Sec. 3.4: consolidate the external memory accumulated during
        # deployment (test stream) -- or the train split for a leakage-free run.
        deploy_output = self.cascade_output(
            self.cascade, 0.08, 0.005, "deploy", split=source_split
        ).decide_all(sft_facts)
        selected = select_write_facts(
            sft_facts, deploy_output, [f.fact_id for f in sft_facts]
        )
        examples = build_sft_examples(
            selected,
            max_qa_per_fact=int(self.cfg["sft"]["max_qa_per_fact"]),
            max_examples=int(self.cfg["sft"]["max_examples"]),
        )
        if not examples:
            raise RuntimeError(
                "write-back selected no examples; check router decisions/labels "
                "before running SFT"
            )
        sft_dir = ensure_dir(self.path("sft"))
        write_jsonl(sft_dir / "examples.jsonl", examples)

        with timer:
            from .models.hf import HFBackbone

            stats = {}
            for role, model_name in (
                ("small", self.cfg["model"]["small"]),
                ("large", self.cfg["model"]["large"]),
            ):
                res = run_hf_sft(
                    model_name,
                    examples,
                    self.cfg,
                    sft_dir / role,
                    seed=int(self.cfg["seed"]),
                )
                stats[role] = res.stats
            adapter_kwargs = dict(
                dtype=self.cfg["model"]["dtype"],
                device=self.cfg["model"]["device"],
                device_map=self.cfg["model"].get("device_map") or None,
                feature_max_length=int(self.cfg["model"]["feature_max_length"]),
                qa_max_length=int(self.cfg["model"]["qa_max_length"]),
                batch_size=int(self.cfg["model"]["prompt_batch_size"]),
                generate_batch_size=int(self.cfg["model"]["generate_batch_size"]),
                max_new_tokens=int(self.cfg["model"]["max_new_tokens"]),
            )
            self.sft_small = HFBackbone(
                self.cfg["model"]["small"],
                adapter_path=str(sft_dir / "small"),
                **adapter_kwargs,
            )
            self.sft_large = HFBackbone(
                self.cfg["model"]["large"],
                adapter_path=str(sft_dir / "large"),
                **adapter_kwargs,
            )
            sft_result = SFTResult(
                backend="hf",
                adapter_path=str(sft_dir),
                n_examples=len(examples),
                n_facts=len(selected),
                stats=stats,
            )

        # consolidation validation / eviction on the memories we trained on
        consolidated = evaluate_consolidation(
            selected,
            self.sft_large,
            tau_flush=float(self.cfg["eval"]["tau_flush"]),
            batch_size=int(self.cfg["labeling"]["batch_size"]),
        )

        # re-label with the consolidated model
        labels_sft = ensure_dir(self.path("labels_sft"))
        splits_to_relabel = ["test"]
        if bool(self.cfg["sft"].get("relabel_train", True)):
            splits_to_relabel = ["train", "val", "test"]
        cap = int(self.cfg["sft"].get("relabel_facts", 0) or 0)
        for split in splits_to_relabel:
            facts = self.split_facts(split)
            if cap and len(facts) > cap:
                facts = facts[:cap]
            run_labeling(
                facts,
                self.sft_large,
                self.cfg,
                out_path=labels_sft / f"{split}.jsonl",
                force=True,
            )
        after_labels = {f.fact_id: f.label for f in test_facts}
        transition = transition_matrix(
            {f.fact_id: base_labels[f.fact_id] for f in test_facts}, after_labels
        )
        write_json(sft_dir / "transition.json", transition)
        write_csv(transition_to_rows(transition), sft_dir / "transition.csv")
        (sft_dir / "transition.md").write_text(table2_markdown(transition), encoding="utf-8")
        write_json(sft_dir / "consolidation.json", consolidated)

        # features + routers on the consolidated model
        self._extract_all(force=True, pooling="mean", prefix="features_sft")
        self.sft_cascade = self.fit_cascade(self.path("routers_sft"), feature_prefix="features_sft")

        timer.stop()
        info = {
            "n_examples": sft_result.n_examples,
            "n_selected_facts": len(selected),
            "consolidation": {k: v for k, v in consolidated.items() if k != "flushed_ids"},
            "transition": {k: v for k, v in transition.items() if k != "matrix"},
            "seconds": timer.elapsed,
        }
        self._mark("sft", time=timer.elapsed, **info)
        log.info("write-back done in %s", human_duration(timer.elapsed))
        return info

    # ------------------------------------------------------------------ #
    # stage 5: Pareto sweep
    # ------------------------------------------------------------------ #
    def sweep(self) -> dict:
        timer = Timer().start()
        facts = self.split_facts("test")
        small_fs = self.feature("small", "test")
        large_fs = self.feature("large", "test")
        assert self.cascade is not None
        sp = self.cascade.small.predict_sets([small_fs])
        lp = self.cascade.large.predict_sets([large_fs])
        g = self.cascade.gate.predict([small_fs], sp) if self.cascade.gate else np.zeros(len(sp))
        lambdas = [float(x) for x in self.cfg["sweep"]["lambda_s"]]
        rows: list[dict] = []
        rows += sweep_single(facts, sp, self.cascade.small.train_lambda_s, lambdas, cost=1.0)
        rows += sweep_single(facts, lp, self.cascade.large.train_lambda_s, lambdas, cost=float(self.cfg["compute"]["large_cost"]))
        by_cap: dict[str, list[dict]] = {}
        for cap in self.cfg["sweep"]["escalation_caps"]:
            c_rows = sweep_cascade(
                facts,
                sp,
                lp,
                g,
                lambdas,
                lambda_e=float(self.cfg["gate"]["lambda_e"]),
                train_lambda_s=self.cascade.small.train_lambda_s,
                train_lambda_e=self.cascade.gate.lambda_e if self.cascade.gate else 0.0,
                escalation_cap=float(cap),
                small_cost=float(self.cfg["compute"]["small_cost"]),
                large_cost=float(self.cfg["compute"]["large_cost"]),
            )
            for r in c_rows:
                r["cap"] = float(cap)
                r["name"] = f"cascade_{cap:g}"
            rows += c_rows
            by_cap[f"cascade_{cap:g}"] = c_rows
        write_csv(rows, self.path("results", "pareto_curves.csv"))

        # Figure 2
        from . import plots

        curves = {
            "small": sweep_single(facts, sp, self.cascade.small.train_lambda_s, lambdas, 1.0),
            "large": sweep_single(facts, lp, self.cascade.large.train_lambda_s, lambdas, 4.7),
        }
        curves.update(by_cap)
        plots.plot_pareto(
            curves,
            self.path("figures", "fig2_pareto"),
            title="Main experiment on the held-out test set",
            xlim=(0.5, 0.9),
        )

        gaps = {}
        cap06 = by_cap.get("cascade_0.6", [])
        small_rows = curves["small"]
        large_rows = curves["large"]
        for budget in (0.6, 0.7, 0.8):
            sm = _interp_at(small_rows, budget)
            lg = _interp_at(large_rows, budget)
            ca = _interp_at(cap06, budget)
            if ca is not None and lg is not None and sm is not None:
                gaps[str(budget)] = {
                    "cascade_minus_large": ca - lg,
                    "cascade_minus_small": ca - sm,
                }
        write_json(self.path("results", "pareto_gaps.json"), gaps)
        timer.stop()
        self._mark("sweep", time=timer.elapsed, gaps=gaps)
        return {"curves": rows, "gaps": gaps}

    # ------------------------------------------------------------------ #
    # stage 6: ablations (Appendix D)
    # ------------------------------------------------------------------ #
    def ablate(self) -> dict:
        timer = Timer().start()
        facts = self.split_facts("test")
        lambdas = [float(x) for x in self.cfg["sweep"]["lambda_s"]]
        out_dir = ensure_dir(self.path("ablations"))
        grids = [float(x) for x in self.cfg["sweep"]["storage_grid"]]
        assert self.cascade is not None

        # --- feature ablation, step 1 (small side) ---------------------
        small_abl: dict[str, list[dict]] = {}
        for mode in self.cfg["ablation"]["feature_modes"]:
            router = Router.fit(
                [self.feature("small", "train")],
                [self.feature("small", "val")],
                dict(self.cfg["router"]),
                float(self.cfg["router"]["lambda_s"]),
                feature_mode=mode,
                source="small",
            )
            pred = router.predict_sets([self.feature("small", "test")])
            small_abl[mode] = sweep_single(
                facts, pred, router.train_lambda_s, lambdas, cost=1.0
            )
        write_csv(
            [dict(r, name=k) for k, rows in small_abl.items() for r in rows],
            out_dir / "feature_small.csv",
        )

        # --- feature ablation, step 1 (large vs small+large) ----------
        large_abl: dict[str, list[dict]] = {}
        combos = {
            "large: semantic": (["large"], "semantic"),
            "small+large: semantic": (["small", "large"], "semantic"),
            "large: uncertainty": (["large"], "uncertainty"),
            "small+large: uncertainty": (["small", "large"], "uncertainty"),
            "large: semantic+uncertainty": (["large"], "semantic_uncertainty"),
            "small+large: semantic+uncertainty": (["small", "large"], "semantic_uncertainty"),
        }
        for name, (sources, mode) in combos.items():
            train_sets = [self.feature(s, "train") for s in sources]
            val_sets = [self.feature(s, "val") for s in sources]
            test_sets = [self.feature(s, "test") for s in sources]
            router = Router.fit(
                train_sets,
                val_sets,
                dict(self.cfg["router"]),
                float(self.cfg["router"]["lambda_s"]),
                feature_mode=mode,
                source="+".join(sources),
            )
            pred = router.predict_sets(test_sets)
            large_abl[name] = sweep_single(
                facts, pred, router.train_lambda_s, lambdas, cost=4.7
            )
        write_csv(
            [dict(r, name=k) for k, rows in large_abl.items() for r in rows],
            out_dir / "feature_large.csv",
        )

        # --- pooling and loss ablation --------------------------------
        self._extract_all(force=False, pooling="last_token", prefix="features_last")
        pooling_abl: dict[str, dict[str, list[dict]]] = {"small": {}, "large": {}}
        for source in ("small", "large"):
            for pooling, prefix in (("mean", "features"), ("last_token", "features_last")):
                for loss in self.cfg["ablation"]["losses"]:
                    name = f"{pooling} + {loss}"
                    router = Router.fit(
                        [self.feature(source, "train", prefix)],
                        [self.feature(source, "val", prefix)],
                        dict(self.cfg["router"]),
                        float(self.cfg["router"]["lambda_s"]),
                        feature_mode="semantic",
                        source=source,
                        loss=loss,
                    )
                    pred = router.predict_sets([self.feature(source, "test", prefix)])
                    pooling_abl[source][name] = sweep_single(
                        facts, pred, router.train_lambda_s, lambdas, cost=1.0
                    )
        write_csv(
            [dict(r, name=k, panel=src) for src, d in pooling_abl.items() for k, rows in d.items() for r in rows],
            out_dir / "pooling_loss.csv",
        )

        # --- gate design ablation -------------------------------------
        sp = self.cascade.small.predict_sets([self.feature("small", "test")])
        lp = self.cascade.large.predict_sets([self.feature("large", "test")])
        gate_panels: dict[str, dict[str, list[dict]]] = {}
        gate_modes = ["full"] + [m for m in self.cfg["ablation"]["gate_modes"] if m != "full"]
        for cap in self.cfg["ablation"]["escalation_caps"]:
            panel: dict[str, list[dict]] = {}
            for mode in gate_modes:
                gate = Gate.fit(
                    [self.feature("small", "val")],
                    self.cascade.small,
                    self.cascade.large,
                    [self.feature("large", "val")],
                    dict(self.cfg["gate"]),
                    lambda_e=float(self.cfg["gate"]["lambda_e"]),
                    feature_mode=mode,
                )
                g = gate.predict([self.feature("small", "test")], sp)
                panel[mode] = sweep_cascade(
                    facts,
                    sp,
                    lp,
                    g,
                    lambdas,
                    lambda_e=float(self.cfg["gate"]["lambda_e"]),
                    train_lambda_s=self.cascade.small.train_lambda_s,
                    train_lambda_e=gate.lambda_e,
                    escalation_cap=float(cap),
                    small_cost=float(self.cfg["compute"]["small_cost"]),
                    large_cost=float(self.cfg["compute"]["large_cost"]),
                )
            gate_panels[f"{cap:g}"] = panel
        write_csv(
            [
                dict(r, name=k, cap=cap)
                for cap, panel in gate_panels.items()
                for k, rows in panel.items()
                for r in rows
            ],
            out_dir / "gate_design.csv",
        )

        from . import plots

        plots.plot_two_panel_ablation(
            small_abl, large_abl, self.path("figures", "fig4_feature_ablation")
        )
        plots.plot_two_panel_ablation(
            pooling_abl["small"],
            pooling_abl["large"],
            self.path("figures", "fig5_pooling_loss"),
            left_title="Step 2: small semantic",
            right_title="Step 2: large semantic",
        )
        plots.plot_gate_ablation(gate_panels, self.path("figures", "fig6_gate_ablation"))

        gaps = _escalation_gaps(
            self.cascade,
            facts,
            self.feature("small", "test"),
            self.feature("large", "test"),
            grids,
        )
        write_json(out_dir / "escalation_gaps.json", gaps)
        timer.stop()
        self._mark("ablate", time=timer.elapsed)
        return {"gaps": gaps}

    # ------------------------------------------------------------------ #
    def _plot_transition(self) -> None:
        path = self.path("sft", "transition.json")
        if not path.exists():
            return
        from . import plots
        from .utils import read_json

        result = read_json(path)
        plots.plot_transition_heatmap(
            result["matrix"], result["order"], self.path("figures", "fig7_transition")
        )

    def run_all(self) -> None:
        self.prepare()
        self.train()
        self.evaluate()
        self.export_tables()
        self.sweep()
        self.ablate()
        if bool(self.cfg["sft"].get("enabled", True)):
            self.run_sft()
            self.evaluate(prefix="features_sft", tag="sft")
            self.export_tables()
            self._plot_transition()
        self._save_manifest()


def _interp_at(rows: Sequence[dict], storage: float) -> float | None:
    grid = interp_to_storage_grid(rows, [storage])
    return grid[0]["em"] if grid else None


def _escalation_gaps(cascade, facts, small_fs, large_fs, grids) -> dict:
    """Appendix D.3: cascade-vs-large and cascade-vs-small EM at fixed storage."""
    sp = cascade.small.predict_sets([small_fs])
    lp = cascade.large.predict_sets([large_fs])
    g = cascade.gate.predict([small_fs], sp) if cascade.gate else np.zeros(len(sp))
    lambdas = [float(x) for x in np.linspace(0.02, 0.9, 45)]
    small_rows = sweep_single(facts, sp, cascade.small.train_lambda_s, lambdas, 1.0)
    large_rows = sweep_single(facts, lp, cascade.large.train_lambda_s, lambdas, 4.7)
    out: dict = {}
    for cap in (0.4, 0.6):
        rows = sweep_cascade(
            facts,
            sp,
            lp,
            g,
            lambdas,
            lambda_e=cascade.gate.lambda_e if cascade.gate else 0.008,
            train_lambda_s=cascade.small.train_lambda_s,
            train_lambda_e=cascade.gate.lambda_e if cascade.gate else 0.0,
            escalation_cap=cap,
        )
        per = {}
        for budget in grids:
            ca = _interp_at(rows, float(budget))
            lg = _interp_at(large_rows, float(budget))
            sm = _interp_at(small_rows, float(budget))
            if None not in (ca, lg, sm):
                per[str(budget)] = {
                    "cascade_minus_large": ca - lg,
                    "cascade_minus_small": ca - sm,
                }
        out[f"cap_{cap:g}"] = per
    return out
