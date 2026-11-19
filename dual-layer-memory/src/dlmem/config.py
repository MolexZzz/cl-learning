"""Experiment configuration: YAML defaults + dotted CLI overrides.

Only the real-backbone path is supported: Qwen3-1.7B / Qwen3-8B over a ZsRE
fact corpus.  The paper-scale setup is ``configs/paper_full.yaml``; small GPU
nodes can scale down with ``configs/gpu_smoke.yaml`` and CLI overrides.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 13,
    "run_name": "default",
    "paths": {
        "data_dir": "data",
        "out_dir": "runs",
        "cache_dir": ".cache",
    },
    "model": {
        "small": "Qwen/Qwen3-1.7B",
        "large": "Qwen/Qwen3-8B",
        "dtype": "bfloat16",
        "device": "auto",
        # "auto" shards the 8B model with accelerate; set to null on a single GPU
        "device_map": "auto",
        "feature_max_length": 256,
        "qa_max_length": 768,
        "prompt_batch_size": 16,
        "generate_batch_size": 32,
        "max_new_tokens": 16,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "quantize_4bit": False,
        "chat_template": True,
    },
    "data": {
        "zsre_path": "data/zsre",
        "hf_name": "",
        "hf_config": "",
        "m_probes": 30,
        "max_facts": 0,
        "splits": [0.7, 0.15, 0.15],
    },
    "labeling": {
        "batch_size": 64,
        "max_probes_per_fact": 0,  # 0 = all (paper: all 30)
    },
    "features": {
        "pooling": "mean",  # mean | last_token
        "interp_len": 64,
        "batch_size": 64,
    },
    "router": {
        "hidden": 256,
        "depth": 2,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "epochs": 40,
        "batch_size": 256,
        "patience": 8,
        "loss": "mse",  # mse | margin_ranking
        "margin": 0.1,
        "lambda_s": 0.2,
        "feature_mode": "semantic_uncertainty",  # semantic | uncertainty | semantic_uncertainty
        "large_feature_mode": "semantic_uncertainty",
        # "none" for paper-scale training; "quantile" only helps tiny datasets
        "calibration": "none",
    },
    "gate": {
        "hidden": 128,
        "depth": 2,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "epochs": 40,
        "batch_size": 256,
        "patience": 8,
        "lambda_e": 0.008,
        # score_only | semantic | uncertainty | semantic_uncertainty | full
        "feature_mode": "full",
        # "train" follows App. C; "val" avoids a degenerate gate on small data
        "supervision": "train",
        "oversample": 8,
    },
    "cascade": {
        "escalation_cap": 0.45,
    },
    "sweep": {
        "lambda_s": [0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "lambda_e": [0.002, 0.005, 0.008, 0.02],
        "escalation_caps": [0.4, 0.6],
        "storage_grid": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    },
    "eval": {
        "split": "test",
        "episodes": 300,
        "turns": 500,
        "facts_per_episode": 100,
        "top_k": 3,
        "tau_flush": 0.5,
        "batch_episodes": 4,
        "max_episodes": 0,
        "operating_points": [
            {"lambda_s": 0.2, "lambda_e": 0.008},
            {"lambda_s": 0.08, "lambda_e": 0.005},
        ],
        "targets": {"heuristic": 0.59, "ppl": 0.717},
    },
    "sft": {
        "enabled": True,
        "source": "test",  # consolidate deployment memories (Sec. 3.4)
        "out_dir": "",
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 4,
        "grad_accum": 8,
        "max_length": 512,
        "lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "gradient_checkpointing": True,
        "max_steps": -1,
        "max_examples": 0,
        "max_qa_per_fact": 2,
        "relabel_train": True,
        "relabel_facts": 0,
    },
    "ablation": {
        "feature_modes": [
            "semantic",
            "uncertainty",
            "semantic_uncertainty",
        ],
        "poolings": ["mean", "last_token"],
        "losses": ["mse", "margin_ranking"],
        "gate_modes": [
            "score_only",
            "semantic",
            "uncertainty_only",
            "semantic_uncertainty",
            "full",
        ],
        "escalation_caps": [0.4, 0.6],
    },
    "compute": {
        # Relative costs of one forward pass.  8B / 1.7B ~= 4.7.
        "small_cost": 1.0,
        "large_cost": 4.7,
        "generation_tokens": 16,
        "gen_cost": 1.0,
    },
    "logging": {"level": "INFO"},
}


class Config(dict):
    """Nested dict with attribute access and dotted get/set."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    # convenience ---------------------------------------------------------
    @property
    def out_dir(self) -> Path:
        return Path(self["paths"]["out_dir"]) / str(self["run_name"])

    @property
    def data_dir(self) -> Path:
        return Path(self["paths"]["data_dir"])

    def path(self, *parts: str) -> Path:
        p = self.out_dir
        for part in parts:
            p = p / part
        return p

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_plain(self), f, indent=2)

    def clone(self) -> "Config":
        return Config(copy.deepcopy(dict(self)))


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def deep_update(base: dict, updates: Mapping[str, Any]) -> dict:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = copy.deepcopy(value)
    return base


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        cfg.set_path(key.strip(), _parse_scalar(raw.strip()))
    return cfg


def load_config(
    path: str | Path | None = None, overrides: Iterable[str] = ()
) -> Config:
    cfg = Config(copy.deepcopy(DEFAULTS))
    if path is not None and str(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        if not isinstance(user, dict):
            raise ValueError(f"config {path} must be a mapping")
        deep_update(cfg, user)
    apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg["router"]["loss"] not in {"mse", "margin_ranking"}:
        raise ValueError("router.loss must be 'mse' or 'margin_ranking'")
    if cfg["features"]["pooling"] not in {"mean", "last_token"}:
        raise ValueError("features.pooling must be 'mean' or 'last_token'")
    if cfg["gate"].get("supervision", "train") not in {"train", "val"}:
        raise ValueError("gate.supervision must be 'train' or 'val'")
    if cfg["router"].get("calibration", "none") not in {"none", "quantile"}:
        raise ValueError("router.calibration must be 'none' or 'quantile'")
    if cfg["sft"].get("source", "test") not in {"train", "test"}:
        raise ValueError("sft.source must be 'train' or 'test'")
    if cfg["eval"]["turns"] <= cfg["eval"]["facts_per_episode"]:
        raise ValueError("eval.turns must exceed eval.facts_per_episode")
    if not str(cfg["data"]["zsre_path"] or cfg["data"]["hf_name"]).strip():
        raise ValueError("set data.zsre_path (local dump) or data.hf_name (Hub id)")
    splits = cfg["data"]["splits"]
    if len(splits) != 3 or abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError("data.splits must be three fractions summing to 1")
