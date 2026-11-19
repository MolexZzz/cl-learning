from __future__ import annotations

import pytest

from dlmem.config import Config, load_config, validate_config


def test_paper_config_loads():
    cfg = load_config("configs/paper_full.yaml")
    assert cfg["model"]["small"] == "Qwen/Qwen3-1.7B"
    assert cfg["model"]["large"] == "Qwen/Qwen3-8B"
    assert cfg["data"]["m_probes"] == 30
    assert cfg["eval"]["episodes"] == 300
    assert cfg["eval"]["turns"] == 500
    assert cfg["router"]["calibration"] == "none"
    assert cfg["gate"]["supervision"] == "train"


def test_gpu_smoke_config_and_overrides():
    cfg = load_config(
        "configs/gpu_smoke.yaml",
        overrides=["eval.episodes=1", "model.device_map=null", "run_name=adhoc"],
    )
    assert cfg["eval"]["episodes"] == 1
    assert cfg["model"]["device_map"] is None
    assert cfg.out_dir.name == "adhoc"


def test_invalid_configs_rejected():
    cfg = load_config("configs/paper_full.yaml")
    cfg["data"]["zsre_path"] = ""
    cfg["data"]["hf_name"] = ""
    with pytest.raises(ValueError):
        validate_config(cfg)

    with pytest.raises(ValueError):
        load_config("configs/paper_full.yaml", overrides=["router.loss=ce"])
    with pytest.raises(ValueError):
        load_config("configs/paper_full.yaml", overrides=["gate.supervision=test"])


def test_config_paths():
    cfg = Config({"paths": {"out_dir": "runs"}, "run_name": "x"})
    assert cfg.path("a", "b").as_posix() == "runs/x/a/b"
