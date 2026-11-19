"""Slow consolidation: write-back SFT and validated memory eviction.

Paper Sec. 3.4 / Appendix C:

* selected external memories (write-new / write-update) become supervision:
  question-level QA pairs and sentence-level targets;
* SFT minimises causal LM loss on assistant tokens only;
* a fact leaves external memory only when its probes are answered correctly
  without retrieval (mean EM >= tau_flush).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import Config
from ..data.schema import Fact
from ..memory.qa import build_qa_prompt
from ..utils import ensure_dir, exact_match, get_logger, write_json

log = get_logger("dlmem.sft")


@dataclass
class SFTResult:
    backend: str
    adapter_path: str
    n_examples: int
    n_facts: int
    stats: dict


def select_write_facts(
    facts: Sequence[Fact],
    policy_output,
    fact_ids: Sequence[str],
    labels: Sequence[str] = ("write-new", "write-update"),
) -> list[Fact]:
    write_by_id = {fid: bool(w) for fid, w in zip(fact_ids, policy_output.write)}
    return [f for f in facts if write_by_id.get(f.fact_id, False) and f.label in labels]


def build_sft_examples(
    facts: Sequence[Fact],
    max_qa_per_fact: int = 2,
    max_examples: int = 0,
    include_sentence: bool = True,
) -> list[dict]:
    examples: list[dict] = []
    for fact in facts:
        for probe in fact.probes[:max_qa_per_fact]:
            examples.append(
                {
                    "prompt": build_qa_prompt(probe.question),
                    "response": probe.answer,
                    "fact_id": fact.fact_id,
                    "kind": "qa",
                }
            )
        if include_sentence:
            examples.append(
                {
                    "prompt": "State the following fact.\nAnswer:",
                    "response": fact.text,
                    "fact_id": fact.fact_id,
                    "kind": "sentence",
                }
            )
        if max_examples and len(examples) >= max_examples:
            break
    return examples[:max_examples] if max_examples else examples


# --------------------------------------------------------------------------- #
# HuggingFace SFT
# --------------------------------------------------------------------------- #
def run_hf_sft(
    base_model: str,
    examples: Sequence[dict],
    cfg: Config,
    out_dir: str | Path,
    seed: int = 13,
) -> SFTResult:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sft = cfg["sft"]
    out_dir = ensure_dir(out_dir)
    device = cfg.get_path("model.device", "auto")
    from ..models.backbone import resolve_device, resolve_dtype

    device = resolve_device(device)
    dtype = resolve_dtype(cfg.get_path("model.dtype", "bfloat16"), device)

    log.info("SFT on %s (%d examples) -> %s", base_model, len(examples), out_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    from ..models.hf import _from_pretrained_compat

    model = _from_pretrained_compat(
        AutoModelForCausalLM,
        base_model,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=cfg.get_path("model.attn_implementation", "sdpa"),
    ).to(device)
    model.config.use_cache = False
    if bool(sft.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if bool(sft.get("lora", True)):
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=int(sft.get("lora_r", 16)),
            lora_alpha=int(sft.get("lora_alpha", 32)),
            lora_dropout=float(sft.get("lora_dropout", 0.05)),
            target_modules=list(sft.get("target_modules", [])),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        if bool(sft.get("gradient_checkpointing", True)):
            model.enable_input_require_grads()
        model.print_trainable_parameters()

    max_length = int(sft.get("max_length", 256))

    class SFTDataset(Dataset):
        def __init__(self, rows: Sequence[dict]) -> None:
            self.rows = list(rows)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            messages = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ]
            if getattr(tokenizer, "chat_template", None):
                try:
                    full_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                    prompt_text = tokenizer.apply_chat_template(
                        [messages[0]],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    full_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    prompt_text = tokenizer.apply_chat_template(
                        [messages[0]], tokenize=False, add_generation_prompt=True
                    )
            else:
                prompt_text = row["prompt"]
                full_text = row["prompt"] + row["response"]
            full = tokenizer(full_text, truncation=True, max_length=max_length)
            prompt = tokenizer(prompt_text, truncation=True, max_length=max_length)
            labels = list(full["input_ids"])
            for i in range(min(len(prompt["input_ids"]), len(labels))):
                labels[i] = -100
            if labels and all(x == -100 for x in labels):
                # truncated prompt: keep at least one target token
                labels[-1] = full["input_ids"][-1]
            return {
                "input_ids": full["input_ids"],
                "attention_mask": full["attention_mask"],
                "labels": labels,
            }

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, masks, labels = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
            masks.append(b["attention_mask"] + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long).to(device),
            "attention_mask": torch.tensor(masks, dtype=torch.long).to(device),
            "labels": torch.tensor(labels, dtype=torch.long).to(device),
        }

    ds = SFTDataset(examples)
    loader = DataLoader(
        ds,
        batch_size=int(sft.get("batch_size", 8)),
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(sft.get("lr", 2e-5)))
    grad_accum = int(sft.get("grad_accum", 4))
    epochs = int(sft.get("epochs", 3))
    max_steps = int(sft.get("max_steps", -1))
    model.train()
    step = 0
    losses: list[float] = []
    torch.manual_seed(seed)
    for epoch in range(epochs):
        running, count = 0.0, 0
        opt.zero_grad()
        for i, batch in enumerate(loader):
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            running += float(out.loss.item())
            count += 1
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if max_steps > 0 and step >= max_steps:
                    break
            if max_steps > 0 and step >= max_steps:
                break
        losses.append(running / max(1, count))
        log.info("SFT epoch %d loss %.4f", epoch + 1, losses[-1])
        if max_steps > 0 and step >= max_steps:
            break

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    stats = {"losses": losses, "steps": step, "backend": "hf", "device": device}
    write_json(Path(out_dir) / "sft_meta.json", stats)
    del model, opt
    if device == "cuda":
        import torch as _torch

        _torch.cuda.empty_cache()
    n_facts = len({ex["fact_id"] for ex in examples})
    return SFTResult("hf", str(out_dir), len(examples), n_facts, stats)


# --------------------------------------------------------------------------- #
# Consolidation validation + label transitions
# --------------------------------------------------------------------------- #
def evaluate_consolidation(
    facts: Sequence[Fact],
    backbone,
    tau_flush: float = 0.5,
    batch_size: int = 64,
) -> dict:
    """Zero-shot probe check; return facts eligible for eviction."""
    prompts: list[str] = []
    owners: list[int] = []
    golds: list[str] = []
    for fi, fact in enumerate(facts):
        for probe in fact.probes:
            prompts.append(build_qa_prompt(probe.question))
            owners.append(fi)
            golds.append(probe.answer)
    answers: list[str] = []
    for start in range(0, len(prompts), batch_size):
        answers.extend(backbone.generate(prompts[start : start + batch_size]))
    per_fact: dict[int, list[float]] = {}
    for owner, gold, ans in zip(owners, golds, answers):
        per_fact.setdefault(owner, []).append(exact_match(ans, gold))
    flushed = []
    scores = []
    for fi, fact in enumerate(facts):
        em = float(np.mean(per_fact.get(fi, [0.0])))
        scores.append(em)
        if em >= tau_flush:
            flushed.append(fact.fact_id)
    return {
        "n_checked": len(facts),
        "n_flushed": len(flushed),
        "flush_rate": len(flushed) / max(1, len(facts)),
        "mean_probe_em": float(np.mean(scores)) if scores else 0.0,
        "flushed_ids": flushed,
    }


TRANSITION_ORDER = ("non-write", "write-new", "write-update")


def transition_matrix(before: dict[str, str], after: dict[str, str]) -> dict:
    matrix = {b: {a: 0 for a in TRANSITION_ORDER} for b in TRANSITION_ORDER}
    changed = 0
    for fid, b in before.items():
        a = after.get(fid)
        if a is None:
            continue
        matrix.setdefault(b, {x: 0 for x in TRANSITION_ORDER})
        matrix[b][a] += 1
        changed += int(a != b)
    n = sum(sum(row.values()) for row in matrix.values())
    return {
        "matrix": matrix,
        "order": list(TRANSITION_ORDER),
        "n": n,
        "changed": changed,
        "changed_rate": changed / max(1, n),
        "internalized": matrix.get("write-new", {}).get("non-write", 0)
        + matrix.get("write-update", {}).get("non-write", 0),
        "partial_absorption": matrix.get("write-new", {}).get("write-update", 0),
        "interference": matrix.get("non-write", {}).get("write-update", 0),
    }


def transition_to_rows(result: dict) -> list[dict]:
    rows = []
    for b in result["order"]:
        for a in result["order"]:
            rows.append({"base": b, "after_sft": a, "count": result["matrix"][b][a]})
    return rows
