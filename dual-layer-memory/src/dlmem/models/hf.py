"""HuggingFace Qwen3 backbones: frozen feature extraction + greedy generation.

The paper freezes qwen3-1.7B (small router) and qwen3-8B (large router) and
uses them for both routing features and downstream QA.  This wrapper keeps that
contract: hidden states and token NLLs are read from the final layer, and
generation is greedy.

Memory notes
------------
``encode_batch`` needs the full logits tensor (all positions) to compute the
teacher-forced NLL curve.  On a 151k-vocabulary Qwen3 model that tensor is
large, so batches are modest and the NLL is accumulated in position chunks
without casting the whole logits tensor to fp32.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..utils import get_logger
from .backbone import Backbone, Encoding, resolve_device, resolve_dtype

log = get_logger("dlmem.models.hf")


class HFBackbone(Backbone):
    def __init__(
        self,
        name: str,
        dtype: str = "bfloat16",
        device: str = "auto",
        device_map: str | None = "auto",
        feature_max_length: int = 256,
        qa_max_length: int = 768,
        batch_size: int = 16,
        generate_batch_size: int = 32,
        max_new_tokens: int = 16,
        trust_remote_code: bool = True,
        attn_implementation: str = "sdpa",
        quantize_4bit: bool = False,
        chat_template: bool = True,
        adapter_path: str | None = None,
    ) -> None:
        self.name = name
        self.dtype_name = dtype
        self.device = resolve_device(device)
        self.device_map = device_map if device_map else None
        self.feature_max_length = int(feature_max_length)
        self.qa_max_length = int(qa_max_length)
        self.batch_size = int(batch_size)
        self.generate_batch_size = int(generate_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.quantize_4bit = quantize_4bit
        self.use_chat_template = chat_template
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None
        self.hidden_size = 0

    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info(
            "loading %s on %s (%s, device_map=%s)",
            self.name,
            self.device,
            self.dtype_name,
            self.device_map,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.name, trust_remote_code=self.trust_remote_code, padding_side="left"
        )
        kwargs = dict(
            dtype=resolve_dtype(self.dtype_name, self.device),
            trust_remote_code=self.trust_remote_code,
            attn_implementation=self.attn_implementation,
        )
        if self.device == "cuda" and self.device_map:
            kwargs["device_map"] = self.device_map
        if self.quantize_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=resolve_dtype(self.dtype_name, self.device),
                bnb_4bit_quant_type="nf4",
            )
        self._model = _from_pretrained_compat(AutoModelForCausalLM, self.name, **kwargs)
        if not (self.device == "cuda" and self.device_map):
            self._model = self._model.to(self.device)
        if self.adapter_path:
            from peft import PeftModel

            log.info("attaching LoRA adapter %s", self.adapter_path)
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()
        self._model.requires_grad_(False)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self.hidden_size = int(self._model.config.hidden_size)

    def _input_device(self):
        if self.device != "cuda":
            return self.device
        device_map = getattr(self._model, "hf_device_map", None)
        if device_map:
            for dev in device_map.values():
                if isinstance(dev, int):
                    return f"cuda:{dev}"
                if isinstance(dev, str) and dev.startswith("cuda"):
                    return dev
        return "cuda:0"

    # ------------------------------------------------------------------ #
    def _format_prompts(self, prompts: Sequence[str]) -> list[str]:
        if not self.use_chat_template or not getattr(self._tokenizer, "chat_template", None):
            return list(prompts)
        out = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            try:
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            out.append(text)
        return out

    # ------------------------------------------------------------------ #
    def encode_batch(self, texts: Sequence[str], pooling: str = "mean") -> list[Encoding]:
        import torch

        self._ensure_model()
        device = self._input_device()
        results: list[Encoding] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            enc = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.feature_max_length,
            ).to(device)
            with torch.inference_mode():
                out = self._model(**enc, output_hidden_states=True)
            hidden = out.hidden_states[-1].float()  # (B, L, H)
            mask = enc["attention_mask"].bool()
            lengths = mask.sum(dim=1)
            for i in range(len(batch)):
                length = int(lengths[i].item())
                if self._tokenizer.padding_side == "left":
                    h = hidden[i, -length:]
                else:
                    h = hidden[i, :length]
                emb = h[-1] if pooling == "last_token" else h.mean(dim=0)
                nll_curve = _token_nll(
                    out.logits[i], enc["input_ids"][i], length, self._tokenizer.padding_side
                )
                results.append(
                    Encoding(
                        embedding=emb.detach().cpu().numpy().astype(np.float32),
                        nll_curve=nll_curve,
                        mean_nll=float(nll_curve.mean()) if len(nll_curve) else 0.0,
                        length=length,
                    )
                )
            del out, hidden
        return results

    # ------------------------------------------------------------------ #
    def generate(self, prompts: Sequence[str], max_new_tokens: int | None = None) -> list[str]:
        import torch

        self._ensure_model()
        prompts = self._format_prompts(prompts)
        max_new_tokens = max_new_tokens or self.max_new_tokens
        device = self._input_device()
        outputs: list[str] = []
        for start in range(0, len(prompts), self.generate_batch_size):
            batch = prompts[start : start + self.generate_batch_size]
            enc = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.qa_max_length,
            ).to(device)
            with torch.inference_mode():
                gen = self._model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1] :]
            decoded = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outputs.extend([d.strip() for d in decoded])
        return outputs

    # ------------------------------------------------------------------ #
    def mean_nll(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([e.mean_nll for e in self.encode_batch(texts)], dtype=np.float64)


def _from_pretrained_compat(model_cls, name: str, dtype=None, **kwargs):
    """Transformers renamed ``torch_dtype`` to ``dtype`` in 4.56."""
    if dtype is None:
        return model_cls.from_pretrained(name, **kwargs)
    try:
        return model_cls.from_pretrained(name, dtype=dtype, **kwargs)
    except TypeError:  # transformers < 4.56
        return model_cls.from_pretrained(name, torch_dtype=dtype, **kwargs)


def _token_nll(
    logits, input_ids, length: int, padding_side: str, chunk: int = 64
) -> np.ndarray:
    """Per-token NLL of a sequence, accumulated in position chunks."""
    import torch

    if padding_side == "left":
        ids = input_ids[-length:]
        lg = logits[-length:]
    else:
        ids = input_ids[:length]
        lg = logits[:length]
    if ids.numel() < 2:
        return np.zeros(1, dtype=np.float32)
    nll_parts = []
    span = len(ids) - 1
    for start in range(0, span, chunk):
        stop = min(start + chunk, span)
        logprobs = torch.log_softmax(lg[start:stop].float(), dim=-1)
        targets = ids[start + 1 : stop + 1].to(logprobs.device)
        nll_parts.append(
            -logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        )
    nll = torch.cat(nll_parts)
    return nll.detach().cpu().numpy().astype(np.float32)


def load_backbone(cfg, role: str = "small") -> Backbone:
    """Factory used by every CLI entry point (real Qwen3 checkpoints only)."""
    name = cfg.get_path(f"model.{role}", "")
    if not name:
        raise ValueError(f"model.{role} is not configured")
    return HFBackbone(
        name=name,
        dtype=cfg.get_path("model.dtype", "bfloat16"),
        device=cfg.get_path("model.device", "auto"),
        device_map=cfg.get_path("model.device_map", "auto"),
        feature_max_length=int(cfg.get_path("model.feature_max_length", 256)),
        qa_max_length=int(cfg.get_path("model.qa_max_length", 768)),
        batch_size=int(cfg.get_path("model.prompt_batch_size", 16)),
        generate_batch_size=int(cfg.get_path("model.generate_batch_size", 32)),
        max_new_tokens=int(cfg.get_path("model.max_new_tokens", 16)),
        trust_remote_code=bool(cfg.get_path("model.trust_remote_code", True)),
        attn_implementation=str(cfg.get_path("model.attn_implementation", "sdpa")),
        quantize_4bit=bool(cfg.get_path("model.quantize_4bit", False)),
        chat_template=bool(cfg.get_path("model.chat_template", True)),
    )
