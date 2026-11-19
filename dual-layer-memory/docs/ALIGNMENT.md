# Why results may not match the paper — and how to align them

The code implements the paper's equations and protocol. It cannot guarantee the
paper's *numbers*, because a reproduction inherits every under-specified choice
in the original. This document ranks the sources of divergence and gives a
diagnostic order.

---

## 0. What "aligned" means

The paper's anchors:

| Quantity | Paper value |
|---|---|
| Facts / probes | 201,352 / 6,040,560 (m = 30) |
| Fact label mix (non-write / write-new / write-update) | 23.5 / 45.1 / 31.4 % |
| Stale-knowledge probes | 19.4% |
| No Store EM | 38.02% |
| Full Store EM | 87.79% |
| Full Store precision (write rate) | 76.54% |
| Write Router (0.2) | 83.97% EM @ 64.81% storage |
| Write Router (0.08) | 86.35% EM @ 77.00% storage |
| After SFT: No Store / Full Store | 69.16% / 92.29% |
| Labels changed by SFT | 67.46% |

If No Store and Full Store are far off, do not tune the router: the problem is
data, prompting or retrieval. The first two numbers are the sanity check.

Each quantity depends on a small number of levers:

```
No Store EM  = P(correct | parametric only)
             = f(model, questions, answer format, EM rule)
Full Store EM = P(correct | fact retrieved)
             = f(No Store, retrieval hit rate, context prompt, label "unsolved")
```

---

## 1. Data: the largest source of divergence

The paper says the corpus is derived from ZsRE, filtered to facts with exactly
30 paraphrased probes. Public ZsRE releases differ substantially:

* different numbers of facts and paraphrases per fact;
* different paraphrase generators (crowd vs LLM), hence different difficulty;
* different answer strings/aliases (EM is sensitive to this);
* possibly an LLM-generated expansion to reach 201k facts.

The authors state that code **and dataset** will be released upon acceptance.
Without that exact dump, matching 38.02% / 87.79% to two decimals is not
achievable; a different dump can move both by several points.

**Diagnostics**

```bash
python scripts/prepare_zsre.py --input /path/to/zsre --validate-only
# clean, non-write, write-new, write-update counts per split
python - <<'PY'
from dlmem.config import load_config
from dlmem.data.build import prepare_facts, facts_by_split
from collections import Counter
cfg = load_config("configs/paper_full.yaml")
facts = prepare_facts(cfg)
for split in ("train", "val", "test"):
    fs = facts_by_split(facts, split)
    print(split, len(fs), Counter(f.label for f in fs))
PY
```

Expected after labelling: fact fractions ≈ 0.235 / 0.451 / 0.314; probe classes
≈ 19.4% stale overall; total probes = 30 × facts.

**Levers**

* `data.m_probes` (keep 30 for the paper);
* `data.splits` (paper: 70/15/15, giving ~30,203 test facts);
* field mapping in `data/zsre.py` if your dump uses other keys;
* answer normalisation/EM in `utils.py`.

---

## 2. Model, prompt and decoding

The paper uses qwen3-8B for labelling/QA and qwen3-1.7B/8B for routing features.
Small differences have large effects:

* **Qwen3 variant.** Base vs instruct, and whether "thinking" is enabled.
  This repo disables thinking via `apply_chat_template(..., enable_thinking=False)`
  and caps generation at 16 tokens. If your tokenizer ignores that flag, Qwen3
  may emit a long chain of thought that gets truncated, producing empty or
  refusal-like answers.
* **Answer format.** Prompts ask for a short factual answer. The paper's exact
  prompt is not in the appendix.
* **EM rule.** This repo uses SQuAD-style normalised **containment** (gold span
  inside the prediction), which is standard for ZsRE. Strict equality would
  lower EM.
* **Refusal detection.** Regex-based; Qwen3 refusals vary ("I'm sorry, but I
  cannot…", "I don't have enough information…"). Inspect 100 low-confidence
  predictions and extend `REFUSAL_PATTERNS` if needed.
* **Truncation.** `model.qa_max_length` (default 768) must fit the instruction,
  top-3 memories and question. `feature_max_length` (default 256) must fit the
  fact sentence.

**Diagnostic** — manually inspect 50 zero-shot / memory-supported pairs:

```python
from dlmem.config import load_config
from dlmem.data.build import prepare_facts
from dlmem.models.hf import load_backbone
from dlmem.memory.qa import build_qa_prompt
cfg = load_config("configs/gpu_smoke.yaml")
facts = prepare_facts(cfg)[:5]
b = load_backbone(cfg, "large")
for f in facts:
    q = f.probes[0]
    print("ZS :", b.generate([build_qa_prompt(q.question)])[0])
    print("MEM:", b.generate([build_qa_prompt(q.question, [f.text])])[0])
    print("GOLD:", q.answer)
```

If memory-supported answers are wrong, Full Store EM will be low even with
perfect retrieval. Common causes: truncated context, chat template swallowing
the `[Memory]` block, thinking mode, or answers longer than `max_new_tokens`.

---

## 3. Retrieval

The paper does not specify the retriever. This repo uses BM25 top-3 over fact
sentences. On ZsRE paraphrases, lexical overlap is usually high, but:

* top-1 may be enough and less noisy; top-3 may add distractors;
* a dense retriever may behave differently;
* memory formatting (bullets, ordering) changes the answer.

The streaming result includes `retrieval_hit_rate`. If it is below ~95% while
Full Store is low, fix retrieval before touching the router.

**Levers:** `eval.top_k`, `memory/qa.py` formatting, `memory/store.py` scoring.

---

## 4. Labelling thresholds

`rho < 0.01` and `gamma >= 0.5` are exact boundaries from the paper; keep them.
What can differ:

* whether a probe counts as correct zero-shot (EM rule);
* whether a wrong-but-non-refusal answer is "confident" (paper: any non-refusal
  incorrect answer is stale);
* probe de-duplication (two paraphrases that normalise to the same question);
* unsolved probes: these lower `rho` but are excluded from `m_new/m_update`.

A quick distribution check:

```python
from dlmem.data.labeling import label_statistics
# after prepare: runs/<run>/labels/stats.json
```

If your write rate is far from 76.5%, the router's storage/EM frontier will be
shifted regardless of model quality.

---

## 5. Router and gate

Sources of divergence, in order:

1. **Supervision split.** App. C says to fit routers, freeze them, then build
   gate targets from their greedy actions. This repo defaults to `gate.supervision:
   train` (paper-faithful). On small datasets this can make the gate degenerate;
   set `gate.supervision=val` for a stacking-style alternative.
2. **Reward targets.** `r(W) = rho − λs`, `r(D) = 0`; probe-weighted `rho`.
3. **Gate calibration.** This repo can oversample positive gains (`gate.oversample`,
   default 8) because escalation gains are rare. Set to 1 for the unweighted MSE.
4. **Reward-head calibration.** `router.calibration: none` by default (paper-scale).
   `quantile` is a small-data fix and changes the λs→storage mapping.
5. **Escalation cap.** `cascade.escalation_cap: 0.45` reproduces the paper's
   ~40–49% escalation at the table's operating points. The gate itself decides
   *which* facts to escalate.
6. **λs/λe.** Paper operating points (0.2, 0.008) and (0.08, 0.005).

If storage is too high at λs=0.2, the predicted `rho` is too high: check the
reward-target distribution, class balance, and whether features actually encode
knowledge (an untrained/incorrect hidden-state extraction silently produces
noise).

**Diagnostic**

```bash
python -m dlmem.cli sweep --config configs/paper_full.yaml
# inspect runs/paper_full/results/pareto_curves.csv:
#   at storage 0.65/0.77, does small/large router EM track the paper?
```

The paper's Fig. 2 gap between small and large routers is small (~0.01–0.02 EM).
If your small router is far worse, the 1.7B features or training split differ.

---

## 6. Streaming schedule

Algorithm 1 is ambiguous. This repo uses:

* `T` turns = `K` injection turns + `(T − K)` query turns;
* one query per non-injection turn;
* stratified injection buckets, final turn reserved for a query;
* probe budget `T − K`, extended on demand.

The paper describes "T = 500 turns with K = 100 injections and downstream QA
interactions". If you interpret every turn as a query (500 queries + 100
injections), the metrics shift slightly but the admission/storage picture does
not. The convention is isolated in `data/episodes.py` and asserted by tests.

---

## 7. Write-back SFT

Under-specified in the paper:

* LoRA rank/alpha/LR/epochs (we use r=16, α=32, LR 2e-5, 3 epochs);
* whether both backbones are fine-tuned (we do both, as stated);
* example construction (2 QA pairs + 1 sentence per fact; assistant-only loss);
* which memories are consolidated (we use the test-stream memories written at
  λs=0.08; `sft.source: train` gives a leakage-free variant);
* eviction threshold `tau_flush = 0.5`;
* whether re-labelling covers train/val (we do, for the post-SFT router).

Expected consequences: SFT No Store EM (paper 69.16%) is the most sensitive
number. If it stays near 38%, consolidation failed (LR too low, loss masked
incorrectly, adapter not attached). If it jumps to ~95%, SFT overfit the
evaluation facts or the examples leaked test labels.

---

## 8. Metrics

* **Store precision/recall** use write-new + write-update as positives.
* **Post-SFT storage** is measured against post-SFT labels in this repo, so
  Full Store precision drops after consolidation. The paper's Table 1 footnote
  evaluates SFT recall against *static base-model targets*; if you need that
  convention, pass the base labels into `eval/stream.py::run_streaming`.
* **Refusal rate** depends on the refusal detector.
* **Routing cost** is in forward-pass equivalents, not wall-clock.

---

## 9. Nondeterminism and scale

GPU kernels, batching and `device_map` sharding can change generated tokens.
The paper reports single runs. For stable comparisons, run ≥ 2 seeds and report
means; at minimum fix `seed`, `do_sample=False`, and the same Transformers
version.

---

## 10. Recommended debugging sequence

| Step | Check | If it fails |
|---|---|---|
| 1 | `prepare` logs: N facts, 30×N probes, split sizes | field mapping / filter / data source |
| 2 | Label mix 23.5/45.1/31.4 (facts), stale ≈ 19.4% (probes) | EM rule, refusal regex, model/prompt |
| 3 | No Store EM ≈ 38.0% | model knowledge, prompts, answer format |
| 4 | Full Store EM ≈ 87.8%, retrieval hit rate > 95% | context truncation, top-k, memory formatting |
| 5 | Baselines ordering: Full > LogReg/MLP > PPL > Heuristic > Random > No Store | features, thresholds, calibration |
| 6 | Router frontier passes near (0.65, 83.97) and (0.77, 86.35) | reward targets, λs, gate, features |
| 7 | SFT No Store ≈ 69%, Full ≈ 92% | LoRA, loss masking, adapter loading |
| 8 | Transition matrix ≈ 67% changed, ~35% internalised | SFT data scope, relabelling |

Record the numbers at every step in `manifest.json`/logs; that is the fastest
way to localise a mismatch to one stage.
