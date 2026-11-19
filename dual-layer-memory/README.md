# Dual-Layer Agentic Memory — reproduction

Independent implementation of

> Wenzhi Li, Dong Nie, Tongtong Lyu, Rui Lan, Peiyao Wang, Lingzi Hong,
> Weihang Pan, Binbin Lin, Boyuan Pan, Yao Hu.
> **Dual-Layer Agentic Memory with Fast Write Routing and Slow Consolidation.**
> arXiv:[2608.22215v2](https://arxiv.org/abs/2608.22215v2), 2026.

The repository implements the paper's full experimental framework on **real
Qwen3 backbones and a ZsRE fact corpus**: behavioural labelling, online
streaming benchmark, small-to-large write router with an escalation gate,
external memory and retrieval, baselines, write-back LoRA SFT, Pareto sweeps,
ablations, paper tables and figures.

There is **no simulator backend**: every experiment runs Qwen3-1.7B / Qwen3-8B
(or the smaller checkpoints in `configs/gpu_smoke.yaml`). The code was built against the paper and validated with CPU unit tests plus a
tiny local Qwen3 checkpoint for the HuggingFace path; the full paper-scale run
requires a CUDA node and the ZsRE dump, and has not been executed in this
workspace.

> 中文说明与对齐分析见 [`docs/复现报告.md`](docs/复现报告.md)。

---

## 1. What is implemented

| Paper component | Code |
|---|---|
| Operational taxonomy non-write / write-new / write-update (Sec. 3.1, Eq. 9) | [`data/labeling.py`](src/dlmem/data/labeling.py), [`data/schema.py`](src/dlmem/data/schema.py) |
| ZsRE corpus, strict `m = 30` probe filter (App. A.1) | [`data/zsre.py`](src/dlmem/data/zsre.py), [`data/build.py`](src/dlmem/data/build.py) |
| Streaming episodes, P1–P4, stratified injection + causal queries (App. A.2, Alg. 1) | [`data/episodes.py`](src/dlmem/data/episodes.py) |
| Semantic + epistemic-uncertainty features (Sec. 3.3.3, Eqs. 2–4) | [`data/features.py`](src/dlmem/data/features.py) |
| Reward-regression routers + escalation gate (Sec. 3.3, App. C) | [`models/router.py`](src/dlmem/models/router.py), [`models/heads.py`](src/dlmem/models/heads.py) |
| Cost-aware cascade with escalation budgets (Fig. 2) | [`models/router.py`](src/dlmem/models/router.py), [`eval/sweep.py`](src/dlmem/eval/sweep.py) |
| BM25 external memory + memory-supported QA | [`memory/store.py`](src/dlmem/memory/store.py), [`memory/qa.py`](src/dlmem/memory/qa.py) |
| Static / heuristic / supervised baselines (Sec. 4.2, App. B) | [`policies/`](src/dlmem/policies) |
| Online evaluation: EM/F1/refusal, Store P/R/F1, routing cost | [`eval/stream.py`](src/dlmem/eval/stream.py), [`eval/metrics.py`](src/dlmem/eval/metrics.py) |
| Write-back SFT + validated eviction (Sec. 3.4, Eq. 8) | [`train/sft.py`](src/dlmem/train/sft.py) |
| Label-transition matrix (Table 2) | [`train/sft.py`](src/dlmem/train/sft.py), [`plots.py`](src/dlmem/plots.py) |
| Feature / pooling / loss / gate ablations (App. D) | [`pipeline.py`](src/dlmem/pipeline.py), [`plots.py`](src/dlmem/plots.py) |
| Frozen Qwen3 features, greedy QA, LoRA SFT | [`models/hf.py`](src/dlmem/models/hf.py) |

---

## 2. Requirements

* Linux + CUDA (the paper used one node with 4×NVIDIA H20, 96 GB each).
* Python ≥ 3.10, PyTorch ≥ 2.2, Transformers ≥ 4.51, PEFT ≥ 0.14.
* The ZsRE dump. The paper's exact release (201,352 facts × 30 probes) is not
  public as of writing; see [§3](#3-data) and [`docs/ALIGNMENT.md`](docs/ALIGNMENT.md).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,analysis]"
pip install -r requirements-gpu.txt
```

Preflight (loads a model, extracts features, generates one answer):

```bash
python scripts/check_gpu.py --model Qwen/Qwen3-1.7B
```

---

## 3. Data

Place the corpus under `data/zsre/` in one of the supported layouts:

```json
{"input": "Kurt Vonnegut was born in Indianapolis.",
 "output": [{"question": "Where was Kurt Vonnegut born?", "answer": "Indianapolis"}]}
```

```json
{"text": "...", "subject": "...", "relation": "...", "object": "...",
 "probes": [{"question": "...", "answer": "..."}]}
```

Validate/normalise a dump:

```bash
python scripts/prepare_zsre.py --input /path/to/zsre --out data/zsre
python scripts/prepare_zsre.py --input data/zsre --validate-only
```

The loader keeps facts with **exactly 30 deduplicated probes**, then assigns a
70/15/15 split. Details and field aliases: [`docs/DATASET.md`](docs/DATASET.md).

> **Data caveat.** The paper reports N = 201,352 facts / 6,040,560 probes.
> Public ZsRE variants differ in size and paraphrase generation. The pipeline is
> data-agnostic, but exact paper numbers require the authors' corpus; the first
> thing to check on a new dump is the fact/probe count and the label mix
> (23.5 / 45.1 / 31.4).

---

## 4. Running

Validate the whole stack on a few hundred facts (real models, minutes):

```bash
python -m dlmem.cli prepare  --config configs/gpu_smoke.yaml
python -m dlmem.cli train    --config configs/gpu_smoke.yaml
python -m dlmem.cli evaluate --config configs/gpu_smoke.yaml
```

Paper scale:

```bash
python -m dlmem.cli prepare  --config configs/paper_full.yaml   # labels, episodes, features
python -m dlmem.cli train    --config configs/paper_full.yaml   # routers + gate
python -m dlmem.cli evaluate --config configs/paper_full.yaml   # Table 1, base framework
python -m dlmem.cli sweep    --config configs/paper_full.yaml   # Fig. 2
python -m dlmem.cli ablate   --config configs/paper_full.yaml   # Figs. 4-6, App. D
python -m dlmem.cli sft      --config configs/paper_full.yaml   # write-back + Table 2 + SFT eval
```

or `bash scripts/run_full.sh`. Every stage is cached and resumable; `--force`
rebuilds. Stage costs and memory knobs: [`REPRODUCTION.md`](REPRODUCTION.md).

Artifacts land in `runs/<run_name>/`:

```
config.json, manifest.json
labels/{train,val,test}.jsonl           behavioural labels
features/{small,large}-{split}.npz      semantic + uncertainty features
episodes/test.json                      streaming schedules
routers/                                small/large/gate heads
results/table1.{csv,md}                 Table 1
results/pareto_curves.csv               Fig. 2 data
ablations/*.csv                         Figs. 4-6 data
sft/{examples.jsonl,transition.md,...}  write-back + Table 2
figures/*.png|pdf
```

---

## 5. Target results (from the paper)

These are the numbers the implementation targets, **not** measurements from
this workspace:

| Policy | QA EM | Store | Routing cost |
|---|---|---|---|
| No Store | 38.02% | 0% | 0 |
| Full Store | 87.79% | 100% | 0 |
| Heuristic Store | 73.56% | 59.08% | 2×Gen8B |
| PPL-Cond. Store | 78.95% | 71.74% | 1×Fwd8B |
| Logistic Regression | 83.84% | 68.50% | 1×Fwd8B |
| MLP Classifier | 83.18% | 65.36% | 1×Fwd8B |
| Write Router (0.2) | 83.97% | 64.81% | Fwd1.7B + 49.0% Fwd8B |
| Write Router (0.08) | 86.35% | 77.00% | Fwd1.7B + 39.7% Fwd8B |
| SFT: No Store | 69.16% | 0% | 0 |
| SFT: Full Store | 92.29% | 100% | 0 |
| SFT: Write Router (0.2) | 89.77% | 32.08% | Fwd1.7B + 45.8% Fwd8B |
| SFT: Write Router (0.08) | 90.71% | 47.85% | Fwd1.7B + 43.3% Fwd8B |

---

## 6. Why a run may not match the paper

The implementation follows the paper's equations and protocol, but exact
alignment depends on choices the paper does not fully specify. The most
important ones, in order of impact:

1. **Fact corpus.** The 201k/30-probe ZsRE release is not public; a different
   dump changes the label mix, the upper bound and every percentage.
2. **Model/prompt.** Qwen3 variant (instruct vs base), thinking mode, decoding,
   answer format and `max_new_tokens` change EM and refusal behaviour.
3. **Label noise.** Refusal detection, answer normalisation and the
   `rho ≥ 0.01` / `gamma ≥ 0.5` boundaries determine which facts are
   write-worthy.
4. **Retrieval.** The paper does not name the retriever; this repo uses BM25
   top-3. A weak retriever depresses Full Store EM.
5. **Router training.** Reward regression targets, λs/λe, gate supervision split,
   escalation cap and any calibration move the storage–EM operating point.
6. **Streaming schedule.** Algorithm 1 is ambiguous about whether injection
   turns also carry queries; this repo uses `T − K` query turns.
7. **Write-back.** LoRA rank/LR/epochs, example construction, eviction threshold
   and re-labelling scope are under-specified.
8. **Metric conventions.** Store precision/recall targets and refusal rate
   definitions differ across papers.

A practical debugging order is in [`docs/ALIGNMENT.md`](docs/ALIGNMENT.md):
first reproduce **No Store ≈ 38.0%** and **Full Store ≈ 87.8%** on the paper's
corpus, then the label mix, then the baselines, then the router. If Full Store
is low, the problem is retrieval/QA/labels, not the router.

---

## 7. Repository layout

```
src/dlmem/
  config.py            YAML config + dotted overrides
  utils.py             seeding, IO, SQuAD normalisation, EM/F1, refusal detection
  pipeline.py          prepare -> train -> evaluate -> sweep -> ablate -> sft
  cli.py               `python -m dlmem.cli <stage>`
  data/                schema, ZsRE loader, labelling, features, episodes
  models/              HF Qwen3 wrapper, MLP heads, routers + cascade
  memory/              BM25 store, prompts
  policies/            static, heuristic, supervised baselines
  eval/                streaming benchmark, metrics, Pareto sweeps
  train/sft.py         write-back SFT, eviction, transitions
  plots.py, tables.py  figures and paper tables
configs/
  paper_full.yaml      paper scale (Qwen3-1.7B/8B, ZsRE)
  gpu_smoke.yaml       real-model smoke test (Qwen3-0.6B/1.7B, 400 facts)
tests/                 24 unit tests + optional real-model test
scripts/               GPU preflight, ZsRE preparation, full run
docs/                  dataset, architecture, alignment, deviations, 中文报告
```

---

## 8. Tests

```bash
pytest tests -q
```

Covers taxonomy boundaries, prompt round-trips, episode invariants
(P1–P4/determinism), feature shapes/round-trips, router/gate training and
save/load, cascade caps, configuration validation and HF token-NLL math.
A real-model integration test is skipped unless requested:

```bash
DLMEM_HF_MODEL=Qwen/Qwen3-0.6B pytest tests/test_hf_integration.py -q -s
```

---

## 9. Citation

```bibtex
@article{li2026dual,
  title   = {Dual-Layer Agentic Memory with Fast Write Routing and Slow Consolidation},
  author  = {Li, Wenzhi and Nie, Dong and Lyu, Tongtong and Lan, Rui and Wang, Peiyao
             and Hong, Lingzi and Pan, Weihang and Lin, Binbin and Pan, Boyuan and Hu, Yao},
  journal = {arXiv preprint arXiv:2608.22215},
  year    = {2026}
}
```

MIT licensed; see [`LICENSE`](LICENSE).
