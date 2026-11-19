# Architecture

## Data flow

```
  ZsRE dump ──▶ data/build.py        facts + 70/15/15 splits
                     │
       ┌─────────────┼───────────────────────────┐
       ▼             ▼                           ▼
 labelling       episodes                    features
 zero-shot vs    Alg. 1, P1–P4              e, u (Eqs. 2–4)
 memory          injection/query streams     mean-pool + NLL
       │             │                           │
       └─────────────┴────────────┬──────────────┘
                                  ▼
                        models/router.py            memory/store.py
                        small/large reward heads    BM25 external memory
                        + escalation gate                 │
                                  │                         │
                                  ▼                         ▼
                        CascadeRouter               eval/stream.py
                        (Fwd1.7B + x% Fwd8B)        QA EM/F1, store P/R/F1
                                  │                         │
                                  └───────────┬─────────────┘
                                              ▼
                                    train/sft.py (write-back)
                                    LoRA SFT → eviction → re-label
                                              │
                                              ▼
                                    results/, ablations/, figures/
```

## Modules

| File | Responsibility |
|---|---|
| `config.py` | Defaults, YAML merge, dotted `--set` overrides, run directory |
| `utils.py` | Seeding, JSONL IO, SQuAD normalisation, EM/F1, refusal regexes |
| `data/schema.py` | `Fact`, `Probe`, `label_from_counts`, offline EM helpers |
| `data/build.py` | Corpus construction/caching, split assignment |
| `data/zsre.py` | Flexible ZsRE loader with the strict `m`-probe filter |
| `data/labeling.py` | Behavioural probe classification and fact aggregation |
| `data/features.py` | Mean pooling, NLL interpolation, FeatureSet save/load |
| `data/episodes.py` | Stratified injections, probe budgeting, causal streaming |
| `models/backbone.py` | Backbone interface and device/dtype resolution |
| `models/hf.py` | Qwen3 features, chunked NLL, greedy generation, adapters |
| `models/heads.py` | MLP (LayerNorm/GELU/dropout), training loops, persistence |
| `models/router.py` | Reward routers, optional calibration, gate, cascade |
| `memory/store.py` | BM25 index over written facts, top-k retrieval |
| `memory/qa.py` | Zero-shot and memory-supported prompt construction |
| `policies/base.py` | `PolicyOutput`, static policies, threshold helpers |
| `policies/heuristic.py` | Self-probing store, perplexity-conditional store |
| `policies/supervised.py` | Logistic regression, BCE MLP, `ArrayPolicy` |
| `eval/metrics.py` | Store counts, expected EM, aggregate QA metrics |
| `eval/stream.py` | Lockstep streaming evaluation across episodes |
| `eval/sweep.py` | λ sweeps, cascade decisions, frontier interpolation |
| `train/sft.py` | SFT examples, LoRA training, eviction, transition matrix |
| `pipeline.py` | `Experiment` orchestration and artifact layout |
| `plots.py` / `tables.py` | Figures 2/4/5/6/7 and Tables 1/2 |
| `cli.py` | Stage entry points |

## Artifact layout

```
runs/<run_name>/
  config.json                 resolved configuration
  manifest.json               stage timings and summaries
  labels/{train,val,test}.jsonl
  labels/stats.json           per-split label distribution
  features/{small,large}-{split}.npz
  episodes/test.json
  routers/{small_head.pt,large_head.pt,gate_head.pt,cascade.json}
  routers/training_summary.json
  results/base_stream_<policy>.json
  results/table1.{csv,md}
  results/pareto_curves.csv
  results/pareto_gaps.json
  ablations/{feature_small,feature_large,pooling_loss,gate_design}.csv
  ablations/escalation_gaps.json
  sft/{examples.jsonl,sft_meta.json,consolidation.json,transition.json,transition.md}
  labels_sft/, features_sft/, routers_sft/
  results/sft_stream_<policy>.json
  figures/*.png, *.pdf
```

## Interfaces

**Backbone**

```python
encode_batch(texts, pooling="mean") -> list[Encoding(embedding, nll_curve, mean_nll, length)]
generate(prompts, max_new_tokens=None) -> list[str]
mean_nll(texts) -> np.ndarray
```

`load_backbone(cfg, role)` returns the configured Qwen3 wrapper.

**Policy** — `decide_all(facts) -> PolicyOutput` (`write`, optional `escalate`,
`cost`, `score`, `cost_label`). Policies are pointwise; the streaming evaluator
applies decisions at injection time.

**Router** — `fit` on feature sets, `predict_sets` returns rewards/score/actions,
`save`/`load` persist the head. The cascade adds a gate that ranks facts for
escalation under a budget.

## Extending

* **New retriever**: replace `ExternalMemory.retrieve` (dense bi-encoder) and
  adjust `memory/qa.py`.
* **New consolidation**: replace `train/sft.py::run_hf_sft` with full
  fine-tuning, online LoRA or knowledge editing; eviction and transition
  analysis are model-agnostic.
* **New features**: extend `FeatureSet` and `models/router.py::build_inputs`;
  ablations read modes from YAML.
* **New baselines**: implement `WritePolicy` and add it in
  `Experiment.build_policies`.
