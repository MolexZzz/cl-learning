# Reproduction guide (Qwen3 + ZsRE, CUDA)

Everything here runs real models. There is no simulator fallback. The full
paper-scale run needs one CUDA node comparable to the paper's 4×H20 (96 GB).

---

## 0. Environment

```bash
git clone <this-repo> dual-layer-memory
cd dual-layer-memory

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analysis]"
pip install -r requirements-gpu.txt

python scripts/check_gpu.py --model Qwen/Qwen3-1.7B
```

The preflight must print the GPU names, load the tokenizer/model, run one
feature extraction and one generation. Fix the stack before any long job.

Versions that matter: PyTorch ≥ 2.2, Transformers ≥ 4.51 (Qwen3), PEFT ≥ 0.14
(LoRA), `datasets` ≥ 2.19 (optional, for Hub loading). Qwen3-8B in bf16 needs
~16 GB of weights; 1.7B another ~3.4 GB; both fit a single 80 GB GPU with
activations, and comfortably on the paper's node.

---

## 1. Data

The paper filters ZsRE to facts with exactly **30** paraphrase probes:

```
N = 201,352 facts,  N × m = 6,040,560 probes
```

Prepare or validate a dump:

```bash
python scripts/prepare_zsre.py --input /path/to/zsre --out data/zsre
python scripts/prepare_zsre.py --input data/zsre --validate-only
```

Accepted layouts are listed in [`docs/DATASET.md`](DATASET.md). The loader
de-duplicates probes, drops facts without exactly 30, shuffles with the seed and
assigns 70/15/15 splits.

> **Critical:** the exact 201k/30-probe dump is not public as of writing (the
> authors say code and dataset will be released upon acceptance). A different
> ZsRE variant will produce different absolute numbers. See
> [`docs/ALIGNMENT.md`](ALIGNMENT.md) §1 before interpreting a mismatch.

---

## 2. Pilot run (recommended first)

Validate the whole loop on a few hundred facts before launching the multi-day
job:

```bash
CONFIG=configs/paper_full.yaml
python -m dlmem.cli prepare  --config $CONFIG \
  --set run_name=pilot --set data.max_facts=500 \
  --set eval.episodes=1 --set eval.facts_per_episode=20 --set eval.turns=100
python -m dlmem.cli train    --config $CONFIG --set run_name=pilot
python -m dlmem.cli evaluate --config $CONFIG --set run_name=pilot
```

Check, in order:

1. `runs/pilot/labels/stats.json` — fact mix ≈ 23.5 / 45.1 / 31.4 %, stale
   probes ≈ 19.4 %, write rate ≈ 76.5 %.
2. No Store EM ≈ 38% and Full Store EM ≈ 88% (on the paper's dump; a pilot of
   500 facts has wide error bars).
3. `results/table1.md` — ordering Full > supervised > PPL > heuristic > random
   > No Store.

If (1) or (2) is far off, stop: the router cannot fix a data or prompting
problem. Debug order: [`docs/ALIGNMENT.md`](ALIGNMENT.md) §10.

---

## 3. Full run

```bash
bash scripts/run_full.sh configs/paper_full.yaml
```

or stage by stage:

```bash
python -m dlmem.cli prepare  --config configs/paper_full.yaml
python -m dlmem.cli train    --config configs/paper_full.yaml
python -m dlmem.cli evaluate --config configs/paper_full.yaml
python -m dlmem.cli sweep    --config configs/paper_full.yaml
python -m dlmem.cli ablate   --config configs/paper_full.yaml
python -m dlmem.cli sft      --config configs/paper_full.yaml
```

### 3.1 `prepare`

1. **Labelling** (`data/labeling.py`): for each of 6,040,560 probes, one
   zero-shot and one memory-supported greedy generation with qwen3-8B
   (12.08M generations), classified into internal/missing/stale/unsolved, then
   aggregated by Eq. 9. Artifacts: `labels/{split}.jsonl`, `labels/stats.json`.
2. **Episodes** (`data/episodes.py`): 300 test episodes, 100 facts each,
   500 turns = 100 injection turns + 400 query turns. Artifact:
   `episodes/test.json`.
3. **Features** (`data/features.py`): mean-pooled final hidden states and
   token-NLL curves, for both backbones. Artifacts:
   `features/{small,large}-{split}.npz`.

Approximate cost on 4×H20: labelling dominates (roughly 10–20 h for the 8B
model, depending on batch size and GPU utilisation); features add 1–3 h. Measure
on the pilot and extrapolate by fact count.

### 3.2 `train`

Trains the small and large reward routers (MSE on `r(W)=rho−λs`, `r(D)=0`) and
then the escalation gate on frozen-router predictions. Artifacts:
`routers/{small,large,gate}_head.pt`, `routers/training_summary.json`.

### 3.3 `evaluate`

Runs the streaming benchmark for all baselines and both router operating
points. 300 episodes × 400 queries = 120k generations per policy on the 8B
model; policies run sequentially. Artifacts: `results/base_stream_*.json`,
`results/table1.{csv,md}`.

### 3.4 `sweep`

λs sweep for small/large/cascade (caps 0.4/0.6), no generation. Artifacts:
`results/pareto_curves.csv`, `results/pareto_gaps.json`, `figures/fig2_pareto.*`.

### 3.5 `ablate`

Feature source/type (Fig. 4), pooling × loss (Fig. 5), gate inputs (Fig. 6),
plus Appendix D.3 escalation gaps. Router retraining only; no generation.
Artifacts: `ablations/*.csv`, `figures/fig4–6.*`.

### 3.6 `sft`

1. Select the external memories written at λs=0.08 (deployment/test stream).
2. Build QA pairs + sentence targets; LoRA-SFT both backbones with
   assistant-only loss (Eq. 8).
3. Validate with zero-shot probes; evict facts with mean EM ≥ 0.5.
4. Re-label the test split (and optionally train/val) with the adapted 8B model.
5. Build the Table 2 transition matrix.
6. Re-extract features, retrain routers/gate, re-evaluate No Store / Full Store /
   Router_SFT.

This repeats the labelling cost for the re-labelled split; `sft.relabel_facts`
caps it if needed. Artifacts: `sft/`, `labels_sft/`, `features_sft/`,
`routers_sft/`, updated `results/table1.md`, `figures/fig7_transition.*`.

**Total:** order of 2–4 days on one 4×H20 node, dominated by the two labelling
passes and SFT. Use the pilot to calibrate.

---

## 4. Caching and resume

* `prepare` caches `facts-<hash>.jsonl`, labels per split, episodes and features.
  The fact-cache key includes the dump's file sizes/mtimes, so editing data
  invalidates it.
* Every stage reloads existing artifacts; `--force` on `prepare` re-labels and
  re-extracts.
* `manifest.json` records stage timings and summaries; a partial run is
  auditable.
* `run_name` isolates experiments: `--set run_name=seed2`.

---

## 5. Memory and throughput knobs

| Symptom | Knobs |
|---|---|
| OOM during labelling/QA | `model.generate_batch_size`, `model.qa_max_length`, `labeling.batch_size` |
| OOM during feature extraction | `model.prompt_batch_size`, `model.feature_max_length` |
| OOM during streaming | `eval.batch_episodes`, `model.generate_batch_size` |
| OOM during SFT | `sft.batch_size`, `sft.grad_accum`, `sft.max_length`, enable `sft.gradient_checkpointing` |
| Slow labelling | raise `labeling.batch_size`/`generate_batch_size` until memory-bound |
| 8B does not fit one GPU | `model.quantize_4bit: true` (bitsandbytes, Linux/CUDA) or shard with `device_map: auto` |

`device_map: auto` shards with accelerate on CUDA; on a single GPU set it to
`null`. `feature_max_length` only needs to cover the fact sentence (256 is
generous); `qa_max_length` must cover instruction + top-k memories + question.

---

## 6. Expected paper results

| Policy | EM | Storage | Escalation |
|---|---|---|---|
| No Store | 38.02% | 0% | – |
| Full Store | 87.79% | 100% | – |
| Write Router (0.2) | 83.97% | 64.81% | 49.0% |
| Write Router (0.08) | 86.35% | 77.00% | 39.7% |
| SFT No Store | 69.16% | 0% | – |
| SFT Full Store | 92.29% | 100% | – |
| Write Router_SFT (0.2) | 89.77% | 32.08% | 45.8% |
| Write Router_SFT (0.08) | 90.71% | 47.85% | 43.3% |

Deviations and under-specified choices are catalogued in
[`docs/DEVIATIONS.md`](DEVIATIONS.md).

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `no facts survived normalisation` | wrong field names or not 30 probes; inspect with `--validate-only` |
| Label mix far from 23/45/31 | EM/refusal rules or a different corpus; see ALIGNMENT §2–4 |
| Full Store EM ≪ 87% | retrieval misses, context truncation, chat template/thinking mode |
| No Store EM ≪ 38% | model/prompt/answer format; inspect 50 pairs |
| Router stores ~100% at λs=0.2 | predicted rho compressed; check features and reward targets |
| Gate never escalates | no disagreement on supervision split; `gate.supervision=val` or inspect targets |
| SFT No Store stays ~38% | adapter not attached, loss masking wrong, LR too low |
| SFT No Store ~95% | too many examples from evaluated facts / too many epochs |
| CUDA OOM at 8B | lower batch sizes, `quantize_4bit`, or shard `device_map` |
