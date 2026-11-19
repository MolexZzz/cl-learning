# Dataset and benchmark details

## 1. Fact corpus

Each record is a `(subject, relation, object)` fact plus `m` paraphrase probes:

```json
{
  "fact_id": "zsre-0000123",
  "subject": "Kurt Vonnegut",
  "relation": "was born in",
  "object": "Indianapolis",
  "text": "Kurt Vonnegut was born in Indianapolis.",
  "probes": [
    {"question": "Where was Kurt Vonnegut born?", "answer": "Indianapolis"},
    {"question": "What is Kurt Vonnegut's birthplace?", "answer": "Indianapolis"}
  ]
}
```

`text` is written to external memory and used for semantic/uncertainty features.
`probes` are used for behavioural labelling and downstream QA. The paper keeps
facts with **exactly 30** probes; the loader de-duplicates questions
(SQuAD-style normalisation) and drops records whose probe count is not exactly
`m`.

### Supported layouts

`dlmem.data.zsre` normalises all of the following:

1. **Original / KnowEdit style**

   ```json
   {"input": "Kurt Vonnegut was born in Indianapolis.",
    "output": [{"question": "Where was Kurt Vonnegut born?",
                "answer": "Indianapolis"}]}
   ```

2. **Probe-list style**

   ```json
   {"text": "...", "subject": "...", "relation": "...", "object": "...",
    "probes": [{"question": "...", "answer": "..."}]}
   ```

3. **Query style**

   ```json
   {"src": "...", "subject": "...", "relation": "...", "object": "...",
    "queries": [{"query": "...", "answer": "..."}]}
   ```

4. **HuggingFace dataset ids** via `data.hf_name` / `--hf-name`, optionally with
   `data.hf_config`. The loader tries the train/test/validation splits.

A directory is scanned recursively for `*.json`, `*.jsonl`, `*.json.gz`;
`manifest.json` and `dataset_stats.json` are skipped.

### Splits

Facts are shuffled with the configured seed and split 70 / 15 / 15. With the
paper's 201,352 facts this gives ~30,203 test facts, matching the 300 episodes ×
100 facts = 30,000 unique facts in the stream (drawn without replacement).

The exact 201,352-fact / 30-probe dump is not public as of writing. Different
ZsRE variants change the absolute numbers; the filter and taxonomy are the parts
that must be preserved. See [`ALIGNMENT.md`](ALIGNMENT.md) §1.

---

## 2. Behavioural labelling (Appendix A.1)

For a probe `q_ij` and fact `f_i`, generate twice with greedy decoding:

* zero-shot: prompt = `q_ij`
* memory-supported: prompt = `f_i + q_ij`

Compare against gold `a_ij`:

| zero-shot | with memory | probe class |
|---|---|---|
| correct | (any) | `internal` |
| refusal | correct | `missing` (`m_new`) |
| wrong, non-refusal | correct | `stale` (`m_update`) |
| wrong | wrong | `unsolved` |

Refusal detection uses the regex list in `dlmem.utils.REFUSAL_PATTERNS`. Fact-level
aggregation (Eq. 9):

```text
rho_i   = (m_new + m_update) / m
gamma_i = m_update / (m_new + m_update)          # 0 if no memory-dependent probe

non-write     if rho_i < 0.01
write-update  if rho_i >= 0.01 and gamma_i >= 0.5
write-new     otherwise
```

Implementation: `dlmem.data.labeling.label_facts` / `label_from_counts`. Per-fact
counts are cached in `runs/<run>/labels/<split>.jsonl`; labels are never read
from file metadata.

Sanity checks on the paper's distribution:

* fact mix: non-write : write-new : write-update = 23.5 : 45.1 : 31.4 (%)
* stale probes: 19.4% of all probes
* Full Store EM ≈ 1 − unsolved rate ≈ 87.8%
* No Store EM ≈ 38.0%

---

## 3. Streaming episodes (Appendix A.2, Algorithm 1)

An episode is `T` turns over `K` facts:

* `K` turns carry injections (one fact each, injected exactly once);
* the remaining `T − K` turns carry queries sampled from probes of facts already
  injected.

This resolves Algorithm 1's ambiguity: Stage 2 budgets `T − K` probes (it
initialises `c = K`), while Stage 3 loops over `T` turns. Injections do not also
carry a query. For the paper's `T=500, K=100`, that is 400 QA interactions per
episode.

Injection turns are stratified: fact `k` draws uniformly from
`[⌊kT/K⌋, ⌊(k+1)T/K⌋−1]`, with the final turn reserved for a query. The probe
budget is one per fact plus a near-uniform remainder, capped by `m` and by the
query turns remaining after injection; the streaming loop can extend a fact's
quota on demand if a stratified gap is longer than expected.

Every episode is asserted against:

* **P1 coverage** — every fact injected once and probed ≥ 1 time;
* **P2 causality** — a query's turn is ≥ its fact's injection turn;
* **P3 liveness** — at least one injection at turn 0;
* **P4 distractors** — the fact pool preserves the test-split class mix.

Seeds: `seed + 1000 + episode_id`; episodes are bit-exactly reproducible.

---

## 4. Metrics

**Downstream QA** (per query turn, averaged over turns/episodes)

* Exact Match: SQuAD-normalised **containment** (`Paris` matches `The answer is
  Paris`), the standard ZsRE/SQuAD-style criterion;
* token-level F1;
* refusal rate.

**Storage decision quality** (per injected fact; positives = write-new or
write-update)

* storage ratio = written / injected;
* Store Precision = TP / (TP + FP);
* Store Recall = TP / (TP + FN);
* Store F1.

**Routing cost** (relative forward-pass units)

* No/Random/Full Store: 0;
* PPL-Conditional: 1 × Fwd8B;
* Logistic / MLP: 1 × Fwd8B;
* Heuristic Store: 2 × Gen8B (question generation + answer generation);
* Write Router: 1 × Fwd1.7B + escalated fraction × Fwd8B (4.7× ratio).

**Offline EM proxy** (Pareto frontiers)

```text
EM(stored)    = (m − unsolved) / m
EM(discarded) = internal / m
```

Exact under the labelling model; reproduces the Full Store / No Store endpoints.
The streaming benchmark is the end-to-end check with real retrieval/generation.

---
