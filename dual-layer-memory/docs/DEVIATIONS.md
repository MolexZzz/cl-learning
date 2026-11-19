# Deviations and unresolved choices

This repository contains no simulator backend. Every experiment calls a real
HuggingFace Qwen3 checkpoint. The list below records where the implementation
had to choose between under-specified alternatives, and where it intentionally
departs from a literal reading of the paper.

## 1. Full-scale execution status

The paper-scale run (201k facts, 6M probes, 300×500 episodes, 8B backbones) was
not executed while building this repository: there was no CUDA node, no model
weights and no ZsRE dump. The CUDA code paths are implemented and unit-tested
where possible (token NLL, config, routers, episodes), and a real-model smoke
config is provided, but **the paper's numbers have not been reproduced here**.

## 2. Gate supervision split

App. C: freeze the routers, then build gate targets from their greedy actions.
Literal implementation uses the training split, which is the default
(`gate.supervision: train`). On small datasets the routers overfit and never
disagree, so every gate target is `-lambda_e` and the gate collapses. For small
runs, set `gate.supervision: val`. The paper-scale run should keep `train`.

## 3. Gate class imbalance

Escalation gains are rare (the small router is usually right). The gate is
trained with positive rows oversampled (`gate.oversample`, default 8) to avoid
a constant negative predictor. Set to 1 for plain MSE on the natural
distribution. The escalation budget (`cascade.escalation_cap`) then selects the
top-ranked facts.

## 4. Reward-head calibration

`router.calibration: none` is the paper-scale default. A monotone quantile
calibration (`quantile`) is available because with a few hundred facts MSE
regression shrinks predicted `rho` toward the class mean and destroys the
absolute λs threshold. It is a small-data workaround, not part of the paper.

## 5. Large-router features

Eq. 7 writes `z_large = [e_large]` (semantic only), while Appendix D.1 reports
semantic+uncertainty as the strongest overall frontier. The default is
semantic+uncertainty for both routers; set `router.large_feature_mode: semantic`
for the literal Eq. 7 variant.

## 6. Episode convention

Algorithm 1's Stage 2 initialises `c = K` and fills `T − K` probe slots, while
Stage 3 loops over all `T` turns. We adopt the Stage-2 reading:

* `K` injection turns (no query on those turns);
* `T − K` query turns, one query each;
* stratified injections, probe budget `T − K`, on-demand extension if a gap is
  longer than expected.

This yields a feasible schedule for `T=500, K=100, m=30`; injections never
violate causality (P2) and every fact is probed (P1). If you prefer a query on
every turn, change `synthesize_episode` to release probes at injection turns and
sample on all `T` turns; the budget becomes `T` probes.

## 7. Write-back scope

Sec. 3.4 consolidates external memories accumulated during deployment, and
Table 2 tracks the held-out test split. The default (`sft.source: test`) builds
SFT examples from test-stream memories written at λs=0.08. This is the intended
lifecycle, not train/test leakage: consolidation is evaluated on the memories
the agent actually accumulated. For a leakage-free transfer estimate, set
`sft.source: train`.

## 8. Heuristic Store

App. B describes generating a factual question from the fact, answering it and
judging confidence. We use the fact's first probe as that question (it is
already a factual question about the fact), combine zero-shot correctness with
answer NLL, and calibrate the threshold on validation to the paper's storage
level (59%).

## 9. Retrieval

The paper does not specify the retriever. This repo uses BM25 top-3 over fact
sentences. `retrieval_hit_rate` is reported separately from admission quality,
so you can swap in a dense retriever without changing the evaluation code.

## 10. Post-SFT storage metrics

Table 1's footnote evaluates Write Router_SFT recall against static base-model
targets. This implementation measures storage against the post-SFT labels, so
Full Store precision drops after consolidation (many facts became non-write).
This is internally consistent; switch the labels in
`eval/stream.py::run_streaming` to reproduce the static-target convention.

## 11. Model and prompt details

* Qwen3 chat templates are applied with `enable_thinking=False`; if a tokenizer
  rejects the flag, the default template is used.
* Greedy decoding everywhere (`do_sample=False`), `max_new_tokens=16` for QA
  and labelling, `max_new_tokens=12–16` in the smoke config.
* Feature extraction uses the final hidden layer, mean-pooled over non-pad
  tokens; NLL is teacher-forced per token.
* `torch_dtype`/`dtype` is handled for Transformers < 4.56 and ≥ 4.56.
* On MPS, bf16 is downgraded to fp16; the paper-scale 8B run is CUDA-only.

## 12. SFT hyperparameters

The paper does not give LoRA rank, LR, epochs or example construction. Defaults:
LoRA r=16, α=32, dropout 0.05, LR 2e-5, 3 epochs, effective batch 32
(4×8), max length 512, two QA pairs plus one sentence-level target per fact,
assistant-token-only loss. Both backbones are adapted, as stated in Sec. 4.2.
Tune these if SFT No Store EM does not move toward 69%.

## 13. Dataset release

The paper's 201,352-fact / 6,040,560-probe ZsRE variant is not public as of this
writing. The loader accepts the common ZsRE layouts and enforces exactly
`m = 30` probes, but a different dump changes all absolute numbers. See
[`ALIGNMENT.md`](ALIGNMENT.md) §1.

## 14. Scale and parallelism

* Both backbones are loaded in one process (`device_map="auto"`), which fits on
  one 4×H20 node.
* Streaming evaluation batches across episodes (`eval.batch_episodes`) but runs
  policies sequentially.
* Labelling and feature extraction are single-process; they can be sharded by
  fact range by running `prepare` on disjoint `data.max_facts` slices and
  merging the cached files, but no launcher is provided.
