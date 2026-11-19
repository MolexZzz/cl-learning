# cl-learning

Reproduction of **Dual-Layer Agentic Memory with Fast Write Routing and Slow
Consolidation** (Li et al., arXiv:2608.22215v2), with real Qwen3 backbones and a
ZsRE-based streaming benchmark.

- Project documentation: [`dual-layer-memory/README.md`](dual-layer-memory/README.md)
- Chinese report: [`dual-layer-memory/docs/复现报告.md`](dual-layer-memory/docs/复现报告.md)
- Reproduction guide: [`dual-layer-memory/REPRODUCTION.md`](dual-layer-memory/REPRODUCTION.md)

## Quick start

```bash
cd dual-layer-memory
pip install -e ".[dev,analysis]"
pip install -r requirements-gpu.txt
python scripts/check_gpu.py --model Qwen/Qwen3-1.7B
python -m dlmem.cli prepare  --config configs/paper_full.yaml
```

The repository contains no simulator backend; every experiment runs real Qwen3
checkpoints. See the project README for the dataset, GPU requirements and
alignment notes.
