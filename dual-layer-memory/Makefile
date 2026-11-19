PYTHON ?= python3
CONFIG ?= configs/paper_full.yaml

.PHONY: install install-gpu test gpu-check smoke full clean

install:
	$(PYTHON) -m pip install -e ".[dev,analysis]"

install-gpu:
	$(PYTHON) -m pip install -e ".[dev,analysis,hf]"

test:
	MPLCONFIGDIR=$${TMPDIR:-/tmp}/dlmem-mpl $(PYTHON) -m pytest tests -q

gpu-check:
	$(PYTHON) scripts/check_gpu.py --config $(CONFIG)

# Real models, a few hundred facts: validates a fresh GPU node end-to-end.
smoke:
	$(PYTHON) -m dlmem.cli prepare  --config configs/gpu_smoke.yaml
	$(PYTHON) -m dlmem.cli train    --config configs/gpu_smoke.yaml
	$(PYTHON) -m dlmem.cli evaluate --config configs/gpu_smoke.yaml

# Paper-scale run (see REPRODUCTION.md for timings).
full:
	bash scripts/run_full.sh $(CONFIG)

clean:
	rm -rf runs data .pytest_cache build dist src/*.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
