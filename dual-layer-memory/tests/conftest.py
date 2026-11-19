"""Test configuration: make ``src`` importable and keep matplotlib headless."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for path in (SRC, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="dlmem-test-mpl-"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
