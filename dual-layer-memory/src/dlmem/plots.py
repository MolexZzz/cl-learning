"""Figures (paper Figures 2, 4, 5, 6 and the Table 2 heat-map)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="dlmem-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .utils import ensure_dir  # noqa: E402

_STYLE = {
    "small": {"color": "#6a8caf", "marker": "o"},
    "large": {"color": "#c0504d", "marker": "s"},
}


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _frontier_xy(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda r: r["storage"])
    return (
        np.asarray([r["storage"] for r in ordered]),
        np.asarray([r["em"] for r in ordered]),
    )


def plot_pareto(
    curves: Mapping[str, Sequence[dict]],
    path: str | Path,
    title: str = "EM-storage frontier",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    styles = {
        "small": dict(color="#6a8caf", marker="o", ls="-", label="small router"),
        "large": dict(color="#c0504d", marker="s", ls="-", label="large router"),
        "cascade_0.4": dict(color="#4f8a5b", marker="^", ls="--", label="cascade, 0.4"),
        "cascade_0.6": dict(color="#8064a2", marker="v", ls="--", label="cascade, 0.6"),
    }
    for name, rows in curves.items():
        x, y = _frontier_xy(rows)
        style = styles.get(name, {"marker": "o", "ls": "-", "label": name})
        ax.plot(x, y, ms=4, lw=1.6, **style)
    ax.set_xlabel("Test storage")
    ax.set_ylabel("Test EM")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, path)


def plot_two_panel_ablation(
    left: Mapping[str, Sequence[dict]],
    right: Mapping[str, Sequence[dict]],
    path: str | Path,
    left_title: str = "Step 1: small-side feature ablation",
    right_title: str = "Step 1: large vs. small+large",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    palette = ["#4f81bd", "#c0504d", "#9bbb59", "#8064a2", "#4bacc6", "#f79646"]
    for ax, data, title in ((axes[0], left, left_title), (axes[1], right, right_title)):
        for i, (name, rows) in enumerate(data.items()):
            x, y = _frontier_xy(rows)
            ax.plot(x, y, marker="o", ms=3, lw=1.4, color=palette[i % len(palette)], label=name)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Test storage")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Test EM")
    _save(fig, path)


def plot_gate_ablation(
    panels: Mapping[str, Mapping[str, Sequence[dict]]],
    path: str | Path,
) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, max(1, n), figsize=(5.0 * max(1, n), 3.8), sharey=True)
    if n == 1:
        axes = [axes]
    palette = ["#4f81bd", "#c0504d", "#9bbb59", "#8064a2", "#4bacc6"]
    for ax, (cap, data) in zip(axes, panels.items()):
        for i, (name, rows) in enumerate(data.items()):
            x, y = _frontier_xy(rows)
            ax.plot(x, y, marker="o", ms=3, lw=1.4, color=palette[i % len(palette)], label=name)
        ax.set_title(f"Gate ablation, cap={cap}", fontsize=9)
        ax.set_xlabel("Test storage")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Test EM")
    _save(fig, path)


def plot_transition_heatmap(
    matrix: Mapping[str, Mapping[str, int]],
    order: Sequence[str],
    path: str | Path,
    title: str = "Post-write-back label transitions",
) -> None:
    data = np.asarray([[matrix[b][a] for a in order] for b in order], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(order)), order, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)), [f"{b} ({int(r.sum())})" for b, r in zip(order, data)], fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            val = int(data[i, j])
            color = "white" if data[i, j] > data.max() * 0.6 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=9, color=color)
    ax.set_xlabel("SFT labels")
    ax.set_ylabel("Base labels")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_storage_over_time(
    trajectories: Mapping[str, Sequence[float]],
    path: str | Path,
    title: str = "External memory growth",
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, ys in trajectories.items():
        ax.plot(range(len(ys)), ys, lw=1.6, label=name)
    ax.set_xlabel("Streaming turn")
    ax.set_ylabel("Stored facts")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    _save(fig, path)
