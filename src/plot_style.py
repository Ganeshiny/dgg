#!/usr/bin/env python3
"""Shared constants and helpers for ARC tuning/ablation figures.

Single source of truth for ontology/model/variant ordering, the categorical
colour assignment, and small stats/rendering helpers, so every figure in
plot_arc_tuning.py / plot_arc_ablations.py / plot_arc_bins.py uses the same
visual language: a colour always means the same model, a hatch always means
the same input-modality variant.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ontologies — always the panel/facet dimension, left to right in this order.
# ---------------------------------------------------------------------------
ONTOLOGY_ORDER = ["molecular_function", "biological_process", "cellular_component"]
ONTOLOGY_LABEL = {
    "molecular_function": "Molecular function",
    "biological_process": "Biological process",
    "cellular_component": "Cellular component",
}
ONTOLOGY_SHORT = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}

# ---------------------------------------------------------------------------
# Models — baseline-to-proposed narrative order, kept fixed everywhere so the
# same colour always names the same architecture across every figure.
# ---------------------------------------------------------------------------
MODEL_ORDER = ["MLP", "GCN", "GAT", "Hybrid", "Hybrid_JK"]

# CVD-validated 8-hue categorical palette, fixed order (see dataviz skill
# references/palette.md). Only the first 5 slots are used, in sequence, so
# the validated adjacent-pair colour-vision-deficiency separation holds.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]
MODEL_COLOR = dict(zip(MODEL_ORDER, CATEGORICAL_PALETTE))
MODEL_MARKER = {"MLP": "o", "GCN": "s", "GAT": "^", "Hybrid": "D", "Hybrid_JK": "P"}

# ---------------------------------------------------------------------------
# Ablation variants (input modality) — encoded as a hatch, not a second hue,
# so model-identity colour never has to compete with modality for the same
# channel. Solid / 45 degree / crossed, the same trio everywhere.
# ---------------------------------------------------------------------------
VARIANT_ORDER = ["full", "seq_only", "struct_only"]
VARIANT_LABEL = {"full": "Full (seq + struct)", "seq_only": "Sequence only", "struct_only": "Structure only"}
VARIANT_HATCH = {"full": "", "seq_only": "//", "struct_only": "xx"}

# ---------------------------------------------------------------------------
# Metrics — never mix a bounded higher-is-better metric with Smin (unbounded,
# lower-is-better) on one axis. Each metric gets its own figure/axis.
# ---------------------------------------------------------------------------
METRIC_ORDER = ["Micro_Fmax", "Macro_Fmax", "Micro_AUPRC", "Macro_AUPRC", "Micro_AUROC", "Macro_AUROC", "Smin"]
METRIC_LABEL = {
    "Micro_Fmax": "Micro-F$_{max}$",
    "Macro_Fmax": "Macro-F$_{max}$",
    "Micro_AUPRC": "Micro-AUPRC",
    "Macro_AUPRC": "Macro-AUPRC",
    "Micro_AUROC": "Micro-AUROC",
    "Macro_AUROC": "Macro-AUROC",
    "Smin": "S$_{min}$",
}
METRIC_HIGHER_IS_BETTER = {m: (m != "Smin") for m in METRIC_ORDER}

# Validation-side metrics use a different (lowercase, prefixed) naming
# convention than test-side metrics in this codebase — kept separate on
# purpose rather than normalised, to avoid silently mismatching a key.
VALIDATION_METRIC_ORDER = [
    "validation_micro_fmax", "validation_macro_fmax",
    "validation_micro_aupr", "validation_macro_aupr",
    "validation_micro_auroc", "validation_macro_auroc",
    "validation_smin",
]
VALIDATION_METRIC_LABEL = {
    "validation_micro_fmax": "Micro-F$_{max}$",
    "validation_macro_fmax": "Macro-F$_{max}$",
    "validation_micro_aupr": "Micro-AUPRC",
    "validation_macro_aupr": "Macro-AUPRC",
    "validation_micro_auroc": "Micro-AUROC",
    "validation_macro_auroc": "Macro-AUROC",
    "validation_smin": "S$_{min}$",
}
VALIDATION_METRIC_HIGHER_IS_BETTER = {m: (not m.endswith("smin")) for m in VALIDATION_METRIC_ORDER}

# ---------------------------------------------------------------------------
# Homology / information-content bin ordering — these are ordinal categories
# (increasing similarity to the training set / increasing term specificity),
# not alphabetic strings. Sorting them lexicographically silently scrambles
# the x-axis, which is a correctness bug, not a style choice.
# ---------------------------------------------------------------------------
BIN_ORDER = {
    "homology": ["no_hit", "<30%", "30-40%", "40-60%", ">=60%"],
    "ic": ["no_positive_terms", "<2_bits", "2-4_bits", "4-6_bits", ">=6_bits"],
}
BIN_AXIS_LABEL = {
    "homology": "Max. train-set sequence identity",
    "ic": "Max. information content of positive terms",
}

# ---------------------------------------------------------------------------
# Nature-style rcParams. Sans-serif, small point sizes, hairline solid
# gridlines on y only, no top/right spine, fonttype 42 so PDF text stays
# selectable/editable rather than outlined (most journals require this).
# ---------------------------------------------------------------------------
def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Nature figure widths, in inches (89 mm single column / 183 mm double column).
SINGLE_COLUMN_IN = 3.5
DOUBLE_COLUMN_IN = 7.2


def colorblind_audit() -> dict[str, float]:
    """Return minimum pairwise colour distances under common simulations.

    Model markers and input hatches carry identity as a second channel; this
    audit still guards the fixed palette against indistinguishable hues.
    """
    matrices = {
        "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                                [0.114503, 0.786281, 0.099216],
                                [-0.003882, -0.048116, 1.051998]]),
        "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                                  [0.280085, 0.672501, 0.047413],
                                  [-0.011820, 0.042940, 0.968881]]),
        "grayscale": np.array([[0.2126, 0.7152, 0.0722],
                               [0.2126, 0.7152, 0.0722],
                               [0.2126, 0.7152, 0.0722]]),
    }
    rgb = []
    for name in MODEL_ORDER:
        value = MODEL_COLOR[name].lstrip("#")
        rgb.append(np.array([int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]))
    result = {}
    for label, matrix in matrices.items():
        simulated = np.clip(np.asarray(rgb) @ matrix.T, 0, 1)
        distances = [np.linalg.norm(simulated[i] - simulated[j])
                     for i in range(len(simulated)) for j in range(i)]
        result[label] = float(min(distances))
    return result


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def label_panel(ax: plt.Axes, letter: str) -> None:
    """Bold panel label (a, b, c...) at the top-left, just outside the axes."""
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def mean_and_error(values: np.ndarray, kind: str = "sd") -> tuple[float, float, int]:
    """Return (mean, error, n) over finite values only. kind: sd | sem | ci95.

    Defaults to SD, matching this repo's existing "mean +/- SD over seeds"
    convention. n is the count of finite (non-NaN) values actually seen —
    callers should surface it, since a bar or point looks identical whether
    it summarises 5 seeds or 1.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(values.mean())
    if n == 1:
        return mean, 0.0, 1
    sd = float(values.std(ddof=1))
    if kind == "sd":
        return mean, sd, n
    if kind == "sem":
        return mean, sd / np.sqrt(n), n
    if kind == "ci95":
        from scipy import stats
        return mean, sd / np.sqrt(n) * float(stats.t.ppf(0.975, df=n - 1)), n
    raise ValueError(f"Unknown error kind: {kind!r}")


ERROR_KIND_LABEL = {"sd": "mean ± SD", "sem": "mean ± SEM", "ci95": "mean ± 95% CI"}


def jitter(n: int, width: float = 0.06, seed: int = 0) -> np.ndarray:
    """Deterministic jitter so a strip plot doesn't re-shuffle every re-run."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


def annotate_insufficient_data(ax: plt.Axes, message: str = "insufficient data") -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="#898781", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
