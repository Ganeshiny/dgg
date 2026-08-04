#!/usr/bin/env python3
"""Shared constants and helpers for ARC tuning/ablation figures.

Single source of truth for ontology/model/variant ordering, the categorical
colour assignment, and small stats/rendering helpers, so every figure in
plot_arc_tuning.py / plot_arc_ablations.py / plot_arc_bins.py uses the same
visual language: a colour always means the same model, a hatch always means
the same input-modality variant.
"""
from __future__ import annotations

import hashlib
import json
import os
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
# Metrics use separate axes when their scales or optimization directions differ.
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


# ---------------------------------------------------------------------------
# Target journal. Figure widths and the main-text display-item cap differ by
# publisher, so they live in one switchable table rather than scattered
# literals. Override with DGG_JOURNAL=bmc in the environment.
# ---------------------------------------------------------------------------
JOURNAL_SPECS = {
    # Nature: 89 mm single / 183 mm double column, ~4 main display items.
    "nature": {"single_mm": 89.0, "double_mm": 183.0, "main_item_cap": 4,
               "raster": "tiff", "main_dpi": 600, "supp_dpi": 300},
    # BMC: 85 mm single / 170 mm double column, no hard main-text cap.
    "bmc": {"single_mm": 85.0, "double_mm": 170.0, "main_item_cap": None,
            "raster": "png", "main_dpi": 600, "supp_dpi": 300},
}
JOURNAL = os.environ.get("DGG_JOURNAL", "bmc").strip().lower()
if JOURNAL not in JOURNAL_SPECS:
    raise ValueError(f"Unknown DGG_JOURNAL={JOURNAL!r}; expected one of {sorted(JOURNAL_SPECS)}")
SPEC = JOURNAL_SPECS[JOURNAL]

_MM_PER_IN = 25.4
SINGLE_COLUMN_IN = SPEC["single_mm"] / _MM_PER_IN
DOUBLE_COLUMN_IN = SPEC["double_mm"] / _MM_PER_IN
MAIN_ITEM_CAP = SPEC["main_item_cap"]

# Figure tiers. Main-text figures are read at full column width and carry the
# argument, so they get larger minimum type; supplementary may run denser.
MAIN, SUPPLEMENTARY = "main", "supplementary"
TIER_MIN_FONT_PT = {MAIN: 7.0, SUPPLEMENTARY: 5.0}
TIER_DPI = {MAIN: SPEC["main_dpi"], SUPPLEMENTARY: SPEC["supp_dpi"]}


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
        distances = [(float(np.linalg.norm(simulated[i] - simulated[j])), MODEL_ORDER[j], MODEL_ORDER[i])
                     for i in range(len(simulated)) for j in range(i)]
        result[label] = min(distances)
    return result


# Below this RGB separation, two models are not reliably told apart by hue
# alone and must carry a second channel (marker shape, or x-axis position and
# tick label). Measured worst pairs on the locked palette: Hybrid vs MLP in
# grayscale (0.112) and Hybrid_JK vs MLP under deuteranopia (0.264). Hybrid vs
# GCN, often assumed to be the risky pair, is comfortably separated everywhere.
CVD_FLOOR = 0.30


def report_colorblind_audit(require_secondary_channel: bool = True) -> list[str]:
    """Print the worst pair per simulation; return warnings for sub-floor pairs."""
    warnings_out = []
    for label, (distance, first, second) in sorted(colorblind_audit().items()):
        flag = "" if distance >= CVD_FLOOR else "  <-- below floor"
        print(f"  colour audit [{label:12s}] worst pair: {first} vs {second} d={distance:.3f}{flag}")
        if distance < CVD_FLOOR:
            warnings_out.append(
                f"{first}/{second} separation {distance:.3f} under {label} is below {CVD_FLOOR}; "
                "hue alone is insufficient")
    if warnings_out and require_secondary_channel:
        print("  -> secondary channel required (marker shape and/or axis position); "
              "every figure in this set carries one.")
    return warnings_out


# ---------------------------------------------------------------------------
# Palette-drift guard. Manual iteration across three plotting scripts is
# exactly how a model quietly changes colour between figures, so the mapping
# is fingerprinted and checked rather than trusted.
# ---------------------------------------------------------------------------
PALETTE_FINGERPRINT = "56a06a743fcd9eb0"


def palette_fingerprint() -> str:
    payload = json.dumps({"models": MODEL_ORDER,
                          "colors": [MODEL_COLOR[m] for m in MODEL_ORDER],
                          "markers": [MODEL_MARKER[m] for m in MODEL_ORDER],
                          "variants": VARIANT_ORDER}, sort_keys=True)
    return hashlib.blake2s(payload.encode(), digest_size=8).hexdigest()


def assert_palette_locked() -> str:
    """Fail loudly if the model->colour/marker mapping drifted."""
    actual = palette_fingerprint()
    if PALETTE_FINGERPRINT and actual != PALETTE_FINGERPRINT:
        raise RuntimeError(
            f"Model colour/marker mapping changed (fingerprint {actual}, expected "
            f"{PALETTE_FINGERPRINT}). Every figure must share one mapping; update "
            "PALETTE_FINGERPRINT deliberately if the change is intended.")
    return actual


# ---------------------------------------------------------------------------
# Caption fragments. Defined once so the stats statement cannot drift between
# an axis label and the figure legend a reviewer actually reads.
# ---------------------------------------------------------------------------
ERROR_CAPTION = {
    "sd": "Error bars show mean ± s.d. across n = 5 training seeds",
    "sem": "Error bars show mean ± s.e.m. across n = 5 training seeds",
    "ci95": "Error bars show mean ± 95% CI across n = 5 training seeds",
}

MACRO_AUPRC_CAVEAT = (
    "Macro-AUPRC values are provisional: archived tables were produced with a "
    "trapezoidal PR estimator that is upward-biased for tied scores "
    "(see docs/figure_data_integrity.md); regenerate before reporting."
)
CONSTANT_PREDICTOR_CAVEAT = (
    "MLP under structure-only receives zeroed features and ignores graph edges, "
    "so it emits a constant score and acts as a null control, not a model."
)


def provenance(script: str, source: str, extra: str = "") -> str:
    tail = f" {extra}" if extra else ""
    return f"Generated by {script} from {source}.{tail}"


def savefig(fig: plt.Figure, path: Path, tier: str = SUPPLEMENTARY,
            formats: tuple[str, ...] | None = None) -> None:
    """Write a figure at its tier's resolution, vector first.

    Vector (PDF) is always written because journals require it for line art;
    the raster companion is TIFF or PNG depending on the target journal.
    """
    if tier not in TIER_DPI:
        raise ValueError(f"Unknown tier {tier!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dpi = TIER_DPI[tier]
    if formats is None:
        # Manuscript source, reviewer-friendly vector, and both common raster
        # companions are emitted together so figures never need to be rerun
        # merely to satisfy a supplementary-file format request.
        formats = ("svg", "pdf", "png", "tiff")
    try:
        # Respect figures that deliberately use constrained/GridSpec layout
        # (notably the heatmap with its dedicated colorbar row).
        if fig.get_layout_engine() is None:
            fig.tight_layout()
    except Exception:
        pass
    for suffix in formats:
        # Uncompressed 600 dpi TIFF runs to tens of MB per figure; LZW is
        # lossless and accepted by publishers, so there is no reason to ship
        # the raw form.
        extra = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(path.with_suffix(f".{suffix}"), bbox_inches="tight", dpi=dpi, **extra)
    plt.close(fig)


def check_min_font(fig: plt.Figure, tier: str) -> list[str]:
    """Return human-readable warnings for text below the tier's floor.

    Font size is set in points and rendered at a fixed physical width, so a
    label that is legible on screen can still be under the journal's minimum
    in print. This checks the actual rendered text objects.
    """
    floor = TIER_MIN_FONT_PT[tier]
    small = set()
    for text in fig.findobj(plt.Text):
        try:
            size = float(text.get_fontsize())
        except (TypeError, ValueError):
            continue
        if text.get_text().strip() and size < floor:
            small.add(round(size, 2))
    if not small:
        return []
    return [f"text at {sorted(small)} pt is below the {tier} minimum of {floor} pt"]


def label_panel(ax: plt.Axes, letter: str) -> None:
    """Bold panel label (a, b, c...) at the top-left, just outside the axes."""
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def format_bar_value(value: float) -> str:
    """Compact, deterministic numeric text for values printed on bars."""
    value = float(value)
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:.0f}"
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def label_vertical_bars(
    ax: plt.Axes,
    bars,
    values,
    errors=None,
    *,
    labels=None,
    fontsize: float = 5.0,
    rotation: float = 90,
    padding: float = 2.0,
) -> None:
    """Print exact values above vertical bars, beyond any upper error cap."""
    patches = list(getattr(bars, "patches", bars))
    values = np.asarray(values, dtype=float)
    if errors is None:
        upper_errors = np.zeros(len(values), dtype=float)
    else:
        error_array = np.asarray(errors, dtype=float)
        upper_errors = error_array if error_array.ndim == 1 else error_array[-1]
        upper_errors = np.nan_to_num(upper_errors, nan=0.0, posinf=0.0, neginf=0.0)
    rendered_tops = []
    for index, (patch, value) in enumerate(zip(patches, values)):
        if not np.isfinite(value):
            continue
        error = float(upper_errors[index]) if index < len(upper_errors) else 0.0
        top = float(patch.get_y() + patch.get_height()) + max(error, 0.0)
        text = labels[index] if labels is not None else format_bar_value(value)
        ax.annotate(
            str(text),
            (patch.get_x() + patch.get_width() / 2.0, top),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
            clip_on=False,
        )
        rendered_tops.append(top)
    if rendered_tops:
        ymin, ymax = ax.get_ylim()
        span = max(ymax - ymin, 1e-9)
        required = max(rendered_tops) + span * (0.13 if rotation else 0.08)
        if required > ymax:
            ax.set_ylim(ymin, required)


def label_horizontal_bars(
    ax: plt.Axes,
    bars,
    values,
    errors=None,
    *,
    labels=None,
    fontsize: float = 5.2,
    padding: float = 2.5,
) -> None:
    """Print exact values to the right of horizontal bars/error caps."""
    patches = list(getattr(bars, "patches", bars))
    values = np.asarray(values, dtype=float)
    if errors is None:
        right_errors = np.zeros(len(values), dtype=float)
    else:
        error_array = np.asarray(errors, dtype=float)
        right_errors = error_array if error_array.ndim == 1 else error_array[-1]
        right_errors = np.nan_to_num(right_errors, nan=0.0, posinf=0.0, neginf=0.0)
    rendered_tips = []
    for index, (patch, value) in enumerate(zip(patches, values)):
        if not np.isfinite(value):
            continue
        error = float(right_errors[index]) if index < len(right_errors) else 0.0
        tip = float(patch.get_x() + patch.get_width()) + max(error, 0.0)
        text = labels[index] if labels is not None else format_bar_value(value)
        ax.annotate(
            str(text),
            (tip, patch.get_y() + patch.get_height() / 2.0),
            xytext=(padding, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=fontsize,
            clip_on=False,
        )
        rendered_tips.append(tip)
    if rendered_tips:
        xmin, xmax = ax.get_xlim()
        span = max(xmax - xmin, 1e-9)
        required = max(rendered_tips) + span * 0.08
        if required > xmax:
            ax.set_xlim(xmin, required)


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
