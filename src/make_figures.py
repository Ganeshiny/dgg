#!/usr/bin/env python3
"""Build the manuscript figure set in two tiers from one command.

  python src/make_figures.py                    # both tiers
  python src/make_figures.py --tier main        # main text only
  DGG_JOURNAL=bmc python src/make_figures.py    # retarget column widths

Main text carries the minimum set that supports the argument; everything else
is auto-numbered into supplementary/ with a generated legend file. The split is
deliberate: three encodings of the same Fmax data (strip, bar, heatmap) is
redundancy, so the strip plot — the only one showing seed-level points rather
than summary bars alone — represents that data in the main text and the other
two are demoted.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_arc_ablations as abl
import plot_arc_bins as bins
from plot_style import (
    MAIN,
    MAIN_ITEM_CAP,
    METRIC_ORDER,
    SUPPLEMENTARY,
    JOURNAL,
    apply_style,
    assert_palette_locked,
    report_colorblind_audit,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ABLATIONS = REPO / "arc_tuning_cafa/ablations/nominal_30_identity_80_coverage"
DEFAULT_BINCSV = DEFAULT_ABLATIONS / "bin_evaluation/bin_metrics.csv"
DEFAULT_ARCHIVE = REPO / "plots/arc_tuning_cafa/ablations/ablation_test_metrics.csv"

# Main-text figures, in order. Each entry is (filename stem, caption).
MAIN_FIGURES = [
    ("figure1_ablation_faceted_micro_fmax",
     "Architecture and input-modality ablation. Micro-F$_{max}$ on the held-out test split for "
     "five architectures under three input modalities, across the three GO sub-ontologies. Points "
     "Rows are input modality and columns are ontology; points are individual training seeds (n = 5) "
     "and the horizontal bar is the mean ± s.d. Modality is faceted rather than encoded as a marker "
     "shape, so no symbol decoding is required. Full and sequence-only are near-identical for every "
     "architecture, while structure-only collapses — the sequence representation carries the signal."),
    ("figure2_metric_family_micro_fmax_vs_micro_auroc",
     "Model ranking depends on the metric family. Identical layout and colour mapping in both rows; "
     "only the metric changes. F$_{max}$ (top) favours the graph-aware Hybrid variants, while AUROC "
     "(bottom) favours the MLP/GCN baselines. Neither family alone supports a single best-model "
     "claim, and the two must be reported together."),
    ("figure3_homology_micro_fmax",
     "Generalisation across sequence-homology bins. Micro-F$_{max}$ by maximum BLAST identity of each "
     "test protein against the training set, per input modality (rows) and ontology (columns). "
     "Marker area is proportional to the square root of the number of test proteins in the bin; "
     "points are not connected across bins with insufficient support."),
]


def _copy(src_dir: Path, stem_src: str, dest_dir: Path, stem_dest: str) -> list[str]:
    """Copy every rendered format of one figure, renaming to its final stem."""
    moved = []
    for path in sorted(src_dir.glob(f"{stem_src}.*")):
        if path.suffix.lower() in {".png", ".pdf", ".tiff", ".svg"}:
            target = dest_dir / f"{stem_dest}{path.suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            moved.append(target.name)
    return moved


def build(tier_wanted: str, out_root: Path, ablations_root: Path, bin_csv: Path,
          archive: Path, logs: Path, support_root: Path, err: str) -> None:
    apply_style()
    print(f"Target journal: {JOURNAL}")
    print("Palette fingerprint:", assert_palette_locked())
    report_colorblind_audit()

    work = out_root / "_work"
    work.mkdir(parents=True, exist_ok=True)
    main_dir = out_root / "main_text"
    supp_dir = out_root / "supplementary"

    # ---- ablation data (shared by figures 1 and 2 and by the supplement) ----
    df = abl.read_results(ablations_root, logs)
    expected = 3 * len(abl.MODEL_ORDER) * len(abl.VARIANT_ORDER) * abl.EXPECTED_SEEDS
    if len(df) < expected and archive.exists():
        archived = pd.read_csv(archive)
        if len(archived) >= expected:
            print(f"Using consolidated archive {archive} ({len(archived)} rows) "
                  f"over {len(df)} locally reconstructed rows.")
            df = archived
    if df.empty:
        raise SystemExit("No ablation data found.")
    abl.audit_ablation_integrity(df, support_root, ablations_root, work)
    abl.report_coverage(abl.coverage_table(df))

    supp_entries: list[tuple[str, str]] = []

    # ---------------- MAIN TEXT ----------------
    if tier_wanted in ("main", "both"):
        abl.plot_strip_faceted(df, work, "Micro_Fmax", err, MAIN)
        _copy(work, "faceted_micro_fmax", main_dir, MAIN_FIGURES[0][0])

        abl.plot_metric_family_composite(df, work, "Micro_Fmax", "Micro_AUROC", err, MAIN)
        _copy(work, "metric_family_micro_fmax_vs_micro_auroc", main_dir, MAIN_FIGURES[1][0])

        bin_frame = bins.load_audited_bins(bin_csv, support_root.parent, logs_hint=None)
        if bin_frame is not None:
            bins.plot_bin_grid(bin_frame[bin_frame.bin_type == "homology"], work, "homology",
                               bins.BIN_ORDER["homology"], "Micro_Fmax", 10, err, MAIN)
            _copy(work, "homology_micro_fmax", main_dir, MAIN_FIGURES[2][0])

        legend = ["# Main-text figure legends", ""]
        for i, (stem, caption) in enumerate(MAIN_FIGURES, start=1):
            if list(main_dir.glob(f"{stem}.*")):
                legend.append(f"**Figure {i}.** {caption}")
                legend.append("")
        (main_dir / "figure_legends.md").write_text("\n".join(legend))
        # Count rendered figures only — figure_legends.md also matches "figure*".
        n_main = len({p.stem for p in main_dir.glob("figure*.*")
                      if p.suffix.lower() in {".pdf", ".tiff", ".png", ".svg"}})
        print(f"Main text: {n_main} figures -> {main_dir}")
        if MAIN_ITEM_CAP and n_main > MAIN_ITEM_CAP:
            print(f"  WARNING: {n_main} main-text display items exceeds the {JOURNAL} "
                  f"guidance of {MAIN_ITEM_CAP}.")

    # ---------------- SUPPLEMENTARY ----------------
    if tier_wanted in ("supplementary", "both"):
        # Every remaining ablation metric, both encodings, plus the heatmap.
        for metric in METRIC_ORDER:
            if metric not in df:
                continue
            abl.plot_strip_faceted(df, work, metric, err, SUPPLEMENTARY)
            supp_entries.append((f"faceted_{metric.lower()}",
                                 f"{metric.replace('_', '-')} with input modality as rows and "
                                 f"ontology as columns. {abl._metric_note(metric)}"))
            abl.plot_strip(df, work, metric, err, SUPPLEMENTARY)
            supp_entries.append((f"strip_{metric.lower()}",
                                 f"Per-seed {metric.replace('_', '-')} for all architectures and "
                                 f"input modalities. {abl._metric_note(metric)}"))
            abl.plot_dynamite(df, work, metric, err, SUPPLEMENTARY)
            supp_entries.append((f"dynamite_{metric.lower()}",
                                 f"Bar-and-error-bar rendering of the same {metric.replace('_', '-')} "
                                 f"data shown per-seed elsewhere; retained for readers who expect "
                                 f"this encoding. {abl._metric_note(metric)}"))
        abl.plot_heatmap(df, work, "Micro_Fmax", SUPPLEMENTARY)
        supp_entries.append(("micro_fmax_model_input_heatmap",
                             "Mean Micro-F$_{max}$ per architecture and input modality as a "
                             "quick-reference grid; colour scale clipped to the observed range."))

        # All bin-stratified metrics, both stratification axes.
        bin_frame = bins.load_audited_bins(bin_csv, support_root.parent, logs_hint=None)
        if bin_frame is not None:
            for bin_type in ("homology", "ic"):
                subset = bin_frame[bin_frame.bin_type == bin_type]
                if subset.empty:
                    continue
                for metric in METRIC_ORDER:
                    if metric not in subset:
                        continue
                    stem = f"{bin_type}_{metric.lower()}"
                    if tier_wanted == "both" and stem == "homology_micro_fmax":
                        continue  # promoted to main text
                    bins.plot_bin_grid(subset, work, bin_type, bins.BIN_ORDER[bin_type],
                                       metric, 10, err, SUPPLEMENTARY)
                    supp_entries.append((stem,
                                         f"{metric.replace('_', '-')} stratified by "
                                         f"{'sequence homology' if bin_type == 'homology' else 'information content'} "
                                         f"bin, per input modality and ontology."))

        lines = ["# Supplementary figure legends", ""]
        number = 0
        for stem, caption in supp_entries:
            if not list(work.glob(f"{stem}.*")):
                continue
            number += 1
            names = _copy(work, stem, supp_dir, f"figureS{number}_{stem}")
            if not names:
                number -= 1
                continue
            lines.append(f"**Supplementary Figure S{number}.** {caption}")
            lines.append("")
        (supp_dir / "figure_legends.md").write_text("\n".join(lines))
        print(f"Supplementary: {number} figures -> {supp_dir}")

    print(f"Working renders kept in {work}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["main", "supplementary", "both"], default="both")
    ap.add_argument("--output-dir", type=Path, default=REPO / "plots/figures")
    ap.add_argument("--ablations-root", type=Path, default=DEFAULT_ABLATIONS)
    ap.add_argument("--bin-csv", type=Path, default=DEFAULT_BINCSV)
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--logs-dir", type=Path, default=REPO / "logs")
    ap.add_argument("--support-root", type=Path,
                    default=REPO / "preprocessing/data_arc_rebuild_2026_07_14/datasets/threshold_30")
    ap.add_argument("--err", choices=["sd", "sem", "ci95"], default="sd")
    args = ap.parse_args()
    build(args.tier, args.output_dir.resolve(), args.ablations_root.resolve(),
          args.bin_csv.resolve(), args.archive.resolve(), args.logs_dir.resolve(),
          args.support_root.resolve(), args.err)


if __name__ == "__main__":
    main()
