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
DEFAULT_BINCSV = REPO / "plots/arc_tuning_cafa/bin_evaluation/bin_metrics.csv"
DEFAULT_ARCHIVE = REPO / "plots/arc_tuning_cafa/ablations/ablation_test_metrics.csv"

# Main-text figures, in order. Each entry is (filename stem, caption).
MAIN_FIGURES = [
    ("figure1_ablation_dynamite_micro_fmax",
     "Architecture and input-modality ablation. Micro-F$_{max}$ on the held-out test split for "
     "five architectures under three input modalities and three GO sub-ontologies. Input modality "
     "is stacked vertically as panel rows; within every panel, non-additive model scores are shown "
     "as separate grouped bars with mean ± s.d. across five seeds. Full and sequence-only are "
     "near-identical, while structure-only collapses — the sequence representation carries the signal."),
    ("figure2_metric_family_micro_fmax_vs_micro_auroc",
     "Model ranking depends on the metric family. Identical layout and colour mapping in both rows; "
     "only the metric changes. F$_{max}$ (top) favours the graph-aware Hybrid variants, while AUROC "
     "(bottom) favours the MLP/GCN baselines. Neither family alone supports a single best-model "
     "claim, and the two must be reported together."),
    ("figure3_homology_micro_fmax",
     "Generalisation across sequence-homology bins. Micro-F$_{max}$ by maximum BLAST identity of each "
     "test protein against the training set, per input modality (rows) and ontology (columns). "
     "Models are separate grouped bars; exact test-bin counts are printed in brackets on the "
     "x-axis, and bins without finite data are omitted."),
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
    ablation_audit = abl.audit_ablation_integrity(df, support_root, ablations_root, work)
    ablation_coverage = abl.coverage_table(df)
    abl.report_coverage(ablation_coverage)
    table_dir = out_root / "supplementary_tables"
    abl.export_ablation_tables(df, ablation_coverage, out_root)
    ablation_audit.to_csv(table_dir / "supp_table_ablation_integrity.csv", index=False)
    (table_dir / "README.md").write_text(
        "# Supplementary tables\n\n"
        "- `supp_table_ablation_seed_metrics.csv`: every seed-level ablation value.\n"
        "- `supp_table_ablation_metric_summary.csv`: n, mean, SD, median, quartiles, min and max.\n"
        "- `supp_table_ablation_heatmap_micro_fmax.csv`: exact heatmap values.\n"
        "- `supp_table_mlp_structure_constant_control.csv`: null-control Macro-AUPRC/AUROC audit.\n"
        "- `supp_table_bin_metrics_audited_raw.csv`: every audited split/model/bin row.\n"
        "- `supp_table_bin_metric_summary.csv`: per-bin seed summaries for every metric.\n"
        "- `supp_table_bin_support.csv`: exact examples and positive-label support.\n"
        "- `supp_table_bin_smin.csv`: Smin bin values (table-only).\n\n"
        "The current archive contains test-bin predictions only. Validation rows are added by "
        "`run_arc_bin_eval.slurm`, which now evaluates both `valid` and `test`.\n"
    )

    supp_entries: list[tuple[str, str]] = []

    # ---------------- MAIN TEXT ----------------
    if tier_wanted in ("main", "both"):
        abl.plot_dynamite(df, work, "Micro_Fmax", err, MAIN)
        _copy(work, "dynamite_micro_fmax", main_dir, MAIN_FIGURES[0][0])

        abl.plot_metric_family_composite(df, work, "Micro_Fmax", "Micro_AUROC", err, MAIN)
        _copy(work, "metric_family_micro_fmax_vs_micro_auroc", main_dir, MAIN_FIGURES[1][0])

        bin_frame = bins.load_audited_bins(bin_csv, support_root.parent, logs_hint=None)
        if bin_frame is not None:
            test_bins = bin_frame[(bin_frame.bin_type == "homology") &
                                  (bin_frame.evaluation_split == "test")]
            if not test_bins.empty:
                bins.plot_bin_grid(test_bins, work, "homology", bins.BIN_ORDER["homology"],
                                   "Micro_Fmax", 10, err, MAIN)
                _copy(work, "test_homology_micro_fmax", main_dir, MAIN_FIGURES[2][0])

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
        # One clear encoding per metric: vertically faceted grouped bars. The
        # raw seeds and full summaries are exported as tables below.
        for metric in METRIC_ORDER:
            if metric not in df:
                continue
            abl.plot_dynamite(df, work, metric, err, SUPPLEMENTARY)
            supp_entries.append((f"dynamite_{metric.lower()}",
                                 f"Vertically faceted grouped-bar rendering of "
                                 f"{metric.replace('_', '-')}; rows are input modalities and columns "
                                 f"are ontologies. {abl._metric_note(metric)}"))
        abl.plot_heatmap(df, work, "Micro_Fmax", SUPPLEMENTARY)
        supp_entries.append(("micro_fmax_model_input_heatmap",
                             "Mean Micro-F$_{max}$ per architecture and input modality as a "
                             "quick-reference grid; colour scale clipped to the observed range."))

        # All non-Smin bin metrics. Smin is table-only; empty bins/panels are
        # omitted by plot_bin_grid, and validation appears automatically once
        # the split-aware ARC evaluator has generated those rows.
        bin_frame = bins.load_audited_bins(bin_csv, support_root.parent, logs_hint=None)
        if bin_frame is not None:
            bins.export_bin_tables(bin_frame, out_root)
            for evaluation_split in bin_frame.evaluation_split.dropna().astype(str).unique():
                for bin_type in ("homology", "ic"):
                    subset = bin_frame[(bin_frame.bin_type == bin_type) &
                                       (bin_frame.evaluation_split == evaluation_split)]
                    if subset.empty:
                        continue
                    for metric in METRIC_ORDER:
                        if metric == "Smin" or metric not in subset:
                            continue
                        stem = f"{evaluation_split}_{bin_type}_{metric.lower()}"
                        if tier_wanted == "both" and stem == "test_homology_micro_fmax":
                            continue  # promoted to main text
                        if bins.plot_bin_grid(subset, work, bin_type, bins.BIN_ORDER[bin_type],
                                              metric, 10, err, SUPPLEMENTARY):
                            supp_entries.append((stem,
                                f"{metric.replace('_', '-')} for the {evaluation_split} split, "
                                f"stratified by {'sequence homology' if bin_type == 'homology' else 'information content'}; "
                                "exact bin counts are printed on the x-axis and empty cells are omitted."))

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
