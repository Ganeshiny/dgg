from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "arc slurms" / "run_full_benchmark.slurm"
SETUP = PROJECT_ROOT / "scripts" / "setup_arc_sota.sh"
ALL_FIGURES = PROJECT_ROOT / "arc slurms" / "run_all_figures.slurm"
PLOT = PROJECT_ROOT / "src" / "plot_baselines_only.py"


class AddedMethodContractTests(unittest.TestCase):
    def test_heal_has_setup_preflight_runner_normalization_and_stage(self):
        benchmark = BENCHMARK.read_text()
        setup = SETUP.read_text()

        self.assertIn("setup_heal()", setup)
        self.assertIn("github.com/ZhonghuiGu/HEAL.git", setup)
        self.assertIn("esm1b_t33_650M_UR50S.pt", setup)
        self.assertIn('"mkl<2024.1"', setup)
        self.assertIn('CURRENT_METHOD=heal', benchmark)
        self.assertIn('scripts/run_heal_arc.py', benchmark)
        self.assertIn('--method heal', benchmark)
        self.assertIn('stage 10b_heal run_heal', benchmark)

    def test_gat_go_has_strict_official_feature_integration(self):
        benchmark = BENCHMARK.read_text()
        setup = SETUP.read_text()
        runner = (PROJECT_ROOT / "scripts" / "run_gat_go_arc.py").read_text()

        self.assertIn("github.com/bl-2633/GAT-GO.git", setup)
        self.assertIn("90ec6d1067a893d4a51be715e41daf9fa4732952", setup)
        self.assertIn("--runtime-smoke-test", setup)
        self.assertIn("dgg_setup_tools", setup)
        self.assertIn("git_run clone", setup)
        self.assertIn("proteins_missing_or_invalid", runner)
        self.assertIn("labels_used_for_inference", runner)
        self.assertIn("Never access obj['label']", runner)
        self.assertIn('CURRENT_METHOD=gat_go', benchmark)
        self.assertIn('--method gat_go', benchmark)
        self.assertIn('stage 10c_gat_go run_gat_go', benchmark)

    def test_deepgraphgo_has_cpu_environment_and_synthetic_query_mapping(self):
        benchmark = BENCHMARK.read_text()
        setup = SETUP.read_text()
        runner = (PROJECT_ROOT / "scripts" / "run_deepgraphgo_arc.py").read_text()

        self.assertIn("github.com/yourh/DeepGraphGO.git", setup)
        self.assertIn("efdb1cb9425f4f48e4613c0a89e603f5542bcb19", setup)
        self.assertIn("--runtime-smoke-test", setup)
        self.assertIn('"pytorch=1.6.0" cpuonly', setup)
        self.assertIn("query_identifiers_synthetic", runner)
        self.assertIn("DGGQ{index:06d}", runner)
        self.assertIn('CURRENT_METHOD=deepgraphgo', benchmark)
        self.assertIn('--method deepgraphgo', benchmark)
        self.assertIn('stage 10d_deepgraphgo run_deepgraphgo', benchmark)

    def test_heal_downloads_the_esm1b_contact_regression_sidecar(self):
        """load_model_and_alphabet_local reads it even with return_contacts=False."""
        setup = SETUP.read_text()
        benchmark = BENCHMARK.read_text()

        self.assertIn("esm1b_t33_650M_UR50S-contact-regression.pt", setup)
        self.assertIn("HEAL_ESM1B_REGRESSION_URL", setup)
        self.assertIn("fair-esm/regression/", setup)
        self.assertIn('${HEAL_ESM1B_MODEL%.pt}-contact-regression.pt', benchmark)

    def test_deepgraphgo_uses_a_split_zip_reader_and_not_dtrx(self):
        """data.zip is the last member of a 7-volume Info-ZIP spanned archive."""
        setup = SETUP.read_text()

        self.assertIn("extract_split_zip", setup)
        self.assertIn("p7zip", setup)
        self.assertIn("7z x", setup)
        # dtrx and unzip both reject spanned archives, and `zip -s 0` silently
        # truncates them; none may come back as the extraction path. Prose
        # explaining that is fine, so match invocation and installation only.
        self.assertNotIn("dtrx -f", setup)
        self.assertNotIn('"dtrx==', setup)
        self.assertNotIn("zip -q -s 0", setup)
        # torch 1.6.0 imports `future` but does not declare it.
        self.assertIn('"future==0.18.3"', setup)
        self.assertIn("ppi_mat.npz", setup)

    def test_benchmark_supports_cpu_only_and_prediction_only_runs(self):
        benchmark = BENCHMARK.read_text()
        cpu_job = PROJECT_ROOT / "arc slurms" / "run_benchmark_cpu.slurm"

        self.assertIn("DGG_BENCHMARK_SKIP_EVALUATION", benchmark)
        self.assertIn("gpu_required()", benchmark)
        self.assertIn("DGG_BENCHMARK_REQUIRE_GPU", benchmark)
        self.assertIn("DGG_BENCHMARK_REQUIRE_METHODS", benchmark)
        self.assertTrue(cpu_job.is_file())
        self.assertIn("DGG_BENCHMARK_REQUIRE_GPU", cpu_job.read_text())
        # DeepGraphGO is pinned to a CPU-only PyTorch build, so it must not be
        # in the set that forces a GPU allocation.
        gpu_line = next(
            line for line in benchmark.splitlines() if line.startswith("GPU_METHODS=")
        )
        self.assertIn("heal", gpu_line)
        self.assertIn("gat_go", gpu_line)
        self.assertNotIn("deepgraphgo", gpu_line)

    def test_gat_go_can_report_coverage_without_consuming_gpu_time(self):
        runner = (PROJECT_ROOT / "scripts" / "run_gat_go_arc.py").read_text()
        setup = SETUP.read_text()

        self.assertIn("--report-coverage-only", runner)
        self.assertIn("report_coverage_only", runner)
        self.assertIn("--report-coverage-only", setup)
        # The strict path must survive: a normal run still refuses partial
        # coverage rather than scoring a subset silently.
        self.assertIn("GAT-GO feature audit failed for", runner)

    def test_deepgoplus_and_deepgose_are_withheld_from_figures(self):
        plot = PLOT.read_text()
        stratified = (PROJECT_ROOT / "src" / "plot_stratified.py").read_text()

        self.assertIn("EXCLUDED_FROM_PLOTS", plot)
        self.assertIn('frozenset({"deepgoplus", "deepgose"})', plot)
        external = plot.split("EXTERNAL_PRETRAINED_METHODS = (")[1].split(")")[0]
        self.assertNotIn("deepgoplus", external)
        self.assertNotIn("deepgose", external)
        focus = stratified.split("FOCUS_METHODS = [")[1].split("]")[0]
        self.assertNotIn("deepgoplus", focus)
        self.assertNotIn("deepgose", focus)
        for method in ("heal", "gat_go", "deepgraphgo"):
            self.assertIn(method, focus)

    def test_pin_repo_always_forces_checkout(self):
        """A SLURM job killed mid-checkout leaves HEAD correct but the
        working tree incomplete; pin_repo must repair that even when HEAD
        already equals the pinned revision, not only on the branch that
        moves HEAD."""
        setup = SETUP.read_text()
        pin_repo_body = setup.split("pin_repo() {")[1].split("\n}\n")[0]
        self.assertIn("checkout -f -- .", pin_repo_body)
        # That line must sit after the if-block that only fires when HEAD
        # needs to move, i.e. outside it, so it always executes.
        if_block_end = pin_repo_body.index("fi")
        force_checkout_pos = pin_repo_body.index("checkout -f -- .")
        self.assertGreater(force_checkout_pos, if_block_end)

    def test_gat_go_layout_is_normalized_after_flat_gdown_download(self):
        """gdown --folder mirrors the Drive folder's actual (flat) layout;
        the checkpoint and go2index.pt do not land under trained_models/ or
        data/data_splits/ on their own."""
        setup = SETUP.read_text()
        self.assertIn("normalize_gat_go_layout()", setup)
        self.assertIn("normalize_gat_go_layout \"${GATGO_ROOT}\"", setup)

    def test_gat_go_normalizes_before_deciding_whether_to_redownload(self):
        """GATGO_DATA_URL is a single personal Google Drive link Google
        rate-limits on repeated access. A previous run's files sitting flat
        on disk (not yet relocated) must be normalized *before* the
        if-check that decides whether to invoke gdown, or every retry
        re-triggers a ~25.6 GB download into that quota wall even when
        nothing new actually needs fetching."""
        setup_body = SETUP.read_text()
        gat_go_fn = setup_body.split("setup_gat_go() {")[1].split("\n}\n")[0]
        first_normalize_pos = gat_go_fn.index("normalize_gat_go_layout")
        download_decision_pos = gat_go_fn.index(
            'if [[ ! -s "${GATGO_ROOT}/trained_models/GAT-GO_modelweights.pt"'
        )
        self.assertLess(
            first_normalize_pos, download_decision_pos,
            "normalize_gat_go_layout must run before the download-needed check",
        )

    def test_deepgraphgo_is_excluded_from_the_submission_chain(self):
        """No pretrained DeepGraphGO checkpoint exists upstream (verified
        against the pinned commit: no models/ directory, no GitHub Release).
        main.py trains from scratch; nothing here does that, so it must not
        be submitted until that is a scoped, separate task."""
        submit = (PROJECT_ROOT / "arc slurms" / "submit_new_baselines.sh").read_text()
        # A comment explaining the exclusion is fine; actually requesting the
        # method (setup methods or benchmark methods) is not.
        code = submit.split("# Layout and why:")[1]
        self.assertNotIn("deepgraphgo", code)
        # The wiring itself must stay intact for when training is added.
        self.assertIn("setup_deepgraphgo()", SETUP.read_text())

    def test_gat_go_is_excluded_from_the_submission_chain(self):
        """The official precomputed features cover only 68/1508 (4.5%) of
        the locked ARC query set (confirmed by
        --report-coverage-only during setup); the runner's strict preflight
        correctly refuses that as full coverage, so submitting it would just
        fail the GPU job. Removed here per user decision; the state-dict
        remap and all other GAT-GO wiring stay in place."""
        submit = (PROJECT_ROOT / "arc slurms" / "submit_new_baselines.sh").read_text()
        code = submit.split("# Layout and why:")[1]
        self.assertNotIn("gat_go", code)
        # The wiring itself must stay intact for whenever fuller feature
        # coverage becomes available.
        self.assertIn("setup_gat_go()", SETUP.read_text())
        self.assertIn(
            "def remap_legacy_gat_go_state_dict",
            (PROJECT_ROOT / "scripts" / "run_gat_go_arc.py").read_text(),
        )

    def test_trapezoid_is_resolved_compatibly_with_numpy_1_and_2(self):
        """numpy 2.0 renamed trapz -> trapezoid. ARC's env has numpy < 2, so
        calling np.trapezoid directly crashed the entire stratified figure
        build there while passing locally on numpy >= 2."""
        for relative in ("src/plot_stratified.py", "src/export_supplementary_tables.py"):
            source = (PROJECT_ROOT / relative).read_text()
            self.assertNotIn("np.trapezoid(", source, relative)
            self.assertIn('getattr(np, "trapezoid", None)', source, relative)
            self.assertIn("np.trapz", source, relative)

    def test_complete_figure_tree_is_built_after_scoring(self):
        benchmark = BENCHMARK.read_text()
        figures = ALL_FIGURES.read_text()
        self.assertIn("run_all_figures.slurm", benchmark)
        for renderer in ("make_figures.py", "plot_arc_tuning.py", "plot_reviewer_figures.py",
                         "plot_benchmark.py", "plot_stratified.py", "export_supplementary_tables.py"):
            self.assertIn(renderer, figures)
        self.assertNotIn("src/plot_stratified.py", benchmark)
        self.assertNotIn("src/export_supplementary_tables.py", benchmark)
        self.assertNotIn("benchmark_cli plot", benchmark)

    def test_heal_has_external_pretrained_provenance(self):
        """Without this, comparison_audit() falls through to 'requires
        manual review', which is wrong provenance for a published table."""
        source = (PROJECT_ROOT / "src" / "export_supplementary_tables.py").read_text()
        external = source.split("EXTERNAL_PRETRAINED = {")[1].split("}")[0]
        self.assertIn("heal", external)

    def test_heal_computes_esm_once_per_protein_not_once_per_ontology(self):
        """ESM-1b (650M params) dominates HEAL's GPU cost and does not depend
        on the ontology. The loop must be protein-outer/task-inner so the
        encoder runs once per protein rather than three times."""
        runner = (PROJECT_ROOT / "scripts" / "run_heal_arc.py").read_text()
        body = runner.split("def main(")[1]
        # The structure loop must enclose the ESM call, and the per-ontology
        # head evaluation must sit inside it, after that single ESM call.
        structure_loop = body.index("for index, structure_path in enumerate(structures")
        esm_call = body.index("esm_model(", structure_loop)
        head_call = body.index("models[task](batch)", structure_loop)
        self.assertLess(structure_loop, esm_call)
        self.assertLess(esm_call, head_call)
        # Exactly one ESM invocation in the whole inference path.
        self.assertEqual(body.count("esm_model("), 1)

    def test_exclusion_set_is_consistent_across_every_output_path(self):
        """DeepGOPlus/DeepGO-SE kept reappearing in the top-level CAFA and
        coverage figures because those are a separate, older code path that
        never consulted the exclusion. Every module that emits a figure or a
        published table must agree."""
        canonical = {"deepgoplus", "deepgose"}
        plot = PLOT.read_text()
        self.assertIn('frozenset({"deepgoplus", "deepgose"})', plot)
        evaluate = (PROJECT_ROOT / "src" / "benchmark" / "evaluate.py").read_text()
        self.assertIn('excluded = {"deepgoplus", "deepgose"}', evaluate)
        supp = (PROJECT_ROOT / "src" / "export_supplementary_tables.py").read_text()
        self.assertIn('EXCLUDED_FROM_TABLES = {"deepgoplus", "deepgose"}', supp)
        # And the figures must actually apply it, not merely define it.
        self.assertIn("available = set(results.method) - excluded", evaluate)
        self.assertIn("drop_excluded(metrics)", supp)

    def test_supplementary_table_s7_is_not_bypassed_by_a_different_column_name(self):
        """S7 reads paired_differences_vs_deepgreengo.csv, whose method column
        is named 'competitor', not 'method' - unlike every other supplementary
        table. drop_excluded() silently no-ops on a column it doesn't find, so
        calling it without column='competitor' here would let DeepGOPlus/
        DeepGO-SE straight through even with every other table clean."""
        supp = (PROJECT_ROOT / "src" / "export_supplementary_tables.py").read_text()
        s7_block = supp.split('paired_path = results / "paired_differences_vs_deepgreengo.csv"')[1]
        s7_block = s7_block.split("write_table(paired,")[0]
        self.assertIn('column="competitor"', s7_block)
        self.assertIn("drop_excluded(pd.read_csv(paired_path)", s7_block)

    def test_top_level_figures_label_every_method_and_emit_svg(self):
        """A method missing from the label map silently fell back to its raw
        key, which is how HEAL was rendered lowercase as 'heal'."""
        evaluate = (PROJECT_ROOT / "src" / "benchmark" / "evaluate.py").read_text()
        for method, label in (
            ("heal", "HEAL"), ("gat_go", "GAT-GO"), ("deepgraphgo", "DeepGraphGO"),
            ("dpfunc", "DPFunc"), ("blast", "BLAST"), ("diamond", "DIAMOND"),
        ):
            self.assertIn(f'"{method}": "{label}"', evaluate)
        # Vector output for the manuscript.
        self.assertIn('for suffix in ("svg", "pdf", "png"):', evaluate)
        self.assertNotIn('for suffix in ("png", "pdf"):', evaluate)

    def test_micro_aupr_gets_a_confidence_interval(self):
        """Only Fmax and Smin had CI columns written, so the Micro AUPR panel
        was drawn as bare bars with no uncertainty."""
        evaluate = (PROJECT_ROOT / "src" / "benchmark" / "evaluate.py").read_text()
        self.assertIn('for metric in ("micro_aupr", "macro_aupr"):', evaluate)
        self.assertIn('row[f"{metric}_ci_low"]', evaluate)
        self.assertIn('row[f"{metric}_ci_high"]', evaluate)
        # The errorbar branch must no longer be restricted to fmax/smin.
        self.assertNotIn('if metric in ("cafa_fmax", "cafa_smin"):', evaluate)

    def test_no_directional_guidance_text_in_benchmark_figures(self):
        for relative in (
            "src/benchmark/evaluate.py",
            "src/plot_baselines_only.py",
            "src/plot_stratified.py",
        ):
            source = (PROJECT_ROOT / relative).read_text()
            self.assertNotIn("higher is better", source, relative)
            self.assertNotIn("lower is better", source, relative)
            self.assertNotIn("higher values indicate", source, relative)

    def test_stratification_covers_the_graph_based_methods(self):
        """run_stratified's METHODS is an allowlist; a method absent from it is
        silently never stratified, which is why HEAL was missing from
        plots/stratified despite completing successfully."""
        source = (PROJECT_ROOT / "src" / "benchmark" / "run_stratified.py").read_text()
        methods_block = source.split("METHODS = [")[1].split("]")[0]
        for method in ("heal", "gat_go", "deepgraphgo"):
            self.assertIn(f'"{method}"', methods_block)
        # And the stratified figure must be willing to draw it.
        focus = (PROJECT_ROOT / "src" / "plot_stratified.py").read_text()
        focus_block = focus.split("FOCUS_METHODS = [")[1].split("]")[0]
        self.assertIn('"heal"', focus_block)

    def test_struct2go_is_removed_and_plot_registers_new_methods(self):
        benchmark = BENCHMARK.read_text()
        setup = SETUP.read_text()
        source = PLOT.read_text()

        self.assertNotIn("struct2go", benchmark.lower())
        self.assertNotIn("struct2go", setup.lower())
        self.assertNotIn('"struct2go":', source.lower())
        self.assertIn('"heal": "HEAL (PDB-only)"', source)
        self.assertIn('"gat_go": "GAT-GO"', source)
        self.assertIn('"deepgraphgo": "DeepGraphGO"', source)
