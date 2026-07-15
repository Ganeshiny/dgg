import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import sys

def fix_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out = []
    in_plot_C = False
    in_plot_F = False
    in_plot_BDE = False
    unindent = False
    
    for i, line in enumerate(lines):
        if line.startswith("def plot_C_ic_bins"):
            in_plot_C = True; in_plot_F = False; in_plot_BDE = False
        elif line.startswith("def plot_F_depth_bins"):
            in_plot_C = False; in_plot_F = True; in_plot_BDE = False
        elif line.startswith("def plot_BDE_pr_curves"):
            in_plot_C = False; in_plot_F = False; in_plot_BDE = True
        elif line.startswith("def plot_G_coverage"):
            in_plot_C = False; in_plot_F = False; in_plot_BDE = False; unindent = False
        elif line.startswith("def plot_summary_fmax"):
            in_plot_C = False; in_plot_F = False; in_plot_BDE = False; unindent = False

        if in_plot_C and "x_positions = np.arange(len(bins))" in line:
            unindent = True
        if in_plot_F and "fig, ax = plt.subplots(figsize=(15, 5))" in line or "fig, axes = plt.subplots" in line:
            # wait, in plot_F, let's just trigger unindent on the first thing after if mask
            pass
        if in_plot_F and "x_positions = np.arange(len(bins))" in line:
            unindent = True
        if in_plot_BDE and "fig, ax = plt.subplots" in line:
            unindent = True

        # Stop unindenting if we drop back to the outer loop level (4 spaces)
        if unindent and len(line) - len(line.lstrip()) <= 4 and line.strip() != "":
            unindent = False

        # Add ic masking to plot_C
        if in_plot_C and "y_true = y_true[mask]" in line:
            out.append(line)
            out.append("            ic = ic[mask]\n")
            continue

        if in_plot_BDE and "for mname in MODEL_ORDER_COVERAGE:" in line:
            line = line.replace("MODEL_ORDER_COVERAGE", "MODEL_ORDER")

        if "local_order = MODEL_ORDER_PERFORMANCE" in line:
            line = line.replace("MODEL_ORDER_PERFORMANCE", "MODEL_ORDER")

        if "* >0.6 bin is memorization regime" in line:
            line = line.replace("* >0.6 bin is memorization regime\\n(random split)", "* >0.6 bin: elevated identity due to MMseqs2\\nbilateral coverage limitation (see Methods)")

        # Perform unindent
        if unindent:
            if line.startswith("            "):
                line = line[4:]
            elif line.startswith("                "):
                line = line[4:]
            elif line.startswith("                    "):
                line = line[4:]

        out.append(line)

    content = "".join(out)
    content = content.replace("['Hybrid', 'Hybrid_JK', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']", "['Hybrid_JK', 'Hybrid', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

fix_file("plot_sota_comparison.py")
print("Patch successful!")
