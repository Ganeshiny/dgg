import os
import re

with open("plot_sota_comparison.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Add --mode to parse_args inside main()
# Original code:
#     parser.add_argument('--common_subset', action='store_true')
#     parser.add_argument('--supplementary', action='store_true', help="Generate supplementary figures with Hybrid_JK")
#     args = parser.parse_args()

new_args = """    parser.add_argument('--common_subset', action='store_true')
    parser.add_argument('--supplementary', action='store_true', help="Generate supplementary figures with Hybrid_JK")
    parser.add_argument('--mode', type=str, default='dl_only', choices=['dl_only', 'baselines_only', 'all'])
    args = parser.parse_args()
    
    global OUT_DIR
    if args.mode != 'dl_only':
        OUT_DIR = os.path.join(PROJECT_DIR, f'plots_sota_comparison_{args.mode}')
    else:
        OUT_DIR = os.path.join(PROJECT_DIR, 'plots_sota_comparison')
    os.makedirs(OUT_DIR, exist_ok=True)
"""
code = code.replace("""    parser.add_argument('--common_subset', action='store_true')
    parser.add_argument('--supplementary', action='store_true', help="Generate supplementary figures with Hybrid_JK")
    args = parser.parse_args()""", new_args)

# 2. Fix MODEL_ORDER setting inside main()
# Original code:
#     if args.supplementary:
#         MODEL_ORDER = MODEL_ORDER_SUPPLEMENTARY
#         MODEL_ORDER_COVERAGE = ['Hybrid', 'Hybrid_JK', 'DPFunc', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']
#     else:
#         MODEL_ORDER = MODEL_ORDER_PERFORMANCE
#         MODEL_ORDER_COVERAGE = ['Hybrid', 'DPFunc', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']

new_model_order = """    if args.mode == 'baselines_only':
        MODEL_ORDER = ['Hybrid_JK', 'Hybrid', 'BLAST', 'DIAMOND', 'Naive']
    elif args.mode == 'all':
        MODEL_ORDER = ['Hybrid_JK', 'Hybrid', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap', 'BLAST', 'DIAMOND', 'Naive']
    else:
        if args.supplementary:
            MODEL_ORDER = MODEL_ORDER_SUPPLEMENTARY
        else:
            MODEL_ORDER = MODEL_ORDER_PERFORMANCE

    if args.supplementary:
        MODEL_ORDER_COVERAGE = ['Hybrid', 'Hybrid_JK', 'DPFunc', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']
    else:
        MODEL_ORDER_COVERAGE = ['Hybrid', 'DPFunc', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']
"""
code = code.replace("""    if args.supplementary:
        MODEL_ORDER = MODEL_ORDER_SUPPLEMENTARY
        MODEL_ORDER_COVERAGE = ['Hybrid', 'Hybrid_JK', 'DPFunc', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']
    else:
        MODEL_ORDER = MODEL_ORDER_PERFORMANCE
        MODEL_ORDER_COVERAGE = ['Hybrid', 'DPFunc', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']""", new_model_order)

# 3. Fix plot_C_ic_bins `ic = ic[mask]` bug
# Original code:
#         if mask is not None:
#             y_true = y_true[mask]
#             ic = ic[mask]
#             # Depending on context, ic or prot_identity or ic_bins may need masking.
#             # It's better to mask them directly after this call.
#
#         active_bins = [b for b in bins if ((ic >= b[0]) & (ic < b[1])).sum() > 0]

fix_c = """        if mask is not None:
            y_true = y_true[mask]
            ic = compute_ic(y_true)
            
        active_bins = [b for b in bins if ((ic >= b[0]) & (ic < b[1])).sum() > 0]"""
code = code.replace("""        if mask is not None:
            y_true = y_true[mask]
            ic = ic[mask]
            # Depending on context, ic or prot_identity or ic_bins may need masking.
            # It's better to mask them directly after this call.

        active_bins = [b for b in bins if ((ic >= b[0]) & (ic < b[1])).sum() > 0]""", fix_c)


# 4. Fix plot_F_depth_bins `term_depths` bug
# Original code:
#         preds, mask = load_predictions(ont_full, ont_short, prot_list, goterms, valid_mask=valid_mask)
#         if mask is not None:
#             y_true = y_true[mask]
#
#         active_bins = [b for b in bins if len([i for i, t in enumerate(goterms) if b[0] <= depths.get(t, 0) < b[1]]) > 0]

fix_f = """        preds, mask = load_predictions(ont_full, ont_short, prot_list, goterms, valid_mask=valid_mask)
        if mask is not None:
            y_true = y_true[mask]

        term_depths = np.array([depths.get(t, 0) for t in goterms])
        active_bins = [b for b in bins if len([i for i, t in enumerate(goterms) if b[0] <= depths.get(t, 0) < b[1]]) > 0]"""

code = code.replace("""        preds, mask = load_predictions(ont_full, ont_short, prot_list, goterms, valid_mask=valid_mask)
        if mask is not None:
            y_true = y_true[mask]

        active_bins = [b for b in bins if len([i for i, t in enumerate(goterms) if b[0] <= depths.get(t, 0) < b[1]]) > 0]""", fix_f)

with open("plot_sota_comparison.py", "w", encoding="utf-8") as f:
    f.write(code)
