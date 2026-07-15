import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
def test_bar_position_calculation():
    MODEL_SET = ["Hybrid_JK", "Hybrid", "BLAST", "DIAMOND", "Naive"]
    FULL_MODEL_ORDER = ["Hybrid", "TransFun", "DeepFRI_Seq", "DeepFRI_Cmap"]
    
    # Actually, the code loops over MODEL_ORDER directly.
    import plot_sota_comparison as psc
    n_models = len(psc.MODEL_ORDER)
    bar_width = 0.15
    
    # Buggy:
    buggy_positions = {}
    for mi, mname in enumerate(psc.MODEL_ORDER):
        if mname not in MODEL_SET:
            continue
        offset = (mi - n_models/2 + 0.5) * bar_width
        buggy_positions[mname] = round(offset, 4)
    
    # Correct:
    plotted_models = [m for m in psc.MODEL_ORDER if m in MODEL_SET]
    n_plotted = len(plotted_models)
    correct_positions = {}
    for mi, mname in enumerate(plotted_models):
        offset = (mi - n_plotted/2 + 0.5) * bar_width
        correct_positions[mname] = round(offset, 4)
        
    print("Buggy offsets:", buggy_positions)
    print("Correct offsets:", correct_positions)
    
    if buggy_positions != correct_positions:
        print("✅ Bug confirmed: Bar offsets are calculated using absolute indices with gaps.")
    else:
        print("❌ Bug not found here.")

if __name__ == '__main__':
    test_bar_position_calculation()
