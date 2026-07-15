def test_color_collision():
    import plot_sota_comparison as psc
    current_deepfri_seq = psc.PALETTE.get("DeepFRI_Seq", psc.PALETTE.get("DeepFRI", "#888888"))
    current_deepfri_cmap = psc.PALETTE.get("DeepFRI_Cmap", psc.PALETTE.get("DeepFRI", "#888888"))
    
    print(f"DeepFRI_Seq color: {current_deepfri_seq}")
    print(f"DeepFRI_Cmap color: {current_deepfri_cmap}")
    
    if current_deepfri_seq == current_deepfri_cmap:
        print("✅ Bug confirmed: Colors are IDENTICAL (fallback to DeepFRI or gray).")
        return False
    return True

if __name__ == '__main__':
    test_color_collision()
