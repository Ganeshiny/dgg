import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

def plot_random_cmap():
    cmap_dir = "preprocessing/data/structure_files/tmp_cmap_files"
    cmap_files = glob.glob(os.path.join(cmap_dir, "*.npz"))
    
    if not cmap_files:
        print(f"No cmap files found in {cmap_dir}. Please make sure cmaps have been generated.")
        return
        
    # Select a random cmap file
    random_file = random.choice(cmap_files)
    prot_id = os.path.basename(random_file).replace('.npz', '')
    print(f"Selected random contact map: {prot_id}")
    
    # Load data
    data = np.load(random_file, allow_pickle=True)
    c_alpha = data['C_alpha']
    c_beta = data['C_beta']
    plddt = data['plddt']
    seqres = data['seqres']
    
    print(f"Sequence length: {len(plddt)}")
    
    # Setup the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Contact Map & pLDDT Profile for {prot_id}", fontsize=16)
    
    # Plot C-alpha
    im1 = axes[0].imshow(c_alpha, cmap='viridis_r', vmin=0, vmax=20)
    axes[0].set_title("C-alpha Distances (Å)")
    axes[0].set_xlabel("Residue Index")
    axes[0].set_ylabel("Residue Index")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot C-beta
    im2 = axes[1].imshow(c_beta, cmap='viridis_r', vmin=0, vmax=20)
    axes[1].set_title("C-beta Distances (Å)")
    axes[1].set_xlabel("Residue Index")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot pLDDT / B-factors
    axes[2].plot(plddt, color='blue', linewidth=2)
    axes[2].set_title("pLDDT / B-factor Scores")
    axes[2].set_xlabel("Residue Index")
    axes[2].set_ylabel("Score")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    # For AlphaFold, pLDDT > 70 is good, > 90 is great. For PDB B-factors, lower is better.
    # Add a horizontal line at 50 if the scores are in the 0-100 range (suggesting AlphaFold)
    if np.max(plddt) <= 100 and np.min(plddt) >= 0:
        axes[2].axhline(y=50, color='red', linestyle='--', label='Masking Threshold (50)')
        axes[2].set_ylim(0, 100)
        axes[2].legend()
    
    plt.tight_layout()
    
    # Save the plot
    out_file = f"cmap_plot_{prot_id}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    print(f"Plot saved to: {out_file}")

if __name__ == "__main__":
    plot_random_cmap()
