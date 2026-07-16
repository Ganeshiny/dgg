import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

def plot_random_cmap():
    cmap_dir = "preprocessing/data_arc_rebuild_2026_07_14/structure_files/tmp_cmap_files"
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
    seqres = data['seqres']
    
    print(seqres)
    
    # Setup the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Contact Map for {prot_id}", fontsize=16)
    
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
    
    plt.tight_layout()
    
    # Save the plot
    out_file = f"cmap_checks/cmap_plot_{prot_id}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    print(f"Plot saved to: {out_file}")

if __name__ == "__main__":
    plot_random_cmap()