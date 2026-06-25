import os
import subprocess
import argparse
import numpy as np
from tqdm import tqdm

def create_deepfri_npz(orig_npz_path, new_npz_path):
    """
    DeepGreenGO saves npz with seqres (from FASTA) and C_alpha (from PDB).
    Often len(seqres) != C_alpha.shape[0].
    DeepFRI requires len(seqres) == C_alpha.shape[0].
    This function creates a new npz where seqres is truncated to match C_alpha.
    """
    data = np.load(orig_npz_path)
    c_alpha = data['C_alpha']
    seqres_raw = data['seqres']
    seq_str = str(seqres_raw.item()) if seqres_raw.ndim == 0 else str(seqres_raw)
    
    # Truncate sequence to match contact map size
    trunc_seq = seq_str[:c_alpha.shape[0]]
    
    # Save new npz with only the keys DeepFRI needs: C_alpha, C_beta, seqres
    save_dict = {
        'C_alpha': c_alpha,
        'seqres': trunc_seq
    }
    if 'C_beta' in data:
        save_dict['C_beta'] = data['C_beta']
        
    np.savez_compressed(new_npz_path, **save_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_list', default='preprocessing/data/split_files/_test.txt')
    parser.add_argument('-test_fasta', default='preprocessing/data/split_files/_test_sequences.fasta')
    parser.add_argument('-cmap_dir', default='preprocessing/data/structure_files/tmp_cmap_files')
    parser.add_argument('-out_dir', default='baselines/deepfri_results')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    deepfri_dir = os.path.join(project_root, 'baselines', 'DeepFRI')
    out_dir = os.path.join(project_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    test_list = os.path.join(project_root, args.test_list)
    test_fasta = os.path.join(project_root, args.test_fasta)
    cmap_dir = os.path.join(project_root, args.cmap_dir)

    # 1. Create catalogue CSV and DeepFRI-compatible NPZ files for Contact Maps
    catalogue_csv = os.path.join(out_dir, "test_cmap_catalogue.csv")
    deepfri_npz_dir = os.path.join(out_dir, "deepfri_cmaps")
    os.makedirs(deepfri_npz_dir, exist_ok=True)
    
    with open(test_list) as f:
        test_prots = [line.strip() for line in f if line.strip()]

    print("Generating DeepFRI-compatible contact maps...")
    with open(catalogue_csv, 'w') as f:
        for prot in tqdm(test_prots):
            orig_npz_path = os.path.join(cmap_dir, f"{prot}.npz")
            if os.path.exists(orig_npz_path):
                new_npz_path = os.path.join(deepfri_npz_dir, f"{prot}.npz")
                create_deepfri_npz(orig_npz_path, new_npz_path)
                f.write(f"{prot},{new_npz_path}\n")

    # 2. Run DeepFRI
    ontologies = ['mf', 'bp', 'cc']

    for mode in ['seq', 'cmap']:
        for ont in ontologies:
            out_prefix = os.path.join(out_dir, f"deepfri_{mode}_{ont.upper()}")
            print(f"=== Running DeepFRI {mode.upper()} mode for {ont.upper()} ===")
            
            cmd = [
                "conda", "run", "-n", "deepfri",
                "python", "predict.py",
                "-ont", ont,
                "-o", out_prefix
            ]
            if mode == 'seq':
                cmd.extend(["--fasta_fn", test_fasta])
            else:
                cmd.extend(["--cmap_csv", catalogue_csv])
                
            subprocess.run(cmd, cwd=deepfri_dir, check=True)

    print(f"DeepFRI predictions saved to {out_dir}")

if __name__ == "__main__":
    main()
