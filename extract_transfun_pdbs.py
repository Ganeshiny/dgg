import os
import gemmi

def get_test_set_ids(split_file):
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def convert_cif_gz_to_pdb_chain(cif_gz_path, pdb_path, chain_id):
    try:
        # Read the CIF structure
        doc = gemmi.cif.read_file(cif_gz_path)
        block = doc[-1]
        st = gemmi.make_structure_from_block(block)
        
        # Keep only the target chain and rename it to 'A'
        for model in st:
            # Delete from end to avoid index shifting
            for i in range(len(model)-1, -1, -1):
                if model[i].name != chain_id:
                    del model[i]
                else:
                    model[i].name = 'A'
            
        # Write to PDB
        st.write_pdb(pdb_path)
        return True
    except Exception as e:
        print(f"Error converting {cif_gz_path}: {e}")
        return False

def main():
    test_split_file = 'preprocessing/data/split_files/_test.txt'
    struct_dir = 'preprocessing/data/structure_files'
    out_dir = 'SOTA/TransFun/test_pdbs'
    
    os.makedirs(out_dir, exist_ok=True)
    
    test_ids = get_test_set_ids(test_split_file)
    print(f"Found {len(test_ids)} proteins in test set.")
    
    success_count = 0
    missing_count = 0
    error_count = 0
    
    for pid in test_ids:
        if '_' in pid:
            parts = pid.split('_')
            pdb_id = parts[0]
            chain_id = parts[1]
        else:
            pdb_id = pid
            chain_id = 'A'
            
        cif_gz = os.path.join(struct_dir, f"{pdb_id}.cif.gz")
        if not os.path.exists(cif_gz):
            cif_gz = os.path.join(struct_dir, f"{pdb_id}.cif") 
        
        if not os.path.exists(cif_gz):
            cif_gz = os.path.join(struct_dir, f"AF-{pid}-F1-model_v4.cif.gz")
            if not os.path.exists(cif_gz):
                cif_gz = os.path.join(struct_dir, f"{pid}.pdb")
                if not os.path.exists(cif_gz):
                    print(f"Warning: Structure for {pid} not found!")
                    missing_count += 1
                    continue
            
        pdb_out = os.path.join(out_dir, f"{pid}.pdb")
        if os.path.exists(pdb_out):
            success_count += 1
            continue
            
        if cif_gz.endswith('.gz'):
            res = convert_cif_gz_to_pdb_chain(cif_gz, pdb_out, chain_id)
        elif cif_gz.endswith('.pdb'):
            import shutil
            shutil.copy(cif_gz, pdb_out)
            res = True
        else:
            res = convert_cif_gz_to_pdb_chain(cif_gz, pdb_out, chain_id)
                
        if res:
            success_count += 1
        else:
            error_count += 1
            
        if (success_count + error_count + missing_count) % 500 == 0:
            print(f"Processed {success_count + error_count + missing_count} / {len(test_ids)}...")
            
    print(f"\nDone. Success: {success_count}, Missing: {missing_count}, Errors: {error_count}")

if __name__ == '__main__':
    main()
