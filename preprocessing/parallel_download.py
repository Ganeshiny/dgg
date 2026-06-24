import os
import sys
import requests
import concurrent.futures

def download_cif(pdb_id, output_dir):
    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
    out_path = os.path.join(output_dir, f"{pdb_id}.cif.gz")
    if os.path.exists(out_path):
        return True # already downloaded
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download {pdb_id}: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {pdb_id}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parallel_download.py <csv_file> <output_dir>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_file, 'r') as f:
        content = f.read().strip()
    
    pdb_ids = [token.strip() for token in content.split(',') if token.strip()]
    print(f"Found {len(pdb_ids)} PDB IDs to download.")

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(download_cif, pdb_id, output_dir): pdb_id for pdb_id in pdb_ids}
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1
            if success_count % 100 == 0:
                print(f"Downloaded {success_count}/{len(pdb_ids)}...")

    print(f"Finished downloading. Success: {success_count}/{len(pdb_ids)}")
