import pandas as pd
from pathlib import Path

# 1. Load protein sequences and compute their lengths
def load_fasta_lengths(fasta_path):
    lengths = {}
    with open(fasta_path, 'r') as f:
        current_id = None
        current_len = 0
        for line in f:
            if line.startswith('>'):
                if current_id:
                    lengths[current_id] = current_len
                current_id = line[1:].strip().split()[0]
                current_len = 0
            else:
                current_len += len(line.strip())
        if current_id:
            lengths[current_id] = current_len
    return lengths

train_lengths = load_fasta_lengths('preprocessing/data/train_sequences.fasta')
test_lengths = load_fasta_lengths('preprocessing/data/test_sequences.fasta')

# 2. Load the raw BLAST results
try:
    blast = pd.read_csv('preprocessing/data/blast_identity_raw.tsv', 
                        sep='\t', header=None,
                        names=['qseqid', 'sseqid', 'pident', 'length'])
except FileNotFoundError:
    print("Error: blast_identity_raw.tsv not found.")
    exit(1)

# Group by test protein (qseqid) to find the top match
idx_max = blast.groupby('qseqid')['pident'].idxmax()
top_hits = blast.loc[idx_max].copy()

# Filter for the high-identity pairs (>60%)
high_identity = top_hits[top_hits['pident'] > 60.0].copy()
total_high = len(high_identity)

# 3. Compute length ratios
def get_length_ratio(row):
    q_len = test_lengths.get(row['qseqid'], 0)
    s_len = train_lengths.get(row['sseqid'], 0)
    
    if q_len == 0 or s_len == 0:
        return 1.0 # Ignore if missing
        
    shorter = min(q_len, s_len)
    longer = max(q_len, s_len)
    return shorter / longer

high_identity['length_ratio'] = high_identity.apply(get_length_ratio, axis=1)

# 4. Count the trap vs genuine leakage
# MMseqs2 used cov=0.8, so if ratio < 0.8, MMseqs *could not* cluster them.
trap_applies = high_identity[high_identity['length_ratio'] < 0.8]
genuine_leak = high_identity[high_identity['length_ratio'] >= 0.8]

print(f"Total High-Identity Test Proteins (>60%): {total_high}")
print(f"Explained by Multi-Domain Trap (Length Ratio < 0.8): {len(trap_applies)} ({(len(trap_applies)/total_high)*100:.1f}%)")
print(f"Genuine Leakage/Clustering Failure (Length Ratio >= 0.8): {len(genuine_leak)} ({(len(genuine_leak)/total_high)*100:.1f}%)")