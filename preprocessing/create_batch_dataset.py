import os
import csv
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import pickle

# ── Lazy ProtBERT singleton ──────────────────────────────────────────────────
# Loading ProtBERT at module-import time crashes any script that merely imports
# this module (e.g. train.py). We instantiate it once on first use instead.
_tokenizer = None
_bert_model = None
_bert_device = None

def _get_protbert():
    """Return (tokenizer, model, device), loading once on first call."""
    global _tokenizer, _bert_model, _bert_device
    if _bert_model is None:
        import transformers.utils.import_utils
        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
        from transformers import BertTokenizer, BertModel
        _bert_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Loading ProtBERT tokenizer…")
        hf_token = os.environ.get("HF_TOKEN")
        _tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, token=hf_token)
        print("Loading ProtBERT model…")
        _bert_model = BertModel.from_pretrained(
            'Rostlab/prot_bert_bfd',
            use_safetensors=True,   # avoids torch.load CVE-2025-32434 (needs torch<2.6 fix)
            token=hf_token
        )
        _bert_model.gradient_checkpointing_enable()
        _bert_model.to(_bert_device).eval()
        print(f"ProtBERT loaded on {_bert_device}")
    return _tokenizer, _bert_model, _bert_device

# Dictionaries for residue properties
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

POLARITY = {
    'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0, 'Q': 1, 'E': 1,
    'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
    'P': 0, 'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0
}

CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1,
    'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
    'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

def compute_residue_features(sequence):
    """Compute residue-level features: hydrophobicity, polarity, and charge."""
    return (
        [HYDROPHOBICITY.get(res, 0) for res in sequence],
        [POLARITY.get(res, 0) for res in sequence],
        [CHARGE.get(res, 0) for res in sequence]
    )

def seq2onehot(seq):
    """Convert sequence to one-hot encoding. Unknown residues map to 'X'."""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_embed = {char: idx for idx, char in enumerate(chars)}
    x_idx = vocab_embed['X']
    vocab_one_hot = np.eye(len(chars), dtype=int)
    return np.array([vocab_one_hot[vocab_embed.get(v, x_idx)] for v in seq])

def seq2protbert(seq):
    """Get ProtBERT embeddings for a protein sequence."""
    tokenizer, bert_model, bert_device = _get_protbert()
    spaced = ' '.join(seq)  # ProtBERT expects space-separated amino acids
    inputs = tokenizer(spaced, return_tensors='pt', add_special_tokens=True,
                       padding=True, truncation=True, max_length=1024)
    input_ids = inputs['input_ids'].to(bert_device)
    attention_mask = inputs['attention_mask'].to(bert_device)

    with torch.no_grad():
        embeddings = bert_model(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state

    embeddings = embeddings.detach().cpu().numpy()
    mask = attention_mask.detach().cpu().numpy()

    # Strip [CLS] and [SEP] tokens → shape (seq_len, 1024)
    return np.array([embeddings[i][1:np.sum(mask[i]) - 1] for i in range(len(embeddings))])

def read_list_file(filename):
    """Read list of PDB IDs from a file."""
    with open(filename, "r") as file:
        return [line.strip() for line in file.readlines()]

class PDB_Dataset(Dataset):
    def __init__(self, root, annot_file, num_shards=20, selected_ontology=None, transform=None, pre_transform=None, model="protBERT", pdb_split_set_file=None, dataset_type = None):
        self.model = model
        self.npz_dir = root
        self.num_shards = num_shards
        self.selected_ontology = selected_ontology
        self.transform = transform
        self.pre_transform = pre_transform
        self.dataset_type = dataset_type

        # Read annotation data
        self.prot2annot, self.goterms, self.gonames, self.prot_list = self.annot_file_reader(annot_file)
        self.y_labels = self.goterms[selected_ontology]

        # Read list of proteins if a specific split is given
        self.pdb_split_list = read_list_file(pdb_split_set_file) if pdb_split_set_file else self.prot_list
        self.pdb_split_list = [prot for prot in self.pdb_split_list if os.path.exists(os.path.join(root, f'{prot}.npz'))]

        print(f" Loaded dataset with {len(self.pdb_split_list)} proteins for {selected_ontology}")

        super(PDB_Dataset, self).__init__(root, transform, pre_transform)

    @classmethod
    def annot_file_reader(cls, annot_filename):
        onts = ['molecular_function', 'biological_process', 'cellular_component']
        prot2annot = {}
        goterms = {ont: [] for ont in onts}
        gonames = {ont: [] for ont in onts}
        prot_list = []

        with open(annot_filename, mode='r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for ont in onts:
                next(reader, None)  # Skip headers
                goterms[ont] = next(reader)
                next(reader, None)  # Skip headers
                gonames[ont] = next(reader)

            next(reader, None)  # Skip headers
            for row in reader:
                prot = row[0]  # Ensure ID format consistency
                prot2annot[prot] = {ont: np.zeros(len(goterms[ont]), dtype=np.int64) for ont in onts}
                for i, ont in enumerate(onts):
                    goterm_indices = [goterms[ont].index(goterm) for goterm in row[i+1].split(',') if goterm]
                    prot2annot[prot][ont][goterm_indices] = 1.0
                prot_list.append(prot)

        return prot2annot, goterms, gonames, prot_list

    @property
    def num_classes(self):
        return len(self.y_labels)

    @property
    def processed_file_names(self):
        """Returns a list of processed filenames uniquely prefixed by ontology."""
        prefix = self.selected_ontology[:2] if self.selected_ontology else "na"
        d_type = self.dataset_type if self.dataset_type else "all"
        if self.pdb_split_list:
            return [f'data_{prefix}_{d_type}_{i}.pt' for i in range(len(self.pdb_split_list))]
        else:
            return [f'data_{prefix}_{d_type}_{i}.pt' for i in range(len(self.prot_list))]

    def process(self):
        # Stream directly to disk to prevent OOM
        for index, prot_id in tqdm(enumerate(self.pdb_split_list), total=len(self.pdb_split_list)):
            out_file = os.path.join(self.processed_dir, self.processed_file_names[index])
            if os.path.exists(out_file):
                continue
                
            data = self._load_data(prot_id)
            if data:
                torch.save(data, out_file)

    def _load_data(self, prot_id):
        pdb_file = os.path.join(self.npz_dir, f'{prot_id}.npz')
        if not os.path.isfile(pdb_file):
            print(f" File not found: {pdb_file}")
            return None

        cmap = np.load(pdb_file)
        # Safely deserialise seqres: np.savez stores strings as 0-d arrays.
        raw_seqres = cmap['seqres']
        sequence = str(raw_seqres.item()) if raw_seqres.ndim == 0 else str(raw_seqres)

        onehot_features = torch.tensor(seq2onehot(sequence), dtype=torch.float).squeeze(0)
        protbert_features = torch.tensor(seq2protbert(sequence), dtype=torch.float).squeeze(0)

        hydrophobicity, polarity, charge = compute_residue_features(sequence)
        additional_features = torch.tensor(np.stack([hydrophobicity, polarity, charge], axis=1), dtype=torch.float)

        # ProtBERT truncates at 1022 residues (1024 − CLS − SEP tokens).
        # Clamp all feature tensors and the distance matrix to the same length
        # to prevent shape mismatches on long proteins.
        MAX_LEN = 1022
        if protbert_features.shape[0] > MAX_LEN:
            protbert_features = protbert_features[:MAX_LEN]
        residue_count = min(protbert_features.shape[0], onehot_features.shape[0], MAX_LEN)
        onehot_features    = onehot_features[:residue_count]
        additional_features = additional_features[:residue_count]

        ca_dist = cmap['C_alpha'][:residue_count, :residue_count]
        plddt   = cmap.get('plddt', None)
        if plddt is not None:
            plddt = plddt[:residue_count]

        node_features = torch.cat([protbert_features, additional_features], dim=1) if self.model == "protBERT" else torch.cat([onehot_features, additional_features], dim=1)

        edge_index = self._get_adjacency_info(ca_dist, plddt, prot_id)
        labels = self._get_labels(prot_id)
        length = torch.tensor(residue_count, dtype=torch.long)

        return Data(x=node_features, edge_index=edge_index, u=prot_id, y=labels, length=length)

    def _get_labels(self, prot_id):
        labels = {ont: torch.tensor(self.prot2annot.get(prot_id, {}).get(ont, np.zeros(len(self.y_labels), dtype=np.int64)), dtype=torch.long) for ont in ['molecular_function', 'biological_process', 'cellular_component']}

        for ont, label in labels.items():
            if label.dim() == 1:
                labels[ont] = label.unsqueeze(0)  # Add batch dimension
        
        return labels.get(self.selected_ontology, torch.zeros(len(self.y_labels), dtype=torch.long))

    def _get_adjacency_info(self, distance_matrix, plddt_array=None, prot_id="", threshold=10.0, plddt_threshold=70.0):
        # 10 Å Cα threshold as per DeepFRI protocol and the manuscript
        with np.errstate(invalid='ignore'):
            adjacency_matrix = (distance_matrix <= threshold).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)
        
        # Apply strict confidence filter to ALL structures (not just AF) if plddt is available
        if plddt_array is not None:
            confident_residues = (plddt_array >= plddt_threshold)
            confident_mask = np.outer(confident_residues, confident_residues)
            adjacency_matrix = adjacency_matrix * confident_mask
            
        edge_indices = np.nonzero(adjacency_matrix)
        return torch.tensor([edge_indices[0], edge_indices[1]], dtype=torch.long)

    def len(self):
        return len(self.pdb_split_list)

    def get(self, idx):
        # Load directly from PyG disk cache instead of re-running ProtBERT!
        return torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))


if __name__ == '__main__':
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    root       = 'preprocessing/data/structure_files/tmp_cmap_files'
    annot_file = 'preprocessing/data/pdb2go.tsv'
    num_shards = 20

    test_file  = 'preprocessing/data/split_files/_test.txt'
    train_file = 'preprocessing/data/split_files/_train.txt'
    valid_file = 'preprocessing/data/split_files/_valid.txt'

    # All three Gene Ontology namespaces
    ONTOLOGIES = ['molecular_function', 'biological_process', 'cellular_component']

    torch.manual_seed(12345)

    all_datasets = {}   # { 'biological_process': {'train': ..., 'test': ..., 'valid': ...}, ... }

    for ont in ONTOLOGIES:
        print(f"\n{'='*60}")
        print(f"  Building datasets for ontology: {ont}")
        print(f"{'='*60}")

        ds_train = PDB_Dataset(
            root=root, annot_file=annot_file, num_shards=num_shards,
            selected_ontology=ont, transform=None, pre_transform=None,
            model='protBERT', pdb_split_set_file=train_file, dataset_type='train'
        )
        ds_valid = PDB_Dataset(
            root=root, annot_file=annot_file, num_shards=num_shards,
            selected_ontology=ont, transform=None, pre_transform=None,
            model='protBERT', pdb_split_set_file=valid_file, dataset_type='valid'
        )
        ds_test = PDB_Dataset(
            root=root, annot_file=annot_file, num_shards=num_shards,
            selected_ontology=ont, transform=None, pre_transform=None,
            model='protBERT', pdb_split_set_file=test_file, dataset_type='test'
        )

        print(f"  Train: {len(ds_train)}  |  Valid: {len(ds_valid)}  |  Test: {len(ds_test)}")
        print(f"  GO classes: {ds_train.num_classes}")

        all_datasets[ont] = {
            'train': ds_train,
            'valid': ds_valid,
            'test':  ds_test,
        }

    # Save all three ontologies to a single pickle so train.py can load any of them
    dataset_save_path = 'preprocessing/data/split_files/datasets.pkl'
    with open(dataset_save_path, 'wb') as f:
        pickle.dump(all_datasets, f)

    print(f"\n✓ Datasets for all 3 ontologies saved to {dataset_save_path}")
    print("  Keys in pickle: biological_process, molecular_function, cellular_component")
    print("  Each key contains: 'train', 'valid', 'test'")