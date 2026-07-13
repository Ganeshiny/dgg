import pickle
import os
import sys

class DummyDataset:
    pass

class _Up(pickle.Unpickler):
    def find_class(self, m, n):
        if m == 'numpy._core.multiarray':
            m = 'numpy.core.multiarray'
        if m == 'torch.utils.data.dataset' or n == 'Dataset':
            return DummyDataset
        if n == 'PDB_Dataset':
            return DummyDataset
        return super().find_class(m, n)

sys.modules['preprocessing'] = type('module', (), {})()
sys.modules['preprocessing.create_batch_dataset'] = type('module', (), {'PDB_Dataset': DummyDataset})()

with open('preprocessing/data/split_files/datasets.pkl', 'rb') as f:
    ds = _Up(f).load()

train_pkl = set(ds['biological_process']['train'].pdb_split_list)
test_pkl = set(ds['biological_process']['test'].pdb_split_list)
valid_pkl = set(ds['biological_process']['valid'].pdb_split_list)

with open('preprocessing/data/split_files/_train.txt') as f: train_txt = set([x.strip() for x in f])
with open('preprocessing/data/split_files/_test.txt') as f: test_txt = set([x.strip() for x in f])
with open('preprocessing/data/split_files/_valid.txt') as f: valid_txt = set([x.strip() for x in f])

print(f'Train PKL: {len(train_pkl)}, Train TXT: {len(train_txt)}, Match: {len(train_pkl.intersection(train_txt))}')
print(f'Test PKL: {len(test_pkl)}, Test TXT: {len(test_txt)}, Match: {len(test_pkl.intersection(test_txt))}')
print(f'Valid PKL: {len(valid_pkl)}, Valid TXT: {len(valid_txt)}, Match: {len(valid_pkl.intersection(valid_txt))}')
