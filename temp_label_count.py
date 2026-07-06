import pickle

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        return super().find_class(module, name)

import __main__
from preprocessing.create_batch_dataset import PDB_Dataset
__main__.PDB_Dataset = PDB_Dataset

with open('preprocessing/data/split_files/datasets.pkl', 'rb') as f:
    d = RenameUnpickler(f).load()

print('BP Labels:', len(d['biological_process']['test'].y_labels))
print('MF Labels:', len(d['molecular_function']['test'].y_labels))
print('CC Labels:', len(d['cellular_component']['test'].y_labels))
