import torch
import pickle
from pathlib import Path
import os 
from Bio.PDB.MMCIFParser import MMCIFParser
import gzip
from Bio.SeqUtils import seq1
import obonet
import numpy as np
import argparse
import glob
import multiprocessing
import csv

def calculate_class_weights(dataset, device):
    # Calculate the number of classes in the dataset
    num_classes = dataset[0].y.size(1)
    print("Number of classes:", num_classes)

    # Initialize class counters
    class_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)


    # Count the number of examples in each class
    for data in dataset:
        class_counts += data.y.sum(dim=0).float().to(device)
        #print(class_counts)
        

    # Calculate class weights by taking the inverse of class frequency
    class_weights = 1.0 / (class_counts / class_counts.sum())
    print(class_weights)
    return class_weights.to(device)

def save_alpha_weights(alpha, filename):
    with open(filename, 'wb') as f:
        pickle.dump(alpha, f)
    print(f'Alpha weights saved to {filename}')

def load_alpha_weights(filename):
    with open(filename, 'rb') as f:
        alpha_weights = pickle.load(f)
    return alpha_weights

def get_seqs(fname):
    with gzip.open(fname, "rt") as handle:
        parser = MMCIFParser()
        pdb_id = os.path.split(fname)[1].split(".")[0] 
        structure = parser.get_structure(pdb_id, handle)
        chains = {f"{pdb_id}_{chain.id}":seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}
    return chains

def write_seqs_from_cifdir(dirpath, fname):
    structure_dir = Path(dirpath)
    seqs_file = open(fname, "w")
    for file in structure_dir.glob("*"):
        chain_dir = get_seqs(file)
        for key in chain_dir:
            #unknown_percentage = chain_dir[key].count("X")/len(chain_dir[key])
            #print(f"seq:{chain_dir[key]}, percentage:{unknown_percentage}")
            #if unknown_percentage <= 0.2:
            seqs_file.write(f">{key}\n{chain_dir[key]}\n")
    return seqs_file

def read_seqs_file(seqs_file):
    pdb2seq = {}
    with open(seqs_file, "r") as fasta_handle:
        for line in fasta_handle:
            if ">" in line:
                key  = line.strip().replace(">", "")
            else:
                unknown_percentage = line.strip().count("X")/len(line.strip())
                if unknown_percentage <= 0.2:
                    pdb2seq[key] = line.strip() 
                #else:
                    #print(f"X character percentage of {pdb2seq[key]} is: ", unknown_percentage)
    return pdb2seq

def load_go_graph(fname):
    go_graph = obonet.read_obo(fname)
    #print(f"DEBUG: {go_graph}, and the number of nodes: {len(go_graph.nodes)}")
    return go_graph

# Global constants for structure preprocessing to prevent training/inference mismatches
CA_DISTANCE_THRESHOLD = 10.0
PLDDT_THRESHOLD = 70.0
