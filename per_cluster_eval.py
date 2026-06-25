import torch
from torch_geometric.loader import DataLoader
import numpy as np
import argparse
import pickle
import pandas as pd
import os
import __main__
from preprocessing.create_batch_dataset import PDB_Dataset
__main__.PDB_Dataset = PDB_Dataset

from model import get_model
from evals import get_micro_fmax, get_auroc, compute_ic, get_smin


def evaluate_per_cluster(model, loader, device, cluster_mapping, ic):
    model.eval()
    all_preds = []
    all_labels = []
    all_prots = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred_probs = torch.sigmoid(out)

            all_preds.append(pred_probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
            # data.u is stored as a plain string scalar by PyG's Data.to()
            if isinstance(data.u, (list, tuple)):
                all_prots.extend(data.u)
            else:
                all_prots.append(data.u)

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)

    # Group by cluster
    cluster_results = {}

    for i, prot in enumerate(all_prots):
        cid = cluster_mapping.get(prot, "Unknown")
        if cid not in cluster_results:
            cluster_results[cid] = {'true': [], 'pred': []}
        cluster_results[cid]['true'].append(y_true[i])
        cluster_results[cid]['pred'].append(y_pred[i])

    # Calculate metrics per cluster
    results = []
    for cid, cdata in cluster_results.items():
        ctrue = np.vstack(cdata['true'])
        cpred = np.vstack(cdata['pred'])
        size = len(cdata['true'])

        # Only evaluate clusters with at least some positive labels
        if np.sum(ctrue) > 0 and size > 1:
            try:
                fmax = get_micro_fmax(ctrue, cpred)
                auroc = get_auroc(ctrue, cpred, average='micro')
                smin = get_smin(ctrue, cpred, ic)
            except Exception as e:
                print(f"Cluster {cid}: metric error — {e}")
                continue

            results.append({
                'Cluster_ID': cid,
                'Size': size,
                'Micro_Fmax': fmax,
                'Micro_AUROC': auroc,
                'Smin': smin
            })

    df = pd.DataFrame(results)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument('--model', type=str, default=None,
                        help="Architecture: GCN | GAT | MLP | Hybrid. "
                             "Auto-detected from path if omitted.")
    parser.add_argument('--ontology', type=str, default='biological_process',
                        choices=['molecular_function', 'biological_process',
                                 'cellular_component'])
    parser.add_argument('--dataset_path', type=str,
                        default='preprocessing/data/split_files/datasets.pkl')
    parser.add_argument('--cluster_map', type=str,
                        default='preprocessing/data/split_files/cluster_mapping.pkl')
    parser.add_argument('--output', type=str,
                        default='runs/cluster_performance.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    print("Loading datasets...")
    with open(args.dataset_path, 'rb') as f:
        datasets = pickle.load(f)

    # Support new nested format {ontology: {train/valid/test}}
    # and old flat format {train/valid/test} for backwards compatibility
    if args.ontology in datasets:
        train_dataset = datasets[args.ontology]['train']
        test_dataset  = datasets[args.ontology]['test']
    elif 'train' in datasets:
        print(f"  [warn] Pickle is old flat format; patching ontology to '{args.ontology}'.")
        train_dataset = datasets['train']
        test_dataset  = datasets['test']
        for ds in (train_dataset, test_dataset):
            ds.selected_ontology = args.ontology
            ds.y_labels = ds.goterms[args.ontology]
    else:
        raise KeyError(f"Pickle does not contain ontology '{args.ontology}'. "
                       f"Available keys: {list(datasets.keys())}")

    # Compute IC from training labels
    print("Computing IC from training set...")
    all_train_labels = []
    for data in train_dataset:
        all_train_labels.append(data.y.numpy())
    all_train_labels = np.vstack(all_train_labels)
    ic = compute_ic(all_train_labels)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Loading cluster mapping...")
    with open(args.cluster_map, 'rb') as f:
        cluster_mapping = pickle.load(f)

    # Infer architecture from path if not provided
    arch = args.model
    if arch is None:
        arch = "Hybrid"
        for m in ["GCN", "GAT", "MLP", "Hybrid"]:
            if m in os.path.basename(args.model_path):
                arch = m
                break

    input_size   = len(train_dataset[0].x[0])
    hidden_sizes = [1024, 912]
    output_size  = train_dataset.num_classes

    print(f"Loading {arch} model from {args.model_path}...")
    model = get_model(arch, input_size, hidden_sizes, output_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device,
                                     weights_only=False))
    model.to(device)

    print("Evaluating per cluster...")
    df = evaluate_per_cluster(model, test_loader, device, cluster_mapping, ic)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved cluster performance to {args.output}")
    print(df.describe())


if __name__ == "__main__":
    main()
