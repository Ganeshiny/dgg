import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import argparse
import pickle
import json
import os
import sys
import platform
import csv

from model import get_model, HybridGNN
from evals import evaluate_all, compute_ic
import __main__
from preprocessing.create_batch_dataset import PDB_Dataset
__main__.PDB_Dataset = PDB_Dataset
from focal_loss import FocalLoss
from utils import load_alpha_weights

def parse_args():
    parser = argparse.ArgumentParser(description="DeepGreenGO Training Script")
    parser.add_argument('--model', type=str, default='Hybrid', choices=['GCN', 'GAT', 'Hybrid', 'MLP'], help='Model architecture to use.')
    parser.add_argument('--loss', type=str, default='Focal', choices=['BCE', 'Focal'], help='Loss function to use.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability.')
    parser.add_argument('--focal_gamma', type=float, default=4.0, help='Gamma for Focal Loss.')
    parser.add_argument('--hidden_sizes', type=str, default='1024,912', help='Comma-separated hidden sizes.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads (for GAT/Hybrid).')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps.')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience.')
    parser.add_argument('--dataset_path', type=str, default='preprocessing/data/split_files/datasets.pkl', help='Path to pre-split datasets.')
    parser.add_argument('--ontology', type=str, default='biological_process', choices=['molecular_function', 'biological_process', 'cellular_component'], help='GO ontology to train on.')
    parser.add_argument('--output_dir', type=str, default='runs/', help='Directory to save logs and models.')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Required alongside deterministic=True

def load_datasets(path, ontology):
    print(f"Loading datasets from {path} for {ontology}...")
    with open(path, 'rb') as f:
        datasets = pickle.load(f)

    # New format: datasets[ontology]['train'/'valid'/'test']
    # Support both the new nested format and the old flat format for backwards compatibility.
    if ontology in datasets:
        # New nested format — each ontology has its own pre-built dataset objects
        ont_datasets = datasets[ontology]
        return ont_datasets['train'], ont_datasets['valid'], ont_datasets['test']
    elif 'train' in datasets:
        # Old flat format (biological_process only) — patch ontology labels at runtime
        print(f"  [warn] Pickle uses old flat format; patching ontology to '{ontology}'.")
        datasets['train'].selected_ontology = ontology
        datasets['valid'].selected_ontology = ontology
        datasets['test'].selected_ontology  = ontology
        datasets['train'].y_labels = datasets['train'].goterms[ontology]
        datasets['valid'].y_labels = datasets['valid'].goterms[ontology]
        datasets['test'].y_labels  = datasets['test'].goterms[ontology]
        return datasets['train'], datasets['valid'], datasets['test']
    else:
        raise KeyError(
            f"Pickle at '{path}' does not contain ontology '{ontology}' "
            f"and is not in the expected flat format. "
            f"Available keys: {list(datasets.keys())}"
        )

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Gather hardware info
    hardware_info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'gpu_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        hardware_info['gpu_name'] = torch.cuda.get_device_name(0)

    # Gather reproducibility info
    import subprocess
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        git_commit = 'unknown'

    reproducibility_info = {
        'git_commit': git_commit,
        'command_line': ' '.join(sys.argv)
    }

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Abbrev ontology
    ont_short = {'molecular_function': 'mf', 'biological_process': 'bp', 'cellular_component': 'cc'}[args.ontology]
    
    # If tuning is enabled or lr/dropout are non-default, include them in the run name to avoid collisions
    run_name = f"{ont_short}_{args.model}_{args.loss}_s{args.seed}"
    # In case we're doing hyperparameter tuning, we might be writing to a different output_dir, 
    # but let's make sure the config is saved properly.
    run_dir = os.path.join(args.output_dir, run_name)
    
    # Append a unique suffix if the directory already exists (e.g. tuning runs)
    if os.path.exists(run_dir):
        import time
        run_dir = f"{run_dir}_{int(time.time())}"
        
    os.makedirs(run_dir, exist_ok=True)

    # Log hyperparameters
    config = vars(args)
    config['hardware'] = hardware_info
    config['reproducibility'] = reproducibility_info
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Load data
    train_dataset, valid_dataset, test_dataset = load_datasets(args.dataset_path, args.ontology)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Compute IC based on Train Set
    all_train_labels = []
    for data in train_dataset:
        all_train_labels.append(data.y.numpy())
    all_train_labels = np.vstack(all_train_labels)
    ic = compute_ic(all_train_labels)

    # Model Setup
    input_size = len(train_dataset[0].x[0])
    hidden_sizes = [int(h) for h in args.hidden_sizes.split(',')]
    output_size = train_dataset.num_classes

    # Handle model kwargs gracefully (only pass num_heads and dropout to Hybrid and GAT if supported)
    if args.model.lower() in ["hybrid", "deepgreengo", "rarelabelgnn"]:
        model = HybridGNN(input_size, hidden_sizes, output_size, num_attention_heads=args.num_heads, dropout=args.dropout)
    elif args.model.lower() == "gat":
        # Our current GATModel doesn't accept dropout in its constructor in model.py, but we could pass num_heads
        model = get_model(args.model, input_size, hidden_sizes, output_size)
    else:
        model = get_model(args.model, input_size, hidden_sizes, output_size)
    model.to(device)

    # Loss Setup
    if args.loss == 'Focal':
        CLASS_WEIGHT_PATH = "model_and_weight_files/alpha_weights.pkl"
        try:
            alpha = load_alpha_weights(CLASS_WEIGHT_PATH)
            criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)
        except FileNotFoundError:
            print(f"Alpha weights not found. Using default FocalLoss (alpha=0.25, gamma={args.focal_gamma}).")
            criterion = FocalLoss(alpha=0.25, gamma=args.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_valid_fmax = 0
    epochs_no_improve = 0

    def evaluate(loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                pred_probs = torch.sigmoid(out)
                all_preds.append(pred_probs.cpu().numpy())
                all_labels.append(data.y.cpu().numpy())
        
        y_true = np.vstack(all_labels)
        y_pred = np.vstack(all_preds)
        metrics = evaluate_all(y_true, y_pred, ic)
        return metrics, y_true, y_pred

    log_file = open(os.path.join(run_dir, 'training_log.csv'), 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Epoch', 'Train_Loss', 'Valid_Micro_Fmax', 'Valid_Macro_Fmax', 'Valid_Smin', 'Valid_Macro_AUROC'])

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        
        for i, data in enumerate(train_loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.float())
            loss = loss / args.accumulation_steps
            loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * args.accumulation_steps

        train_loss = total_loss / len(train_loader)

        # Evaluate
        valid_metrics, val_y_true, val_y_pred = evaluate(valid_loader)
        micro_fmax = valid_metrics['Micro_Fmax']
        macro_fmax = valid_metrics['Macro_Fmax']
        smin = valid_metrics['Smin']
        macro_auroc = valid_metrics['Macro_AUROC']
        
        print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Val MiFmax: {micro_fmax:.4f} | Val MaFmax: {macro_fmax:.4f} | Val Smin: {smin:.4f}")
        log_writer.writerow([epoch, train_loss, micro_fmax, macro_fmax, smin, macro_auroc])
        log_file.flush()

        scheduler.step(macro_fmax)

        if macro_fmax > best_valid_fmax:
            best_valid_fmax = macro_fmax
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
            
            with open(os.path.join(run_dir, 'valid_metrics.json'), 'w') as f:
                json.dump(valid_metrics, f, indent=4)
            print("  --> New best model saved!")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    log_file.close()

    print("Training finished. Evaluating on Test Set...")
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth'),
                                     map_location=device, weights_only=False))
    test_metrics, test_y_true, test_y_pred = evaluate(test_loader)
    
    print("\n--- TEST METRICS ---")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
        
    # Save raw predictions for post-hoc plotting (PR/ROC curves)
    np.save(os.path.join(run_dir, 'test_y_true.npy'), test_y_true)
    np.save(os.path.join(run_dir, 'test_y_pred.npy'), test_y_pred)

if __name__ == "__main__":
    main()
