import os
import subprocess
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_experiment(model, loss, seed=12345):
    print(f"=== Running Experiment: Model={model}, Loss={loss} ===")
    cmd = [
        "python", "train.py",
        "--model", model,
        "--loss", loss,
        "--epochs", "50",
        "--batch_size", "32",
        "--lr", "1e-4",
        "--seed", str(seed)
    ]
    subprocess.run(cmd, check=True)

def collect_results(runs_dir="runs/"):
    results = []
    for run_name in os.listdir(runs_dir):
        metrics_file = os.path.join(runs_dir, run_name, "test_metrics.json")
        config_file = os.path.join(runs_dir, run_name, "config.json")
        if os.path.exists(metrics_file) and os.path.exists(config_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            res = {
                "Model": config["model"],
                "Loss": config["loss"],
                "Seed": config["seed"],
                **metrics
            }
            results.append(res)
    return pd.DataFrame(results)

def plot_results(df, output_dir="plots/"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    metrics_to_plot = ["Fmax", "Macro_AUROC", "Micro_AUROC", "Smin"]
    
    for metric in metrics_to_plot:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y=metric, hue="Loss", capsize=.1)
        plt.title(f"{metric} Comparison across Models and Losses")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"))
        plt.close()
        
    print(f"Plots saved to {output_dir}")

def main():
    models = ["MLP", "GCN", "GAT", "Hybrid"]
    losses = ["BCE", "Focal"]
    
    # Run all combinations
    for model in models:
        for loss in losses:
            # We skip some combinations if we want, but for full ablation we do all
            # Try/catch to continue if one fails (e.g., out of memory)
            try:
                run_experiment(model, loss)
            except subprocess.CalledProcessError as e:
                print(f"Experiment {model}-{loss} failed with error: {e}")
                
    # Collect and plot
    df = collect_results()
    print("=== Collected Results ===")
    print(df.to_string())
    
    # Save raw results
    df.to_csv("runs/all_results.csv", index=False)
    
    plot_results(df)

if __name__ == "__main__":
    main()
