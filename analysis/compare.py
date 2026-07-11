import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_comparison_reports(base_log_dir="logs", output_dir="reports/comparison"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Define baselines
    baselines = {
        "Continuous": os.path.join(base_log_dir, "continuous", "metrics.csv"),
        "Gumbel-Softmax": os.path.join(base_log_dir, "gumbel", "metrics.csv"),
        "REINFORCE": os.path.join(base_log_dir, "reinforce", "metrics.csv")
    }
    
    data = {}
    for name, path in baselines.items():
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
        else:
            print(f"Warning: Data for {name} not found at {path}")

    if not data:
        print("No baseline data found to compare.")
        return

    # Metrics to compare
    metrics_to_plot = [
        ("val/mse", "MSE vs Epoch (Validation)"),
        ("val/psnr", "PSNR vs Epoch (Validation)"),
        ("val/reward_total", "Total Reward vs Epoch (Validation)"),
        ("val_comm/vocab_entropy", "Vocabulary Entropy vs Epoch"),
        ("val_comm/vocab_utilization", "Vocabulary Utilization vs Epoch"),
        ("loss/reconstruction", "Reconstruction Loss vs Step (Training)")
    ]

    for metric, title in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        for name, df in data.items():
            if metric in df.columns:
                # For validation metrics, drop NAs to just plot the eval steps
                plot_data = df[['step', metric]].dropna()
                if not plot_data.empty:
                    plt.plot(plot_data['step'], plot_data[metric], label=name, marker='o' if 'val' in metric else None)
        
        plt.title(title)
        plt.xlabel("Global Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Safe filename
        safe_name = metric.replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"{safe_name}.png"))
        plt.close()
        
    print(f"Comparison reports generated in {output_dir}")

if __name__ == "__main__":
    generate_comparison_reports()
